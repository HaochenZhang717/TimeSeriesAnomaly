import torch
import wandb
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

class FlowTSPretrain(object):
    def __init__(
            self,optimizer, model, train_loader,
            val_loader, max_epochs, device, save_dir,
            wandb_project_name, wandb_run_name, grad_clip_norm,
            early_stop
    ):
        self.optimizer = optimizer
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.device = device
        self.save_dir = save_dir
        self.wandb_project_name = wandb_project_name
        self.wandb_run_name = wandb_run_name
        self.grad_clip_norm = grad_clip_norm
        self.early_stop = early_stop

    def pretrain(self, config):

        # 初始化 wandb
        wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_run_name,
            config=config,
        )

        os.makedirs(self.save_dir, exist_ok=True)

        best_val_loss = float("inf")
        no_improve_epochs = 0
        global_steps = 0

        for epoch in range(self.max_epochs):
            total_loss = 0
            tr_seen = 0
            self.model.train()
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):

                X_normal = batch["orig_signal"].to(self.device)

                self.optimizer.zero_grad()
                loss = self.model(X_normal, anomaly_label=None)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                total_loss += loss.item()
                tr_seen += 1
                global_steps += 1
                # wandb.log({
                #     "train/step_total_loss": loss.item(),
                #     "lr": self.optimizer.param_groups[0]["lr"],
                #     "step": global_steps
                # })
            train_total_avg = total_loss / tr_seen


            """evaluation"""
            self.model.eval()
            with torch.no_grad():
                val_total, val_seen =  0, 0
                for batch in self.val_loader:
                    X_normal = batch["orig_signal"].to(self.device)
                    loss = self.model(X_normal, anomaly_label=None)
                    val_total += loss.item()
                    val_seen += 1

                val_total /= val_seen

                # 记录到 wandb
                wandb.log({
                    "train/total_loss": train_total_avg,
                    "val/total_loss": val_total,
                    "epoch": epoch,
                    "step": global_steps,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

                if val_total < best_val_loss:
                    best_val_loss = val_total
                    no_improve_epochs = 0
                    torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                else:
                    no_improve_epochs += 1

                if self.early_stop== "true" and no_improve_epochs >= 10:
                    print(f"⛔ Early stopping triggered at Step {global_steps}.")
                    break

        wandb.finish()


class FlowTSFinetune(object):
    def __init__(
            self,optimizer, model,
            train_normal_loader: DataLoader,
            val_normal_loader: DataLoader,
            train_anomaly_loader: DataLoader,
            val_anomaly_loader: DataLoader,
            max_iters, device, save_dir,
            wandb_project_name, wandb_run_name, grad_clip_norm,
            pretrained_ckpt, early_stop
    ):
        self.optimizer = optimizer
        self.model = model.to(device)

        self.train_normal_loader = train_normal_loader
        self.val_normal_loader = val_normal_loader

        self.train_anomaly_loader = train_anomaly_loader
        self.val_anomaly_loader = val_anomaly_loader

        self.max_iters = max_iters
        self.device = device
        self.save_dir = save_dir
        self.wandb_project_name = wandb_project_name
        self.wandb_run_name = wandb_run_name
        self.grad_clip_norm = grad_clip_norm
        self.pretrained_ckpt = pretrained_ckpt
        self.early_stop = early_stop

    def finetune(self, config, version, mode):
        if mode == "mixed_data":
            return self.finetune_mixed_data(config, version)
        elif mode == "anomaly_only":
            return self.finetune_anomaly_only(config, version)
        else:
            raise ValueError("mode must be 'mixed_data' or 'anomaly_only'")


    def finetune_mixed_data(self, config, version):

        self.model.prepare_for_finetune(ckpt_path=self.pretrained_ckpt, version=version)

        ema_decay = 0.999
        ema_state_dict = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

        wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_run_name,
            config=config,
        )

        os.makedirs(self.save_dir, exist_ok=True)

        best_val_loss = float("inf")
        no_improve_epochs = 0


        tr_loss_normal = 0
        tr_loss_anomaly = 0
        tr_loss_total = 0
        tr_seen = 0
        """train on a mixed dataset"""
        normal_train_iterator = iter(self.train_normal_loader)
        anomaly_train_iterator = iter(self.train_anomaly_loader)
        self.model.train()
        for step in tqdm(range(self.max_iters), desc=f"Training"):

            self.optimizer.zero_grad()
            normal_batch = next(normal_train_iterator)
            normal_signal = normal_batch["orig_signal"].to(self.device)
            normal_random_anomaly_label = normal_batch["random_anomaly_label"].to(device=self.device, dtype=torch.long)
            loss_on_normal = self.model.finetune_loss(normal_signal, normal_random_anomaly_label, mode="normal")

            anomaly_batch = next(anomaly_train_iterator)
            anomaly_signal = anomaly_batch["orig_signal"].to(self.device)
            anomaly_label = anomaly_batch["anomaly_label"].to(device=self.device, dtype=torch.long)
            anomaly_label = (anomaly_label > 0).to(dtype=torch.long)#for now, we use this

            loss_on_anomaly = self.model.finetune_loss(anomaly_signal, anomaly_label, mode="anomaly")

            total_loss = (loss_on_normal + loss_on_anomaly) * 0.5
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            # 🔥 Update EMA after optimizer.step()
            with torch.no_grad():
                model_state = self.model.state_dict()
                for key in model_state.keys():
                    ema_state_dict[key].mul_(ema_decay).add_(model_state[key], alpha=1-ema_decay)
            # -------------------------------

            tr_loss_total += total_loss.item()
            tr_loss_normal += loss_on_normal.item()
            tr_loss_anomaly += loss_on_anomaly.item()
            tr_seen += 1
            """evaluate every 250 steps"""
            if step % 250 == 0:
                # calculate and log training statistics
                tr_total_avg = tr_loss_total / tr_seen
                tr_normal_avg = tr_loss_normal / tr_seen
                tr_anomaly_avg = tr_loss_anomaly / tr_seen
                # run evaluation
                self.model.eval()

                val_loss_normal = 0
                val_seen = 0
                for normal_batch in self.val_normal_loader:
                    normal_signal = normal_batch["orig_signal"].to(self.device)
                    normal_random_anomaly_label = normal_batch["random_anomaly_label"].to(device=self.device, dtype=torch.long)
                    with torch.no_grad():
                        loss_on_normal = self.model.finetune_loss(
                            normal_signal,
                            normal_random_anomaly_label,
                            mode="normal"
                        )
                    bs = normal_signal.shape[0]
                    val_loss_normal += loss_on_normal.item() * bs
                    val_seen += bs
                val_loss_normal_avg = val_loss_normal / val_seen

                val_loss_anomaly = 0
                val_seen = 0
                for anomaly_batch in self.val_anomaly_loader:
                    anomaly_signal = anomaly_batch["orig_signal"].to(self.device)
                    anomaly_label = anomaly_batch["anomaly_label"].to(device=self.device, dtype=torch.long)
                    anomaly_label = (anomaly_label > 0).to(dtype=torch.long)  # for now, we use this
                    with torch.no_grad():
                        loss_on_anomaly = self.model.finetune_loss(anomaly_signal, anomaly_label, mode="anomaly")
                    bs = anomaly_signal.shape[0]
                    val_loss_anomaly += loss_on_anomaly.item() * bs
                    val_seen += bs
                val_loss_anomaly_avg = val_loss_anomaly / val_seen

                val_loss_total_avg = (val_loss_normal_avg + val_loss_anomaly_avg) * 0.5


                wandb.log({
                    "train/avg_loss_total": tr_total_avg,
                    "train/avg_loss_on_normal": tr_normal_avg,
                    "train/avg_loss_on_anomaly": tr_anomaly_avg,
                    "val/avg_loss_total": val_loss_total_avg,
                    "val/avg_loss_on_normal": val_loss_normal_avg,
                    "val/avg_loss_on_anomaly": val_loss_anomaly_avg,
                    "step": step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })


                self.model.train()
                # reset training statistics
                tr_loss_normal = 0
                tr_loss_anomaly = 0
                tr_loss_total = 0
                tr_seen = 0

                if self.early_stop== "true":
                    # save model and early stop
                    if val_loss_total_avg < best_val_loss:
                        best_val_loss = val_loss_total_avg
                        no_improve_epochs = 0
                        torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                    else:
                        no_improve_epochs += 1


                    if no_improve_epochs >= 10:
                        print(f"⛔ Early stopping triggered at Step {step}.")
                        break
                else:
                    torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                    torch.save(ema_state_dict, f"{self.save_dir}/ema_ckpt.pth")

        wandb.finish()
        return ema_state_dict


    def finetune_anomaly_only(self, config, version):

        self.model.prepare_for_finetune(ckpt_path=self.pretrained_ckpt, version=version)
        ema_decay = 0.999  # you can adjust
        ema_state_dict = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

        wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_run_name,
            config=config,
        )

        os.makedirs(self.save_dir, exist_ok=True)

        best_val_loss = float("inf")
        no_improve_epochs = 0


        tr_loss_anomaly = 0
        tr_loss_total = 0
        tr_seen = 0
        """train on a mixed dataset"""
        anomaly_train_iterator = iter(self.train_anomaly_loader)
        self.model.train()
        for step in tqdm(range(self.max_iters), desc=f"Training"):

            self.optimizer.zero_grad()
            anomaly_batch = next(anomaly_train_iterator)
            anomaly_signal = anomaly_batch["orig_signal"].to(self.device)
            anomaly_label = anomaly_batch["anomaly_label"].to(device=self.device, dtype=torch.long)
            anomaly_label = (anomaly_label > 0).to(dtype=torch.long)#for now, we use this

            loss_on_anomaly = self.model.finetune_loss(anomaly_signal, anomaly_label, mode="anomaly")

            total_loss = loss_on_anomaly
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            # 🔥 Update EMA after optimizer.step()
            with torch.no_grad():
                model_state = self.model.state_dict()
                for key in model_state.keys():
                    ema_state_dict[key].mul_(ema_decay).add_(model_state[key], alpha=1-ema_decay)
            # -------------------------------

            tr_loss_total += total_loss.item()
            tr_loss_anomaly += loss_on_anomaly.item()
            tr_seen += 1
            """evaluate every 250 steps"""
            if step % 250 == 0:
                # calculate and log training statistics
                tr_total_avg = tr_loss_total / tr_seen
                tr_anomaly_avg = tr_loss_anomaly / tr_seen
                # run evaluation
                self.model.eval()


                val_loss_anomaly = 0
                val_seen = 0
                for anomaly_batch in self.val_anomaly_loader:
                    anomaly_signal = anomaly_batch["orig_signal"].to(self.device)
                    anomaly_label = anomaly_batch["anomaly_label"].to(device=self.device, dtype=torch.long)
                    anomaly_label = (anomaly_label > 0).to(dtype=torch.long)  # for now, we use this
                    with torch.no_grad():
                        loss_on_anomaly = self.model.finetune_loss(anomaly_signal, anomaly_label, mode="anomaly")
                    bs = anomaly_signal.shape[0]
                    val_loss_anomaly += loss_on_anomaly.item() * bs
                    val_seen += bs
                val_loss_anomaly_avg = val_loss_anomaly / val_seen

                val_loss_total_avg = val_loss_anomaly_avg


                wandb.log({
                    "train/avg_loss_total": tr_total_avg,
                    # "train/avg_loss_on_normal": tr_normal_avg,
                    "train/avg_loss_on_anomaly": tr_anomaly_avg,
                    "val/avg_loss_total": val_loss_total_avg,
                    # "val/avg_loss_on_normal": val_loss_normal_avg,
                    "val/avg_loss_on_anomaly": val_loss_anomaly_avg,
                    "step": step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })


                self.model.train()
                # reset training statistics
                tr_loss_normal = 0
                tr_loss_anomaly = 0
                tr_loss_total = 0
                tr_seen = 0

                # save model and early stop
                if self.early_stop == "true":
                    if val_loss_total_avg < best_val_loss:
                        best_val_loss = val_loss_total_avg
                        no_improve_epochs = 0
                        torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                    else:
                        no_improve_epochs += 1

                    if no_improve_epochs >= 10:
                        print(f"⛔ Early stopping triggered at Step {step}.")
                        break
                else:
                    torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                    torch.save(ema_state_dict, f"{self.save_dir}/ema_ckpt.pth")


        wandb.finish()
        return ema_state_dict


class FlowTSTrainerTwoTogether(object):
    def __init__(
            self,optimizer, scheduler,
            model, train_loader,
            val_loader, max_epochs,
            device, save_dir,
            wandb_project_name, wandb_run_name,
            grad_clip_norm,
            grad_accum_steps, early_stop, patience,
            ae=None
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.device = device
        self.save_dir = save_dir
        self.wandb_project_name = wandb_project_name
        self.wandb_run_name = wandb_run_name
        self.grad_clip_norm = grad_clip_norm
        self.grad_accum_steps = grad_accum_steps
        self.early_stop = early_stop
        self.patience = patience


        self.ae=ae
        if self.ae is not None:
            self.ae.to(device)

    def unconditional_train(self, config):
        ema_decay = 0.999
        ema_state_dict = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

        wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_run_name,
            config=config,
        )

        os.makedirs(self.save_dir, exist_ok=True)

        best_val_loss = float("inf")
        no_improve_epochs = 0
        global_steps = 0
        model_dtype = next(self.model.parameters()).dtype
        for epoch in range(self.max_epochs):
            total_loss = 0
            tr_seen = 0
            self.model.train()
            self.optimizer.zero_grad()
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):

                X_normal = batch["orig_signal"].to(dtype=model_dtype, device=self.device)

                loss = self.model(X_normal, anomaly_label=None)
                backward_loss = loss / self.grad_accum_steps
                backward_loss.backward()

                global_steps += 1
                total_loss += loss.item()
                tr_seen += 1

                if global_steps % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # 🔥 Update EMA after optimizer.step()
                    with torch.no_grad():
                        model_state = self.model.state_dict()
                        for key in model_state.keys():
                            ema_state_dict[key].mul_(ema_decay).add_(model_state[key], alpha=1 - ema_decay)
                    # -------------------------------



            train_total_avg = total_loss / tr_seen


            """evaluation"""
            self.model.eval()
            with torch.no_grad():
                val_total, val_seen =  0, 0
                for batch in self.val_loader:
                    X_normal = batch["orig_signal"].to(dtype=model_dtype, device=self.device)
                    loss = self.model(X_normal, anomaly_label=None)
                    val_total += loss.item()
                    val_seen += 1

                val_total /= val_seen

                # 记录到 wandb
                wandb.log({
                    "train/total_loss": train_total_avg,
                    "val/total_loss": val_total,
                    "epoch": epoch,
                    "step": global_steps,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })
                if self.early_stop == "true":
                    if val_total < best_val_loss:
                        best_val_loss = val_total
                        no_improve_epochs = 0
                        torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                        torch.save(ema_state_dict, f"{self.save_dir}/ema_ckpt.pth")
                    else:
                        no_improve_epochs += 1

                    if no_improve_epochs >= self.patience:
                        print(f"⛔ Early stopping triggered at Step {global_steps}.")
                        break
                else:
                    torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                    torch.save(ema_state_dict, f"{self.save_dir}/ema_ckpt.pth")
            self.scheduler.step(val_total)
        wandb.finish()

    def conditional_train(self, config):
        ema_decay = 0.999
        ema_state_dict = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

        wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_run_name,
            config=config,
        )

        os.makedirs(self.save_dir, exist_ok=True)

        best_val_loss = float("inf")
        no_improve_epochs = 0
        global_steps = 0
        model_dtype = next(self.model.parameters()).dtype
        for epoch in range(self.max_epochs):
            total_loss = 0
            tr_seen = 0
            self.model.train()
            self.optimizer.zero_grad()
            for batch in tqdm(self.train_loader, desc=f"Train Epoch {epoch}"):

                X_signal = batch["orig_signal"].to(dtype=model_dtype, device=self.device)
                anomaly_label = batch["anomaly_label"].to(dtype=model_dtype, device=self.device)

                loss = self.model(X_signal, anomaly_label=anomaly_label)

                total_loss += loss.item()
                tr_seen += 1
                global_steps += 1
                loss_backward = loss / self.grad_accum_steps
                loss_backward.backward()

                if global_steps % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # 🔥 Update EMA after optimizer.step()
                    with torch.no_grad():
                        model_state = self.model.state_dict()
                        for key in model_state.keys():
                            ema_state_dict[key].mul_(ema_decay).add_(model_state[key], alpha=1 - ema_decay)
                    # -------------------------------


            train_total_avg = total_loss / tr_seen


            """evaluation"""
            self.model.eval()
            with torch.no_grad():
                val_total, val_seen =  0, 0
                # for batch in self.val_loader:
                for batch in tqdm(self.val_loader, desc=f"Eval Epoch {epoch}"):
                    X_signal = batch["orig_signal"].to(dtype=model_dtype, device=self.device)
                    anomaly_label = batch["anomaly_label"].to(dtype=model_dtype, device=self.device)
                    loss = self.model(X_signal, anomaly_label=anomaly_label)
                    val_total += loss.item()
                    val_seen += 1

                val_total /= val_seen

                # 记录到 wandb
                wandb.log({
                    "train/total_loss": train_total_avg,
                    "val/total_loss": val_total,
                    "epoch": epoch,
                    "step": global_steps,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })
                if self.early_stop == "true":
                    if val_total < best_val_loss:
                        best_val_loss = val_total
                        no_improve_epochs = 0
                        torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                        torch.save(ema_state_dict, f"{self.save_dir}/ema_ckpt.pth")
                    else:
                        no_improve_epochs += 1

                    if no_improve_epochs >= self.patience:
                        print(f"⛔ Early stopping triggered at Step {global_steps}.")
                        break
                else:
                    torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                    torch.save(ema_state_dict, f"{self.save_dir}/ema_ckpt.pth")
            self.scheduler.step(val_total)
        wandb.finish()

    def normal_manifold_init_train(self, config):

        wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_run_name,
            config=config,
        )

        os.makedirs(self.save_dir, exist_ok=True)

        best_val_loss = float("inf")
        no_improve_epochs = 0
        global_steps = 0
        model_dtype = next(self.model.parameters()).dtype
        for epoch in range(self.max_epochs):
            total_loss = 0
            tr_seen = 0
            self.model.train()
            self.optimizer.zero_grad()
            for batch in tqdm(self.train_loader, desc=f"Train Epoch {epoch}"):

                X_signal = batch["orig_signal"].to(dtype=model_dtype, device=self.device)
                anomaly_label = batch["random_anomaly_label"].to(dtype=model_dtype, device=self.device)

                _, loss = self.model(X_signal, anomaly_label=anomaly_label)

                total_loss += loss.item()
                tr_seen += 1
                global_steps += 1
                loss_backward = loss / self.grad_accum_steps
                loss_backward.backward()

                if global_steps % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            train_total_avg = total_loss / tr_seen

            """evaluation"""
            self.model.eval()
            with torch.no_grad():
                val_total, val_seen = 0, 0
                # for batch in self.val_loader:
                for batch in tqdm(self.val_loader, desc=f"Eval Epoch {epoch}"):
                    X_signal = batch["orig_signal"].to(dtype=model_dtype, device=self.device)
                    anomaly_label = batch["random_anomaly_label"].to(dtype=model_dtype, device=self.device)
                    _, loss = self.model(X_signal, anomaly_label=anomaly_label)
                    val_total += loss.item()
                    val_seen += 1

                val_total /= val_seen

                # 记录到 wandb
                wandb.log({
                    "train/total_loss": train_total_avg,
                    "val/total_loss": val_total,
                    "epoch": epoch,
                    "step": global_steps,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })
                if self.early_stop == "true":
                    if val_total < best_val_loss:
                        best_val_loss = val_total
                        no_improve_epochs = 0
                        torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                        # torch.save(ema_state_dict, f"{self.save_dir}/ema_ckpt.pth")
                    else:
                        no_improve_epochs += 1

                    if no_improve_epochs >= self.patience:
                        print(f"⛔ Early stopping triggered at Step {global_steps}.")
                        break
                else:
                    torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                    # torch.save(ema_state_dict, f"{self.save_dir}/ema_ckpt.pth")
            self.scheduler.step(val_total)
        wandb.finish()


    def deterministic_flow_train(self, config):
        ema_decay = 0.999
        ema_state_dict = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

        wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_run_name,
            config=config,
        )

        os.makedirs(self.save_dir, exist_ok=True)

        best_val_loss = float("inf")
        no_improve_epochs = 0
        global_steps = 0
        model_dtype = next(self.model.parameters()).dtype
        for epoch in range(self.max_epochs):
            total_loss = 0
            tr_seen = 0
            self.model.train()
            self.optimizer.zero_grad()
            for batch in tqdm(self.train_loader, desc=f"Train Epoch {epoch}"):

                X_signal = batch["orig_signal"].to(dtype=model_dtype, device=self.device)
                anomaly_label = batch["anomaly_label"].to(dtype=model_dtype, device=self.device)
                with torch.no_grad():
                    X_tilde, _ = self.ae(X_signal, anomaly_label=anomaly_label)

                loss = self.model(X_signal, anomaly_label=anomaly_label)

                total_loss += loss.item()
                tr_seen += 1
                global_steps += 1
                loss_backward = loss / self.grad_accum_steps
                loss_backward.backward()

                if global_steps % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # 🔥 Update EMA after optimizer.step()
                    with torch.no_grad():
                        model_state = self.model.state_dict()
                        for key in model_state.keys():
                            ema_state_dict[key].mul_(ema_decay).add_(model_state[key], alpha=1 - ema_decay)
                    # -------------------------------


            train_total_avg = total_loss / tr_seen


            """evaluation"""
            self.model.eval()
            with torch.no_grad():
                val_total, val_seen =  0, 0
                # for batch in self.val_loader:
                for batch in tqdm(self.val_loader, desc=f"Eval Epoch {epoch}"):
                    X_signal = batch["orig_signal"].to(dtype=model_dtype, device=self.device)
                    anomaly_label = batch["anomaly_label"].to(dtype=model_dtype, device=self.device)
                    loss = self.model(X_signal, anomaly_label=anomaly_label)
                    val_total += loss.item()
                    val_seen += 1

                val_total /= val_seen

                # 记录到 wandb
                wandb.log({
                    "train/total_loss": train_total_avg,
                    "val/total_loss": val_total,
                    "epoch": epoch,
                    "step": global_steps,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })
                if self.early_stop == "true":
                    if val_total < best_val_loss:
                        best_val_loss = val_total
                        no_improve_epochs = 0
                        torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                        torch.save(ema_state_dict, f"{self.save_dir}/ema_ckpt.pth")
                    else:
                        no_improve_epochs += 1

                    if no_improve_epochs >= self.patience:
                        print(f"⛔ Early stopping triggered at Step {global_steps}.")
                        break
                else:
                    torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                    torch.save(ema_state_dict, f"{self.save_dir}/ema_ckpt.pth")
            self.scheduler.step(val_total)
        wandb.finish()
















