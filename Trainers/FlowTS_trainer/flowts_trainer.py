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


            """evalaution"""
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

                if no_improve_epochs >= 10:
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
            pretrained_ckpt
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

    def finetune(self, config):

        self.model.prepare_for_finetune(ckpt_path=self.pretrained_ckpt)

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
        for step in range(self.max_iters):
            self.optimizer.zero_grad()
            normal_batch = next(normal_train_iterator)
            normal_signal = normal_batch["orig_signal"].to(self.device)
            normal_random_anomaly_label = normal_batch["random_anomaly_label"].to(device=self.device, dtype=torch.long)
            loss_on_normal = self.model.finetune_loss(normal_signal, normal_random_anomaly_label, mode="normal")

            anomaly_batch = next(anomaly_train_iterator)
            anomaly_signal = anomaly_batch["orig_signal"].to(self.device)
            anomaly_label = anomaly_batch["anomaly_label"].to(device=self.device, dtype=torch.long)
            loss_on_anomaly = self.model.finetune_loss(anomaly_signal, anomaly_label, mode="anomaly")

            total_loss = loss_on_normal + loss_on_anomaly
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

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
                    with torch.no_grad():
                        loss_on_anomaly = self.model.finetune_loss(anomaly_signal, anomaly_label, mode="anomaly")
                    bs = anomaly_signal.shape[0]
                    val_loss_anomaly += loss_on_anomaly.item() * bs
                    val_seen += bs
                val_loss_anomaly_avg = val_loss_anomaly / val_seen

                val_loss_total_avg = val_loss_normal_avg + val_loss_anomaly_avg


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

            if step > self.max_iters:
                break

        wandb.finish()
