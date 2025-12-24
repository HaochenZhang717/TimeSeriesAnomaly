import torch
import wandb
import os
from tqdm import tqdm
from torch.utils.data import DataLoader


class DSPFlowTrainer(object):
    def __init__(
            self,optimizer, scheduler,
            model, train_loader,
            val_loader, max_epochs,
            device, save_dir,
            wandb_project_name, wandb_run_name,
            grad_clip_norm,
            grad_accum_steps, early_stop, patience,
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


    def no_context_train(self, config):
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

                batch["signals"] = batch["signals"].to(dtype=model_dtype, device=self.device)
                batch["attn_mask"] = batch["attn_mask"].to(dtype=torch.bool, device=self.device)
                loss = self.model(batch, mode="no_context")

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
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                ema_state_dict[name].mul_(ema_decay).add_(param, alpha=1 - ema_decay)
                    # -------------------------------

            train_total_avg = total_loss / tr_seen

            """evaluation"""
            self.model.eval()
            with torch.no_grad():
                val_total, val_seen = 0, 0
                # for batch in self.val_loader:
                for batch in tqdm(self.val_loader, desc=f"Eval Epoch {epoch}"):
                    batch["signals"] = batch["signals"].to(dtype=model_dtype, device=self.device)
                    batch["attn_mask"] = batch["attn_mask"].to(dtype=torch.bool, device=self.device)
                    loss = self.model(batch, mode="no_context")
                    val_total += loss.item() * batch["signals"].shape[0]
                    val_seen += batch["signals"].shape[0]

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


    def imputation_train(self, config):
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

                batch["signals"] = batch["signals"].to(dtype=model_dtype, device=self.device)
                batch["missing_signals"] = batch["missing_signals"].to(dtype=model_dtype, device=self.device)
                batch["attn_mask"] = batch["attn_mask"].to(dtype=torch.bool, device=self.device)
                batch["noise_mask"] = batch["noise_mask"].to(dtype=torch.long, device=self.device)
                loss = self.model(batch, mode="imputation")

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
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                ema_state_dict[name].mul_(ema_decay).add_(param, alpha=1 - ema_decay)
                    # -------------------------------

            train_total_avg = total_loss / tr_seen

            if epoch % 1 == 0:
                """evaluation"""
                self.model.eval()
                with torch.no_grad():
                    val_total, val_seen = 0, 0
                    # for batch in self.val_loader:
                    for batch in tqdm(self.val_loader, desc=f"Eval Epoch {epoch}"):
                        batch["signals"] = batch["signals"].to(dtype=model_dtype, device=self.device)
                        batch["missing_signals"] = batch["missing_signals"].to(dtype=model_dtype, device=self.device)
                        batch["attn_mask"] = batch["attn_mask"].to(dtype=torch.bool, device=self.device)
                        batch["noise_mask"] = batch["noise_mask"].to(dtype=torch.long, device=self.device)
                        loss = self.model(batch, mode="imputation")

                        val_total += loss.item() * batch["signals"].shape[0]
                        val_seen += batch["signals"].shape[0]

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
                # torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt_epoch{epoch}.pth")
                self.scheduler.step(val_total)
            else:
                wandb.log({
                    "train/total_loss": train_total_avg,
                    "epoch": epoch,
                    "step": global_steps,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })
        wandb.finish()


    def no_context_no_code_train(self, config):
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

                batch["signals"] = batch["signals"].to(dtype=model_dtype, device=self.device)
                batch["attn_mask"] = batch["attn_mask"].to(dtype=torch.bool, device=self.device)
                loss = self.model(batch, mode="no_context_no_code")

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
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                ema_state_dict[name].mul_(ema_decay).add_(param, alpha=1 - ema_decay)
                    # -------------------------------

            train_total_avg = total_loss / tr_seen

            """evaluation"""
            self.model.eval()
            with torch.no_grad():
                val_total, val_seen = 0, 0
                # for batch in self.val_loader:
                for batch in tqdm(self.val_loader, desc=f"Eval Epoch {epoch}"):
                    batch["signals"] = batch["signals"].to(dtype=model_dtype, device=self.device)
                    batch["attn_mask"] = batch["attn_mask"].to(dtype=torch.bool, device=self.device)
                    loss = self.model(batch, mode="no_context_no_code")
                    val_total += loss.item() * batch["signals"].shape[0]
                    val_seen += batch["signals"].shape[0]

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


    def no_code_imputation_train(self, config):
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

                batch["signals"] = batch["signals"].to(dtype=model_dtype, device=self.device)
                batch["missing_signals"] = batch["missing_signals"].to(dtype=model_dtype, device=self.device)
                batch["attn_mask"] = batch["attn_mask"].to(dtype=torch.bool, device=self.device)
                batch["noise_mask"] = batch["noise_mask"].to(dtype=torch.long, device=self.device)
                loss = self.model(batch, mode="no_code_imputation")

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
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                ema_state_dict[name].mul_(ema_decay).add_(param, alpha=1 - ema_decay)
                    # -------------------------------

            train_total_avg = total_loss / tr_seen

            if epoch % 1 == 0:
                """evaluation"""
                self.model.eval()
                with torch.no_grad():
                    val_total, val_seen = 0, 0
                    # for batch in self.val_loader:
                    for batch in tqdm(self.val_loader, desc=f"Eval Epoch {epoch}"):
                        batch["signals"] = batch["signals"].to(dtype=model_dtype, device=self.device)
                        batch["missing_signals"] = batch["missing_signals"].to(dtype=model_dtype, device=self.device)
                        batch["attn_mask"] = batch["attn_mask"].to(dtype=torch.bool, device=self.device)
                        batch["noise_mask"] = batch["noise_mask"].to(dtype=torch.long, device=self.device)
                        loss = self.model(batch, mode="no_code_imputation")

                        val_total += loss.item() * batch["signals"].shape[0]
                        val_seen += batch["signals"].shape[0]

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
                torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt_epoch{epoch}.pth")
                self.scheduler.step(val_total)
            else:
                wandb.log({
                    "train/total_loss": train_total_avg,
                    "epoch": epoch,
                    "step": global_steps,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })
        wandb.finish()
