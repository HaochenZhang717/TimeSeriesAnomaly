import torch
import wandb
import os
from tqdm import tqdm

class CGATPretrain(object):
    def __init__(
            self,optimizer, scheduler, model, train_loader,
            val_loader, max_epochs, device, save_dir,
            wandb_project_name, wandb_run_name, grad_clip_norm,
            early_stop
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
        self.early_stop = early_stop

    def pretrain(self, config):
        # freeze encoder
        for param in self.model.anomaly_decoder.parameters():
            param.requires_grad = False

        # 初始化 wandb
        wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_run_name,
            config=config
        )

        os.makedirs(self.save_dir, exist_ok=True)

        best_val_loss = float("inf")
        no_improve_epochs = 0
        global_steps = 0

        for epoch in range(self.max_epochs):
            total_loss = reconstruction_loss = kl_loss = 0
            tr_seen = 0
            self.model.train()
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):

                X_occluded = batch["signal_random_occluded"].to(self.device)
                X_normal = batch["orig_signal"].to(self.device)

                self.optimizer.zero_grad()
                z_mean, z_log_var, z = self.model.encoder(X_occluded)
                reconstruction = self.model.normal_decoder(z)
                loss, recon_loss, kl = self.model.loss_function(X_normal, reconstruction, z_mean, z_log_var)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                # 累积统计
                total_loss += loss.item()
                reconstruction_loss += recon_loss.item()
                kl_loss += kl.item()
                tr_seen += 1
                global_steps += 1

                # wandb.log({
                #     "train/step_total_loss": loss.item(),
                #     "train/step_recon_loss": recon_loss.item(),
                #     "train/step_kl_loss": kl.item(),
                #     "lr": self.optimizer.param_groups[0]["lr"],
                #     "step": global_steps
                # })
            train_total_avg = total_loss / tr_seen
            train_recon_avg = reconstruction_loss / tr_seen
            train_kl_avg = kl_loss / tr_seen


            """evaluation"""
            self.model.eval()
            with torch.no_grad():
                val_total, val_recon, val_kl, val_seen = 0, 0, 0, 0
                for batch in self.val_loader:
                    X_occluded = batch["signal_random_occluded"].to(self.device)
                    X_normal = batch["orig_signal"].to(self.device)
                    z_mean, z_log_var, z = self.model.encoder(X_occluded)
                    reconstruction = self.model.normal_decoder(z)
                    loss, recon_loss, kl = self.model.loss_function(X_normal, reconstruction, z_mean, z_log_var)
                    val_total += loss.item()
                    val_recon += recon_loss.item()
                    val_kl += kl.item()
                    val_seen += 1

                val_total /= val_seen
                val_recon /= val_seen
                val_kl /= val_seen

                # 记录到 wandb
                wandb.log({
                    "train/total_loss": train_total_avg,
                    "train/recon_loss": train_recon_avg,
                    "train/kl_loss": train_kl_avg,
                    "val/total_loss": val_total,
                    "val/recon_loss": val_recon,
                    "val/kl_loss": val_kl,
                    "epoch": epoch,
                    "step": global_steps,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

                if self.early_stop == "true":
                    if val_total < best_val_loss:
                        best_val_loss = val_total
                        no_improve_epochs = 0
                        torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                    else:
                        no_improve_epochs += 1

                    if no_improve_epochs >= 10:
                        print(f"⛔ Early stopping triggered at Step {global_steps}.")
                        break
                else:
                    torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")

                self.scheduler.step(val_total)

        wandb.finish()


class CGATFinetune(object):
    def __init__(
            self, optimizer, scheduler, model, train_loader,
            val_loader, max_epochs, device, save_dir,
            wandb_project_name, wandb_run_name, grad_clip_norm,
            early_stop
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
        self.early_stop = early_stop

    def finetune(self, config):
        # freeze encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.normal_decoder.parameters():
            param.requires_grad = False
        # only train anomaly_decoder
        for param in self.model.anomaly_decoder.parameters():
            param.requires_grad = True

        # 初始化 wandb
        wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_run_name,
            config=config
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
                X_occluded = batch["original_occluded_signal"].to(self.device)
                X_target = batch["orig_signal"].to(self.device)
                anomaly_label = batch["anomaly_label"].unsqueeze(-1).to(self.device)

                self.optimizer.zero_grad()

                z_mean, z_log_var, z = self.model.encoder(X_occluded)
                reconstruction = self.model.anomaly_decoder(z, anomaly_label)
                loss = torch.nn.MSELoss()(reconstruction, X_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                # 累积统计
                total_loss += loss.item()
                tr_seen += 1
                global_steps += 1

                # ====== 记录训练到 wandb ======
                # wandb.log({
                #     "train/step_total_loss": loss.item(),
                #     "lr": self.optimizer.param_groups[0]["lr"],
                #     "step": global_steps
                # })
            train_total_avg = total_loss / tr_seen

            """evaluation"""
            self.model.eval()
            val_total, val_seen = 0, 0
            for batch in self.val_loader:
                X_occluded = batch["original_occluded_signal"].to(self.device)
                X_target = batch["orig_signal"].to(self.device)
                anomaly_label = batch["anomaly_label"].unsqueeze(-1).to(self.device)
                with torch.no_grad():
                    z_mean, z_log_var, z = self.model.encoder(X_occluded)
                    reconstruction = self.model.anomaly_decoder(z, anomaly_label)
                loss = torch.nn.MSELoss()(reconstruction, X_target)

                val_total += loss.item()
                val_seen += 1

            val_total /= val_seen
            # 记录到 wandb
            wandb.log({
                "val/total_loss": val_total,
                "train/total_loss": train_total_avg,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch": epoch,
                "step": global_steps
            })

            if self.early_stop == "true":
                if val_total < best_val_loss:
                    best_val_loss = val_total
                    no_improve_epochs = 0
                    torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
                else:
                    no_improve_epochs += 1


                if no_improve_epochs >= 100 and  epoch > self.max_epochs//4:
                    print(f"⛔ Early stopping triggered at Step {global_steps}.")
                    break
            else:
                torch.save(self.model.state_dict(), f"{self.save_dir}/ckpt.pth")
            self.scheduler.step(val_total)

        wandb.finish()

