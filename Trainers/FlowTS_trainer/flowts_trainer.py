import torch
import wandb
import os
from tqdm import tqdm

class FlowTSPretrain(object):
    def __init__(
            self,optimizer, model, train_loader,
            val_loader, max_epochs, device, save_dir,
            wandb_project_name, wandb_run_name, grad_clip_norm,
    ):
        self.optimizer = optimizer
        self.model = model
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

                X_normal = batch["orig_signal"]

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
                    X_normal = batch["orig_signal"]
                    loss = self.model(X_normal, anomaly_label=None)
                    val_total += loss.item()
                    val_seen += 1

                val_total /= val_seen

                # 记录到 wandb
                wandb.log({
                    "train/total_loss": train_total_avg,
                    "val/total_loss": val_total,
                    "epoch": epoch,
                    "step": global_steps
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
