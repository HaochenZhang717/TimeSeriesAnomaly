import torch
import wandb
import os
from tqdm import tqdm
from torch.utils.data import DataLoader


class VRFTrainer(object):
    def __init__(
            self,optimizer, scheduler, model, train_loader,
            val_loader, max_epochs, device, save_dir,
            wandb_project_name, wandb_run_name, grad_clip_norm,
            grad_accum_steps, early_stop, patience
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
            tr_total_loss = 0
            tr_flow_loss = 0
            tr_kl_loss = 0
            tr_seen = 0
            self.model.train()
            self.optimizer.zero_grad()
            for batch in tqdm(self.train_loader, desc=f"Train Epoch {epoch}"):

                X_signal = batch["orig_signal"].to(dtype=model_dtype, device=self.device)
                anomaly_label = batch["anomaly_label"].to(dtype=model_dtype, device=self.device)

                loss, flow_loss, kl_loss = self.model(X_signal, anomaly_label=anomaly_label)

                tr_total_loss += loss.item()
                tr_flow_loss += flow_loss.item()
                tr_kl_loss += kl_loss.item()
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


            train_total_avg = tr_total_loss / tr_seen
            train_flow_avg = tr_flow_loss / tr_seen
            train_kl_avg = tr_kl_loss / tr_seen


            """evaluation"""
            self.model.eval()
            with torch.no_grad():
                val_total_loss = 0
                val_flow_loss = 0
                val_kl_loss = 0
                val_seen = 0
                for batch in tqdm(self.val_loader, desc=f"Eval Epoch {epoch}"):
                    X_signal = batch["orig_signal"].to(dtype=model_dtype, device=self.device)
                    anomaly_label = batch["anomaly_label"].to(dtype=model_dtype, device=self.device)
                    loss, flow_loss, kl_loss = self.model(X_signal, anomaly_label=anomaly_label)

                    val_total_loss += loss.item() * X_signal.shape[0]
                    val_flow_loss += flow_loss.item() * X_signal.shape[0]
                    val_kl_loss += kl_loss.item() * X_signal.shape[0]
                    val_seen += X_signal.shape[0]

                val_total_loss = val_total_loss / val_seen
                val_flow_loss = val_flow_loss / val_seen
                val_kl_loss = val_kl_loss / val_seen

                # 记录到 wandb
                wandb.log({
                    "train/total_loss": train_total_avg,
                    "train/flow_loss": train_flow_avg,
                    "train/kl_loss": train_kl_avg,
                    "val/total_loss": val_total_loss,
                    "val/flow_loss": val_flow_loss,
                    "val/kl_loss": val_kl_loss,
                    "epoch": epoch,
                    "step": global_steps,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })
                if self.early_stop == "true":
                    if val_total_loss < best_val_loss:
                        best_val_loss = val_total_loss
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
            self.scheduler.step(val_total_loss)
        wandb.finish()
