import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import wandb
import numpy as np


class Sampling(nn.Module):
    def forward(self, inputs):
        z_mean, z_log_var = inputs
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class BaseVariationalAutoencoder(nn.Module, ABC):
    model_name = None

    def __init__(
        self,
        seq_len,
        feat_dim,
        latent_dim,
        kl_wt,
        **kwargs
    ):
        super(BaseVariationalAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.kl_wt = kl_wt
        # self.batch_size = batch_size
        self.encoder = None
        self.normal_decoder = None
        self.anomaly_decoder =  None
        self.sampling = Sampling()

    # def fit_on_data_orig(self, train_data, valid_data, max_steps=1000, lr=3e-5, verbose=0,):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.to(device)
    #     global_steps = 0
    #     # train_tensor = torch.FloatTensor(train_data).to(device)
    #     # train_dataset = TensorDataset(train_tensor)
    #     # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    #
    #     train_loader = DataLoader(train_data, batch_size=self.batch_size)
    #     val_loader = DataLoader(valid_data, batch_size=self.batch_size)
    #
    #     optimizer = optim.Adam(self.parameters(), lr=lr)
    #
    #     self.train()
    #     total_loss = 0
    #     reconstruction_loss = 0
    #     kl_loss = 0
    #     tr_seen = 0
    #     for batch in train_loader:
    #         if global_steps > max_steps:
    #             break
    #
    #         X_occluded = batch[0].to(device)
    #         X_normal = batch[1].to(device)
    #
    #
    #         optimizer.zero_grad()
    #
    #         z_mean, z_log_var, z = self.encoder(X_occluded)
    #         reconstruction = self.decoder(z)
    #
    #         loss, recon_loss, kl = self.loss_function(X_normal, reconstruction, z_mean, z_log_var)
    #
    #         # Normalize the loss by the batch size
    #         # loss = loss / X.size(0)
    #         # recon_loss = recon_loss / X.size(0)
    #         # kl = kl / X.size(0)
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         total_loss += loss.item()
    #         reconstruction_loss += recon_loss.item()
    #         kl_loss += kl.item()
    #         tr_seen += 1
    #         global_steps += 1
    #
    #         if global_steps % 1000 == 0 and verbose:
    #             self.eval()
    #             print(f"Train Step {global_steps}/ {max_steps} | Total loss: {total_loss / tr_seen:.4f} | "
    #                   f"Recon loss: {reconstruction_loss / tr_seen:.4f} | "
    #                   f"KL loss: {kl_loss / tr_seen:.4f}")
    #
    #             total_loss = 0
    #             reconstruction_loss = 0
    #             kl_loss = 0
    #             tr_seen = 0
    #
    #             for val_batch in val_loader:
    #                 X_occluded = val_batch[0].to(device)
    #                 X_normal = val_batch[1].to(device)
    #                 with torch.no_grad():
    #                     z_mean, z_log_var, z = self.encoder(X_occluded)
    #                     reconstruction = self.decoder(z)
    #                     loss, recon_loss, kl = self.loss_function(X_normal, reconstruction, z_mean, z_log_var)
    #                 total_loss += loss.item()
    #                 reconstruction_loss += recon_loss.item()
    #                 kl_loss += kl.item()
    #                 tr_seen += 1
    #                 if tr_seen > 500:
    #                     break
    #             print("-" * 80)
    #             print(f"Eval Step {global_steps}/ {max_steps} | Total loss: {total_loss / tr_seen:.4f} | "
    #                   f"Recon loss: {reconstruction_loss / tr_seen:.4f} | "
    #                   f"KL loss: {kl_loss / tr_seen:.4f}")
    #             print("=" * 80)
    #             total_loss = 0
    #             reconstruction_loss = 0
    #             kl_loss = 0
    #             tr_seen = 0
    #             self.train()
    #
    #         # if verbose:
    #         #     print(f"Epoch {epoch + 1}/{max_epochs} | Total loss: {total_loss / len(train_loader):.4f} | "
    #         #         f"Recon loss: {reconstruction_loss / len(train_loader):.4f} | "
    #         #         f"KL loss: {kl_loss / len(train_loader):.4f}")


    # def pretrain(self, train_data, valid_data, max_steps=1000, lr=3e-5, verbose=0, save_dir="none"):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.to(device)
    #
    #     # 初始化 wandb
    #     wandb.init(project="TimeVAE-CGATS-pretraining", name="TimeVAE-CGATS-pretraining", config={"lr": lr, "max_steps": max_steps})
    #     wandb.watch(self, log="all", log_freq=100)
    #
    #     train_loader = DataLoader(train_data, batch_size=self.batch_size)
    #     val_loader = DataLoader(valid_data, batch_size=self.batch_size)
    #
    #     optimizer = optim.Adam(self.parameters(), lr=lr)
    #
    #     global_steps = 0
    #     best_val_loss = float("inf")
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     self.train()
    #     total_loss = reconstruction_loss = kl_loss = 0
    #     tr_seen = 0
    #     no_improve_epochs = 0
    #
    #     for batch in train_loader:
    #         if global_steps > max_steps:
    #             break
    #
    #         X_occluded = batch[0].to(device)
    #         X_normal = batch[1].to(device)
    #
    #         optimizer.zero_grad()
    #         z_mean, z_log_var, z = self.encoder(X_occluded)
    #         reconstruction = self.normal_decoder(z)
    #         loss, recon_loss, kl = self.loss_function(X_normal, reconstruction, z_mean, z_log_var)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # 累积统计
    #         total_loss += loss.item()
    #         reconstruction_loss += recon_loss.item()
    #         kl_loss += kl.item()
    #         tr_seen += 1
    #         global_steps += 1
    #
    #         # ====== 记录训练到 wandb ======
    #         wandb.log({
    #             "train/total_loss": loss.item(),
    #             "train/recon_loss": recon_loss.item(),
    #             "train/kl_loss": kl.item(),
    #             "lr": optimizer.param_groups[0]["lr"],
    #             "step": global_steps
    #         })
    #
    #         # ====== 每 1000 step 评估 ======
    #         if global_steps % 1000 == 0 and verbose:
    #             self.eval()
    #             with torch.no_grad():
    #                 val_total, val_recon, val_kl, val_seen = 0, 0, 0, 0
    #                 for val_batch in val_loader:
    #                     X_occluded = val_batch[0].to(device)
    #                     X_normal = val_batch[1].to(device)
    #                     z_mean, z_log_var, z = self.encoder(X_occluded)
    #                     reconstruction = self.normal_decoder(z)
    #                     loss, recon_loss, kl = self.loss_function(X_normal, reconstruction, z_mean, z_log_var)
    #                     val_total += loss.item()
    #                     val_recon += recon_loss.item()
    #                     val_kl += kl.item()
    #                     val_seen += 1
    #                     if val_seen > 500:
    #                         break
    #
    #                 val_total /= val_seen
    #                 val_recon /= val_seen
    #                 val_kl /= val_seen
    #
    #                 print("-" * 80)
    #                 print(f"[Eval] Step {global_steps}/{max_steps} | "
    #                       f"val_total={val_total:.4f} | val_recon={val_recon:.4f} | val_kl={val_kl:.4f}")
    #                 print("=" * 80)
    #
    #                 # 记录到 wandb
    #                 wandb.log({
    #                     "val/total_loss": val_total,
    #                     "val/recon_loss": val_recon,
    #                     "val/kl_loss": val_kl,
    #                     "step": global_steps
    #                 })
    #
    #                 if val_total < best_val_loss:
    #                     no_improve_epochs = 0
    #                 else:
    #                     no_improve_epochs += 1
    #
    #                 if no_improve_epochs >= 10:
    #                     print(f"⛔ Early stopping triggered at Step {global_steps}.")
    #                     break
    #
    #             # 重置计数
    #             total_loss = reconstruction_loss = kl_loss = 0
    #             tr_seen = 0
    #             self.train()
    #
    #     wandb.finish()
    #
    #
    # def finetune(self, train_data, valid_data, max_steps=1000, lr=3e-5, verbose=0):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.to(device)
    #
    #     # 初始化 wandb
    #     wandb.init(project="TimeVAE-CGATS-finetune", name="TimeVAE-CGATS-finetune", config={"lr": lr, "max_steps": max_steps})
    #     wandb.watch(self, log="all", log_freq=100)
    #
    #     train_loader = DataLoader(train_data, batch_size=self.batch_size)
    #     val_loader = DataLoader(valid_data, batch_size=self.batch_size)
    #
    #     # freeze encoder
    #     for param in self.encoder.parameters():
    #         param.requires_grad = False
    #     # only train anomaly_decoder
    #     for param in self.anomaly_decoder.parameters():
    #         param.requires_grad = True
    #
    #     optimizer = optim.Adam(self.anomaly_decoder.parameters(), lr=lr)
    #
    #     global_steps = 0
    #     best_val_loss = float("inf")
    #     # os.makedirs(save_dir, exist_ok=True)
    #
    #     self.train()
    #     total_loss = 0
    #     tr_seen = 0
    #     no_improve_epochs = 0
    #
    #     for X_occluded, X_anomaly, window_label in train_loader:
    #         if global_steps > max_steps:
    #             break
    #
    #         X_occluded = X_occluded.to(device=device, dtype=torch.float32)
    #         X_anomaly = X_anomaly.to(device=device, dtype=torch.float32)
    #         window_label = window_label.to(device=device, dtype=torch.float32)
    #
    #         optimizer.zero_grad()
    #         z_mean, z_log_var, z = self.encoder(X_occluded)
    #         reconstruction = self.anomaly_decoder(z, window_label)
    #         loss = nn.MSELoss()(reconstruction, X_anomaly)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # 累积统计
    #         total_loss += loss.item()
    #         tr_seen += 1
    #         global_steps += 1
    #
    #         # ====== 记录训练到 wandb ======
    #         wandb.log({
    #             "train/total_loss": loss.item(),
    #             "lr": optimizer.param_groups[0]["lr"],
    #             "step": global_steps
    #         })
    #
    #         # ====== 每 1000 step 评估 ======
    #         if global_steps % 1000 == 0 and verbose:
    #             self.eval()
    #             with torch.no_grad():
    #                 val_total, val_seen = 0, 0
    #                 for X_occluded, X_anomaly, window_label in val_loader:
    #
    #                     X_occluded = X_occluded.to(device=device, dtype=torch.float32)
    #                     X_anomaly = X_anomaly.to(device=device, dtype=torch.float32)
    #                     window_label = window_label.to(device=device, dtype=torch.float32)
    #
    #                     z_mean, z_log_var, z = self.encoder(X_occluded)
    #                     reconstruction = self.anomaly_decoder(z, window_label)
    #                     loss = nn.MSELoss()(reconstruction, X_anomaly)
    #
    #                     val_total += loss.item()
    #                     val_seen += 1
    #                     if val_seen > 500:
    #                         break
    #
    #                 val_total /= val_seen
    #
    #                 print("-" * 80)
    #                 print(f"[Eval] Step {global_steps}/{max_steps} | "
    #                       f"val_total={val_total:.4f} |")
    #                 print("=" * 80)
    #
    #                 # 记录到 wandb
    #                 wandb.log({
    #                     "val/total_loss": val_total,
    #                     "step": global_steps
    #                 })
    #
    #                 if val_total < best_val_loss:
    #                     no_improve_epochs = 0
    #                 else:
    #                     no_improve_epochs += 1
    #
    #                 if no_improve_epochs >= 10:
    #                     print(f"⛔ Early stopping triggered at Step {global_steps}.")
    #                     break
    #
    #                 # ====== 保存最优模型 ======
    #                 # if val_total < best_val_loss:
    #                 #     best_val_loss = val_total
    #                 #     ckpt_path = os.path.join(save_dir, f"best_model_step{global_steps}_val{val_total:.4f}.pt")
    #                 #     torch.save({
    #                 #         "model_state_dict": self.state_dict(),
    #                 #         "optimizer_state_dict": optimizer.state_dict(),
    #                 #         "step": global_steps,
    #                 #         "val_loss": val_total,
    #                 #     }, ckpt_path)
    #                 #     print(f"✅ Saved best model to {ckpt_path}")
    #
    #             # 重置计数
    #             total_loss = reconstruction_loss = kl_loss = 0
    #             tr_seen = 0
    #             self.train()
    #
    #     wandb.finish()


    def normal_forward(self, X_occluded):
        z_mean, z_log_var, z = self.encoder(X_occluded)
        x_decoded = self.normal_decoder(z_mean)
        return x_decoded


    def normal_predict(self, valid_data):
        self.eval()
        val_loader = DataLoader(valid_data, batch_size=16)
        batch = next(iter(val_loader))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_occluded = batch[0].to(device)
        X_normal = batch[1].to(device)

        with torch.no_grad():
            z_mean, z_log_var, z = self.encoder(X_occluded)
            x_decoded = self.normal_decoder(z_mean)
        return x_decoded.cpu().detach().numpy(), X_occluded.cpu().detach().numpy(), X_normal.cpu().detach().numpy()


    def anomaly_predict(self, valid_data):
        self.eval()
        val_loader = DataLoader(valid_data, batch_size=16)
        batch = next(iter(val_loader))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_occluded = batch[0].to(device)
        X_anomaly = batch[1].to(device)
        window_label = batch[2].to(device)

        with torch.no_grad():
            z_mean, z_log_var, z = self.encoder(X_occluded)
            x_decoded = self.anomaly_decoder(z_mean, window_label)
        return x_decoded.cpu().detach().numpy(), X_anomaly.cpu().detach().numpy(), window_label.cpu().detach().numpy()


    def anomaly_inject(self, X_occluded, anomaly_label):
        self.eval()
        with torch.no_grad():
            z_mean, z_log_var, z = self.encoder(X_occluded)
            x_decoded = self.anomaly_decoder(z_mean, anomaly_label)
        return x_decoded

    def get_num_trainable_variables(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_prior_normal_samples(self, num_samples):
        device = next(self.parameters()).device
        Z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.normal_decoder(Z)
        return samples

    def get_prior_anomaly_samples(self, anomaly_labels):
        device = next(self.parameters()).device
        num_samples = len(anomaly_labels)
        Z = torch.randn(num_samples, self.latent_dim).to(device)
        if len(anomaly_labels.shape) == 2:
            anomaly_labels = anomaly_labels.unsqueeze(-1)
        samples = self.anomaly_decoder(Z, anomaly_labels)
        return samples

    def get_prior_samples_given_Z(self, Z):
        Z = torch.FloatTensor(Z).to(next(self.parameters()).device)
        samples = self.decoder(Z)
        return samples.cpu().detach().numpy()

    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_normal_decoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_anomaly_decoder(self, **kwargs):
        raise NotImplementedError


    def _get_reconstruction_loss(self, X, X_recons):
        def get_reconst_loss_by_axis(X, X_recons, dim):
            x_r = torch.mean(X, dim=dim)
            x_c_r = torch.mean(X_recons, dim=dim)
            err = torch.pow(x_r - x_c_r, 2)
            loss = torch.sum(err)
            return loss

        # err = torch.pow(X - X_recons, 2)
        # reconst_loss = torch.sum(err) / err.shape[0]

        reconst_loss = nn.MSELoss()(X, X_recons)
        
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, dim=2)  # by time axis
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, dim=1)  # by feature axis 

        return reconst_loss


    def loss_function(self, X, X_recons, z_mean, z_log_var):
        reconstruction_loss = self._get_reconstruction_loss(X, X_recons)
        # kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        # kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        kl = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
        kl_loss = torch.mean(kl)

        total_loss = reconstruction_loss + self.kl_wt * kl_loss
        # total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def save_weights(self, model_dir):
        if self.model_name is None:
            raise ValueError("Model name not set.")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(model_dir, f"{self.model_name}_encoder_wts.pth"))
        torch.save(self.normal_decoder.state_dict(), os.path.join(model_dir, f"{self.model_name}_normal_decoder_wts.pth"))
        torch.save(self.anomaly_decoder.state_dict(), os.path.join(model_dir, f"{self.model_name}_anomaly_decoder_wts.pth"))

    # def load_pretrain_encoder(self, model_dir):
    #     model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.encoder.load_state_dict(torch.load(os.path.join(model_dir, f"{self.model_name}_encoder_wts.pth"), map_location=model_device))
    #     print("pretrained encoder loaded")
    #     # self.decoder.load_state_dict(torch.load(os.path.join(model_dir, f"{self.model_name}_decoder_wts.pth")))

    # def save(self, model_dir):
    #     os.makedirs(model_dir, exist_ok=True)
    #     self.save_weights(model_dir)
    #     dict_params = {
    #         "seq_len": self.seq_len,
    #         "feat_dim": self.feat_dim,
    #         "latent_dim": self.latent_dim,
    #         "kl_wt": self.kl_wt,
    #         "hidden_layer_sizes": list(self.hidden_layer_sizes) if hasattr(self, 'hidden_layer_sizes') else None,
    #     }
    #     params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
    #     joblib.dump(dict_params, params_file)

if __name__ == "__main__":
    pass