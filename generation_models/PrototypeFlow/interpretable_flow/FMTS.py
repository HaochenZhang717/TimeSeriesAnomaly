import torch
import torch.nn.functional as F
from torch import nn
from einops import reduce
from .transformer import Transformer, MTANDEncoderDecoder
import os



class PrototypeFlow(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            n_heads=4,
            mlp_hidden_times=4,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            num_prototypes=8
    ):
        super(PrototypeFlow, self).__init__()

        self.seq_length = seq_length
        self.feature_size = feature_size

        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size])

        self.prototype_embedding = nn.Embedding(num_prototypes, d_model)

        self.alpha = 3  ## t shifting, change to 1 is the uniform sampling during inference
        self.time_scalar = 1000 ## scale 0-1 to 0-1000 for time embedding

        self.num_timesteps = int(os.environ.get('hucfg_num_steps', '100'))
    
    def output(self, x, t, prototypes, padding_masks):
        if padding_masks is not None:
            x = x * padding_masks.unsqueeze(-1)
        output = self.model(x, t, prototypes, padding_masks)

        return output

    @torch.no_grad()
    def sample(self, shape, prototype_id, padding_masks):
        model_device = next(self.parameters()).device
        if prototype_id != -100:
            prototypes = prototype_id * torch.ones(shape).mean((1, 2)).to(dtype=torch.long, device=model_device)
            prototype_embed = self.prototype_embedding(prototypes)
        else:
            prototype_embed = None
        self.eval()
        zt = torch.randn(shape).to(model_device)  ## init the noise
        ## t shifting from stable diffusion 3
        timesteps = torch.linspace(0, 1, self.num_timesteps+1)
        t_shifted = 1-(self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0)
        for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
            step = t_prev - t_curr
            t_input = torch.tensor([t_curr*self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).to(model_device).view(-1)
            v = self.output(zt.clone(), t_input, prototype_embed, padding_masks)
            zt = zt.clone() + step * v
        return zt

    @torch.no_grad()
    def impute(self, signals, prototypes, attn_mask, noise_mask):
        """
        x_start: (B, T, C)
        anomaly_label: (B, T, C)   1 = missing, 0 = observed
        """
        self.eval()
        prototype_embed = self.prototype_embedding(prototypes)

        zt = torch.randn_like(signals) * noise_mask.unsqueeze(-1) + signals * (1 - noise_mask.unsqueeze(-1))

        # --- identical timestep shifting as unconditional sample ---
        timesteps = torch.linspace(0, 1, self.num_timesteps + 1)
        t_shifted = 1 - (self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0).to(signals.device)

        # 2) Integrate ODE from t=1 → t=0
        for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
            step = t_prev - t_curr
            t_input = torch.tensor([t_curr*self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).to(signals.device).view(-1)
            v = self.output(zt.clone(), t_input, prototype_embed, attn_mask)

            #update missing region ONLY
            zt = zt + step * v * noise_mask.unsqueeze(-1)
            #restore known region
            zt = zt * noise_mask.unsqueeze(-1) + signals * (1 - noise_mask.unsqueeze(-1))

        return zt


    def generate_mts(self, seq_length, batch_size, prototype_id, padding_masks):
        feature_size = self.feature_size
        return self.sample((batch_size, seq_length, feature_size), prototype_id, padding_masks)

    # def _unconditional_loss(self, x_start):
    #     z0 = torch.randn_like(x_start)
    #     z1 = x_start
    #
    #     t = torch.rand(z0.shape[0], 1, 1).to(z0.device)
    #     if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
    #         t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1)).to(z0.device)
    #
    #     z_t =  t * z1 + (1.-t) * z0
    #     target = z1 - z0
    #     model_out = self.output(z_t, t.view(-1) * self.time_scalar, None)
    #     train_loss = F.mse_loss(model_out, target, reduction='none')
    #
    #     train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
    #     train_loss = train_loss.mean()
    #     return train_loss
    #
    #
    # def _impute_loss(self, x_start, anomaly_label):
    #     # x_start == [1,2,3,4,5,6]
    #     # anomaly_label = [0,0,1,1,1,0]
    #     z0_impute = torch.randn_like(x_start) * anomaly_label.unsqueeze(-1) + x_start * (1 - anomaly_label.unsqueeze(-1)) #[1,2,noise,noise,noise,6]
    #     z1 = x_start # [1,2,3,4,5,6]
    #
    #     t = torch.rand(z0_impute.shape[0], 1, 1).to(z0_impute.device)
    #     if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
    #         t = torch.sigmoid(torch.randn(z0_impute.shape[0], 1, 1)).to(z0_impute.device)
    #
    #     z_t = t * z1 + (1. - t) * z0_impute # [1,2,3+noise,4+noise,5+noise,6]
    #
    #     target = (z1 - z0_impute) * anomaly_label.unsqueeze(-1) # [0,0,3-noise, 4-noise, 5-noise, 0]
    #     model_out = self.output(z_t, t.view(-1) * self.time_scalar, None)
    #     model_out = model_out * anomaly_label.unsqueeze(-1)
    #
    #     # train_loss: (B, ..., ...)
    #     train_loss = ((model_out - target) ** 2).mean(-1) #(B, T)
    #     # 只对 anomaly 部分计算误差
    #     masked_loss = train_loss * anomaly_label #(B, T)
    #     # 每个样本 anomaly 的数量
    #     num_anomalies = reduce(anomaly_label, 'b t -> b 1', 'sum')  # shape: (B, 1)
    #
    #     # 每个样本的 loss = sum(masked_loss) / num_anomalies
    #     loss_per_sample = reduce(masked_loss, 'b t -> b 1', 'sum') / num_anomalies
    #
    #     # 最终 batch loss = mean over batch
    #     return loss_per_sample.mean()


    def forward(self, batch, mode):
        if mode=="no_context":
            signals = batch["signals"]
            lengths=batch["lengths"]
            prototypes=batch["prototypes"]
            attn_mask=batch["attn_mask"]
            return self._no_context_loss(signals, lengths, prototypes, attn_mask)
        elif mode=="imputation":
            signals = batch["signals"]
            prototypes = batch["prototypes"]
            attn_mask=batch["attn_mask"]
            noise_mask=batch["noise_mask"]
            return self._imputation_loss(signals, prototypes, attn_mask, noise_mask)
        else:
            raise NotImplementedError("No such mode")


    def _no_context_loss(self, signals, lengths, prototypes, attn_mask):
        if prototypes[0] != -100:
            prototype_embeds = self.prototype_embedding(prototypes)
        else:
            prototype_embeds = None

        z0 = torch.randn_like(signals)
        z1 = signals

        t = torch.rand(z0.shape[0], 1, 1).to(z0.device)
        if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
            t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1)).to(z0.device)

        z_t = t * z1 + (1. - t) * z0
        target = z1 - z0
        model_out = self.output(
            z_t,
            t.view(-1) * self.time_scalar,
            prototype_embeds,
            padding_masks=attn_mask)

        # -------- length-aware mask --------
        # mask: [B, T, 1]
        B, T, C = signals.shape
        device = signals.device
        mask = (
                torch.arange(T, device=device)[None, :, None]
                < lengths[:, None, None]
        ).float()
        # -------- masked MSE --------
        loss = (model_out - target) ** 2
        loss = loss * mask
        # normalize by valid length
        loss = loss.sum(dim=(1, 2)) / (lengths.float() * C)
        loss = loss.mean()

        return loss


    def _imputation_loss(self, signals, prototypes, attn_mask, noise_mask):

        assert (prototypes >= 0).all()
        # if prototypes[0] != -100:
        #     prototype_embeds = self.prototype_embedding(prototypes)
        # else:
        #     prototype_embeds = None

        prototype_embeds = self.prototype_embedding(prototypes)
        z0 = torch.randn_like(signals) * noise_mask.unsqueeze(-1) + signals * (1 - noise_mask.unsqueeze(-1))
        z1 = signals

        t = torch.rand(z0.shape[0], 1, 1).to(z0.device)
        if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
            t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1)).to(z0.device)

        z_t = t * z1 + (1. - t) * z0
        target = (z1 - z0) * noise_mask.unsqueeze(-1)
        model_out = self.output(
            z_t,
            t.view(-1) * self.time_scalar,
            prototype_embeds,
            padding_masks=attn_mask)
        model_out = model_out * noise_mask.unsqueeze(-1)

        # -------- masked MSE --------
        loss = (model_out - target) ** 2
        loss = loss.mean(-1)

        # 只对 anomaly 部分计算误差
        masked_loss = loss * noise_mask  # (B, T)
        # 每个样本 anomaly 的数量
        num_anomalies = reduce(noise_mask, 'b t -> b 1', 'sum')  # shape: (B, 1)

        # 每个样本的 loss = sum(masked_loss) / num_anomalies
        loss_per_sample = reduce(masked_loss, 'b t -> b 1', 'sum') / num_anomalies

        # 最终 batch loss = mean over batch
        return loss_per_sample.mean()



class MTANDPrototypeFlow(nn.Module):
    def __init__(
            self,
            encoder_H,
            encoder_d_h,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            n_heads=4,
            mlp_hidden_times=4,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            num_prototypes=8
    ):
        super(MTANDPrototypeFlow, self).__init__()

        self.seq_length = seq_length
        self.feature_size = feature_size

        self.model = MTANDEncoderDecoder(
            encoder_H=encoder_H,
            encoder_d_h=encoder_d_h,
            n_feat=feature_size,
            n_channel=seq_length,
            n_layer_dec=n_layer_dec,
            n_heads=n_heads, attn_pdrop=attn_pd,
            resid_pdrop=resid_pd,
            mlp_hidden_times=mlp_hidden_times,
            max_len=seq_length, n_embd=d_model,
            conv_params=[kernel_size, padding_size])

        self.prototype_embedding = nn.Embedding(num_prototypes, d_model)

        self.alpha = 3  ## t shifting, change to 1 is the uniform sampling during inference
        self.time_scalar = 1000  ## scale 0-1 to 0-1000 for time embedding

        self.num_timesteps = int(os.environ.get('hucfg_num_steps', '100'))

    def output(self, x, t, prototypes, padding_masks):
        if padding_masks is not None:
            x = x * padding_masks.unsqueeze(-1)
        output = self.model(x, t, prototypes, padding_masks)

        return output

    @torch.no_grad()
    def sample(self, shape, prototype_id, padding_masks):
        model_device = next(self.parameters()).device
        if prototype_id != -100:
            prototypes = prototype_id * torch.ones(shape).mean((1, 2)).to(dtype=torch.long, device=model_device)
            prototype_embed = self.prototype_embedding(prototypes)
        else:
            prototype_embed = None
        self.eval()
        zt = torch.randn(shape).to(model_device)  ## init the noise
        ## t shifting from stable diffusion 3
        timesteps = torch.linspace(0, 1, self.num_timesteps + 1)
        t_shifted = 1 - (self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0)
        for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
            step = t_prev - t_curr
            t_input = torch.tensor([t_curr * self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).to(
                model_device).view(-1)
            v = self.output(zt.clone(), t_input, prototype_embed, padding_masks)
            zt = zt.clone() + step * v
        return zt

    @torch.no_grad()
    def impute(self, signals, prototypes, attn_mask, noise_mask):
        """
        x_start: (B, T, C)
        anomaly_label: (B, T, C)   1 = missing, 0 = observed
        """
        self.eval()
        prototype_embed = self.prototype_embedding(prototypes)

        zt = torch.randn_like(signals) * noise_mask.unsqueeze(-1) + signals * (1 - noise_mask.unsqueeze(-1))

        # --- identical timestep shifting as unconditional sample ---
        timesteps = torch.linspace(0, 1, self.num_timesteps + 1)
        t_shifted = 1 - (self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0).to(signals.device)

        # 2) Integrate ODE from t=1 → t=0
        for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
            step = t_prev - t_curr
            t_input = torch.tensor([t_curr * self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).to(
                signals.device).view(-1)
            v = self.output(zt.clone(), t_input, prototype_embed, attn_mask)

            # update missing region ONLY
            zt = zt + step * v * noise_mask.unsqueeze(-1)
            # restore known region
            zt = zt * noise_mask.unsqueeze(-1) + signals * (1 - noise_mask.unsqueeze(-1))

        return zt

    def generate_mts(self, seq_length, batch_size, prototype_id, padding_masks):
        feature_size = self.feature_size
        return self.sample((batch_size, seq_length, feature_size), prototype_id, padding_masks)

    def forward(self, batch, mode):
        if mode == "no_context":
            signals = batch["signals"]
            lengths = batch["lengths"]
            prototypes = batch["prototypes"]
            attn_mask = batch["attn_mask"]
            return self._no_context_loss(signals, lengths, prototypes, attn_mask)
        elif mode == "imputation":
            signals = batch["signals"]
            prototypes = batch["prototypes"]
            attn_mask = batch["attn_mask"]
            noise_mask = batch["noise_mask"]
            return self._imputation_loss(signals, prototypes, attn_mask, noise_mask)
        else:
            raise NotImplementedError("No such mode")

    # def _no_context_loss(self, signals, lengths, prototypes, attn_mask):
    #     if prototypes[0] != -100:
    #         prototype_embeds = self.prototype_embedding(prototypes)
    #     else:
    #         prototype_embeds = None
    #
    #     z0 = torch.randn_like(signals)
    #     z1 = signals
    #
    #     t = torch.rand(z0.shape[0], 1, 1).to(z0.device)
    #     if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
    #         t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1)).to(z0.device)
    #
    #     z_t = t * z1 + (1. - t) * z0
    #     target = z1 - z0
    #     model_out = self.output(
    #         z_t,
    #         t.view(-1) * self.time_scalar,
    #         prototype_embeds,
    #         padding_masks=attn_mask)
    #
    #     # -------- length-aware mask --------
    #     # mask: [B, T, 1]
    #     B, T, C = signals.shape
    #     device = signals.device
    #     mask = (
    #             torch.arange(T, device=device)[None, :, None]
    #             < lengths[:, None, None]
    #     ).float()
    #     # -------- masked MSE --------
    #     loss = (model_out - target) ** 2
    #     loss = loss * mask
    #     # normalize by valid length
    #     loss = loss.sum(dim=(1, 2)) / (lengths.float() * C)
    #     loss = loss.mean()
    #
    #     return loss

    def _imputation_loss(self, signals, prototypes, attn_mask, noise_mask):

        if (prototypes != -100).all():
            prototype_embeds = self.prototype_embedding(prototypes)
        else:
            prototype_embeds = None

        z0 = torch.randn_like(signals) * noise_mask.unsqueeze(-1) + signals * (1 - noise_mask.unsqueeze(-1))
        z1 = signals

        t = torch.rand(z0.shape[0], 1, 1).to(z0.device)
        if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
            t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1)).to(z0.device)

        z_t = t * z1 + (1. - t) * z0
        target = (z1 - z0) * noise_mask.unsqueeze(-1)
        model_out = self.output(
            z_t,
            t.view(-1) * self.time_scalar,
            prototype_embeds,
            padding_masks=attn_mask)
        model_out = model_out * noise_mask.unsqueeze(-1)

        # -------- masked MSE --------
        loss = (model_out - target) ** 2
        loss = loss.mean(-1)

        # 只对 anomaly 部分计算误差
        masked_loss = loss * noise_mask  # (B, T)
        # 每个样本 anomaly 的数量
        num_anomalies = reduce(noise_mask, 'b t -> b 1', 'sum')  # shape: (B, 1)

        # 每个样本的 loss = sum(masked_loss) / num_anomalies
        loss_per_sample = reduce(masked_loss, 'b t -> b 1', 'sum') / num_anomalies

        # 最终 batch loss = mean over batch
        return loss_per_sample.mean()
