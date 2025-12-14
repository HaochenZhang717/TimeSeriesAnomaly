import torch
import torch.nn.functional as F
from torch import nn
from einops import reduce
from .transformer import Transformer
from .variational_encoder import TemporalVariationalEncoder1D
import os



class VRF_v3(nn.Module):
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

            ve_channels=(16, 32, 64),
            ve_kernel_size=3,
            ve_pool_kernel=2,
            ve_pool_stride=2,
            ve_z_dim=16,

            kl_beta=0.1
    ):
        super(VRF_v3, self).__init__()

        self.seq_length = seq_length
        self.feature_size = feature_size
        self.kl_beta = kl_beta
        self.variational_encoder = TemporalVariationalEncoder1D(
            in_channels=feature_size,
            channels=ve_channels,
            kernel_size=ve_kernel_size,
            pool_kernel=ve_pool_kernel,
            pool_stride=ve_pool_stride,
            z_dim=ve_z_dim,
        )
        self.model = Transformer(
            n_feat=feature_size, n_channel=seq_length,
            n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
            n_heads=n_heads, attn_pdrop=attn_pd,
            resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
            max_len=seq_length + self.variational_encoder.num_tokens,
            real_ts_len=seq_length,
            n_embd=d_model,
            conv_params=[kernel_size, padding_size]
        )

        self.latent_projector = nn.Linear(ve_z_dim, d_model)
        self.alpha = 3  ## t shifting, change to 1 is the uniform sampling during inference
        self.time_scalar = 1000 ## scale 0-1 to 0-1000 for time embedding

        self.num_timesteps = int(os.environ.get('hucfg_num_steps', '100'))
    
    def output(self, x, t, ve_input):
        if ve_input is not None:
            _, mu_t, logvar_t, latent_t = self.variational_encoder(ve_input.permute(0,2,1)) # (B, 3~4, C)
            mu_t = mu_t.permute(0, 2, 1)
            logvar_t = logvar_t.permute(0, 2, 1)
            latent_t = latent_t.permute(0, 2, 1)
        else:
            mu_t = None
            logvar_t = None
            latent_t = self.variational_encoder.sample_prior_latent(x.shape[0]).permute(0,2,1)

        num_tokens = latent_t.shape[1]
        projected_latent = self.latent_projector(latent_t)
        # print(self.variational_encoder.num_tokens)
        # breakpoint()
        # x = torch.cat([projected_latent, x], dim=1)
        output = self.model(x, t, projected_latent, padding_masks=None)
        # return output[:, num_tokens:], mu_t, logvar_t
        return output, mu_t, logvar_t


    @torch.no_grad()
    def impute(self, x_start, anomaly_label, mode, seed):
        """
        x_start: (B, T, C)
        anomaly_label: (B, T, C)   1 = missing, 0 = observed
        """
        self.eval()
        # 1) Init z_t: missing is noise
        if seed is not None:
            g = torch.Generator(device=x_start.device)
            g.manual_seed(seed)
            noise = torch.randn(
                x_start.shape,
                device=x_start.device,
                dtype=x_start.dtype,
                generator=g
            )
        else:
            noise = torch.randn_like(x_start)

        zt = noise * anomaly_label.unsqueeze(-1) + x_start * (1 - anomaly_label.unsqueeze(-1))

        # --- identical timestep shifting as unconditional sample ---
        timesteps = torch.linspace(0, 1, self.num_timesteps + 1)
        t_shifted = 1 - (self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0).to(x_start.device)

        # latent = self.variational_encoder.sample_prior_latent(batch_size=1)
        # 2) Integrate ODE from t=1 → t=0
        step_idx = 0
        for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
            print(f"step: {step_idx}")
            step = t_prev - t_curr
            t_input = torch.tensor([t_curr*self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).to(x_start.device).view(-1)
            if mode == "prior":
                v, _, _ = self.output(zt.clone(), t_input, None)
            elif mode == "posterior":
                v, _, _ = self.output(zt.clone(), t_input, x_start*anomaly_label.unsqueeze(-1))
            else:
                raise ValueError("Unknown Mode!!! mode must be 'prior' or 'posterior'")

            #update missing region ONLY
            zt = zt + step * v * anomaly_label.unsqueeze(-1)

            #restore known region
            zt = zt * anomaly_label.unsqueeze(-1) + x_start * (1 - anomaly_label.unsqueeze(-1))
            if torch.isnan(zt).any() or torch.isinf(zt).any():
                print("NaN at step", step_idx)
                print("zt stats:", zt.min(), zt.max())
                break
            step_idx += 1
        return zt


    def generate_mts(self, batch_size=16):
        feature_size, seq_length = self.feature_size, self.seq_length
        return self.sample((batch_size, seq_length, feature_size))


    def _impute_loss(self, x_start, anomaly_label):
        # x_start == [1,2,3,4,5,6]
        # anomaly_label = [0,0,1,1,1,0]

        z0_impute = torch.randn_like(x_start) * anomaly_label.unsqueeze(-1) + x_start * (1 - anomaly_label.unsqueeze(-1)) #[1,2,noise,noise,noise,6]
        z1 = x_start # [1,2,3,4,5,6]

        t = torch.rand(z0_impute.shape[0], 1, 1).to(z0_impute.device)
        if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
            t = torch.sigmoid(torch.randn(z0_impute.shape[0], 1, 1)).to(z0_impute.device)

        z_t = t * z1 + (1. - t) * z0_impute # [1,2,3+noise,4+noise,5+noise,6]

        target = (z1 - z0_impute) * anomaly_label.unsqueeze(-1) # [0,0,3-noise, 4-noise, 5-noise, 0]

        model_out, mu_t, logvar_t = self.output(z_t, t.view(-1) * self.time_scalar, z1*anomaly_label.unsqueeze(-1))

        model_out = model_out * anomaly_label.unsqueeze(-1)

        """calculate flow loss"""
        # train_loss: (B, ..., ...)
        train_loss = ((model_out - target) ** 2).mean(-1) #(B, T)
        # 只对 anomaly 部分计算误差
        masked_loss = train_loss * anomaly_label #(B, T)
        # 每个样本 anomaly 的数量
        num_anomalies = reduce(anomaly_label, 'b t -> b 1', 'sum')  # shape: (B, 1)
        # 每个样本的 loss = sum(masked_loss) / num_anomalies
        loss_per_sample = reduce(masked_loss, 'b t -> b 1', 'sum') / num_anomalies
        flow_loss = loss_per_sample.mean()

        """kl loss"""
        kl_loss = self.variational_encoder.kl_loss(mu_t, logvar_t)

        return flow_loss + self.kl_beta * kl_loss, flow_loss, kl_loss


    def forward(self, x, anomaly_label):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        # if anomaly_label is None:
        #     return self._unconditional_loss(x_start=x)
        # else:
        #     return self._impute_loss(x_start=x, anomaly_label=anomaly_label)

        return self._impute_loss(x_start=x, anomaly_label=anomaly_label)






