import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math


class ConvMeanPool1DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        pool_kernel=4,
        pool_stride=4,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

        self.pool = nn.AvgPool1d(
            kernel_size=pool_kernel,
            stride=pool_stride
        )


        self.res_proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        """
        x: (B, C, T)
        return: (B, out_channels, T')
        """
        h = self.conv(x)


        h = h + self.res_proj(x)

        h = self.pool(h)
        return h


class TemporalVariationalEncoder1D(nn.Module):
    """
    Produces a temporal latent z_t (B, z_dim, T') instead of a single vector.
    """
    def __init__(
        self,
        in_channels=1,
        seq_len=1800,
        channels=[64, 128, 256],
        kernel_size=7,
        pool_kernel=2,
        pool_stride=2,
        z_dim=16,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_blocks = len(channels)
        T = seq_len
        for _ in range(self.num_blocks):
            T = math.floor((T - pool_kernel) / pool_stride + 1)
        self.num_tokens = T

        blocks = []
        ch_in = in_channels
        for ch_out in channels:
            blocks.append(
                ConvMeanPool1DBlock(
                    in_channels=ch_in,
                    out_channels=ch_out,
                    kernel_size=kernel_size,
                    pool_kernel=pool_kernel,
                    pool_stride=pool_stride,
                )
            )
            ch_in = ch_out

        self.backbone = nn.Sequential(*blocks)  # (B, C_last, T')
        c_last = channels[-1]

        # 1x1 conv heads produce per-time-step variational params
        self.to_mu = nn.Conv1d(c_last, z_dim, kernel_size=1)
        self.to_logvar = nn.Conv1d(c_last, z_dim, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C, T)
        returns:
          h: (B, C_last, T')
          mu_t: (B, z_dim, T')
          logvar_t: (B, z_dim, T')
          z_t: (B, z_dim, T')
        """

        h = self.backbone(x)
        mu_t = self.to_mu(h)
        logvar_t = self.to_logvar(h)

        std_t = torch.exp(0.5 * logvar_t)
        qz_t = Normal(mu_t, std_t)
        z_t = qz_t.rsample()

        return h, mu_t, logvar_t, z_t

    def kl_loss(self, mu_t, logvar_t):
        kl = 0.5 * (
                mu_t.pow(2) + logvar_t.exp() - 1 - logvar_t
        )  # (B, z_dim, T')

        kl = kl.sum(dim=1)  # sum over z_dim → (B, T')
        kl = kl.mean(dim=1)  # mean over time → (B,)
        kl_loss = kl.mean()  # mean over batch → scalar
        return kl_loss

    def sample_prior_latent(self, batch_size):
        device = next(self.parameters()).device
        mu = torch.zeros(batch_size, self.to_mu.out_channels, self.num_tokens, device=device)
        std = torch.ones_like(mu)
        pz = Normal(mu, std)
        return pz.sample()

if __name__ == '__main__':
    # block = ConvMeanPool1DBlock(
    #     in_channels=1,
    #     out_channels=64,
    #     kernel_size=3,
    #     pool_kernel=4,
    #     pool_stride=4,
    # )
    model= TemporalVariationalEncoder1D(
        in_channels=1,
        channels=(16, 32, 64),
        kernel_size=3,
        pool_kernel=4,
        pool_stride=4,
        z_dim=16,
    )
    input = torch.randn(2, 1, 1800)
    h, mu_t, logvar_t, z_t = model(input)
    print(h.shape)
    print(mu_t.shape)
    print(logvar_t.shape)
    print(z_t.shape)