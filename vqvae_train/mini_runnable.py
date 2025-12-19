import os
import json
import math
import random
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
import wandb

import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch

# --------------------------
# Dataset (copied from you)
# --------------------------

def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


class AnomalyDataset(Dataset):
    def __init__(
            self,
            raw_data_path,
            indices_path,
            one_channel,
            max_length,
    ):
        super().__init__()
        self.index_lines = load_jsonl(indices_path)
        self.raw_data_path = raw_data_path
        self.one_channel = one_channel
        self.max_length = max_length

        raw_data = np.load(raw_data_path)
        raw_signal = raw_data["signal"]
        scaler = MinMaxScaler()
        normed_signal = scaler.fit_transform(raw_signal)
        self.data = normed_signal

    def __len__(self):
        return len(self.index_lines)

    def __getitem__(self, index):
        # start, end = self.index_lines[index]
        start = self.index_lines[index]["start"]
        end = self.index_lines[index]["end"]
        if end - start > self.max_length:
            end = start + self.max_length
        # if self.one_channel:
        #     return torch.from_numpy(self.data[start:end, :1]).float()
        # else:
        #     return torch.from_numpy(self.data[start:end]).float()

        ts_dim = self.data.shape[1]
        signal = torch.zeros(self.max_length, ts_dim, dtype=torch.float32)
        signal[:end-start] = torch.from_numpy(self.data[start:end]).float()
        if self.one_channel:
            return signal[:, :1]
        else:
            return signal

def pad_collate_fn(batch):
    """
    batch: list of Tensor [L_i, C]
    returns:
      padded:  [B, T, C]
      lengths: [B]
    """
    lengths = torch.tensor([x.shape[0] for x in batch], dtype=torch.long)
    max_len = lengths.max().item()
    C = batch[0].shape[-1]

    padded = torch.zeros(len(batch), max_len, C, dtype=batch[0].dtype)
    for i, x in enumerate(batch):
        padded[i, :x.shape[0]] = x

    return padded, lengths


# --------------------------
# Utils: pad mask
# --------------------------

def make_valid_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """
    lengths: [B]
    returns: [B, T] float mask, 1 for valid, 0 for pad
    """
    device = lengths.device
    return (torch.arange(T, device=device)[None, :] < lengths[:, None]).float()


# --------------------------
# Vector Quantizer (VQ-VAE)
# --------------------------

class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE (nearest neighbor) with straight-through gradient.

    z_e: [B, T', D]  (encoder outputs)
    z_q: [B, T', D]  (quantized)
    ids: [B, T']     (code indices)
    loss: vq loss (codebook + commitment)
    """
    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e: torch.Tensor):
        B, T, D = z_e.shape
        assert D == self.code_dim

        flat = z_e.reshape(B * T, D)  # [BT, D]
        e = self.codebook.weight      # [K, D]

        # squared euclidean distances: ||x||^2 - 2 x·e + ||e||^2
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ e.t()
            + e.pow(2).sum(dim=1, keepdim=True).t()
        )  # [BT, K]

        ids = torch.argmin(dist, dim=1)             # [BT]
        z_q = self.codebook(ids).view(B, T, D)      # [B, T, D]

        # VQ losses
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss   = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, ids.view(B, T), vq_loss


# --------------------------
# 1D Conv VQ-VAE
# --------------------------

class ResBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        downsample=False,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.downsample = downsample or (stride != 1) or (in_channels != out_channels)

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

        if self.downsample:
            self.skip = nn.Conv1d(
                in_channels, out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.act(out)
        return out


class ResNetEncoder1D(nn.Module):
    """
    Input:  x [B, T, C]
    Output: z_e [B, T', D]
    """
    def __init__(
        self,
        in_channels,
        channels=(64, 128, 256),
        blocks_per_stage=2,
        code_dim=128,
        kernel_size=3,
        down_ratio=2,
        code_len=4
    ):
        super().__init__()

        self.stem = nn.Conv1d(
            in_channels, channels[0],
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

        stages = []
        in_ch = channels[0]

        for stage_idx, out_ch in enumerate(channels):
            for block_idx in range(blocks_per_stage):
                # downsample only at first block of each stage (except stage 0)
                stride = down_ratio if (block_idx == 0 and stage_idx > 0) else 1
                stages.append(
                    ResBlock1D(
                        in_ch, out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                    )
                )
                in_ch = out_ch

        self.stages = nn.Sequential(*stages)

        self.global_pooling = nn.AdaptiveAvgPool1d(code_len)
        self.proj = nn.Conv1d(
            in_ch, code_dim,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        # x: [B, T, C] → [B, C, T]
        x = x.transpose(1, 2)
        h = self.stem(x)
        h = self.stages(h)
        z = self.proj(h)          # [B, D, T']
        z = self.global_pooling(z)
        z = z.transpose(1, 2)     # [B, T', D]
        return z


class UpResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, up_ratio):
        super().__init__()
        padding = kernel_size // 2

        self.upsample = nn.Upsample(scale_factor=up_ratio, mode="nearest")

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.upsample(x)
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.act(out)
        return out


class ResNetDecoder1D(nn.Module):
    """
    Input:  z_q [B, T', D]
    Output: x_hat [B, T, C]
    """
    def __init__(
        self,
        out_channels,
        channels=(256, 128, 64),
        blocks_per_stage=1,
        code_dim=128,
        kernel_size=3,
        up_ratio=2,
        code_len=123,
        seq_len=123
    ):
        super().__init__()

        stages = []
        in_ch = channels[0]

        for out_ch in channels[1:]:
            for _ in range(blocks_per_stage):
                stages.append(
                    UpResBlock1D(
                        in_ch, out_ch,
                        kernel_size=kernel_size,
                        up_ratio=up_ratio,
                    )
                )
                in_ch = out_ch

        self.stages = nn.Sequential(*stages)

        self.head = nn.Conv1d(
            in_ch, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        n_upsample = len(channels) - 1
        T0 = math.ceil(seq_len / (up_ratio ** n_upsample))

        self.input_proj_1 = nn.Linear(code_dim, channels[0])   # channel
        self.input_proj_2 = nn.Linear(code_len, T0)            # time


    def forward(self, z, target_len):
        '''
        z: [B, T, C]
        '''
        z = self.input_proj_1(z)

        z = z.transpose(1, 2)           # [B, C0, 4]
        z = self.input_proj_2(z)
        h = self.stages(z)
        x_hat = self.head(h)       # [B, C, T_recon]
        x_hat = x_hat.transpose(1, 2)

        # crop / pad to target_len
        if x_hat.size(1) > target_len:
            x_hat = x_hat[:, :target_len]
        elif x_hat.size(1) < target_len:
            x_hat = F.pad(x_hat, (0, 0, 0, target_len - x_hat.size(1)))

        return x_hat


class VQVAE1D(nn.Module):
    def __init__(
        self,
        in_channels, encoder_channels, decoder_channels, code_dim, num_codes,
        down_ratio, up_ratio, code_len, seq_len
    ):
        super().__init__()
        # self.encoder = ResNetEncoder1D(
        #     in_channels=in_channels,
        #     channels=(16, 16, 32, 32, 64, 64),
        #     blocks_per_stage=1,
        #     code_dim=code_dim,
        # )
        self.encoder = ResNetEncoder1D(
            in_channels=in_channels,
            channels=encoder_channels,
            blocks_per_stage=1,
            code_dim=code_dim,
            down_ratio=down_ratio,
            code_len=code_len
        )
        self.quantizer = VectorQuantizer(num_codes, code_dim)
        # self.decoder = ResNetDecoder1D(
        #     out_channels=in_channels,
        #     channels=(64, 64, 32, 32, 16, 16),
        #     blocks_per_stage=1,
        #     code_dim=code_dim,
        # )
        self.decoder = ResNetDecoder1D(
            out_channels=in_channels,
            channels=decoder_channels,
            blocks_per_stage=1,
            code_dim=code_dim,
            up_ratio=up_ratio,
            code_len=code_len,
            seq_len=seq_len
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, ids, vq_loss = self.quantizer(z_e)
        x_hat = self.decoder(z_q, x.size(1))
        return x_hat, ids, vq_loss

# --------------------------
# Train / Eval
# --------------------------

@dataclass
class TrainConfig:
    encoder_channels: tuple
    decoder_channels: tuple
    down_ratio: int
    up_ratio: int
    max_length: int

    raw_data_path: str
    indices_path: str
    one_channel: bool = True

    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-2

    hidden: int = 64
    code_dim: int = 128
    code_len: int = 4
    num_codes: int = 1024
    beta: float = 0.25

    recon_loss: str = "mse"   # "mse" or "smoothl1"
    vq_loss_weight: float = 1.0

    device: str = "cuda"
    save_path: str = "vqvae_1d.pt"




@torch.no_grad()
def extract_token_ids(model: VQVAE1D, loader: DataLoader, device: torch.device):
    """
    Returns list of (ids, lengths_Tprime), where ids is [B, T'].
    Note: lengths in T' are approx lengths/8 (floor-ish), so we compute it.
    """
    model.eval()
    all_ids = []
    all_tprime_lens = []

    for x, lengths in loader:
        x = x.to(device)
        lengths = lengths.to(device)
        _, ids, _, z_e, _ = model(x)  # ids [B, T']

        # Approx downsample factor = 8 for 3 stride-2 layers.
        # length after convs depends on padding; this approximation is usually ok.
        tprime = ids.size(1)
        # We can infer valid token lengths by mapping original valid mask through encoder length
        # Simple heuristic:
        tprime_lens = torch.clamp((lengths.float() / 8.0).floor().long(), min=1, max=tprime)

        all_ids.append(ids)
        all_tprime_lens.append(tprime_lens)

    return all_ids, all_tprime_lens


@torch.no_grad()
def _compute_codebook_stats(epoch_ids, num_codes):
    """
    epoch_ids: list of Tensor [B, T']
    """
    flat_ids = torch.cat([ids.reshape(-1) for ids in epoch_ids], dim=0)

    counts = torch.bincount(flat_ids, minlength=num_codes).float()
    total = counts.sum()

    active_code_ratio = (counts > 0).float().mean().item()

    probs = counts / (total + 1e-9)
    perplexity = torch.exp(
        -(probs * torch.log(probs + 1e-9)).sum()
    ).item()

    dead_codes = (counts == 0).sum().item()

    return {
        "active_code_ratio": active_code_ratio,
        "perplexity": perplexity,
        "dead_codes": dead_codes,
        "counts": counts.cpu().numpy(),
    }


def train_vqvae(cfg: TrainConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # -------- wandb --------
    wandb.init(
        project="vqvae-ts",
        name=f"vqvae_K{cfg.num_codes}_D{cfg.code_dim}",
        config=cfg.__dict__,
    )

    # -------- dataset / loader --------
    dataset = AnomalyDataset(
        raw_data_path=cfg.raw_data_path,
        indices_path=cfg.indices_path,
        one_channel=cfg.one_channel,
        max_length=cfg.max_length
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=pad_collate_fn,
    )

    # -------- model --------
    in_channels = 1 if cfg.one_channel else dataset.data.shape[1]
    model = VQVAE1D(
        in_channels=in_channels,
        encoder_channels=cfg.encoder_channels,
        decoder_channels=cfg.decoder_channels,
        code_dim=cfg.code_dim,
        num_codes=cfg.num_codes,
        down_ratio=cfg.down_ratio,
        up_ratio=cfg.up_ratio,
        code_len=cfg.code_len,
        seq_len=cfg.max_length
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    def recon_criterion(x_hat, x, valid_mask):
        # valid_mask: [B, T, 1]
        per = (x_hat - x) ** 2
        per = per * valid_mask
        return per.sum() / (valid_mask.sum() + 1e-6)

    # -------- training --------
    model.train()
    for ep in range(cfg.epochs):
        total = 0.0
        total_rec = 0.0
        total_vq = 0.0

        epoch_ids = []   # 👈 收集整个 epoch 的 code ids

        for x, lengths in loader:
            x = x.to(device)                 # [B, T, C]
            lengths = lengths.to(device)
            B, T, C = x.shape

            valid = make_valid_mask(lengths, T).unsqueeze(-1)  # [B, T, 1]

            x_hat, ids, vq_loss = model(x)
            rec_loss = recon_criterion(x_hat, x, valid)

            loss = rec_loss + cfg.vq_loss_weight * vq_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += loss.item()
            total_rec += rec_loss.item()
            total_vq += vq_loss.item()

            epoch_ids.append(ids.detach())

        # -------- codebook stats --------
        stats = _compute_codebook_stats(epoch_ids, cfg.num_codes)

        # -------- logging --------
        avg_loss = total / len(loader)
        avg_rec = total_rec / len(loader)
        avg_vq = total_vq / len(loader)

        wandb.log({
            "epoch": ep + 1,
            "loss": avg_loss,
            "recon_loss": avg_rec,
            "vq_loss": avg_vq,
            "active_code_ratio": stats["active_code_ratio"],
            "perplexity": stats["perplexity"],
            "dead_codes": stats["dead_codes"],
        })

        # 每 5 个 epoch 画一次 usage histogram
        if (ep + 1) % 5 == 0:
            wandb.log({
                "codebook_usage_hist": wandb.Histogram(stats["counts"])
            })

        print(
            f"[Epoch {ep+1:03d}/{cfg.epochs}] "
            f"loss={avg_loss:.6f}  "
            f"rec={avg_rec:.6f}  "
            f"vq={avg_vq:.6f}  "
            f"active={stats['active_code_ratio']:.3f}  "
            f"ppl={stats['perplexity']:.1f}  "
            f"dead={stats['dead_codes']}"
        )

    # -------- save --------
    # os.makedirs(cfg.save_path, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
        },
        cfg.save_path,
    )
    print(f"Saved: {cfg.save_path}")

    wandb.finish()
    return model





@torch.no_grad()
def extract_code_segments(
    in_channels,
    code_dim,
    num_codes,
    model_path,
    raw_data_path,
    indices_path,
    one_channel,
    device,
    save_path,
    down_ratio,
    up_ratio,
    max_length,
    encoder_channels,
    decoder_channels,
):
    model = VQVAE1D(
        in_channels=in_channels,
        code_dim=code_dim,
        num_codes=num_codes,
        down_ratio=down_ratio,
        up_ratio=up_ratio,
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
    ).to(device)
    model.eval()
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # model.load_state_dict(torch.load(model_path, map_location=device))

    code_segments = defaultdict(list)

    dataset = AnomalyDataset(
        raw_data_path=raw_data_path,
        indices_path=indices_path,
        one_channel=one_channel,
        max_length=max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        collate_fn=pad_collate_fn,
    )

    all_codes = []
    for x, lengths in tqdm(loader, desc="Extracting code ids"):
        x = x.to(device)  # [B, T, C]
        lengths = lengths.to(device)

        z_e = model.encoder(x)
        _, ids, _ = model.quantizer(z_e)  # ids: [B, T']

        B, Tprime = ids.shape

        for b in range(B):
            entry = {
                "ids": ids[b].detach().cpu(),  # [T']
                "orig_len": int(lengths[b].item()),
                "tprime_len": int(Tprime),
            }
            all_codes.append(entry)

    torch.save(all_codes, save_path)
    print(f"[Saved] time-series codes -> {save_path}")



# def plot_code_waveforms(
#     code_segments_path,
#     code_ids=None,
#     num_samples=5,
#     show_mean=True,
#     figsize=(12, 4),
# ):
#     """
#     从保存的 code_segments.pt 中加载并画图
#
#     Args:
#         code_segments_path: extract_code_segments 保存的路径
#         code_ids: 要画的 code 列表（None = 随机选几个）
#         num_samples: 每个 code 画多少条 sample
#         show_mean: 是否画 mean waveform
#     """
#     code_segments = torch.load(code_segments_path)
#
#     all_codes = sorted(code_segments.keys())
#     if code_ids is None:
#         code_ids = all_codes[:8]   # 默认画前 8 个
#
#     for k in code_ids:
#         segs = code_segments[k]    # [N, L]
#
#         plt.figure(figsize=figsize)
#
#         # sample
#         for i in range(min(num_samples, len(segs))):
#             plt.plot(segs[i], color="gray", alpha=0.3)
#
#         # mean
#         if show_mean:
#             mean_wave = segs.mean(axis=0)
#             plt.plot(mean_wave, color="red", linewidth=2, label="mean")
#
#         plt.title(f"Code {k}  |  N={len(segs)}")
#         plt.xlabel("Time")
#         plt.ylabel("Amplitude")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()



def plot_code_waveforms(
    code_segments_path,
    code_ids=None,
    num_samples=5,
    show_mean=True,
    figsize=(12, 4),
):
    """
    Args:
        code_segments_path: extract_code_segments 保存的路径
        code_ids: 要画的 sample 的索引列表，例如 [0, 5, 10]
    """
    code_segments = torch.load(code_segments_path)

    # code_segments 是 list，不是 dict
    total = len(code_segments)
    if code_ids is None:
        code_ids = list(range(min(8, total)))  # 默认画前 8 个

    for idx in code_ids:
        entry = code_segments[idx]
        ids = entry["ids"].numpy()  # [T']
        orig_len = entry["orig_len"]
        tprime_len = entry["tprime_len"]

        plt.figure(figsize=figsize)
        plt.plot(ids, color="gray", alpha=0.7)

        plt.title(f"Sample {idx}  |  orig_len={orig_len}, tprime_len={tprime_len}")
        plt.xlabel("Timestep")
        plt.ylabel("Code ID")
        plt.tight_layout()
        plt.show()

# --------------------------
# Example usage
# --------------------------

if __name__ == "__main__":
    cfg = TrainConfig(
        encoder_channels=(16,16,32,32,64,64),
        decoder_channels=(64,64,32,32,16,16),
        down_ratio=2,
        up_ratio=2,
        max_length=800,

        raw_data_path="../dataset_utils/ECG_datasets/raw_data/106.npz",
        indices_path="../dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/mixed.jsonl",
        one_channel=True,

        batch_size=64,
        epochs=50,
        lr=1e-3,

        hidden=64,
        code_dim=8,
        code_len=4,
        num_codes=300,
        beta=0.25,

        recon_loss="mse",
        vq_loss_weight=1.0,
        device="cuda:7",
        # device="cpu",
        save_path="/root/tianyi/vqvae_save_path/vqvae_1d.pt",
    )

    model = train_vqvae(cfg)

    extract_code_segments(
        in_channels=1,
        code_dim=8,
        num_codes=300,
        model_path="/root/tianyi/vqvae_save_path/vqvae_1d.pt",
        raw_data_path="../dataset_utils/ECG_datasets/raw_data/106.npz",
        indices_path="../dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/mixed.jsonl",
        one_channel=True,
        device="cuda:7",
        save_path="/root/tianyi/vqvae_save_path/code_segments.pt",
        down_ratio=2,
        up_ratio=2,
        max_length=100,
        encoder_channels=(16,16,32,32,64,64),
        decoder_channels=(64,64,32,32,16,16),
    )

    # 2️⃣ 离线画图
    # plot_code_waveforms(
    #     "code_segments.pt",
    #     code_ids=[0, 5, 12, 42],
    #     num_samples=10,
    # )