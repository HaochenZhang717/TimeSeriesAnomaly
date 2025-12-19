from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import wandb
import torch
from vqvae import build_vqvae
from dataset import AnomalyDataset


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


@dataclass
class TrainConfig:
    max_length: int

    raw_data_path: str
    indices_path: str
    one_channel: bool

    # model parameters
    ts_dim:int
    num_class_tokens: int
    code_dim: int
    num_codes: int
    beta: float = 0.25


    # training parameters
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-2


    recon_loss: str = "mse"   # "mse" or "smoothl1"
    vq_loss_weight: float = 1.0

    device: str = "cuda"
    save_path: str = "vqvae_1d.pt"



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
    )

    # -------- model --------
    in_channels = 1 if cfg.one_channel else dataset.data.shape[1]
    model = build_vqvae(
        ts_dim=cfg.ts_dim,
        num_class_tokens=cfg.num_class_tokens,
        code_dim=cfg.code_dim,
        codebook_size=cfg.num_codes,
        max_len=cfg.max_length,
        beta=cfg.beta,
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

        for x, pad_mask in loader:
            x = x.to(device)                 # [B, T, C]
            pad_mask = pad_mask.to(device)
            B, T, C = x.shape

            x_hat, ids, vq_loss = model(x, pad_mask)
            rec_loss = recon_criterion(x_hat, x, pad_mask)

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


if __name__ == "__main__":

    cfg = TrainConfig(
        max_length=100,

        raw_data_path="../dataset_utils/ECG_datasets/raw_data/106.npz",
        indices_path="../dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/mixed.jsonl",
        one_channel=True,

        ts_dim=1,
        num_class_tokens=4,
        code_dim=8,
        num_codes=200,
        beta=0.25,

        batch_size=64,
        epochs=1000,
        lr=1e-4,
        weight_decay=1e-3,

        recon_loss="mse",
        vq_loss_weight=1.0,
        device="cuda:7",
        save_path="/root/tianyi/vqvae_save_path/transformer_vqvae_1d.pt",
        # device="cpu",
        # save_path="./transformer_vqvae_1d.pt",

    )
    model = train_vqvae(cfg)
