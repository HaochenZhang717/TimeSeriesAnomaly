import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, IterableDataset, DataLoader
import json
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import json

import torch
import random

from sklearn.cluster import KMeans




def info_nce_loss(z1, z2, temperature=0.1):
    """
    z1, z2: [B, D]
    """
    B = z1.size(0)

    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.T) / temperature  # cosine since normalized

    # mask self similarity
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # positive pairs
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels], dim=0)

    loss = F.cross_entropy(sim, labels)
    return loss


def augment_anomaly(
    x,
    noise_std=0.02,
    scale_range=(0.8, 1.2),
):
    """
    x: Tensor [L, C]
    """
    x = x.clone()

    # 1) linear (amplitude) scaling
    if scale_range is not None:
        scale = random.uniform(scale_range[0], scale_range[1])
        x = x * scale

    # 2) small Gaussian noise
    if noise_std > 0:
        noise = noise_std * torch.randn_like(x)
        x = x + noise

    return x


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
    ):
        super(AnomalyDataset, self).__init__()
        self.index_lines = load_jsonl(indices_path)
        self.raw_data_path = raw_data_path
        self.one_channel = one_channel

        raw_data = np.load(raw_data_path)
        raw_signal = raw_data["signal"]
        scaler = MinMaxScaler()
        normed_signal = scaler.fit_transform(raw_signal)
        self.data = normed_signal


    def __len__(self):
        return len(self.index_lines)


    def __getitem__(self, index):
        start, end = self.index_lines[index]
        if self.one_channel:
            return torch.from_numpy(self.data[start:end, :1])
        else:
            return torch.from_numpy(self.data[start:end])


def pad_collate_fn(batch):
    """
    batch: list of Tensor [L_i, C]
    """
    lengths = torch.tensor([x.shape[0] for x in batch], dtype=torch.long)
    max_len = lengths.max().item()
    C = batch[0].shape[-1]

    padded = torch.zeros(len(batch), max_len, C)

    for i, x in enumerate(batch):
        padded[i, :x.shape[0]] = x

    return padded, lengths


def pad_collate_fn_pairs(batch):
    """
    batch: list of (x1, x2), each [L_i, C]
    """
    x1_list, x2_list = zip(*batch)

    def pad(seq_list):
        lengths = torch.tensor([x.shape[0] for x in seq_list], dtype=torch.long)
        max_len = lengths.max().item()
        C = seq_list[0].shape[-1]

        padded = torch.zeros(len(seq_list), max_len, C)
        for i, x in enumerate(seq_list):
            padded[i, :x.shape[0]] = x
        return padded, lengths

    x1_pad, len1 = pad(x1_list)
    x2_pad, len2 = pad(x2_list)

    return x1_pad, x2_pad, len1, len2


class ContrastiveWrapper(Dataset):
    def __init__(
        self,
        base_dataset,
        augment_fn,
    ):
        """
        base_dataset: AnomalyDataset
        augment_fn: function(x) -> augmented x
        """
        self.base_dataset = base_dataset
        self.augment = augment_fn

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x = self.base_dataset[index]   # [L, C]

        x1 = self.augment(x)
        x2 = self.augment(x)

        return x1, x2



class AnomalyEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        z_dim=32,
        hidden_channels=64,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.proj = nn.Linear(hidden_channels, z_dim)

    def forward(self, x, lengths=None):
        """
        x: [B, T, C]   (padded)
        lengths: [B]   (valid lengths), optional but recommended
        """
        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        # temporal conv
        h = self.conv(x)            # [B, H, T]
        h = h.transpose(1, 2)       # [B, T, H]

        if lengths is not None:
            # pad-aware masked average pooling
            B, T, H = h.shape
            device = h.device

            mask = (
                torch.arange(T, device=device)[None, :] < lengths[:, None]
            )  # [B, T]

            mask = mask.unsqueeze(-1)   # [B, T, 1]
            h = h * mask

            pooled = h.sum(dim=1) / lengths.unsqueeze(-1)
        else:
            # fallback: simple average
            pooled = h.mean(dim=1)

        z = self.proj(pooled)
        z = F.normalize(z, dim=-1)   # 🔴 very important for InfoNCE

        return z



def train_contrastive_encoder(
    raw_data_path,
    indices_path,
    one_channel=True,
    batch_size=64,
    num_epochs=50,
    lr=1e-3,
    z_dim=32,
    device=None,
    log_every=5,
):
    """
    Train anomaly encoder with contrastive (InfoNCE) objective.

    Returns:
        encoder: trained AnomalyEncoder (frozen, eval mode)
    """

    # -------- device --------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- dataset --------
    base_dataset = AnomalyDataset(
        raw_data_path=raw_data_path,
        indices_path=indices_path,
        one_channel=one_channel,
    )

    contrastive_dataset = ContrastiveWrapper(
        base_dataset=base_dataset,
        augment_fn=augment_anomaly,
    )

    loader = DataLoader(
        contrastive_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=pad_collate_fn_pairs,
    )

    # -------- model --------
    encoder = AnomalyEncoder(
        in_channels=1 if one_channel else base_dataset.data.shape[1],
        z_dim=z_dim,
    ).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    # -------- training --------
    encoder.train()
    for epoch in range(num_epochs):
        total_loss = 0.0

        for x1, x2, len1, len2 in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            len1 = len1.to(device)
            len2 = len2.to(device)

            z1 = encoder(x1, len1)
            z2 = encoder(x2, len2)

            loss = info_nce_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(
                f"[Epoch {epoch+1:03d}/{num_epochs}] "
                f"InfoNCE loss = {avg_loss:.4f}"
            )

    # -------- freeze encoder --------
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    return encoder


def extract_embeddings(
    encoder,
    raw_data_path,
    indices_path,
    one_channel,
    batch_size=64,
    device=None,
):
    """
    Returns:
        Z: np.ndarray [N, D]   embeddings
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AnomalyDataset(
        raw_data_path=raw_data_path,
        indices_path=indices_path,
        one_channel=one_channel,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )

    encoder.eval()
    encoder.to(device)

    all_embeddings = []

    with torch.no_grad():
        for x, lengths in loader:
            x = x.to(device)
            lengths = lengths.to(device)

            z = encoder(x, lengths)     # [B, D]
            all_embeddings.append(z.cpu().numpy())

    Z = np.concatenate(all_embeddings, axis=0)
    return Z


def run_kmeans(Z, K=8, seed=0):
    kmeans = KMeans(
        n_clusters=K,
        n_init=20,
        max_iter=300,
        random_state=seed,
    )
    cluster_ids = kmeans.fit_predict(Z)
    centers = kmeans.cluster_centers_

    return cluster_ids, centers



def save_prototype_jsonl(
    indices_path,
    cluster_ids,
    out_path,
):
    """
    indices_path: 原始 anomaly segment jsonl
    cluster_ids: np.ndarray [N]
    """
    indices = load_jsonl(indices_path)
    assert len(indices) == len(cluster_ids)

    with open(out_path, "w") as f:
        for (start, end), cid in zip(indices, cluster_ids):
            item = {
                "start": int(start),
                "end": int(end),
                "prototype_id": int(cid),
            }
            f.write(json.dumps(item) + "\n")


def train_and_cluster():
    encoder = train_contrastive_encoder(
        raw_data_path="../dataset_utils/ECG_datasets/raw_data/106.npz",
        indices_path="../dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/raw_anomaly_segments.jsonl",
        one_channel=True,
        batch_size=64,
        num_epochs=100,
        lr=1e-3,
        z_dim=32,
        device=torch.device("cuda:0"),
    )

    # encoder 现在可以直接拿去做 embedding / clustering
    print("Encoder training finished.")

    Z = extract_embeddings(
        encoder=encoder,
        raw_data_path="../dataset_utils/ECG_datasets/raw_data/106.npz",
        indices_path="../dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/raw_anomaly_segments.jsonl",
        one_channel=True,
        batch_size=128,
        device=torch.device("cuda:0"),
    )

    print("Embedding shape:", Z.shape)  # [N, z_dim]
    cluster_ids, centers = run_kmeans(Z, K=8)

    save_prototype_jsonl(
        indices_path="../dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/raw_anomaly_segments.jsonl",
        cluster_ids=cluster_ids,
        out_path="anomaly_segments_with_prototype.jsonl",
    )



def load_prototype_jsonl(path):
    """
    Returns:
        segments: list of (start, end)
        cluster_ids: np.ndarray [N]
    """
    data = load_jsonl(path)

    segments = []
    cluster_ids = []

    for item in data:
        segments.append((item["start"], item["end"]))
        cluster_ids.append(item["prototype_id"])

    return segments, np.array(cluster_ids)


class AnomalyDatasetWithPrototype(Dataset):
    def __init__(self, raw_data_path, prototype_jsonl_path, one_channel):
        self.segments, self.cluster_ids = load_prototype_jsonl(prototype_jsonl_path)
        self.one_channel = one_channel

        raw_data = np.load(raw_data_path)
        raw_signal = raw_data["signal"]
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(raw_signal)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        start, end = self.segments[idx]
        seg = self.data[start:end]
        if self.one_channel:
            seg = seg[:, :1]
        return torch.from_numpy(seg), self.cluster_ids[idx]



def visualize_clusters_from_jsonl(
    raw_data_path,
    prototype_jsonl_path,
    one_channel=True,
    K=8,
    num_samples=10,
):
    dataset = AnomalyDatasetWithPrototype(
        raw_data_path=raw_data_path,
        prototype_jsonl_path=prototype_jsonl_path,
        one_channel=one_channel,
    )

    cluster_ids = dataset.cluster_ids

    results = torch.load(
        "/Users/zhc/Documents/PhD/projects/TimeSeriesAnomaly/samples_path/PrototypeFlow/samples.pth",
        map_location='cpu'
    )
    MAX_LEN = 100


    for k in range(K):
        idxs = np.where(cluster_ids == k)[0]
        if len(idxs) == 0:
            continue

        print(f"\nPrototype {k}, #samples = {len(idxs)}")
        chosen = np.random.choice(idxs, min(num_samples, len(idxs)), replace=False)

        plt.figure(figsize=(8, 2))
        for i in chosen:
            seg, _ = dataset[i]
            plt.plot(seg.numpy().flatten()[:MAX_LEN], alpha=0.7)

        plt.title(f"real prototype {k}")
        # plt.show()
        plt.savefig(f"real_prototype_{k}.pdf")
        plt.close()
        # samples = results[k]
        # plt.figure(figsize=(8, 2))
        # for j in range(samples.shape[0]):
        #     plt.plot(samples[j, :])
        # plt.title(f"samples prototype {k}")
        # plt.show()
        print(K)

if __name__ == "__main__":
    visualize_clusters_from_jsonl(
        raw_data_path="../dataset_utils/ECG_datasets/raw_data/106.npz",
        prototype_jsonl_path="anomaly_segments_with_prototype.jsonl",
        one_channel=True,
        K=8,
        num_samples=100
    )






