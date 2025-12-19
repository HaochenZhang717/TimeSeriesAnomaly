import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
from mini_runnable import AnomalyDataset

# code_segments 是一个 list，每个元素是 dict，包含 'ids': Tensor
# codebook_embedding 是 (V, D) 的 Tensor



def find_nearest_neighbors(
    all_embeddings: torch.Tensor,
    anchor_idx: int,
    k: int = 5,
    metric: str = "l2",
):
    """
    all_embeddings: (N, D)
    anchor_idx: index in [0, N)
    k: number of nearest neighbors (excluding itself)
    metric: 'l2' or 'cosine'
    """
    anchor = all_embeddings[anchor_idx]  # (D,)

    if metric == "l2":
        dists = torch.norm(all_embeddings - anchor, dim=1)
    elif metric == "cosine":
        anchor_norm = anchor / anchor.norm()
        emb_norm = all_embeddings / all_embeddings.norm(dim=1, keepdim=True)
        dists = 1 - torch.matmul(emb_norm, anchor_norm)
    else:
        raise ValueError(f"Unknown metric {metric}")

    # 排序，去掉自己
    sorted_idx = torch.argsort(dists)
    neighbors = [i.item() for i in sorted_idx if i != anchor_idx][:k]

    return neighbors, dists[neighbors]



def plot_neighbor_time_series(
    code_segments,
    valid_indices,
    anchor_emb_idx,
    neighbor_emb_indices,
    signal_key="signal",
    max_len=None,
):
    """
    anchor_emb_idx: index in embedding space
    neighbor_emb_indices: list of indices in embedding space
    """
    all_idxs = [anchor_emb_idx] + neighbor_emb_indices
    n = len(all_idxs)

    plt.figure(figsize=(10, 2.5 * n))

    for row, emb_idx in enumerate(all_idxs):
        seg_idx = valid_indices[emb_idx]
        segment = code_segments[seg_idx]

        if signal_key not in segment:
            raise KeyError(f"segment {seg_idx} has no key '{signal_key}'")

        signal = segment[signal_key]
        if isinstance(signal, torch.Tensor):
            signal = signal.cpu().numpy()

        if max_len is not None:
            signal = signal[:max_len]

        plt.subplot(n, 1, row + 1)
        plt.plot(signal, lw=1.5)
        if row == 0:
            plt.title("Anchor Time Series", fontsize=12)
        else:
            plt.title(f"Neighbor {row}", fontsize=11)
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()



def inspect_embedding_neighborhood(
    all_embeddings,
    code_segments,
    valid_indices,
    anchor_emb_idx: int,
    k: int = 5,
    metric: str = "l2",
    signal_key: str = "signal",
):
    neighbors, dists = find_nearest_neighbors(
        all_embeddings, anchor_emb_idx, k=k, metric=metric
    )

    print("Anchor embedding index:", anchor_emb_idx)
    print("Nearest neighbors (embedding indices):")

    for neighbor_idx, (i, d) in enumerate(zip(neighbors, dists)):
        print(f" Neighbor{neighbor_idx}: idx={i}, dist={float(d):.4f}")

    plot_neighbor_time_series(
        code_segments,
        valid_indices,
        anchor_emb_idx,
        neighbors,
        signal_key=signal_key,
    )


def get_time_series_embedding(code_ids, codebook_embedding):
    """
    code_ids: Tensor of shape (T,)
    codebook_embedding: Tensor of shape (V, D)
    """
    emb = codebook_embedding[code_ids]  # shape: (T, D)
    # return emb.mean(dim=0)  # shape: (D,)
    return emb.flatten()  # shape: (D,)



if __name__ == "__main__":

    dataset = AnomalyDataset(
        raw_data_path="../dataset_utils/ECG_datasets/raw_data/106.npz",
        indices_path="../dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/mixed.jsonl",
        one_channel=True,
        max_length=100
    )

    code_segments = torch.load("/Users/zhc/Documents/PhD/projects/TimeSeriesAnomaly/vqvae_save_path/code_segments.pt", map_location='cpu')
    for i, code_segment in enumerate(code_segments):
        code_segment.update({'signal': dataset.__getitem__(i)})

    model_ckpt = torch.load("/Users/zhc/Documents/PhD/projects/TimeSeriesAnomaly/vqvae_save_path/vqvae_1d.pt", map_location='cpu')
    state_dict = model_ckpt["model_state"]
    # 自动查找 codebook 的 key
    for k in state_dict:
        if 'quantizer' in k and 'weight' in k:
            print(f"Found codebook: {k}")
            codebook_embedding = state_dict[k]
            print("Shape:", codebook_embedding.shape)
            break

    # Step 1: Compute embedding for each time series
    all_embeddings = []
    valid_indices = []  # 保存合法的索引，防止某些项有问题

    for i, segment in enumerate(code_segments):
        if 'ids' not in segment:
            continue
        code_ids = segment['ids']  # shape (T,)
        try:
            emb = get_time_series_embedding(code_ids, codebook_embedding)
            all_embeddings.append(emb)
            valid_indices.append(i)
        except Exception as e:
            print(f"Skipping segment {i} due to error: {e}")

    all_embeddings = torch.stack(all_embeddings)  # (N, D)
    all_embeddings_np = all_embeddings.cpu().numpy()

    # inspect_embedding_neighborhood(
    #     all_embeddings=all_embeddings,
    #     code_segments=code_segments,
    #     valid_indices=valid_indices,
    #     anchor_emb_idx=1,  # 任选一个
    #     k=200,
    #     metric="l2",
    #     signal_key="signal",
    # )

    # # Step 2: KMeans Clustering
    # num_clusters = 5
    # kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # cluster_labels = kmeans.fit_predict(all_embeddings_np)
    #
    # # Step 3: PCA for Visualization
    # pca = PCA(n_components=2)
    # emb_2d = pca.fit_transform(all_embeddings_np)
    #
    # # Step 4: Plot
    # plt.figure(figsize=(10, 6))
    # for i in range(num_clusters):
    #     idx = cluster_labels == i
    #     plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], label=f'Cluster {i}', alpha=0.7)
    # plt.title("Clustering of Time Series Code Embeddings (from list)")
    # plt.xlabel("PCA-1")
    # plt.ylabel("PCA-2")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


    # num_anomalies = 423
    # # the first 422 are anomaly
    # anomaly_codes = []
    # for i in range(num_anomalies):
    #     for code in code_segments[i]['ids']:
    #         anomaly_codes.append(code.item())
    #
    # plt.hist(anomaly_codes, bins=len(np.unique(anomaly_codes)))
    # plt.title("Histogram of anomaly codes")
    # plt.show()
    #
    # normal_codes = []
    # for i in range(num_anomalies, len(code_segments)):
    #     for code in code_segments[i]['ids']:
    #         normal_codes.append(code.item())
    # plt.hist(normal_codes, bins=len(np.unique(normal_codes)))
    # plt.title("Histogram of normal codes")
    # plt.show()



    num_anomalies = 423
    anomaly_codes = []
    for i in range(num_anomalies):
        for code in code_segments[i]['ids']:
            anomaly_codes.append(code.item())

    normal_codes = []
    for i in range(num_anomalies, len(code_segments)):
        for code in code_segments[i]['ids']:
            normal_codes.append(code.item())

    anomaly_codes = np.array(anomaly_codes)
    normal_codes = np.array(normal_codes)

    # 统一 bin：code index 是整数
    all_codes = np.concatenate([anomaly_codes, normal_codes])
    bins = np.arange(all_codes.min(), all_codes.max() + 2) - 0.5

    plt.figure(figsize=(8, 4))

    plt.hist(
        anomaly_codes,
        bins=bins,
        alpha=0.6,
        label="Anomaly",
    )

    plt.hist(
        normal_codes,
        bins=bins,
        alpha=0.6,
        label="Normal",
    )

plt.xlabel("VQ code index")
plt.ylabel("Count")
plt.title("VQ Code Usage: Anomaly vs Normal")
plt.legend()
plt.tight_layout()
plt.show()

