import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# code_segments 是一个 list，每个元素是 dict，包含 'ids': Tensor
# codebook_embedding 是 (V, D) 的 Tensor

def get_time_series_embedding(code_ids, codebook_embedding):
    """
    code_ids: Tensor of shape (T,)
    codebook_embedding: Tensor of shape (V, D)
    """
    emb = codebook_embedding[code_ids]  # shape: (T, D)
    # return emb.mean(dim=0)  # shape: (D,)
    return emb.flatten()  # shape: (D,)




code_segments = torch.load("code_segments.pt", map_location='cpu')
model_ckpt = torch.load("vqvae_1d.pt", map_location='cpu')
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

# Step 2: KMeans Clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(all_embeddings_np)

# Step 3: PCA for Visualization
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(all_embeddings_np)

# Step 4: Plot
plt.figure(figsize=(10, 6))
for i in range(num_clusters):
    idx = cluster_labels == i
    plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], label=f'Cluster {i}', alpha=0.7)
plt.title("Clustering of Time Series Code Embeddings (from list)")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()