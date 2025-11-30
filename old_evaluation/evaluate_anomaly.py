import os
import json
import torch
import numpy as np
from pytorch_anomaly_classification_metric import classification_metrics_torch   # <-- 你要把分类metric放这
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def load_normal_windows(filepath, window_size=800, stride=100, num=100):
    """
    读取 CSV → MinMaxScaler → 切 sliding windows → 返回前 num 个

    Args:
        filepath: CSV path
        window_size: window 长度
        stride: 滑动步长
        num: 只返回前 num 个 windows

    Returns:
        windows: shape (num, window_size, feature_dim)
    """
    # 1. Load CSV
    df = pd.read_csv(filepath, header=0)
    data = df.values  # (N, D)

    # 2. MinMaxScaler normalize
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data)

    # 3. Extract sliding windows
    windows = []
    N = len(data_norm)

    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        w = data_norm[start:end]
        windows.append(w)

        if len(windows) >= num:  # 已够 num 个，提前退出
            break

    windows = np.array(windows)
    return windows

# ==============================
# Settings
# ==============================
iterations = 5
output_jsonl = "anomaly_quality.jsonl"

# Clear old file
if os.path.exists(output_jsonl):
    os.remove(output_jsonl)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==============================
# Load dataset
# ==============================
# original data



# ori_data_anomaly = np.load("../output_path/MITDB_anomaly_finetune-dim64-heads4-enc3-dec2/samples/mitdb_anomaly_finetune_norm_truth_800_train.npy")
#
# gen_anomaly_data = np.load("../output_path/MITDB_anomaly_finetune-dim64-heads4-enc3-dec2/ddpm_fake_MITDB_anomaly_finetune.npy")
#
#
#
# ori_data_normal = load_normal_windows(
#     filepath="./Data/datasets/MITDB_normal_segment.csv",
#     window_size=800,
#     stride=100,
#     num=len(gen_anomaly_data))


all_data = torch.load("/root/tianyi/samples_path/flow/2025-11-29-06:31:47/all_data.pt")

ori_data_normal = all_data["orig_normal_train_signal"]
ori_data_anomaly = all_data["orig_anomaly_train_signal"]
gen_anomaly_data = all_data["orig_anomaly_train_signal"]


print("Loaded data:")
print("ori_normal =", ori_data_normal.shape)
print("ori_anomaly =", ori_data_anomaly.shape)
print("gen_anomaly =", gen_anomaly_data.shape)


# ==============================
# Helper: write JSONL
# ==============================
def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


# ==============================
# Summary helpers
# ==============================
def display_scores(scores, name="score"):
    arr = np.array(scores)
    print(f"{name}: mean={arr.mean():.4f}, std={arr.std():.4f}")


# ==============================
# Evaluation Loop
# ==============================
cls_acc_list = []
normal_acc_list = []
anomaly_acc_list = []
auc_list = []


for i in range(iterations):
    print(f"\n================ Iteration {i} ================")

    # 随机选取同样大小的 gen anomaly
    min_len = min(len(ori_data_anomaly), len(gen_anomaly_data))
    idx = np.random.choice(len(gen_anomaly_data), size=min_len, replace=False)
    gen_batch = gen_anomaly_data[idx]

    # ------------------------------
    # Compute classification metrics
    # ------------------------------
    result = classification_metrics_torch(
        ori_normal_data=ori_data_normal,
        ori_anomaly_data=ori_data_anomaly,
        gen_anomaly_data=gen_batch,
        hidden_dim=64,
        max_epochs=200,
        batch_size=64,
        patience=10,
        device=device
    )

    cls_acc = result["test_accuracy"]
    normal_acc = result["test_normal_acc"]
    anomaly_acc = result["test_anomaly_acc"]
    auc = result["auc"]

    cls_acc_list.append(cls_acc)
    normal_acc_list.append(normal_acc)
    anomaly_acc_list.append(anomaly_acc)
    auc_list.append(auc)

    print(f"classification_accuracy   = {cls_acc:.4f}")
    print(f"normal_accuracy          = {normal_acc:.4f}")
    print(f"anomaly_accuracy         = {anomaly_acc:.4f}")
    print(f"AUC                      = {auc:.4f}")

    # ------------------------------
    # Save each iteration to JSONL
    # ------------------------------
    record = {
        "iter": i,
        "classification_accuracy": float(cls_acc),
        "normal_accuracy": float(normal_acc),
        "anomaly_accuracy": float(anomaly_acc),
        "auc": float(auc),
    }
    append_jsonl(output_jsonl, record)


# ==============================
# Print Summary
# ==============================
print("\n====== Summary ======")
display_scores(cls_acc_list, "classification_accuracy")
display_scores(normal_acc_list, "normal_accuracy")
display_scores(anomaly_acc_list, "anomaly_accuracy")
display_scores(auc_list, "auc")

print(f"\nAll anomaly evaluation results saved to: {output_jsonl}")