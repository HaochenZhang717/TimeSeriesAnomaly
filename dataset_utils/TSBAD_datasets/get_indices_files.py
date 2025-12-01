import matplotlib.pyplot as plt
import wfdb
import os
import numpy as np
from scipy.signal import decimate
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm



def extract_windows_from_record(
    signal,
    anomaly_label,
    source_name,
    window_size,
    stride,
    anomaly_map,
    max_anomaly_ratio
):

    class_windows = {k: [] for k in range(0, 2)}  # 0–6

    N = len(anomaly_label)

    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        seg = anomaly_label[start:end]

        # 无效区域
        if -1 in seg:
            continue

        anomaly_vals = seg[seg > 0]

        if len(anomaly_vals) == 0:
            anomaly_type = 0  # normal
        else:
            uniq = np.unique(anomaly_vals)
            if len(uniq) > 1:
                continue
            anomaly_type = int(uniq[0])

            # 异常比例不可超过阈值
            idxs = np.where(seg == anomaly_type)[0]
            if len(idxs) / window_size > max_anomaly_ratio:
                continue

        # 添加
        class_windows[anomaly_type].append({
            "source_file": source_name,
            "start": start,
            "end": end,
            "anomaly_type": anomaly_type
        })

    return class_windows


def build_single_ts_train_val(
    signal,
    anomaly_label,
    source_name,
    output_dir,
    window_size=800,
    stride=100,
    train_ratio=0.8,
    max_anomaly_ratio=0.2,
    anomaly_map={'anomaly':1}
):

    # 映射表
    name_map = {
        0: "normal",
        1: "anomaly",
    }

    # 初始化全局统计
    global_windows = {k: [] for k in range(0, 2)}


    # --- 遍历每个 record ---
    # data = np.load(npz_file)
    # signal = data["signal"]
    # anomaly_label = data["anomaly_label"]

    per_record = extract_windows_from_record(
        signal=signal,
        anomaly_label=anomaly_label,
        source_name=source_name,
        window_size=window_size,
        stride=stride,
        anomaly_map=anomaly_map,
        max_anomaly_ratio=max_anomaly_ratio
    )

    # 汇总
    for k in range(0, 2):
        global_windows[k].extend(per_record[k])


    # ----------- 开始写入 train/validation 文件 ------------
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    stats = {}

    for k in range(0, 2):
        fname = f"{name_map[k]}.jsonl"
        windows = global_windows[k]

        # split
        split_idx = int(len(windows) * train_ratio)
        train_list = windows[:split_idx]
        val_list = windows[split_idx:]

        # 写 train
        with open(os.path.join(train_dir, fname), "w") as f:
            for item in train_list:
                f.write(json.dumps(item) + "\n")

        # 写 val
        with open(os.path.join(val_dir, fname), "w") as f:
            for item in val_list:
                f.write(json.dumps(item) + "\n")

        stats[k] = {
            "total": len(windows),
            "train": len(train_list),
            "val": len(val_list)
        }

        print(f"{fname:12s} total={len(windows):6d}  train={len(train_list):6d}  val={len(val_list):6d}")

    return stats


# ----------------------- 使用示例 -----------------------
if __name__ == "__main__":

    # df = pd.read_csv("data.csv")  # 读 CSV
    # arr = df["column_name"].values  # 取一列并转成 numpy array

    folder = "./raw_data/selected_uts"

    for name in os.listdir(folder):
        full_path = os.path.join(folder, name)
        df = pd.read_csv(full_path)
        signal = df["Data"].values
        anomaly_labels = df["Label"].values
        stats = build_single_ts_train_val(
            signal=f"./raw_data/{name}.npz",
            anomaly_label=anomaly_labels,
            source_name=name.split(".")[0],
            output_dir=f"./indices/slide_windows_{name.split(".")[0]}",
            window_size=800,
            stride=10,
            train_ratio=1.0,
            max_anomaly_ratio=0.2
        )
        print(name)

    # for name in range(100, 235):
    #     print('-'*100)
    #     print(name)
    #     print('-'*100)
    #
    #     stats = build_single_ts_train_val(
    #         npz_file=f"./raw_data/{name}.npz",
    #         output_dir=f"./indices/slide_windows_{name}npz",
    #         window_size=800,
    #         stride=10,
    #         train_ratio=0.5,
    #         max_anomaly_ratio=0.2
    #     )