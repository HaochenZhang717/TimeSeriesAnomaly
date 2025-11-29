import matplotlib.pyplot as plt
import wfdb
import os
import numpy as np
from scipy.signal import decimate
import os
import json
import numpy as np
from tqdm import tqdm



def convert_to_npy(base_path, save_path):
    # base_path = "/Users/zhc/Downloads/MIT-BIH_Atrial_Fibrillation_Database/"
    # save_path = "/Users/zhc/Downloads/AFDB_npy/"
    os.makedirs(save_path, exist_ok=True)

    # 扫描所有记录（以 hea 或 atr 为主）
    records = sorted([f[:-4] for f in os.listdir(base_path) if f.endswith(".dat")])

    for rec in records:
        print("Processing:", rec)

        # --------------------------
        # 读取波形
        # --------------------------
        record = wfdb.rdrecord(os.path.join(base_path, rec))
        signal = record.p_signal.astype(np.float32)  # (N, 2)
        N = signal.shape[0]

        # --------------------------
        # 初始化逐点标签
        # --------------------------
        anomaly_label = np.zeros(N, dtype=np.int8)

        # --------------------------
        # 读取 atr 节律标注
        # --------------------------
        try:
            ann = wfdb.rdann(os.path.join(base_path, rec), "atr")
            aux_note = ann.aux_note
            samples = ann.sample
        except:
            print("⚠ No atr annotation for", rec)
            aux_note = []
            samples = []

        # --------------------------
        # 逐段展开 AFIB → 1，其余 → 0
        # --------------------------
        if len(samples) > 0:
            for i in range(len(samples)):
                label = aux_note[i]
                start = samples[i]

                if i < len(samples) - 1:
                    end = samples[i+1]
                else:
                    end = N   # 最后一段直到录音结束

                if "(AFIB" in label:
                    anomaly_label[start:end] = 1
                else:
                    anomaly_label[start:end] = 0

                print(label, start, "→", end)

        # --------------------------
        # 读取 hea 文件（可选）
        # --------------------------
        with open(os.path.join(base_path, rec + ".hea"), "r") as f:
            hea_text = f.read()

        # --------------------------
        # 保存为 npz
        # --------------------------
        np.savez(
            os.path.join(save_path, rec + ".npz"),
            signal=signal,          # (N, 2)
            fs=record.fs,           # sampling rate
            ann_sample=np.array(samples),
            ann_aux_note=np.array(aux_note),
            anomaly_label=anomaly_label,   # (N,) 0/1
            hea_text=hea_text
        )

    print("\nDone! All files converted to .npz")


def convert_svdb_to_npy():
    base_path = "/Users/zhc/Downloads/mit-bih-supraventricular-arrhythmia-database-1.0.0/"
    save_path = "/Users/zhc/Downloads/SVDB_npy/"
    os.makedirs(save_path, exist_ok=True)
    # "mit-bih-arrhythmia-database-1.0.0"

    # 扫描所有记录（以 hea 或 atr 为主）
    records = sorted([f[:-4] for f in os.listdir(base_path) if f.endswith(".dat")])

    for rec in records:
        print("Processing:", rec)

        # --------------------------
        # 读取波形
        # --------------------------
        record = wfdb.rdrecord(os.path.join(base_path, rec))
        signal = record.p_signal.astype(np.float32)  # (N, 2)
        N = signal.shape[0]

        # --------------------------
        # 初始化逐点标签
        # --------------------------
        anomaly_label = np.zeros(N, dtype=np.int8)

        # --------------------------
        # 读取 atr 节律标注
        # --------------------------
        try:
            ann = wfdb.rdann(os.path.join(base_path, rec), "atr")
            # aux_note = ann.aux_note
            samples = ann.sample
            symbol = ann.symbol
        except:
            print("⚠ No atr annotation for", rec)
            aux_note = []
            samples = []

        # --------------------------
        # 逐段展开 AFIB → 1，其余 → 0
        # --------------------------
        if len(samples) > 0:
            for i in range(len(samples)):
                label = symbol[i]
                start = samples[i]

                if i < len(samples) - 1:
                    end = samples[i+1]
                else:
                    end = N   # 最后一段直到录音结束

                if label not in ["N"]:
                    anomaly_label[start:end] = 1
                else:
                    anomaly_label[start:end] = 0

                print(label, start, "→", end)

        # --------------------------
        # 读取 hea 文件（可选）
        # --------------------------
        with open(os.path.join(base_path, rec + ".hea"), "r") as f:
            hea_text = f.read()

        # --------------------------
        # 保存为 npz
        # --------------------------
        np.savez(
            os.path.join(save_path, rec + ".npz"),
            signal=signal,          # (N, 2)
            fs=record.fs,           # sampling rate
            ann_sample=np.array(samples),
            ann_symbol=np.array(symbol),
            anomaly_label=anomaly_label,   # (N,) 0/1
            hea_text=hea_text
        )

    print("\nDone! All files converted to .npz")


def show_npy(q, data_path):   # q=2 → 250Hz → 125Hz
    # data_path = "/Users/zhc/Downloads/AFDB_npy/"

    records = [f for f in os.listdir(data_path) if f.endswith(".npz")]

    for rec in records:
        print("Processing:", rec)
        record = np.load(os.path.join(data_path, rec))

        signal = record["signal"]          # (N, 2)
        anomaly_label = record["anomaly_label"]  # (N,)
        ann_aux_note = record["ann_aux_note"]
        ann_sample = record["ann_sample"]

        for i, (note, start_time) in enumerate(zip(ann_aux_note, ann_sample)):
            selected_signal = signal[start_time:start_time+5000]
            selected_label = anomaly_label[start_time:start_time+5000]

            selected_signal_ds = decimate(selected_signal, q=q, axis=0)  # 降 q 倍
            selected_anomaly_label_ds = selected_label[::q]

            plt.figure(figsize=(12, 4))
            plt.plot(selected_signal_ds[:, 0], label="ECG channel 1 (downsampled)")
            plt.plot(selected_signal_ds[:, 1], label="ECG channel 2 (downsampled)")
            plt.plot(selected_anomaly_label_ds, label="AFIB Label")
            plt.legend()
            plt.title(i)
            plt.show()

        print("123")
        break


def show_svdb_npy(q):   # q=2 → 250Hz → 125Hz
    # data_path = "/Users/zhc/Downloads/AFDB_npy/"
    data_path="/Users/zhc/Downloads/SVDB_npy/"
    # "mit-bih-arrhythmia-database-1.0.0"
    records = [f for f in os.listdir(data_path) if f.endswith(".npz")]

    for rec in records:
        print("Processing:", rec)
        record = np.load(os.path.join(data_path, rec))

        signal = record["signal"]          # (N, 2)
        anomaly_label = record["anomaly_label"]  # (N,)
        ann_symbol = record["ann_symbol"]
        ann_sample = record["ann_sample"]

        for i, (note, start_time) in enumerate(zip(ann_symbol, ann_sample)):
            # if note in ['S','N']:
            if note in ['N']:
                continue
            selected_signal = signal[start_time:start_time+500]
            selected_label = anomaly_label[start_time:start_time+500]

            # selected_signal_ds = decimate(selected_signal, q=q, axis=0)  # 降 q 倍
            # selected_anomaly_label_ds = selected_label[::q]
            selected_signal_ds = selected_signal
            selected_anomaly_label_ds = selected_label

            plt.figure(figsize=(12, 4))
            plt.plot(selected_signal_ds[:, 0], label="ECG channel 1 (downsampled)")
            plt.plot(selected_signal_ds[:, 1], label="ECG channel 2 (downsampled)")
            plt.plot(selected_anomaly_label_ds, label="anomaly Label")
            plt.legend()
            plt.title(f"{note}-{i}")
            plt.show()

        print("123")
        break


def convert_mitdb_to_npy():

    # anomalies we are going to use for training and testing
    # anomaly_nums = {'V': 7130, 'A': 2546, 'F': 803, 'L': 8075, 'R': 7259, '/': 7028}
    anomaly_map = {'V': 1, 'A': 2, 'F': 3, 'L': 4, 'R': 5, '/': 6}

    base_path = "/Users/zhc/Downloads/mit-bih-arrhythmia-database-1.0.0/"
    save_path = "/Users/zhc/Downloads/MITDB_npy/"
    os.makedirs(save_path, exist_ok=True)


    # 扫描所有记录（以 hea 或 atr 为主）
    records = sorted([f[:-4] for f in os.listdir(base_path) if f.endswith(".dat")])

    for rec in records:
        print("Processing:", rec)

        # --------------------------
        # 读取波形
        # --------------------------
        record = wfdb.rdrecord(os.path.join(base_path, rec))
        signal = record.p_signal.astype(np.float32)  # (N, 2)
        N = signal.shape[0]

        # --------------------------
        # 初始化逐点标签
        # --------------------------
        anomaly_label = np.zeros(N, dtype=np.int8)

        # --------------------------
        # 读取 atr 节律标注
        # --------------------------
        try:
            ann = wfdb.rdann(os.path.join(base_path, rec), "atr")
            # aux_note = ann.aux_note
            samples = ann.sample
            symbol = ann.symbol
        except:
            print("⚠ No atr annotation for", rec)
            aux_note = []
            samples = []


        # --------------------------
        if len(samples) > 0:
            for i in range(len(samples)):
                label = symbol[i]
                start = samples[i]

                if i < len(samples) - 1:
                    end = samples[i+1]
                else:
                    end = N   # 最后一段直到录音结束

                if label not in ["N", "|", "+"]:
                    if label in anomaly_map.keys():
                        anomaly_label[start:end] = anomaly_map[label]
                        if anomaly_map[label] > 1:
                            print(anomaly_map[label])
                    else:
                        anomaly_label[start:end] = -1 # some anomaly we do not need

                print(label, start, "→", end)

        # --------------------------
        # 读取 hea 文件（可选）
        # --------------------------
        with open(os.path.join(base_path, rec + ".hea"), "r") as f:
            hea_text = f.read()

        # --------------------------
        # 保存为 npz
        # --------------------------
        mean_signal = np.mean(signal, axis=0)
        std_signal = np.std(signal, axis=0)
        normed_signal = (signal - mean_signal) / std_signal

        np.savez(
            os.path.join(save_path, rec + ".npz"),
            signal=signal,          # (N, 2)
            normed_signal=normed_signal,
            fs=record.fs,           # sampling rate
            ann_sample=np.array(samples),
            ann_symbol=np.array(symbol),
            anomaly_label=anomaly_label,   # (N,) 0/1
            hea_text=hea_text
        )

    print("\nDone! All files converted to .npz")


def show_mitdb_npy():   # q=2 → 250Hz → 125Hz
    # data_path = "/Users/zhc/Downloads/AFDB_npy/"
    data_path="/Users/zhc/Downloads/mitDB_npy/"
    records = [f for f in os.listdir(data_path) if f.endswith(".npz")]
    # anomaly_types = dict()
    anomaly_map = {'V': 1, 'A': 2, 'F': 3, 'L': 4, 'R': 5, '/': 6}


    for rec in records:
        print("Processing:", rec)
        record = np.load(os.path.join(data_path, rec))

        signal = record["signal"]          # (N, 2)
        anomaly_label = record["anomaly_label"]  # (N,)
        ann_symbol = record["ann_symbol"]
        ann_sample = record["ann_sample"]


        for i, (note, start_time) in enumerate(zip(ann_symbol, ann_sample)):
            # if note in ['S','N']:
            # if note in ["N", "|", "+"]:
            #     continue
            # if note not in anomaly_types.keys():
            #     anomaly_types.update({note: 1})
            # else:
            #     anomaly_types[note] += 1

            if note not in anomaly_map.keys():
                continue
            selected_signal = signal[start_time:start_time+2000]
            selected_label = anomaly_label[start_time:start_time+2000]

            # selected_signal_ds = decimate(selected_signal, q=q, axis=0)  # 降 q 倍
            # selected_anomaly_label_ds = selected_label[::q]
            selected_signal_ds = selected_signal
            selected_anomaly_label_ds = selected_label

            plt.figure(figsize=(12, 4))
            plt.plot(selected_signal_ds[:, 0], label="ECG channel 1 (downsampled)")
            plt.plot(selected_signal_ds[:, 1], label="ECG channel 2 (downsampled)")
            plt.plot(selected_anomaly_label_ds, label="anomaly Label")
            plt.legend()
            plt.title(f"{note}-{i}")
            plt.show()

        # print("123")
        # break

    # print(anomaly_types)


def extract_windows_from_record(
    signal,
    anomaly_label,
    source_name,
    window_size,
    stride,
    anomaly_map,
    max_anomaly_ratio
):

    class_windows = {k: [] for k in range(0, 7)}  # 0–6

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
    npz_file,
    output_dir,
    window_size=800,
    stride=100,
    train_ratio=0.8,
    max_anomaly_ratio=0.2,
    anomaly_map={'V':1,'A':2,'F':3,'L':4,'R':5,'/':6}
):

    # 映射表
    name_map = {
        0: "normal",
        1: "V",
        2: "A",
        3: "F",
        4: "L",
        5: "R",
        6: "slash"
    }

    # 初始化全局统计
    global_windows = {k: [] for k in range(0, 7)}


    # --- 遍历每个 record ---
    data = np.load(npz_file)
    signal = data["signal"]
    anomaly_label = data["anomaly_label"]

    per_record = extract_windows_from_record(
        signal=signal,
        anomaly_label=anomaly_label,
        source_name=npz_file,
        window_size=window_size,
        stride=stride,
        anomaly_map=anomaly_map,
        max_anomaly_ratio=max_anomaly_ratio
    )

    # 汇总
    for k in range(0, 7):
        global_windows[k].extend(per_record[k])


    # ----------- 开始写入 train/validation 文件 ------------
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    stats = {}

    for k in range(0, 7):
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

    stats = build_single_ts_train_val(
        npz_file="./raw_data/100.npz",
        output_dir="./indices/slide_windows_100npz",
        window_size=800,
        stride=10,
        train_ratio=0.5,
        max_anomaly_ratio=0.2
    )