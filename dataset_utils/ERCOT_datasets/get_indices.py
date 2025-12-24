import matplotlib.pyplot as plt
import wfdb
import os
import numpy as np
from scipy.signal import decimate
import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path



def has_exactly_one_anomaly_segment(label):
    """
    label: 1D array-like of 0/1
    """
    label = np.asarray(label).astype(int)

    # 找到从 0 -> 1 的位置
    starts = np.where((label[:-1] == 0) & (label[1:] == 1))[0]

    # 如果第一个点就是 1，也算一个 segment
    if label[0] == 1:
        num_segments = len(starts) + 1
    else:
        num_segments = len(starts)

    return num_segments == 1


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
    min_start,
    window_size,
    stride,
    anomaly_map,
    max_anomaly_ratio
):

    class_windows = {k: [] for k in range(0, 7)}  # 0–6

    N = len(anomaly_label)

    for start in range(min_start, N - window_size + 1, stride):
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


def build_normal_ts(
    source_name,
    signal,
    anomaly_label,
    output_dir,
    min_start,
    window_size=800,
    stride=100,
    max_anomaly_ratio=0.2,
    anomaly_map={'anomaly':1}
):

    # 映射表
    name_map = {
        0: "normal",
    }

    # 初始化全局统计
    global_windows = {k: [] for k in range(0, 1)}


    per_record = extract_windows_from_record(
        signal=signal,
        anomaly_label=anomaly_label,
        source_name=source_name,
        min_start=min_start,
        window_size=window_size,
        stride=stride,
        anomaly_map=anomaly_map,
        max_anomaly_ratio=max_anomaly_ratio
    )

    # 汇总
    for k in range(0, 1):
        global_windows[k].extend(per_record[k])


    # ----------- 开始写入 train/validation 文件 ------------
    stats = {}
    for k in range(0, 1):
        fname = f"{name_map[k]}_{window_size}.jsonl"
        windows = global_windows[k]

        # 写 train
        with open(f"{output_dir}_{fname}", "w") as f:
            for item in windows:
                f.write(json.dumps(item) + "\n")

        print(f"{fname:12s} total={len(windows):6d}")

    return stats



#----------------------utils to get anomaly segments------------------------
def get_anomaly_segments(labels, anomaly_type):
    """
    输入: labels = 0/1 的 array
    输出: list of (start_idx, end_idx)
    """
    labels = np.array(labels)
    idx = np.where(labels == anomaly_type)[0]  # 找出所有标为 1 的点
    segments = []

    if len(idx) == 0:
        return segments

    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            # 继续同一段
            prev = i
        else:
            # 上一段结束
            segments.append((start, prev))
            start = i
            prev = i

    # 记得补上最后一段
    segments.append((start, prev))
    return segments


def get_normal_segments(labels):
    """
    输入: labels = 0/1 的 array
    输出: list of (start_idx, end_idx)
    """
    labels = np.array(labels)
    idx = np.where(labels == 0)[0]  # 找出所有标为 1 的点
    segments = []

    if len(idx) == 0:
        return segments

    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            # 继续同一段
            prev = i
        else:
            # 上一段结束
            segments.append((start, prev))
            start = i
            prev = i

    # 记得补上最后一段
    segments.append((start, prev))
    return segments


def extract_more_windows_containing_segments(
    signal,
    segments,
    cluster_ids,
    window_size,
    step=1,  # 滑窗步长，可以调大速度更快
    jsonl_path=None,
):
    """
    从信号中截取长度为 window_size 的窗口，
    条件：
        1) 窗口要完全包含某一个 anomaly segment (start,end)
        2) 窗口内 anomaly ratio 在 ratio_range 内
    返回：
        windows: [num_windows, window_size]
        windows_label: [num_windows, window_size]
        window_starts: 每个窗口的起点 index
    """
    jsonl_file = open(jsonl_path, "w") if jsonl_path is not None else None

    T = len(signal)


    for (seg_start, seg_end), cluster_id in zip(segments, cluster_ids):

        # 为完全包含异常段，需要窗口满足：
        # start <= seg_start AND start+window_size-1 >= seg_end
        earliest = seg_end - window_size + 1
        latest = seg_start

        # 合法窗口起点范围
        valid_range_start = max(0, earliest)
        valid_range_end   = min(latest, T - window_size)

        if valid_range_start > valid_range_end:
            # 异常段比窗口还长，无解
            continue

        # 遍历所有可能起点
        for start in range(valid_range_start, valid_range_end + 1, step):
            end = start + window_size
            record = {
                "ts_start": int(start),
                "ts_end": int(end),
                "anomaly_start": int(seg_start),
                "anomaly_end": int(seg_end),
                "anomaly_type": 1,
                "prototype_id": int(cluster_id),
            }
            jsonl_file.write(json.dumps(record) + "\n")



def extract_windows_containing_segments(
    signal,
    labels,
    segments,
    cluster_ids,
    window_size,
    length_range=(0.01, 0.50),
    step=1,  # 滑窗步长，可以调大速度更快
    jsonl_path=None,
    anomaly_type=1
):
    """
    从信号中截取长度为 window_size 的窗口，
    条件：
        1) 窗口要完全包含某一个 anomaly segment (start,end)
        2) 窗口内 anomaly ratio 在 ratio_range 内
    返回：
        windows: [num_windows, window_size]
        windows_label: [num_windows, window_size]
        window_starts: 每个窗口的起点 index
    """
    jsonl_file = open(jsonl_path, "w") if jsonl_path is not None else None

    min_length, max_length = length_range
    min_ratio = min_length / window_size
    max_ratio = max_length / window_size

    T = len(signal)

    windows = []
    windows_label = []
    window_starts = []

    if cluster_ids is not None:
        for (seg_start, seg_end), cluster_id in zip(segments, cluster_ids):

            earliest = seg_end - window_size + 1
            latest = seg_start

            valid_range_start = max(0, earliest)
            valid_range_end   = min(latest, T - window_size)

            if valid_range_start > valid_range_end:
                # 异常段比窗口还长，无解
                continue

            min_seg_len = float("inf")
            max_seg_len = 0
            # 遍历所有可能起点
            for start in range(valid_range_start, valid_range_end + 1, step):
                end = start + window_size

                label_win = labels[start:end]

                if not np.array_equal(np.unique(label_win), np.array([0, anomaly_type])):
                    continue

                if not has_exactly_one_anomaly_segment(label_win):
                    continue

                anomaly_ratio = label_win.sum() / window_size

                if min_ratio <= anomaly_ratio <= max_ratio:
                    windows.append(signal[start:end])
                    windows_label.append(label_win)
                    window_starts.append(start)


                    # ====== 在窗口内部重新统计“连续 1 段”的长度 ======
                    idx = np.where(label_win == anomaly_type)[0]
                    if len(idx) > 0:
                        # 找出所有连续段
                        seg_start_idx = idx[0]
                        prev = idx[0]
                        for i in idx[1:]:
                            if i == prev + 1:
                                prev = i
                            else:
                                # 前一段结束
                                seg_len = prev - seg_start_idx + 1
                                min_seg_len = min(min_seg_len, seg_len)
                                max_seg_len = max(max_seg_len, seg_len)
                                # 新的一段开始
                                seg_start_idx = i
                                prev = i
                        # 别忘了最后一段
                        seg_len = prev - seg_start_idx + 1
                        min_seg_len = min(min_seg_len, seg_len)
                        max_seg_len = max(max_seg_len, seg_len)

                    # plt.plot(signal[start:end,0], label="signal channel 0")
                    # plt.plot(label_win, label="anomaly label")
                    # plt.show()

                    record = {
                        "ts_start": int(start),
                        "ts_end": int(end),
                        "anomaly_start": int(seg_start),
                        "anomaly_end": int(seg_end),
                        "anomaly_type": 1,
                        "cluster_id": int(cluster_id),
                    }
                    jsonl_file.write(json.dumps(record) + "\n")

    else:
        for seg_start, seg_end in segments:

            earliest = seg_end - window_size + 1
            latest = seg_start

            valid_range_start = max(0, earliest)
            valid_range_end = min(latest, T - window_size)

            if valid_range_start > valid_range_end:
                # 异常段比窗口还长，无解
                continue

            min_seg_len = float("inf")
            max_seg_len = 0
            # 遍历所有可能起点
            for start in range(valid_range_start, valid_range_end + 1, step):
                end = start + window_size

                label_win = labels[start:end]

                if not np.array_equal(np.unique(label_win), np.array([0, anomaly_type])):
                    continue

                if not has_exactly_one_anomaly_segment(label_win):
                    continue

                anomaly_ratio = label_win.sum() / window_size

                if min_ratio <= anomaly_ratio <= max_ratio:
                    windows.append(signal[start:end])
                    windows_label.append(label_win)
                    window_starts.append(start)

                    # ====== 在窗口内部重新统计“连续 1 段”的长度 ======
                    idx = np.where(label_win == anomaly_type)[0]
                    if len(idx) > 0:
                        # 找出所有连续段
                        seg_start_idx = idx[0]
                        prev = idx[0]
                        for i in idx[1:]:
                            if i == prev + 1:
                                prev = i
                            else:
                                # 前一段结束
                                seg_len = prev - seg_start_idx + 1
                                min_seg_len = min(min_seg_len, seg_len)
                                max_seg_len = max(max_seg_len, seg_len)
                                # 新的一段开始
                                seg_start_idx = i
                                prev = i
                        # 别忘了最后一段
                        seg_len = prev - seg_start_idx + 1
                        min_seg_len = min(min_seg_len, seg_len)
                        max_seg_len = max(max_seg_len, seg_len)

                    # plt.plot(signal[start:end,0], label="signal channel 0")
                    # plt.plot(label_win, label="anomaly label")
                    # plt.show()

                    record = {
                        "ts_start": int(start),
                        "ts_end": int(end),
                        "anomaly_start": int(seg_start),
                        "anomaly_end": int(seg_end),
                        "anomaly_type": 1,
                    }
                    jsonl_file.write(json.dumps(record) + "\n")

    if min_seg_len == float("inf"):
        min_seg_len = None
        max_seg_len = None

    return (
        np.array(windows),
        np.array(windows_label),
        np.array(window_starts),
        min_seg_len,
        max_seg_len
    )



def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


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

# ----------------------- 使用示例 -----------------------
if __name__ == "__main__":

    names = [
        "coast",
        "east",
        "fwest",
        "ncent",
        "north",
        "scent",
        "south",
        "west",
    ]


    # get indices for all normal series
    for name in names:
        print(name)

        raw_signal = np.load(f"./raw_data/{name}.npy")
        anomaly_label = np.load(f"./raw_data/label.npy")


        stats = build_normal_ts(
            source_name=name,
            signal=raw_signal,
            anomaly_label=anomaly_label,
            min_start=1230,
            output_dir=f"./indices/{name}",
            window_size=1000,
            stride=1,
            max_anomaly_ratio=0.7,
            anomaly_map={'anomaly': 1}
        )

        stats = build_normal_ts(
            source_name=name,
            signal=raw_signal,
            anomaly_label=anomaly_label,
            min_start=1230,
            output_dir=f"./indices/{name}",
            window_size=200,
            stride=1,
            max_anomaly_ratio=0.7,
            anomaly_map={'anomaly': 1}
        )


    # get indices for all anomaly
    for name in names:
        raw_signal = np.load(f"./raw_data/{name}.npy")
        anomaly_label = np.load(f"./raw_data/label.npy")
        segments = get_anomaly_segments(anomaly_label, anomaly_type=1)
        print(name)
        print(f"总共有 {len(segments)} 段 anomaly")
        segments_info_list = []
        for segment in segments:
            segments_info_list.append(
                {
                    "start": int(segment[0]),
                    "end": int(segment[1]),
                }
            )

        with open(f"./indices/anomaly_segments_{name}.jsonl", "w") as f:
            for item in segments_info_list:
                f.write(json.dumps(item) + "\n")

        windows, window_labels, starts, min_anomaly_length, max_anomaly_length = extract_windows_containing_segments(
            signal=raw_signal,
            labels=anomaly_label,
            segments=segments,
            cluster_ids=None,
            window_size=1000,
            length_range=(190, 193),  # 调这个
            step=10,
            jsonl_path=f"./indices/{name}_anomaly.jsonl",
            anomaly_type=1
        )


    # anomaly_type_maps = {'anomaly': 1}
    #
    # for k, v in anomaly_type_maps.items():
    #     segments = get_anomaly_segments(anomaly_label, anomaly_type=v)
    #
    #     print(f"总共有 {len(segments)} 段 anomaly")
    #     lengths = []
    #     for i, (s, e) in enumerate(segments):
    #         print(f"Segment {i}: start = {s}, end = {e}, length = {e - s + 1}")
    #         lengths.append(e - s + 1)
    #     print(max(lengths))
    #     print(min(lengths))
    #
    #     windows, window_labels, starts, min_anomaly_length, max_anomaly_length = extract_windows_containing_segments(
    #         signal=raw_signal,
    #         labels=anomaly_label,
    #         segments=segments,
    #         cluster_ids=None,
    #         window_size=1000,
    #         length_range=(66, 608),  # 调这个
    #         step=100,
    #         jsonl_path=f"./indices/slide_windows_{name}npz/train/{k}.jsonl",
    #         anomaly_type=v
    #     )

