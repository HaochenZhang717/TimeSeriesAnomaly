import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, IterableDataset, DataLoader
import json
import pandas as pd


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


class TSBADDataset(Dataset):
    def __init__(
            self,
            raw_data_paths,
            indices_paths,
            seq_len,
            max_anomaly_length,
    ):
        super(TSBADDataset, self).__init__()
        self.seq_len = seq_len
        # self.max_anomaly_ratio = max_anomaly_ratio
        self.max_anomaly_length = max_anomaly_length
        self.slide_windows = []
        self.anomaly_labels = []

        indices_paths = [indices_paths]
        raw_data_paths = [raw_data_paths]

        for indices_path, raw_data_path in zip(indices_paths, raw_data_paths):

            # raw_data = np.load(raw_data_path)

            df_raw = pd.read_csv(raw_data_path)
            raw_signal = df_raw["Data"].values
            anomaly_label = df_raw["Label"].values

            if len(raw_signal.shape) == 1:
                raw_signal = raw_signal.reshape(-1, 1)

            # raw_signal = raw_data["signal"]
            # anomaly_label = raw_data["anomaly_label"]
            scaler = MinMaxScaler()
            normed_signal = scaler.fit_transform(raw_signal)
            index_lines= load_jsonl(indices_path)

            for index_line in index_lines:
                start_index = index_line["start"]
                end_index = index_line["end"]
                self.slide_windows.append(normed_signal[start_index:end_index])
                self.anomaly_labels.append(anomaly_label[start_index:end_index])

    def __getitem__(self, index):
        signal = self.slide_windows[index]
        anomaly_label = self.anomaly_labels[index]

        random_anomaly_length = np.random.randint(0, self.max_anomaly_length)
        anomaly_start = np.random.randint(0, self.max_anomaly_length - random_anomaly_length)
        anomaly_end = anomaly_start + random_anomaly_length

        random_anomaly_label = np.zeros_like(anomaly_label)
        random_anomaly_label[anomaly_start:anomaly_end] = 1
        signal_random_occluded = signal * (1 - random_anomaly_label[:, None])
        original_occluded_signal = signal * (1 - anomaly_label[:, None])
        sample = {
            "orig_signal": signal,
            "anomaly_label": anomaly_label,
            "original_occluded_signal": original_occluded_signal,
            "random_anomaly_label": random_anomaly_label,
            "signal_random_occluded": signal_random_occluded,
        }
        # sample = {
        #     "orig_signal": signal[:24],
        #     "anomaly_label": anomaly_label[:24],
        #     "original_occluded_signal": original_occluded_signal[:24],
        #     "random_anomaly_label": random_anomaly_label[:24],
        #     "signal_random_occluded": signal_random_occluded[:24],
        # }
        return sample

    def __len__(self):
        return len(self.slide_windows)



class IterableTSBADDataset(IterableDataset):
    def __init__(
            self,
            raw_data_paths,
            indices_paths,
            seq_len,
            max_anomaly_length,
    ):
        super(IterableTSBADDataset, self).__init__()
        self.seq_len = seq_len
        # self.max_anomaly_ratio = max_anomaly_ratio
        self.max_anomaly_length = max_anomaly_length
        self.slide_windows = []
        self.anomaly_labels = []

        indices_paths = [indices_paths]
        raw_data_paths = [raw_data_paths]

        for indices_path, raw_data_path in zip(indices_paths, raw_data_paths):

            # raw_data = np.load(raw_data_path)
            # raw_signal = raw_data["signal"]
            # anomaly_label = raw_data["anomaly_label"]

            df_raw = pd.read_csv(raw_data_path)
            raw_signal = df_raw["Data"].values
            anomaly_label = df_raw["Label"].values
            if len(raw_signal.shape) == 1:
                raw_signal = raw_signal.reshape(-1, 1)

            scaler = MinMaxScaler()
            normed_signal = scaler.fit_transform(raw_signal)
            index_lines= load_jsonl(indices_path)

            for index_line in index_lines:
                start_index = index_line["start"]
                end_index = index_line["end"]
                self.slide_windows.append(normed_signal[start_index:end_index])
                self.anomaly_labels.append(anomaly_label[start_index:end_index])

    def __iter__(self):
        while True:
            index = np.random.randint(len(self.slide_windows))
            signal = self.slide_windows[index]
            anomaly_label = self.anomaly_labels[index]

            random_anomaly_length = np.random.randint(0, self.max_anomaly_length)
            anomaly_start = np.random.randint(0, self.max_anomaly_length - random_anomaly_length)
            anomaly_end = anomaly_start + random_anomaly_length

            random_anomaly_label = np.zeros_like(anomaly_label)
            random_anomaly_label[anomaly_start:anomaly_end] = 1
            signal_random_occluded = signal * (1 - random_anomaly_label[:, None])
            original_occluded_signal = signal * (1 - anomaly_label[:, None])
            sample = {
                "orig_signal": signal,
                "anomaly_label": anomaly_label,
                "original_occluded_signal": original_occluded_signal,
                "random_anomaly_label": random_anomaly_label,
                "signal_random_occluded": signal_random_occluded,
            }

            yield sample



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = IterableTSBADDataset(
        raw_data_paths="./raw_data/selected_uts/030_WSD_id_2_WebService_tr_4499_1st_4599.csv",
        indices_paths="./indices/slide_windows_030_WSD_id_2_WebService_tr_4499_1st_4599/train/normal.jsonl",
        seq_len=800,
        max_anomaly_length=20,
    )

    data_loader = DataLoader(dataset, batch_size=1)
    one_loader = iter(data_loader)

    for i in range(20):
        example = next(one_loader)
        plt.plot(example["orig_signal"][0])
        plt.show()