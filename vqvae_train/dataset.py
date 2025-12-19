import json
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch


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

        ts_dim = self.data.shape[-1]
        signal = torch.zeros(self.max_length, ts_dim, dtype=torch.float32)
        real_length = end - start
        signal[:real_length] = torch.from_numpy(self.data[start:end])

        if self.one_channel:
            signal = signal[:, :1]

        pad_mask = torch.zeros(1, 1, self.max_length, dtype=torch.long)
        pad_mask[:,:,:real_length] = 1

        return signal, pad_mask