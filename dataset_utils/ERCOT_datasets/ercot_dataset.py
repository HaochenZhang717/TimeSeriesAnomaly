import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, IterableDataset, DataLoader
import json
import matplotlib.pyplot as plt
import random
import math


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data





class ImputationERCOTDataset(Dataset):
    def __init__(
            self,
            raw_data_paths,
            indices_paths,
            seq_len,
            one_channel,
            # use_prototype,
            max_infill_length,
    ):
        super(ImputationERCOTDataset, self).__init__()
        self.seq_len = seq_len
        self.one_channel = one_channel
        # self.use_prototype = use_prototype
        self.max_infill_length = max_infill_length
        self.raw_data_paths = raw_data_paths
        self.indices_paths = indices_paths

        self.normed_signal_list = []
        self.index_lines_list = []
        for raw_data_path, indices_path in zip(self.raw_data_paths, self.indices_paths):
            raw_data = np.load(raw_data_path)
            raw_signal = np.expand_dims(raw_data, axis=-1)
            scaler = MinMaxScaler()
            normed_signal = scaler.fit_transform(raw_signal)
            index_lines = load_jsonl(indices_path)
            self.normed_signal_list.append(normed_signal)
            self.index_lines_list.append(index_lines)

        self.global_index = []
        for region_id, index_lines in enumerate(self.index_lines_list):
            for i in range(len(index_lines)):
                self.global_index.append((region_id, i))


    def __getitem__(self, index):
        which_list, which_index = self.global_index[index]

        ts_start = self.index_lines_list[which_list][which_index]["ts_start"]
        ts_end = self.index_lines_list[which_list][which_index]["ts_end"]
        anomaly_start = self.index_lines_list[which_list][which_index]["anomaly_start"]
        anomaly_end = self.index_lines_list[which_list][which_index]["anomaly_end"]

        relative_anomaly_start = anomaly_start - ts_start
        relative_anomaly_end = anomaly_end - ts_start

        if self.one_channel:
            signal = torch.from_numpy(self.normed_signal_list[which_list][ts_start:ts_end, :1])
        else:
            signal = torch.from_numpy(self.normed_signal_list[which_list][ts_start:ts_end])


        anomaly_label = torch.zeros(ts_end - ts_start, dtype=torch.long)
        anomaly_label[relative_anomaly_start:relative_anomaly_end] = 1
        T = anomaly_label.shape[0]
        # ===== attention mask =====
        # normal + target anomaly are visible
        context_mask = torch.zeros(T, dtype=torch.long)
        context_mask[anomaly_label == 0] = 1
        context_mask[relative_anomaly_start:relative_anomaly_end] = 1

        # ===== noise mask =====
        noise_mask = torch.zeros(T, dtype=torch.long)
        noise_mask[relative_anomaly_start:relative_anomaly_end] = 1

        # ===== missing signals =====
        missing_signals = torch.zeros(self.max_infill_length, signal.shape[-1])
        infill_length = anomaly_end - anomaly_start
        missing_signals[:infill_length] = signal[relative_anomaly_start:relative_anomaly_end]

        return {
            'signals': signal,
            'missing_signals': missing_signals,
            'attn_mask': context_mask,
            'noise_mask': noise_mask,
        }


    def __len__(self):
        return len(self.global_index)



class ImputationNormalERCOTDataset(Dataset):
    def __init__(
            self,
            raw_data_paths,
            indices_paths,
            seq_len,
            one_channel,
            min_infill_length,
            max_infill_length,
    ):
        super(ImputationNormalERCOTDataset, self).__init__()
        self.seq_len = seq_len
        self.one_channel = one_channel
        self.raw_data_paths = raw_data_paths
        self.indices_paths = indices_paths
        self.min_infill_length = min_infill_length
        self.max_infill_length = max_infill_length

        self.normed_signal_list = []
        self.index_lines_list = []
        for raw_data_path, indices_path in zip(self.raw_data_paths, self.indices_paths):
            raw_data = np.load(raw_data_path)
            raw_signal = np.expand_dims(raw_data, axis=-1)
            scaler = MinMaxScaler()
            normed_signal = scaler.fit_transform(raw_signal)
            index_lines = load_jsonl(indices_path)
            self.normed_signal_list.append(normed_signal)
            self.index_lines_list.append(index_lines)

        self.global_index = []
        for region_id, index_lines in enumerate(self.index_lines_list):
            for i in range(len(index_lines)):
                self.global_index.append((region_id, i))


    def __getitem__(self, index):
        which_list, which_index = self.global_index[index]

        ts_start = self.index_lines_list[which_list][which_index]["start"]
        ts_end = self.index_lines_list[which_list][which_index]["end"]
        ts_length = ts_end - ts_start

        infill_length = self.min_infill_length + math.floor(torch.sigmoid(torch.rand(1)).item() * (self.max_infill_length - self.min_infill_length))
        relative_anomaly_start = random.randint(0, ts_length - infill_length)
        relative_anomaly_end = relative_anomaly_start + infill_length


        if self.one_channel:
            signal = torch.from_numpy(self.normed_signal_list[which_list][ts_start:ts_end, :1])
        else:
            signal = torch.from_numpy(self.normed_signal_list[which_list][ts_start:ts_end])


        anomaly_label = torch.zeros(ts_end - ts_start, dtype=torch.long)
        anomaly_label[relative_anomaly_start:relative_anomaly_end] = 1
        T = anomaly_label.shape[0]
        # ===== attention mask =====
        # normal + target anomaly are visible
        context_mask = torch.zeros(T, dtype=torch.long)
        context_mask[anomaly_label == 0] = 1
        context_mask[relative_anomaly_start:relative_anomaly_end] = 1

        # ===== noise mask =====
        noise_mask = torch.zeros(T, dtype=torch.long)
        noise_mask[relative_anomaly_start:relative_anomaly_end] = 1

        # ===== missing signals =====
        missing_signals = torch.zeros(self.max_infill_length, signal.shape[-1])
        missing_signals[:infill_length] = signal[relative_anomaly_start:relative_anomaly_end]

        return {
            'signals': signal,
            'missing_signals': missing_signals,
            'attn_mask': context_mask,
            'noise_mask': noise_mask,
        }


    def __len__(self):
        return len(self.global_index)



class NoContextNormalERCOTDataset(Dataset):
    def __init__(
            self,
            raw_data_paths,
            indices_paths,
            seq_len,
            one_channel,
            min_infill_length,
            max_infill_length,
    ):
        super(NoContextNormalERCOTDataset, self).__init__()
        self.seq_len = seq_len
        self.one_channel = one_channel

        self.raw_data_paths = raw_data_paths
        self.indices_paths = indices_paths

        self.min_infill_length = min_infill_length
        self.max_infill_length = max_infill_length

        self.normed_signal_list = []
        self.index_lines_list = []
        for raw_data_path, indices_path in zip(self.raw_data_paths, self.indices_paths):
            raw_data = np.load(raw_data_path)
            raw_signal = np.expand_dims(raw_data, axis=-1)
            scaler = MinMaxScaler()
            normed_signal = scaler.fit_transform(raw_signal)
            index_lines = load_jsonl(indices_path)
            self.normed_signal_list.append(normed_signal)
            self.index_lines_list.append(index_lines)

        self.global_index = []
        for region_id, index_lines in enumerate(self.index_lines_list):
            for i in range(len(index_lines)):
                self.global_index.append((region_id, i))

    def __getitem__(self, index):


        which_list, which_index = self.global_index[index]

        ts_start = self.index_lines_list[which_list][which_index]["start"]
        ts_end = self.index_lines_list[which_list][which_index]["end"]
        ts_length = ts_end - ts_start
        infill_length = random.randint(self.min_infill_length, self.max_infill_length)

        relative_anomaly_start = random.randint(0, ts_length - infill_length)
        relative_anomaly_end = relative_anomaly_start + infill_length


        if self.one_channel:
            signal = torch.from_numpy(self.normed_signal_list[which_list][ts_start:ts_end, :1])
        else:
            signal = torch.from_numpy(self.normed_signal_list[which_list][ts_start:ts_end])

        # ===== missing signals =====
        missing_signals = torch.zeros(self.max_infill_length, signal.shape[-1])
        missing_signals[:infill_length] = signal[relative_anomaly_start:relative_anomaly_end]

        context_mask = torch.zeros(self.max_infill_length, dtype=torch.long)
        context_mask[:infill_length] = 1
        return {
            'signals': missing_signals,
            'attn_mask': context_mask,
        }


    def __len__(self):
        return len(self.global_index)



class NoContextAnomalyERCOTDataset(Dataset):
    def __init__(
            self,
            raw_data_paths,
            indices_paths,
            seq_len,
            one_channel,
    ):
        super(NoContextAnomalyERCOTDataset, self).__init__()
        self.seq_len = seq_len
        self.one_channel = one_channel

        self.raw_data_paths = raw_data_paths
        self.indices_paths = indices_paths


        self.normed_signal_list = []
        self.index_lines_list = []
        for raw_data_path, indices_path in zip(self.raw_data_paths, self.indices_paths):
            raw_data = np.load(raw_data_path)
            raw_signal = np.expand_dims(raw_data, axis=-1)
            scaler = MinMaxScaler()
            normed_signal = scaler.fit_transform(raw_signal)
            index_lines = load_jsonl(indices_path)
            self.normed_signal_list.append(normed_signal)
            self.index_lines_list.append(index_lines)

        self.global_index = []
        for region_id, index_lines in enumerate(self.index_lines_list):
            for i in range(len(index_lines)):
                self.global_index.append((region_id, i))

    def __getitem__(self, index):


        which_list, which_index = self.global_index[index]

        ts_start = self.index_lines_list[which_list][which_index]["start"]
        ts_end = self.index_lines_list[which_list][which_index]["end"]
        ts_length = ts_end - ts_start

        data = self.normed_signal_list[which_list][ts_start:ts_end]
        signal = torch.zeros(self.seq_len, data.shape[-1])
        signal[:ts_length] = torch.from_numpy(data)

        if self.one_channel:
            signal = signal[:, :1]

        context_mask = torch.zeros(self.seq_len, dtype=torch.long)
        context_mask[:ts_length] = 1


        return {
            'signals': signal,
            'attn_mask': context_mask,
        }


    def __len__(self):
        return len(self.global_index)


if __name__ == '__main__':
    # dataset = NoContextNormalERCOTDataset(
    #     raw_data_paths=["./raw_data/coast.npy", "./raw_data/east.npy"],
    #     indices_paths=["./indices/coast_normal_200.jsonl", "./indices/east_normal_200.jsonl"],
    #     seq_len=200,
    #     one_channel=1,
    #     min_infill_length=190,
    #     max_infill_length=193,
    # )

    dataset = ImputationERCOTDataset(
        raw_data_paths=["./raw_data/coast.npy", "./raw_data/east.npy"],
        indices_paths=["./indices/coast_anomaly.jsonl", "./indices/east_anomaly.jsonl"],
        seq_len=1000,
        one_channel=1,
        max_infill_length=193,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        print(batch)