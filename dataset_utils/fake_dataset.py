import torch
import numpy as np


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, normal_data_path, maximum_anomaly_length, minimum_anomaly_length, one_channel):
        self.normal_data = torch.load(normal_data_path)
        num_data, seq_len, feat_size = self.normal_data['samples'].shape
        self.num_data = num_data
        self.seq_len = seq_len
        self.max_anomaly_length = maximum_anomaly_length
        self.min_anomaly_length = minimum_anomaly_length
        self.one_channel = one_channel

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        normal_data = self.normal_data['samples'][index]

        random_anomaly_length = np.random.randint(self.min_anomaly_length, self.max_anomaly_length)
        anomaly_start = np.random.randint(0, self.max_anomaly_length - random_anomaly_length)
        anomaly_end = anomaly_start + random_anomaly_length


        random_anomaly_label = torch.zeros_like(normal_data.sum(-1))
        random_anomaly_label[anomaly_start:anomaly_end] = 1
        if self.one_channel:
            sample = {
                "original_signal": normal_data[:, :1],
                "random_anomaly_label": random_anomaly_label,
            }
        else:
            sample = {
                "original_signal": normal_data,
                "random_anomaly_label": random_anomaly_label,
            }

        return sample