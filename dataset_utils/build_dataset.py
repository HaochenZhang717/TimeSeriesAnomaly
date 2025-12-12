from .ECG_datasets import ECGDataset, IterableECGDataset
from .TSBAD_datasets import TSBADDataset, IterableTSBADDataset
import torch
import numpy as np


dataset_name_map = {
    'ECG': {'non_iterable': ECGDataset, 'iterable': IterableECGDataset},
    'TSBAD': {'non_iterable': TSBADDataset, 'iterable': IterableTSBADDataset},
}



def build_dataset(
        dataset_name: str,
        dataset_type: str,
        raw_data_paths,
        indices_paths,
        seq_len,
        max_anomaly_length,
        min_anomaly_length,
        one_channel,
        limited_data_size
    ):
    assert dataset_name in dataset_name_map.keys()

    if dataset_name in ['ECG']:
        dataset_cls = dataset_name_map[dataset_name][dataset_type]
        return dataset_cls(
            raw_data_paths, indices_paths, seq_len,
            max_anomaly_length, min_anomaly_length,
            one_channel, limited_data_size
        )
    elif dataset_name in ['TSBAD']:
        dataset_cls = dataset_name_map[dataset_name][dataset_type]
        return dataset_cls(raw_data_paths, indices_paths, seq_len, max_anomaly_length, min_anomaly_length)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')
