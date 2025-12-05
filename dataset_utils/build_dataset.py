from .ECG_datasets import ECGDataset, IterableECGDataset
from .TSBAD_datasets import TSBADDataset, IterableTSBADDataset

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
    ):
    assert dataset_name in dataset_name_map.keys()
    dataset_cls = dataset_name_map[dataset_name][dataset_type]
    return dataset_cls(raw_data_paths, indices_paths, seq_len, max_anomaly_length)