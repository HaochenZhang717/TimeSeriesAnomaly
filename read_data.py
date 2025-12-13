import pandas as pd
import numpy as np
import wfdb
import ast
import random

import matplotlib.pyplot as plt

LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
LEADS_TO_PLOT = [0]  # II, V1, V5

def plot_ecg_sample(ecg, title=None):
    """
    ecg: (T, 12)
    """
    t = np.arange(ecg.shape[0]) / 100.0  # seconds

    plt.figure(figsize=(12, 4))
    offset = 0.0
    for lead_idx in LEADS_TO_PLOT:
        plt.plot(t, ecg[:, lead_idx] + offset, label=LEAD_NAMES[lead_idx])
        offset += 1.2

    plt.xlabel("Time (s)")
    plt.yticks([])
    plt.legend(loc='upper right')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def pick_samples_by_class(X, Y, target_class, k=3):
    idxs = [i for i, labs in enumerate(Y) if target_class in labs]
    chosen = random.sample(idxs, k)
    return X[chosen], chosen


def pick_samples_by_scp_code(X, Y_df, scp_code, k=3):
    """
    X: ECG array (N, T, 12)
    Y_df: full dataframe (ptbxl_database.csv)
    scp_code: e.g. 'PVC'
    """
    idxs = [
        i for i, codes in enumerate(Y_df.scp_codes)
        if scp_code in codes
    ]

    print(f"Found {len(idxs)} ECGs with scp_code = {scp_code}")

    chosen = random.sample(idxs, k)
    return X[chosen], chosen

if __name__ == "__main__":

    # path = 'path/to/ptbxl/'
    path = '/Users/zhc/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    sampling_rate=100

    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    # Split data into train and test
    test_fold = 10
    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

    # classes = ['NORM', 'MI', 'CD', 'STTC', 'HYP']
    #
    # for cls in classes:
    #     samples, idxs = pick_samples_by_class(X_train, y_train, cls, k=2)
    #     for i, ecg in enumerate(samples):
    #         plot_ecg_sample(
    #             ecg,
    #             title=f"{cls} | sample idx {idxs[i]}"
    #         )

    # =========================
    # Visualize PVC samples
    # =========================

    samples, idxs = pick_samples_by_scp_code(
        X_train,
        Y[Y.strat_fold != test_fold],  # 对齐 X_train
        scp_code='PVC',
        k=2
    )

    for i, ecg in enumerate(samples):
        plot_ecg_sample(
            ecg,
            title=f"PVC | ECG idx {idxs[i]}"
        )