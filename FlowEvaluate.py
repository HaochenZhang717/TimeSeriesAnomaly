import numpy as np

from generation_models import FM_TS
from Trainers import FlowTSPretrain
from dataset_utils import ECGDataset, IterableECGDataset
import argparse
import torch
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from evaluation_utils import run_anomaly_quality_test, classification_metrics_torch
from evaluation_utils import predictive_score_metrics, discriminative_score_metrics
from tqdm import tqdm

def get_evaluate_args():
    parser = argparse.ArgumentParser(description="parameters for flow-ts pretraining")

    """time series general parameters"""
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--feature_size", type=int, required=True)

    """model parameters"""
    parser.add_argument("--n_layer_enc", type=int, required=True)
    parser.add_argument("--n_layer_dec", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True)

    """data parameters"""
    parser.add_argument("--max_anomaly_ratio", type=float, required=True)
    parser.add_argument("--raw_data_paths_train", type=str, required=True)
    parser.add_argument("--raw_data_paths_val", type=str, required=True)
    parser.add_argument("--normal_indices_paths_train", type=str, required=True)
    parser.add_argument("--normal_indices_paths_val", type=str, required=True)
    parser.add_argument("--anomaly_indices_paths_train", type=str, required=True)
    parser.add_argument("--anomaly_indices_paths_val", type=str, required=True)

    """training parameters"""
    parser.add_argument("--batch_size", type=int, required=True)


    """save and load parameters"""
    parser.add_argument("--model_ckpt", type=str, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)

    """sample parameters"""
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--generated_path", type=str, required=True)

    return parser.parse_args()


def evaluate():
    args = get_evaluate_args()

    device = torch.device("cuda:%d" % args.gpu_id)
    model = FM_TS(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
    ).to(device)
    model.prepare_for_finetune(ckpt_path=None)
    model.load_state_dict(torch.load(args.model_ckpt))
    model.eval()


    normal_train_set = IterableECGDataset(
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.normal_indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_ratio=args.max_anomaly_ratio,
    )

    normal_val_set = ECGDataset(
        raw_data_paths=args.raw_data_paths_val,
        indices_paths=args.normal_indices_paths_val,
        seq_len=args.seq_len,
        max_anomaly_ratio=args.max_anomaly_ratio,
    )

    anomaly_train_set = IterableECGDataset(
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.anomaly_indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_ratio=args.max_anomaly_ratio,
    )

    anomaly_val_set = ECGDataset(
        raw_data_paths=args.raw_data_paths_val,
        indices_paths=args.anomaly_indices_paths_val,
        seq_len=args.seq_len,
        max_anomaly_ratio=args.max_anomaly_ratio,
    )

    all_data = {
        "orig_normal_train_signal": torch.from_numpy(np.stack(normal_train_set.slide_windows, axis=0)),
        "orig_normal_train_label": torch.from_numpy(np.stack(normal_train_set.anomaly_labels, axis=0)),
        "orig_anomaly_train_signal": torch.from_numpy(np.stack(anomaly_train_set.slide_windows, axis=0)),
        "orig_anomaly_train_label": torch.from_numpy(np.stack(anomaly_train_set.anomaly_labels, axis=0)),
        "orig_normal_val_signal": torch.from_numpy(np.stack(normal_val_set.slide_windows, axis=0)),
        "orig_normal_val_label": torch.from_numpy(np.stack(normal_val_set.anomaly_labels, axis=0)),
        "orig_anomaly_val_signal": torch.from_numpy(np.stack(anomaly_val_set.slide_windows, axis=0)),
        "orig_anomaly_val_label": torch.from_numpy(np.stack(anomaly_val_set.anomaly_labels, axis=0)),
    }


    if args.num_samples > 0:
        normal_train_loader = torch.utils.data.DataLoader(normal_train_set, batch_size=args.batch_size)
        normal_train_iterator = iter(normal_train_loader)
        args.num_samples = len(normal_train_set.slide_windows)
        num_cycle = int(args.num_samples // args.batch_size) + 1

        all_anomaly_samples = []
        all_anomaly_labels = []
        for _ in tqdm(range(num_cycle), desc="Generating samples"):
            anomaly_label = next(normal_train_iterator)['random_anomaly_label'].to(device)
            samples = model.generate_mts(
                batch_size=args.batch_size,
                anomaly_label=anomaly_label,
            ).cpu()
            all_anomaly_samples.append(samples)
            all_anomaly_labels.append(anomaly_label)

        all_samples = torch.cat(all_anomaly_samples, dim=0)
        all_anomaly_labels = torch.cat(all_anomaly_labels, dim=0)
        to_save = {
        "all_samples": all_samples,
        "all_anomaly_labels": all_anomaly_labels,
        }
        os.makedirs(args.generated_path, exist_ok=True)
        torch.save(to_save,f"{args.generated_path}/generated.pt")
    else:
        assert args.generated_path is not None
        generated = torch.load(f"{args.generated_path}/generated.pt", map_location=device)
        all_samples = generated['all_samples']
        all_anomaly_labels = generated['all_anomaly_labels']

    all_data.update({
        "all_samples": all_samples,
        "all_anomaly_labels": all_anomaly_labels,
    })
    torch.save(all_data, f"{args.generated_path}/all_data.pt")


    breakpoint()
    print(all_data["orig_normal_train_signal"].mean())
    print(all_data["orig_anomaly_train_signal"].mean())
    print(all_data["all_samples"].mean())

    old_eval_result = classification_metrics_torch(
    ori_normal_data=all_data["orig_normal_train_signal"],
    ori_anomaly_data=all_data["orig_anomaly_train_signal"],
    gen_anomaly_data=all_data["all_samples"],
    hidden_dim=64,
    max_epochs=2000,
    batch_size=64,
    patience=20,
    device=device
    )

    for k, v in old_eval_result.items():
        print(f"{k}: {v}")


    # default_metrics = run_anomaly_quality_test(
    #     train_normal_signal=all_data["orig_normal_train_signal"],
    #     train_anomaly_signal=all_data["orig_anomaly_train_signal"],
    #     train_anomaly_label=all_data["orig_anomaly_train_label"],
    #     test_normal_signal=all_data["orig_normal_val_signal"],
    #     test_anomaly_signal=all_data["orig_anomaly_val_signal"],
    #     test_anomaly_label=all_data["orig_anomaly_val_label"],
    #     model=GRUClassifier(input_dim=args.feature_size, hidden_dim=128).to(device),
    #     device=device,
    #     lr=1e-4,
    #     bs=64,
    #     mode="interval"
    # )
    #
    # flow_metrics = run_anomaly_quality_test(
    #     train_normal_signal=all_data["orig_normal_train_signal"],
    #     train_anomaly_signal=all_data["all_samples"],
    #     train_anomaly_label=all_data["all_anomaly_labels"],
    #     test_normal_signal=all_data["orig_normal_val_signal"],
    #     test_anomaly_signal=all_data["orig_anomaly_val_signal"],
    #     test_anomaly_label=all_data["orig_anomaly_val_label"],
    #     model=GRUClassifier(input_dim=args.feature_size, hidden_dim=128).to(device),
    #     device=device,
    #     lr=1e-4,
    #     bs=64,
    #     mode="interval"
    # )
    # print("default metrics:")
    # print(default_metrics)
    # print("Flow metrics:")
    # print(flow_metrics)


def evaluate_pretrain():
    args = get_evaluate_args()

    device = torch.device("cuda:%d" % args.gpu_id)
    model = FM_TS(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
    ).to(device)
    model.load_state_dict(torch.load(args.model_ckpt))
    model.eval()

    normal_train_set = ECGDataset(
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.normal_indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_ratio=args.max_anomaly_ratio,
    )

    if args.num_samples > 0: # 15601

        num_cycle = int(args.num_samples // args.batch_size) + 1
        all_samples = []
        for _ in tqdm(range(num_cycle), desc="Generating samples"):
            samples = model.generate_mts(
                batch_size=args.batch_size,
                anomaly_label=None,
            ).cpu()
            all_samples.append(samples)

        all_samples = torch.cat(all_samples, dim=0)
        os.makedirs(args.generated_path, exist_ok=True)
        torch.save(all_samples,f"{args.generated_path}/generated_normal.pt")
    else:
        assert args.generated_path is not None
        all_samples = torch.load(
            f"{args.generated_path}/generated_normal.pt",
            map_location=device
        )


    # old_eval_result = classification_metrics_torch(
    # ori_normal_data=all_data["orig_normal_train_signal"],
    # ori_anomaly_data=all_data["orig_anomaly_train_signal"],
    # gen_anomaly_data=all_data["all_samples"],
    # hidden_dim=64,
    # max_epochs=2000,
    # batch_size=64,
    # patience=20,
    # device=device
    # )
    #
    # for k, v in old_eval_result.items():
    #     print(f"{k}: {v}")


    # default_metrics = run_anomaly_quality_test(
    #     train_normal_signal=all_data["orig_normal_train_signal"],
    #     train_anomaly_signal=all_data["orig_anomaly_train_signal"],
    #     train_anomaly_label=all_data["orig_anomaly_train_label"],
    #     test_normal_signal=all_data["orig_normal_val_signal"],
    #     test_anomaly_signal=all_data["orig_anomaly_val_signal"],
    #     test_anomaly_label=all_data["orig_anomaly_val_label"],
    #     model=GRUClassifier(input_dim=args.feature_size, hidden_dim=128).to(device),
    #     device=device,
    #     lr=1e-4,
    #     bs=64,
    #     mode="interval"
    # )
    #
    # flow_metrics = run_anomaly_quality_test(
    #     train_normal_signal=all_data["orig_normal_train_signal"],
    #     train_anomaly_signal=all_data["all_samples"],
    #     train_anomaly_label=all_data["all_anomaly_labels"],
    #     test_normal_signal=all_data["orig_normal_val_signal"],
    #     test_anomaly_signal=all_data["orig_anomaly_val_signal"],
    #     test_anomaly_label=all_data["orig_anomaly_val_label"],
    #     model=GRUClassifier(input_dim=args.feature_size, hidden_dim=128).to(device),
    #     device=device,
    #     lr=1e-4,
    #     bs=64,
    #     mode="interval"
    # )
    # print("default metrics:")
    # print(default_metrics)
    # print("Flow metrics:")
    # print(flow_metrics)


if __name__ == "__main__":
    evaluate_pretrain()









