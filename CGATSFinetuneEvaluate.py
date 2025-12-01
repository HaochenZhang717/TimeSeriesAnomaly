from generation_models import TimeVAECGATS
from Trainers import CGATFinetune
from dataset_utils import ECGDataset, IterableECGDataset
import argparse
import torch
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import os
from tqdm import tqdm
from evaluation_utils import calculate_robustTAD


def get_evaluate_args():
    parser = argparse.ArgumentParser(description="parameters for TimeVAE-CGATS pretraining")

    """time series general parameters"""
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--feature_size", type=int, required=True)

    """model parameters"""
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--trend_poly", type=int, required=True)
    parser.add_argument("--kl_wt", type=float, required=True)
    parser.add_argument("--hidden_layer_sizes", type=json.loads, required=True)
    parser.add_argument("--custom_seas", type=json.loads, required=True)


    """data parameters"""
    parser.add_argument("--max_anomaly_ratio", type=float, required=True)
    parser.add_argument("--raw_data_paths_train", type=str, required=True)
    parser.add_argument("--raw_data_paths_val", type=str, required=True)
    parser.add_argument("--indices_paths_train", type=str, required=True)
    parser.add_argument("--indices_paths_val", type=str, required=True)

    """training parameters"""
    parser.add_argument("--batch_size", type=int, required=True)

    """save and load parameters"""
    parser.add_argument("--model_ckpt", type=str, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)

    """sample parameters"""
    parser.add_argument("--need_to_generate", type=int, required=True)
    parser.add_argument("--generated_path", type=str, required=True)

    return parser.parse_args()


def evaluate_anomaly():
    device = torch.device("cuda:%d" % args.gpu_id)
    args = get_evaluate_args()
    model = TimeVAECGATS(
        hidden_layer_sizes=args.hidden_layer_sizes,
        trend_poly=args.trend_poly,
        custom_seas=args.custom_seas,
        use_residual_conn=True,
        seq_len=args.seq_len,
        feat_dim=args.feature_size,
        latent_dim=args.latent_dim,
        kl_wt=args.kl_wt,
    ).to(device)
    model.load_state_dict(torch.load(args.model_ckpt))
    model.eval()

    normal_train_set = IterableECGDataset(
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.normal_indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_ratio=args.max_anomaly_ratio,
    )

    if args.need_to_generate:
        num_samples = len(normal_train_set.slide_windows)
        num_cycle = int(num_samples // args.batch_size) + 1
        all_samples = []
        all_anomaly_labels = []
        normal_train_loader = torch.utils.data.DataLoader(normal_train_set, batch_size=args.batch_size)
        normal_train_iterator = iter(normal_train_loader)
        for _ in tqdm(range(num_cycle), desc="Generating samples"):
            anomaly_label = next(normal_train_iterator)['random_anomaly_label'].to(device)
            samples = model.get_prior_anomaly_samples(anomaly_labels=anomaly_label).cpu()
            all_samples.append(samples)
            all_anomaly_labels.append(anomaly_label)
        all_samples = torch.cat(all_samples, dim=0)
        all_anomaly_labels = torch.cat(all_anomaly_labels, dim=0)
        os.makedirs(args.generated_path, exist_ok=True)
        to_save = {
            "all_samples": all_samples,
            "all_anomaly_labels": all_anomaly_labels,
        }
        torch.save(to_save,f"{args.generated_path}/generated_anomaly.pt")
    else:
        assert args.generated_path is not None
        to_load = torch.load(
            f"{args.generated_path}/generated_anomaly.pt",
            map_location=device
        )
        all_samples = to_load["all_samples"]
        all_anomaly_labels = to_load["all_anomaly_labels"]

    anomaly_train_set = ECGDataset(
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.anomaly_indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_ratio=args.max_anomaly_ratio,
    )

    orig_data = torch.from_numpy(np.stack(anomaly_train_set.slide_windows, axis=0))
    orig_labels = torch.from_numpy(np.stack(anomaly_train_set.anomaly_labels, axis=0))

    normal_accuracy, anomaly_accuracy = calculate_robustTAD(
        anomaly_weight=5.0, feature_size=args.feature_size,
        ori_data=orig_data, ori_labels=orig_labels,
        gen_data=all_samples,
        gen_labels=all_anomaly_labels,
        device=device,
        lr=1e-4,
        max_epochs=2000,
        batch_size=64,
        patience=20)

    print(f"Normal Accuracy: {normal_accuracy:.4f}")
    print(f"Anomaly Accuracy: {anomaly_accuracy:.4f}")



