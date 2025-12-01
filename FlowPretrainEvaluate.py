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
    parser.add_argument("--need_to_generate", type=int, required=True)
    parser.add_argument("--generated_path", type=str, required=True)

    return parser.parse_args()


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

    if args.need_to_generate: # 15601
        num_samples = len(normal_train_set.slide_windows)
        num_cycle = int(num_samples // args.batch_size) + 1
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

    orig_data = torch.from_numpy(np.stack(normal_train_set.slide_windows, axis=0)).to(device)
    generated_data = all_samples.to(device)

    predictive_scores = []
    discriminative_scores = []
    for i in range(5):
        predictive_scores.append(
            predictive_score_metrics(
                ori_data=orig_data,
                gen_data=generated_data
            )
        )
        disc_score, fake_acc, real_acc = discriminative_score_metrics(
                ori_data=orig_data,
                gen_data=generated_data
            )

        discriminative_scores.append(disc_score)

    pred_mean = np.mean(predictive_scores)
    pred_std = np.std(predictive_scores)

    disc_mean = np.mean(discriminative_scores)
    disc_std = np.std(discriminative_scores)

    print("Predictive mean:", pred_mean, "std:", pred_std)
    print("Discriminative mean:", disc_mean, "std:", disc_std)




if __name__ == "__main__":
    evaluate_pretrain()









