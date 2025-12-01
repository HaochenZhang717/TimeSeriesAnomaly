from generation_models import TimeVAECGATS
from Trainers import CGATFinetune
from dataset_utils import ECGDataset
import argparse
import torch
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import os
from tqdm import tqdm
import numpy as np
from evaluation_utils import predictive_score_metrics, discriminative_score_metrics


def save_args_to_jsonl(args, output_path):
    args_dict = vars(args)
    with open(output_path, "w") as f:
        json.dump(args_dict, f)
        f.write("\n")  # JSONL 一行一个 JSON


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
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--grad_clip_norm", type=float, required=True)

    """save and load parameters"""
    parser.add_argument("--pretrained_ckpt", type=str, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)


    """sample parameters"""
    parser.add_argument("--need_to_generate", type=int, required=True)
    parser.add_argument("--generated_path", type=str, required=True)


    return parser.parse_args()


def evaluate_pretrain():
    args = get_evaluate_args()
    device = torch.device("cuda:%d" % args.gpu_id)

    model = TimeVAECGATS(
        hidden_layer_sizes=args.hidden_layer_sizes,
        trend_poly=args.trend_poly,
        custom_seas=args.custom_seas,
        use_residual_conn=True,
        seq_len=args.seq_len,
        feat_dim=args.feature_size,
        latent_dim=args.latent_dim,
        kl_wt = args.kl_wt,
    ).to(device)

    '''during pretraining, we did not update parameters in anomaly decoder, so we can just load'''
    pretrained_state_dict = torch.load(args.pretrained_ckpt)
    model.load_state_dict(pretrained_state_dict)
    model.eval()

    normal_train_set = ECGDataset(
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_ratio=args.max_anomaly_ratio,
    )

    if args.need_to_generate:
        num_samples = len(normal_train_set.slide_windows)
        num_cycle = int(num_samples // args.batch_size) + 1
        all_samples = []
        for _ in tqdm(range(num_cycle), desc="Generating samples"):
            with torch.no_grad():
                samples = model.get_prior_normal_samples(args.batch_size).detach().cpu()
            all_samples.append(samples)
        all_samples = torch.cat(all_samples, dim=0)
        os.makedirs(args.generated_path, exist_ok=True)
        torch.save(all_samples, f"{args.generated_path}/generated_normal.pt")
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