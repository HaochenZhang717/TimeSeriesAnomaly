from Trainers import FlowTSTrainerTwoTogether
from generation_models import FM_TS_Two_Together, fast_build_autoencoder
from dataset_utils import build_dataset
import argparse
import torch
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from tqdm import tqdm
import numpy as np
from evaluation_utils import calculate_robustTAD, evaluate_model_long_sequence


def save_args_to_jsonl(args, output_path):
    args_dict = vars(args)
    with open(output_path, "w") as f:
        json.dump(args_dict, f)
        f.write("\n")  # JSONL 一行一个 JSON


def get_args():
    parser = argparse.ArgumentParser(description="parameters for flow-ts pretraining")


    """what to do"""
    parser.add_argument(
        "--what_to_do", type=str, required=True,
        choices=[
            "autoencoder_train",
            "flow_training",],
        help="what to do"
    )


    """time series general parameters"""
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--feature_size", type=int, required=True)
    parser.add_argument("--one_channel", type=int, required=True)


    """model parameters"""
    parser.add_argument("--n_layer_enc", type=int, required=True)
    parser.add_argument("--n_layer_dec", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True)


    """data parameters"""
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--max_anomaly_length", type=int, required=True)
    parser.add_argument("--min_anomaly_length", type=int, required=True)
    parser.add_argument("--raw_data_paths_train", type=str, required=True)
    parser.add_argument("--raw_data_paths_val", type=str, required=True)
    parser.add_argument("--indices_paths_train", type=str, required=True)
    parser.add_argument("--indices_paths_val", type=str, required=True)
    parser.add_argument("--limited_data_size", type=int, required=True)

    """training parameters"""
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_epochs", type=int, required=True)
    parser.add_argument("--grad_clip_norm", type=float, required=True)
    parser.add_argument("--grad_accum_steps", type=int, required=True)
    parser.add_argument("--early_stop", type=str, required=True)
    parser.add_argument("--patience", type=int, required=True)


    """wandb parameters"""
    parser.add_argument("--wandb_project", type=str,required=True)
    parser.add_argument("--wandb_run", type=str, required=True)

    """save and load parameters"""
    parser.add_argument("--ckpt_dir", type=str, required=True)

    """parameters for conditional sample"""
    parser.add_argument("--cond_eval_model_ckpt", type=str, required=True)
    parser.add_argument("--generated_path", type=str, required=True)
    parser.add_argument("--generated_file", type=str, required=True)
    parser.add_argument("--normal_data_path", type=str, required=True)
    parser.add_argument("--cond_num_samples", type=int, required=True)

    """parameters for unconditional sample"""
    parser.add_argument("--uncond_eval_model_ckpt", type=str, required=True)
    parser.add_argument("--uncond_num_samples", type=int, required=True)

    """parameters for anomaly evaluation"""
    parser.add_argument("--eval_train_size", type=int, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)

    return parser.parse_args()



def autoencoder_train(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")
    model = fast_build_autoencoder(feat_dim=args.feature_size, max_len=args.seq_len)
    normal_train_set = build_dataset(
        args.dataset_name,
        'non_iterable',
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_length=args.max_anomaly_length,
        min_anomaly_length=args.min_anomaly_length,
        one_channel=args.one_channel,
        limited_data_size=args.limited_data_size,
    )

    train_loader = torch.utils.data.DataLoader(normal_train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(normal_train_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    optimizer= torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,  # multiply LR by 0.5
        patience=2,  # wait 3 epochs with no improvement
        threshold=1e-4,  # improvement threshold
        min_lr=1e-6,  # min LR clamp
    )

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    trainer = FlowTSTrainerTwoTogether(
        optimizer=optimizer,
        scheduler=scheduler,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=args.max_epochs,
        device=device,
        save_dir=args.ckpt_dir,
        wandb_run_name=args.wandb_run,
        wandb_project_name=args.wandb_project,
        grad_clip_norm=args.grad_clip_norm,
        grad_accum_steps=args.grad_accum_steps,
        early_stop=args.early_stop,
        patience=args.patience,
    )

    trainer.normal_manifold_init_train(config=vars(args))


def main():
    args = get_args()
    if args.what_to_do == "autoencoder_train":
        autoencoder_train(args)
    elif args.what_to_do == "flow_train":
        raise NotImplementedError("Flow training not implemented.")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()


