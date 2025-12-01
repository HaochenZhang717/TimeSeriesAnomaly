from Trainers.FlowTS_trainer.flowts_trainer import FlowTSFinetune
from generation_models import FM_TS
from Trainers import FlowTSPretrain
from dataset_utils import ECGDataset, IterableECGDataset
import argparse
import torch
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os

def save_args_to_jsonl(args, output_path):
    args_dict = vars(args)
    with open(output_path, "w") as f:
        json.dump(args_dict, f)
        f.write("\n")  # JSONL 一行一个 JSON


def get_finetune_args():
    parser = argparse.ArgumentParser(description="parameters for flow-ts pretraining")

    """time series general parameters"""
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--feature_size", type=int, required=True)

    """model parameters"""
    parser.add_argument("--n_layer_enc", type=int, required=True)
    parser.add_argument("--n_layer_dec", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True)
    parser.add_argument("--version", type=int, required=True)

    """data parameters"""
    parser.add_argument("--max_anomaly_ratio", type=float, required=True)
    parser.add_argument("--raw_data_paths_train", type=str, required=True)
    parser.add_argument("--raw_data_paths_val", type=str, required=True)
    parser.add_argument("--normal_indices_paths_train", type=str, required=True)
    parser.add_argument("--normal_indices_paths_val", type=str, required=True)
    parser.add_argument("--anomaly_indices_paths_train", type=str, required=True)
    parser.add_argument("--anomaly_indices_paths_val", type=str, required=True)

    """training parameters"""
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_iters", type=int, required=True)
    parser.add_argument("--grad_clip_norm", type=float, required=True)
    parser.add_argument("--mode", type=str, required=True)

    """wandb parameters"""
    parser.add_argument("--wandb_project", type=str,required=True)
    parser.add_argument("--wandb_run", type=str, required=True)

    """save and load parameters"""
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--pretrained_ckpt", type=str, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)

    return parser.parse_args()


def finetune():
    args = get_finetune_args()

    timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d-%H:%M:%S")
    args.ckpt_dir = f"{args.ckpt_dir}/{timestamp}"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")

    model = FM_TS(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
    )

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

    """train loaders are on IterableDataset"""
    normal_train_loader = torch.utils.data.DataLoader(normal_train_set, batch_size=args.batch_size)
    anomaly_train_loader = torch.utils.data.DataLoader(anomaly_train_set, batch_size=args.batch_size)
    """val loaders are on Dataset"""
    normal_val_loader = torch.utils.data.DataLoader(normal_val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    anomaly_val_loader = torch.utils.data.DataLoader(anomaly_val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    optimizer= torch.optim.Adam(model.parameters(), lr=args.lr)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    trainer = FlowTSFinetune(
        optimizer=optimizer,
        model=model,
        train_normal_loader=normal_train_loader,
        val_normal_loader=normal_val_loader,
        train_anomaly_loader=anomaly_train_loader,
        val_anomaly_loader=anomaly_val_loader,
        max_iters=args.max_iters,
        device=device,
        save_dir=args.ckpt_dir,
        wandb_run_name=args.wandb_run,
        wandb_project_name=args.wandb_project,
        grad_clip_norm=args.grad_clip_norm,
        pretrained_ckpt=args.pretrained_ckpt,
    )
    trainer.finetune(config=vars(args), version=args.version, mode=args.mode)

if __name__ == "__main__":
    finetune()
