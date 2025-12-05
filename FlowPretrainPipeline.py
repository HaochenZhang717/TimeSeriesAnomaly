from generation_models import FM_TS, LastLayerPerturbFlow
from Trainers import FlowTSPretrain
# from dataset_utils import ECGDataset
from dataset_utils import build_dataset
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


def get_pretrain_args():
    parser = argparse.ArgumentParser(description="parameters for flow-ts pretraining")

    """time series general parameters"""
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--feature_size", type=int, required=True)

    """model parameters"""
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--n_layer_enc", type=int, required=True)
    parser.add_argument("--n_layer_dec", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True)
    parser.add_argument("--early_stop", type=str, required=True)

    """data parameters"""
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--max_anomaly_length", type=int, required=True)
    parser.add_argument("--raw_data_paths_train", type=str, required=True)
    parser.add_argument("--raw_data_paths_val", type=str, required=True)
    parser.add_argument("--indices_paths_train", type=str, required=True)
    parser.add_argument("--indices_paths_val", type=str, required=True)

    """training parameters"""
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--grad_clip_norm", type=float, required=True)

    """wandb parameters"""
    parser.add_argument("--wandb_project", type=str,required=True)
    parser.add_argument("--wandb_run", type=str, required=True)

    """save and load parameters"""
    parser.add_argument("--ckpt_dir", type=str, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)

    return parser.parse_args()

def pretrain():
    args = get_pretrain_args()

    timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d-%H:%M:%S")
    args.ckpt_dir = f"{args.ckpt_dir}/{timestamp}"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")

    if args.model_type == "FM_TS":
        model = FM_TS(
            seq_length=args.seq_len,
            feature_size=args.feature_size,
            n_layer_enc=args.n_layer_enc,
            n_layer_dec=args.n_layer_dec,
            d_model=args.d_model,
            n_heads=args.n_heads,
            mlp_hidden_times=4,
        )
    elif args.model_type == "LastLayerPerturbFlow":
        model = LastLayerPerturbFlow(
            seq_length=args.seq_len,
            feature_size=args.feature_size,
            n_layer_enc=args.n_layer_enc,
            n_layer_dec=args.n_layer_dec,
            d_model=args.d_model,
            n_heads=args.n_heads,
            mlp_hidden_times=4,
        )
    else:
        raise ValueError(f"{args.model_type} is not supported")

    # pretrain_dataset_train = ECGDataset(
    #     args.raw_data_paths_train,
    #     args.indices_paths_train,
    #     args.seq_len,
    #     args.max_anomaly_ratio,
    # )
    #
    # pretrain_dataset_val = ECGDataset(
    #     args.raw_data_paths_val,
    #     args.indices_paths_val,
    #     args.seq_len,
    #     args.max_anomaly_ratio,
    # )

    pretrain_dataset_train = build_dataset(
        args.dataset_name,
        'non_iterable',
        args.raw_data_paths_train,
        args.indices_paths_train,
        args.seq_len,
        args.max_anomaly_length,
    )

    pretrain_dataset_val = build_dataset(
        args.dataset_name,
        'non_iterable',
        args.raw_data_paths_val,
        args.indices_paths_val,
        args.seq_len,
        args.max_anomaly_length,
    )


    train_loader = torch.utils.data.DataLoader(pretrain_dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(pretrain_dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)


    optimizer= torch.optim.Adam(model.parameters(), lr=args.lr)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    trainer = FlowTSPretrain(
        optimizer=optimizer,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=args.epochs,
        device=device,
        save_dir=args.ckpt_dir,
        wandb_run_name=args.wandb_run,
        wandb_project_name=args.wandb_project,
        grad_clip_norm=args.grad_clip_norm,
        early_stop=args.early_stop,
    )
    trainer.pretrain(config=vars(args))

if __name__ == "__main__":
    pretrain()
