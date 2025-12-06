from Trainers import FlowTSFinetune
from generation_models import FM_TS, LastLayerPerturbFlow
from dataset_utils import build_dataset
import argparse
import torch
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from tqdm import tqdm
import numpy as np
from evaluation_utils import calculate_robustTAD


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
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--n_layer_enc", type=int, required=True)
    parser.add_argument("--n_layer_dec", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--early_stop", type=str, required=True)


    """data parameters"""
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--max_anomaly_length", type=int, required=True)
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
    parser.add_argument("--generated_path", type=str, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)

    return parser.parse_args()


def evaluate_finetune_anomaly_quality(
    args,
    model,
    normal_train_set,
    anomaly_train_set,
    ema_state_dict
    ):
    device = torch.device("cuda:%d" % args.gpu_id)
    model.load_state_dict(ema_state_dict)
    model.eval()

    num_samples = len(normal_train_set.slide_windows)
    num_cycle = int(num_samples // args.batch_size) + 1
    all_samples = []
    all_anomaly_labels = []
    normal_train_loader = torch.utils.data.DataLoader(normal_train_set, batch_size=args.batch_size)
    normal_train_iterator = iter(normal_train_loader)
    for _ in tqdm(range(num_cycle), desc="Generating samples"):
        anomaly_label = next(normal_train_iterator)['random_anomaly_label'].to(device).squeeze()
        samples = model.generate_mts(
            batch_size=args.batch_size,
            anomaly_label=anomaly_label,
        ).cpu()
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


    orig_data = torch.from_numpy(np.stack(anomaly_train_set.slide_windows, axis=0))
    orig_labels = torch.from_numpy(np.stack(anomaly_train_set.anomaly_labels, axis=0))


    precisions = []
    recalls = []
    f1s = []
    for _ in range(5):
        precision, recall, f1 = calculate_robustTAD(
            anomaly_weight=5.0,
            feature_size=args.feature_size,
            ori_data=orig_data,
            ori_labels=orig_labels,
            gen_data=all_samples,
            gen_labels=all_anomaly_labels,
            device=device,
            lr=1e-4,
            max_epochs=2000,
            batch_size=64,
            patience=20)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1s)
    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1 = np.std(f1s)
    print(f"precision: {mean_precision}+-{std_precision}")
    print(f"recall: {mean_recall}+-{std_recall}")
    print(f"f1: {mean_f1}+-{std_f1}")

    result = {
        "precision_mean": float(mean_precision),
        "precision_std": float(std_precision),
        "recall_mean": float(mean_recall),
        "recall_std": float(std_recall),
        "f1_mean": float(mean_f1),
        "f1_std": float(std_f1),
        "timestamp": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
    }

    output_record = {
        "args": vars(args),
        "result": result,
    }

    save_path = os.path.join(args.generated_path, "evaluation_results.jsonl")
    os.makedirs(args.generated_path, exist_ok=True)

    with open(save_path, "a") as f:
        f.write(json.dumps(output_record) + "\n")




def finetune():
    args = get_finetune_args()

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


    # normal_train_set = IterableECGDataset(
    #     raw_data_paths=args.raw_data_paths_train,
    #     indices_paths=args.normal_indices_paths_train,
    #     seq_len=args.seq_len,
    #     max_anomaly_ratio=args.max_anomaly_ratio,
    # )
    #
    # normal_val_set = ECGDataset(
    #     raw_data_paths=args.raw_data_paths_val,
    #     indices_paths=args.normal_indices_paths_val,
    #     seq_len=args.seq_len,
    #     max_anomaly_ratio=args.max_anomaly_ratio,
    # )
    #
    # anomaly_train_set = IterableECGDataset(
    #     raw_data_paths=args.raw_data_paths_train,
    #     indices_paths=args.anomaly_indices_paths_train,
    #     seq_len=args.seq_len,
    #     max_anomaly_ratio=args.max_anomaly_ratio,
    # )
    #
    # anomaly_val_set = ECGDataset(
    #     raw_data_paths=args.raw_data_paths_val,
    #     indices_paths=args.anomaly_indices_paths_val,
    #     seq_len=args.seq_len,
    #     max_anomaly_ratio=args.max_anomaly_ratio,
    # )

    normal_train_set = build_dataset(
        args.dataset_name,
        'iterable',
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.normal_indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_length=args.max_anomaly_length,
    )

    normal_val_set = build_dataset(
        args.dataset_name,
        'non_iterable',
        raw_data_paths=args.raw_data_paths_val,
        indices_paths=args.normal_indices_paths_val,
        seq_len=args.seq_len,
        max_anomaly_length=args.max_anomaly_length,
    )

    anomaly_train_set = build_dataset(
        args.dataset_name,
        'iterable',
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.anomaly_indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_length=args.max_anomaly_length,
    )

    anomaly_val_set = build_dataset(
        args.dataset_name,
        'non_iterable',
        raw_data_paths=args.raw_data_paths_val,
        indices_paths=args.anomaly_indices_paths_val,
        seq_len=args.seq_len,
        max_anomaly_length=args.max_anomaly_length,
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
        early_stop=args.early_stop,
    )
    ema_state_dict = trainer.finetune(
        config=vars(args),
        version=args.version,
        mode=args.mode
    )

    evaluate_finetune_anomaly_quality(
        args,
        trainer.model,
        normal_train_set,
        anomaly_train_set,
        ema_state_dict
    )


if __name__ == "__main__":
    finetune()
