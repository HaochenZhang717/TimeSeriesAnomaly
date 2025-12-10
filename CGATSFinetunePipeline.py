from generation_models import TimeVAECGATS
from Trainers import CGATFinetune
from dataset_utils import build_dataset
import argparse
import torch
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import os
from tqdm import tqdm
from evaluation_utils import calculate_robustTAD, evaluate_model_long_sequence
import numpy as np


def save_args_to_jsonl(args, output_path):
    args_dict = vars(args)
    with open(output_path, "w") as f:
        json.dump(args_dict, f)
        f.write("\n")  # JSONL 一行一个 JSON


def get_args():
    parser = argparse.ArgumentParser(description="parameters for TimeVAE-CGATS pretraining")
    parser.add_argument(
        "--what_to_do",
        type=str,
        required=True,
        choices=["finetune", "sample_anomaly"]
    )

    """time series general parameters"""
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--feature_size", type=int, required=True)
    parser.add_argument("--one_channel", type=int, required=True)

    """model parameters"""
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--trend_poly", type=int, required=True)
    parser.add_argument("--kl_wt", type=float, required=True)
    parser.add_argument("--hidden_layer_sizes", type=json.loads, required=True)
    parser.add_argument("--custom_seas", type=json.loads, required=True)


    """data parameters"""
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--max_anomaly_length", type=float, required=True)
    parser.add_argument("--min_anomaly_length", type=float, required=True)
    parser.add_argument("--raw_data_paths_train", type=str, required=True)
    parser.add_argument("--normal_indices_paths_train", type=str, required=True)
    parser.add_argument("--anomaly_indices_paths_train", type=str, required=True)

    """training parameters"""
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--grad_clip_norm", type=float, required=True)
    parser.add_argument("--early_stop", type=str, required=True)

    """wandb parameters"""
    parser.add_argument("--wandb_project", type=str,required=True)
    parser.add_argument("--wandb_run", type=str, required=True)

    """save and load parameters"""
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--pretrained_ckpt", type=str, required=True)

    """sample parameters"""
    parser.add_argument("--tuned_ckpt", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--generated_path", type=str, required=True)
    parser.add_argument("--eval_train_size", type=int, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)

    return parser.parse_args()


def finetune(args):
    # timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d-%H:%M:%S")
    # args.ckpt_dir = f"{args.ckpt_dir}/{timestamp}"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")

    model = TimeVAECGATS(
        hidden_layer_sizes=args.hidden_layer_sizes,
        trend_poly=args.trend_poly,
        custom_seas=args.custom_seas,
        use_residual_conn=True,
        seq_len=args.seq_len,
        feat_dim=args.feature_size,
        latent_dim=args.latent_dim,
        kl_wt = args.kl_wt,
    )

    '''during pretraining, we did not update parameters in anomaly decoder, so we can just load'''
    pretrained_state_dict = torch.load(args.pretrained_ckpt)
    model.load_state_dict(pretrained_state_dict)


    data_set = build_dataset(
        args.dataset_name,
        'non_iterable',
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.anomaly_indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_length=args.max_anomaly_length,
        min_anomaly_length=args.min_anomaly_length,
        one_channel=args.one_channel,
    )

    train_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=False)


    optimizer= torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,  # multiply LR by 0.5
        patience=2,  # wait 3 epochs with no improvement
        threshold=1e-4,  # improvement threshold
        min_lr=1e-6,  # min LR clamp
    )

    trainer = CGATFinetune(
        optimizer=optimizer,
        scheduler=scheduler,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=args.epochs,
        device=f"cuda:{args.gpu_id}",
        save_dir=args.ckpt_dir,
        wandb_run_name=args.wandb_run,
        wandb_project_name=args.wandb_project,
        grad_clip_norm=args.grad_clip_norm,
        early_stop=args.early_stop,
    )
    trainer.finetune(config=vars(args))


def sample_anomaly(args):
    device = torch.device("cuda:%d" % args.gpu_id)
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
    model.load_state_dict(torch.load(args.tuned_ckpt))
    model.eval()

    normal_train_dataset = build_dataset(
        args.dataset_name,
        'iterable',
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.normal_indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_length=args.max_anomaly_length,
        min_anomaly_length=args.min_anomaly_length,
        one_channel=args.one_channel,
    )
    num_samples = args.num_samples
    num_cycle = int(num_samples // args.batch_size) + 1
    all_samples = []
    all_anomaly_labels = []
    normal_train_loader = torch.utils.data.DataLoader(normal_train_dataset, batch_size=args.batch_size)
    normal_train_iterator = iter(normal_train_loader)
    for _ in tqdm(range(num_cycle), desc="Generating samples"):
        anomaly_label = next(normal_train_iterator)['random_anomaly_label'].to(device)
        samples = model.get_prior_anomaly_samples(anomaly_labels=anomaly_label).cpu()
        all_samples.append(samples)
        all_anomaly_labels.append(anomaly_label)

    gen_data = torch.cat(all_samples, dim=0)
    gen_labels = torch.cat(all_anomaly_labels, dim=0)

    os.makedirs(args.generated_path, exist_ok=True)
    to_save = {
        "all_samples": all_samples,
        "all_anomaly_labels": all_anomaly_labels,
    }
    torch.save(to_save, f"{args.generated_path}/generated_anomaly.pt")

    # anomaly_train_set = ECGDataset(
    #     raw_data_paths=args.raw_data_paths_train,
    #     indices_paths=args.anomaly_indices_paths_train,
    #     seq_len=args.seq_len,
    #     max_anomaly_length=args.max_anomaly_length,
    # )

    anomaly_train_set = build_dataset(
        args.dataset_name,
        'non_iterable',
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.anomaly_indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_length=args.max_anomaly_length,
        min_anomaly_length=args.min_anomaly_length,
        one_channel=args.one_channel,
    )


    orig_data = torch.from_numpy(np.stack(anomaly_train_set.slide_windows, axis=0))
    orig_labels = torch.from_numpy(np.stack(anomaly_train_set.anomaly_labels, axis=0))
    orig_data = orig_data[:, :, :args.feature_size]


    # do evaluation as in ImagenTime
    # scores = evaluate_model_long_sequence(orig_data, gen_data[:len(orig_data)], device)
    # breakpoint()

    # do my evaluation
    precisions = []
    recalls = []
    f1s = []
    normal_accuracies = []
    anomaly_accuracies = []
    for _ in range(5):
        random_indices = torch.randperm(len(gen_data))[:args.eval_train_size]
        sampled_gen_data = gen_data[random_indices]
        sampled_gen_labels = gen_labels[random_indices]

        normal_accuracy, anomaly_accuracy, precision, recall, f1 = calculate_robustTAD(
            anomaly_weight=5.0,
            feature_size=args.feature_size,
            ori_data=orig_data,
            ori_labels=orig_labels,
            gen_data=sampled_gen_data,
            gen_labels=sampled_gen_labels,
            device=device,
            lr=1e-5,
            max_epochs=2000,
            batch_size=64,
            patience=20,
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        normal_accuracies.append(normal_accuracy)
        anomaly_accuracies.append(anomaly_accuracy)

    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1s)
    mean_normal_accuracy = np.mean(normal_accuracies)
    mean_anomaly_accuracy = np.mean(anomaly_accuracies)

    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1 = np.std(f1s)
    std_normal_accuracy = np.std(normal_accuracies)
    std_anomaly_accuracy = np.std(anomaly_accuracies)

    print(f"precision: {mean_precision}+-{std_precision}")
    print(f"recall: {mean_recall}+-{std_recall}")
    print(f"f1: {mean_f1}+-{std_f1}")
    print(f"normal_accuracy: {mean_normal_accuracy}+-{std_normal_accuracy}")
    print(f"anomaly_accuracy: {mean_anomaly_accuracy}+-{std_anomaly_accuracy}")

    result = {
        "precision_mean": float(mean_precision),
        "precision_std": float(std_precision),
        "recall_mean": float(mean_recall),
        "recall_std": float(std_recall),
        "f1_mean": float(mean_f1),
        "f1_std": float(std_f1),
        "normal_accuracy_mean": float(mean_normal_accuracy),
        "normal_accuracy_std": float(std_normal_accuracy),
        "anomaly_accuracy_mean": float(mean_anomaly_accuracy),
        "anomaly_accuracy_std": float(std_anomaly_accuracy),
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



if __name__ == "__main__":

    args = get_args()
    if args.what_to_do == "finetune":
        finetune(args)
    elif args.what_to_do == "sample_anomaly":
        sample_anomaly(args)
    else:
        raise ValueError("what_to_do must be 'finetune' or 'sample_anomaly'")