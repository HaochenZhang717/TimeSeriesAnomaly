from Trainers import FlowTSTrainerTwoTogether
from generation_models import FM_TS_Two_Together
from dataset_utils import build_dataset, FakeDataset
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
        choices=["conditional_training", "unconditional_training",
                 "conditional_evaluate",
                 "conditional_sample_on_real_anomaly", "conditional_sample_on_real_normal",
                 "conditional_sample_on_fake", "unconditional_sample",
                 "anomaly_evaluate", "unconditional_evaluate"],
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


# def evaluate_finetune_anomaly_quality(
#     args,
#     model,
#     normal_train_set,
#     anomaly_train_set,
#     ema_state_dict
#     ):
#     device = torch.device("cuda:%d" % args.gpu_id)
#     model.load_state_dict(ema_state_dict)
#     model.eval()
#
#     num_samples = len(normal_train_set.slide_windows)
#     num_cycle = int(num_samples // args.batch_size) + 1
#     all_samples = []
#     all_anomaly_labels = []
#     normal_train_loader = torch.utils.data.DataLoader(normal_train_set, batch_size=args.batch_size)
#     normal_train_iterator = iter(normal_train_loader)
#     for _ in tqdm(range(num_cycle), desc="Generating samples"):
#         anomaly_label = next(normal_train_iterator)['random_anomaly_label'].to(device).squeeze()
#         samples = model.generate_mts(
#             batch_size=args.batch_size,
#             anomaly_label=anomaly_label,
#         ).cpu()
#         all_samples.append(samples)
#         all_anomaly_labels.append(anomaly_label)
#     all_samples = torch.cat(all_samples, dim=0)
#     all_anomaly_labels = torch.cat(all_anomaly_labels, dim=0)
#     os.makedirs(args.generated_path, exist_ok=True)
#     to_save = {
#         "all_samples": all_samples,
#         "all_anomaly_labels": all_anomaly_labels,
#     }
#     torch.save(to_save,f"{args.generated_path}/generated_anomaly.pt")
#
#
#     orig_data = torch.from_numpy(np.stack(anomaly_train_set.slide_windows, axis=0))
#     orig_labels = torch.from_numpy(np.stack(anomaly_train_set.anomaly_labels, axis=0))
#
#
#     precisions = []
#     recalls = []
#     f1s = []
#     for _ in range(5):
#         precision, recall, f1 = calculate_robustTAD(
#             anomaly_weight=5.0,
#             feature_size=args.feature_size,
#             ori_data=orig_data,
#             ori_labels=orig_labels,
#             gen_data=all_samples,
#             gen_labels=all_anomaly_labels,
#             device=device,
#             lr=1e-4,
#             max_epochs=2000,
#             batch_size=64,
#             patience=20)
#         precisions.append(precision)
#         recalls.append(recall)
#         f1s.append(f1)
#
#     mean_precision = np.mean(precisions)
#     mean_recall = np.mean(recalls)
#     mean_f1 = np.mean(f1s)
#     std_precision = np.std(precisions)
#     std_recall = np.std(recalls)
#     std_f1 = np.std(f1s)
#     print(f"precision: {mean_precision}+-{std_precision}")
#     print(f"recall: {mean_recall}+-{std_recall}")
#     print(f"f1: {mean_f1}+-{std_f1}")
#
#     result = {
#         "precision_mean": float(mean_precision),
#         "precision_std": float(std_precision),
#         "recall_mean": float(mean_recall),
#         "recall_std": float(std_recall),
#         "f1_mean": float(mean_f1),
#         "f1_std": float(std_f1),
#         "timestamp": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
#     }
#
#     output_record = {
#         "args": vars(args),
#         "result": result,
#     }
#
#     save_path = os.path.join(args.generated_path, "evaluation_results.jsonl")
#     os.makedirs(args.generated_path, exist_ok=True)
#
#     with open(save_path, "a") as f:
#         f.write(json.dumps(output_record) + "\n")




# def train():
#     args = get_args()
#
#     # timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d-%H:%M:%S")
#     # args.ckpt_dir = f"{args.ckpt_dir}/{timestamp}"
#     os.makedirs(args.ckpt_dir, exist_ok=True)
#     save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")
#
#
#     model = FM_TS_Two_Together(
#         seq_length=args.seq_len,
#         feature_size=args.feature_size,
#         n_layer_enc=args.n_layer_enc,
#         n_layer_dec=args.n_layer_dec,
#         d_model=args.d_model,
#         n_heads=args.n_heads,
#         mlp_hidden_times=4,
#     )
#
#     normal_train_set = build_dataset(
#         args.dataset_name,
#         'iterable',
#         raw_data_paths=args.raw_data_paths_train,
#         indices_paths=args.normal_indices_paths_train,
#         seq_len=args.seq_len,
#         max_anomaly_length=args.max_anomaly_length,
#     )
#
#     normal_val_set = build_dataset(
#         args.dataset_name,
#         'non_iterable',
#         raw_data_paths=args.raw_data_paths_val,
#         indices_paths=args.normal_indices_paths_val,
#         seq_len=args.seq_len,
#         max_anomaly_length=args.max_anomaly_length,
#     )
#
#     anomaly_train_set = build_dataset(
#         args.dataset_name,
#         'iterable',
#         raw_data_paths=args.raw_data_paths_train,
#         indices_paths=args.anomaly_indices_paths_train,
#         seq_len=args.seq_len,
#         max_anomaly_length=args.max_anomaly_length,
#     )
#
#     anomaly_val_set = build_dataset(
#         args.dataset_name,
#         'non_iterable',
#         raw_data_paths=args.raw_data_paths_val,
#         indices_paths=args.anomaly_indices_paths_val,
#         seq_len=args.seq_len,
#         max_anomaly_length=args.max_anomaly_length,
#     )
#
#
#     """train loaders are on IterableDataset"""
#     normal_train_loader = torch.utils.data.DataLoader(normal_train_set, batch_size=args.batch_size)
#     anomaly_train_loader = torch.utils.data.DataLoader(anomaly_train_set, batch_size=args.batch_size)
#     """val loaders are on Dataset"""
#     normal_val_loader = torch.utils.data.DataLoader(normal_val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
#     anomaly_val_loader = torch.utils.data.DataLoader(anomaly_val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
#
#     optimizer= torch.optim.Adam(model.parameters(), lr=args.lr)
#
#     device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
#     trainer = FlowTSTrainerTwoTogether(
#         optimizer=optimizer,
#         model=model,
#         train_normal_loader=normal_train_loader,
#         val_normal_loader=normal_val_loader,
#         train_anomaly_loader=anomaly_train_loader,
#         val_anomaly_loader=anomaly_val_loader,
#         max_iters=args.max_iters,
#         device=device,
#         save_dir=args.ckpt_dir,
#         wandb_run_name=args.wandb_run,
#         wandb_project_name=args.wandb_project,
#         grad_clip_norm=args.grad_clip_norm,
#         early_stop=args.early_stop,
#     )
#     ema_state_dict = trainer.train(
#         config=vars(args),
#     )
#
#     evaluate_finetune_anomaly_quality(
#         args,
#         trainer.model,
#         normal_train_set,
#         anomaly_train_set,
#         ema_state_dict
#     )


def unconditional_train(args):
    # args = get_args()

    # timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d-%H:%M:%S")
    # args.ckpt_dir = f"{args.ckpt_dir}/{timestamp}"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")

    model = FM_TS_Two_Together(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
    )

    normal_train_set = build_dataset(
        args.dataset_name,
        'non_iterable',
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_length=args.max_anomaly_length,
        min_anomaly_length=args.min_anomaly_length,
        one_channel=args.one_channel,
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

    trainer.unconditional_train(config=vars(args))



def conditional_train(args):
    # args = get_args()

    # timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d-%H:%M:%S")
    # args.ckpt_dir = f"{args.ckpt_dir}/{timestamp}"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")

    model = FM_TS_Two_Together(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
    )

    anomaly_train_set = build_dataset(
        args.dataset_name,
        'non_iterable',
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_length=args.max_anomaly_length,
        min_anomaly_length=args.min_anomaly_length,
        one_channel=args.one_channel,
        limited_data_size=args.limited_data_size
    )

    train_loader = torch.utils.data.DataLoader(anomaly_train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(anomaly_train_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    optimizer= torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,  # multiply LR by 0.5
        patience=5,  # wait 3 epochs with no improvement
        threshold=1e-4,  # improvement threshold
        min_lr=1e-5,  # min LR clamp
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

    trainer.conditional_train(config=vars(args))



# def conditional_sample_on_real_anomaly(args):
#     # args = get_args()
#     device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
#     model = FM_TS_Two_Together(
#         seq_length=args.seq_len,
#         feature_size=args.feature_size,
#         n_layer_enc=args.n_layer_enc,
#         n_layer_dec=args.n_layer_dec,
#         d_model=args.d_model,
#         n_heads=args.n_heads,
#         mlp_hidden_times=4,
#     )
#     model.load_state_dict(torch.load(args.cond_eval_model_ckpt))
#     model.to(device)
#     model.eval()
#     anomaly_train_set = build_dataset(
#         args.dataset_name,
#         'iterable',
#         raw_data_paths=args.raw_data_paths_train,
#         indices_paths=args.indices_paths_train,
#         seq_len=args.seq_len,
#         max_anomaly_length=args.max_anomaly_length,
#         min_anomaly_length=args.min_anomaly_length,
#         one_channel=args.one_channel,
#     )
#
#     train_loader = torch.utils.data.DataLoader(
#         anomaly_train_set,
#         batch_size=args.batch_size,
#     )
#
#     # num_samples = len(anomaly_train_set.slide_windows)
#     num_samples = args.cond_num_samples
#     num_cycle = int(num_samples // args.batch_size) + 1
#     train_iterator = iter(train_loader)
#
#     all_samples = []
#     all_real = []
#     all_anomaly_labels = []
#     for _ in tqdm(range(num_cycle), desc="Generating samples"):
#         a_batch = next(train_iterator)
#         anomaly_label = a_batch['anomaly_label'].to(device).squeeze(-1)#i changed this
#         real_signal = a_batch['orig_signal'].to(device)[:, :, : args.feature_size]
#         samples = model.impute(
#             x_start=real_signal,
#             anomaly_label=anomaly_label,
#         ).cpu()
#         all_samples.append(samples)
#         all_real.append(real_signal)
#         all_anomaly_labels.append(anomaly_label)
#
#     all_samples = torch.cat(all_samples, dim=0)
#     all_anomaly_labels = torch.cat(all_anomaly_labels, dim=0)
#     all_real = torch.cat(all_real, dim=0)
#     to_save = {
#         "samples": all_samples,
#         "real": all_real,
#         "anomaly_labels": all_anomaly_labels,
#     }
#     os.makedirs(args.generated_path, exist_ok=True)
#     torch.save(to_save, f"{args.generated_path}/generated_anomaly_on_real_anomaly.pt")
#     # scores = evaluate_model_long_sequence(to_save["real"], to_save["samples"], device)
#     # print(f"Scores: {scores}")
#     # breakpoint()
#     # print(f"Scores: {scores}")


def conditional_sample_on_real_anomaly(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    model = FM_TS_Two_Together(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
    )
    model.load_state_dict(torch.load(args.cond_eval_model_ckpt))
    model.to(device)
    model.eval()

    anomaly_train_set = build_dataset(
        args.dataset_name,
        'iterable',
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_length=args.max_anomaly_length,
        min_anomaly_length=args.min_anomaly_length,
        one_channel=args.one_channel,
        limited_data_size=args.limited_data_size
    )

    train_loader = torch.utils.data.DataLoader(
        anomaly_train_set,
        batch_size=args.batch_size,
    )

    num_samples = args.cond_num_samples
    num_cycle = int(num_samples // args.batch_size) + 1
    train_iterator = iter(train_loader)

    K = 10  # number of samples per real input

    all_samples = []
    all_real = []
    all_anomaly_labels = []

    for _ in tqdm(range(num_cycle), desc="Generating samples"):
        batch = next(train_iterator)

        anomaly_label = batch['anomaly_label'].to(device).squeeze(-1)   # (B, T)
        real_signal = batch['orig_signal'].to(device)[:, :, :args.feature_size]  # (B, T, C)

        B = real_signal.shape[0]

        # -------- multiple stochastic flow samples --------
        batch_samples = []
        for _ in range(K):
            samples = model.impute(
                x_start=real_signal,
                anomaly_label=anomaly_label,
            )  # (B, T, C)
            batch_samples.append(samples.unsqueeze(1))  # (B, 1, T, C)

        batch_samples = torch.cat(batch_samples, dim=1)  # (B, K, T, C)

        all_samples.append(batch_samples.cpu())
        all_real.append(real_signal.cpu())
        all_anomaly_labels.append(anomaly_label.cpu())

    # -------- concat over batches --------
    all_samples = torch.cat(all_samples, dim=0)          # (N, K, T, C)
    all_real = torch.cat(all_real, dim=0)                # (N, T, C)
    all_anomaly_labels = torch.cat(all_anomaly_labels, dim=0)  # (N, T)

    to_save = {
        "samples": all_samples,
        "real": all_real,
        "anomaly_labels": all_anomaly_labels,
    }

    os.makedirs(args.generated_path, exist_ok=True)
    torch.save(
        to_save,
        f"{args.generated_path}/generated_anomaly_on_real_anomaly_multi_sample.pt"
    )


def conditional_sample_on_fake(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = FM_TS_Two_Together(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
    )
    model.load_state_dict(torch.load(args.cond_eval_model_ckpt))
    model.to(device)
    model.eval()

    fake_normal_data = FakeDataset(
        normal_data_path=args.normal_data_path,
        maximum_anomaly_length=args.max_anomaly_length,
        minimum_anomaly_length=args.min_anomaly_length,
        one_channel=args.one_channel,
    )

    fake_normal_loader = torch.utils.data.DataLoader(
        fake_normal_data,
        batch_size=args.batch_size,
    )


    all_samples = []
    all_real = []
    all_anomaly_labels = []
    for batch in tqdm(fake_normal_loader, desc="Generating samples"):
        anomaly_label = batch['random_anomaly_label'].to(device)
        real_signal = batch['original_signal'].to(device)
        samples = model.impute(
            x_start=real_signal,
            anomaly_label=anomaly_label,
        ).cpu()
        all_samples.append(samples)
        all_real.append(real_signal)
        all_anomaly_labels.append(anomaly_label)

    all_samples = torch.cat(all_samples, dim=0)
    all_anomaly_labels = torch.cat(all_anomaly_labels, dim=0)
    all_real = torch.cat(all_real, dim=0)
    to_save = {
        "samples": all_samples,
        "real": all_real,
        "anomaly_labels": all_anomaly_labels,
    }
    os.makedirs(args.generated_path, exist_ok=True)
    torch.save(to_save, f"{args.generated_path}/generated_anomaly_on_fake.pt")


def conditional_sample_on_real_normal(args):

    def get_fake_anomaly_labels():
        random_anomaly_length = np.random.randint(args.min_anomaly_length, args.max_anomaly_length)
        anomaly_start = np.random.randint(0, args.seq_len - random_anomaly_length)
        anomaly_end = anomaly_start + random_anomaly_length
        random_anomaly_label = torch.zeros((1, args.seq_len))
        random_anomaly_label[0, anomaly_start:anomaly_end] = 1
        return random_anomaly_label

    def get_batch_fake_anomaly_labels():
        batch_fake_anomaly_labels = []
        for _ in range(args.batch_size):
            batch_fake_anomaly_labels.append(get_fake_anomaly_labels())
        batch_fake_anomaly_labels = torch.cat(batch_fake_anomaly_labels, dim=0)
        return batch_fake_anomaly_labels

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    model = FM_TS_Two_Together(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
    )
    model.load_state_dict(torch.load(args.cond_eval_model_ckpt))
    model.to(device)
    model.eval()
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs! DataParallel enabled.")
    #     model = torch.nn.DataParallel(model)

    normal_train_set = build_dataset(
        args.dataset_name,
        'iterable',
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_length=args.max_anomaly_length,
        min_anomaly_length=args.min_anomaly_length,
        one_channel=args.one_channel,
    )


    normal_loader = torch.utils.data.DataLoader(
        normal_train_set,
        batch_size=args.batch_size,
    )

    num_samples = args.cond_num_samples
    num_cycle = int(num_samples // args.batch_size) + 1
    train_iterator = iter(normal_loader)

    all_samples = []
    all_real = []
    all_anomaly_labels = []
    num_generated = 0
    for _ in tqdm(range(num_cycle), desc="Generating samples"):
        batch = next(train_iterator)
        anomaly_label = get_batch_fake_anomaly_labels().to(device)
        real_signal = batch['orig_signal'].to(device)
        # samples = model.impute(
        #     x_start=real_signal,
        #     anomaly_label=anomaly_label,
        # ).cpu()
        samples = model.impute(real_signal, anomaly_label)
        samples = samples.detach().cpu()

        num_generated += samples.shape[0]
        print(num_generated)
        all_samples.append(samples)
        all_real.append(real_signal)
        all_anomaly_labels.append(anomaly_label)
        if num_generated >= args.cond_num_samples:
            break

    all_samples = torch.cat(all_samples, dim=0)
    all_anomaly_labels = torch.cat(all_anomaly_labels, dim=0)
    all_real = torch.cat(all_real, dim=0)
    to_save = {
        "samples": all_samples,
        "real": all_real,
        "anomaly_labels": all_anomaly_labels,
    }
    os.makedirs(args.generated_path, exist_ok=True)
    torch.save(to_save, f"{args.generated_path}/generated_anomaly_on_real_normal.pt")


def unconditional_sample(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = FM_TS_Two_Together(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
    )
    model.load_state_dict(torch.load(args.uncond_eval_model_ckpt))
    model.to(device)
    model.eval()

    num_cycle = int(args.uncond_num_samples // args.batch_size) + 1

    all_samples = []
    for _ in tqdm(range(num_cycle), desc="Generating samples"):
        samples = model.generate_mts(batch_size=args.batch_size).cpu()
        all_samples.append(samples)

    all_samples = torch.cat(all_samples, dim=0)
    to_save = {
        "samples": all_samples,
    }
    os.makedirs(args.generated_path, exist_ok=True)
    torch.save(to_save, f"{args.generated_path}/generated_normal.pt")



def unconditional_evaluate(args):
    fake_normal_data = torch.load(args.normal_data_path)
    fake_normal_data = fake_normal_data['samples']
    normal_train_set = build_dataset(
        args.dataset_name,
        'non_iterable',
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_length=args.max_anomaly_length,
        min_anomaly_length=args.min_anomaly_length,
        one_channel=args.one_channel,
    )
    real_normal_data = torch.from_numpy(np.stack(normal_train_set.slide_windows, axis=0))[:,:,:args.feature_size]

    device = torch.device(f"cuda:{args.gpu_id}")
    num_data = min(len(real_normal_data), len(fake_normal_data))
    scores = evaluate_model_long_sequence(real_normal_data[:num_data], fake_normal_data[:num_data], device)
    print(scores)
    breakpoint()
    print(scores)




def anomaly_evaluate(args):

    device = torch.device(f"cuda:{args.gpu_id}")
    # all_anomalies = torch.load(
    #     f"{args.generated_path}/generated_anomaly_on_fake.pt",
    #     map_location=device
    # )
    # all_anomalies = torch.load(
    #     f"{args.generated_path}/generated_anomaly_on_real_normal.pt",
    #     map_location=device
    # )
    all_anomalies = torch.load(
        f"{args.generated_path}/{args.generated_file}",
        map_location=device
    )

    gen_data = all_anomalies['samples']
    gen_labels = all_anomalies['anomaly_labels']

    # ---- Step 1: 找出含 NaN 的样本 ----
    nan_mask = torch.isnan(gen_data).any(dim=(1, 2))  # True 表示该样本含 NaN

    print("Samples containing NaN:", nan_mask.sum().item(), "/", gen_data.size(0))

    # ---- Step 2: 删除这些样本 ----
    gen_data = gen_data[~nan_mask]
    gen_labels = gen_labels[~nan_mask]


    anomaly_train_set = build_dataset(
        args.dataset_name,
        'iterable',
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_length=args.max_anomaly_length,
        min_anomaly_length=args.min_anomaly_length,
        one_channel=args.one_channel,
    )
    orig_data = torch.from_numpy(np.stack(anomaly_train_set.slide_windows, axis=0)).to(device)
    orig_labels = torch.from_numpy(np.stack(anomaly_train_set.anomaly_labels, axis=0)).to(device)
    orig_data = orig_data[:, :, :args.feature_size]

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
            patience=20)
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
    qweqwe = args.generated_file.split(".")[0]
    save_path = os.path.join(args.generated_path, f"evaluation_results_{qweqwe}.jsonl")
    os.makedirs(args.generated_path, exist_ok=True)

    with open(save_path, "a") as f:
        f.write(json.dumps(output_record) + "\n")



def main():
    args = get_args()
    if args.what_to_do == "conditional_training":
        conditional_train(args)
    elif args.what_to_do == "conditional_sample_on_real_anomaly":
        conditional_sample_on_real_anomaly(args)
    elif args.what_to_do == "conditional_sample_on_real_normal":
        conditional_sample_on_real_normal(args)
    elif args.what_to_do == "conditional_sample_on_fake":
        conditional_sample_on_fake(args)
    elif args.what_to_do == "unconditional_sample":
        unconditional_sample(args)
    elif args.what_to_do == "unconditional_evaluate":
        unconditional_evaluate(args)
    elif args.what_to_do == "anomaly_evaluate":
        anomaly_evaluate(args)
    elif args.what_to_do == "unconditional_training":
        unconditional_train(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
