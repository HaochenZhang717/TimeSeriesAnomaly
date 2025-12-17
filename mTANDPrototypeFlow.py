from Trainers import PrototypeFlowTSTrainer
from generation_models import MTANDPrototypeFlow
from dataset_utils import ImputationECGDataset
import argparse
import torch
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from tqdm import tqdm
import numpy as np
from evaluation_utils import calculate_robustTAD, evaluate_model_long_sequence


def pad_collate_fn(batch, max_len):
    """
    batch: list of dicts
        {
            'signal': Tensor [L_i, C],
            'prototype_id': int
        }
    """

    B = len(batch)
    C = batch[0]['signal'].shape[-1]

    padded = torch.zeros(B, max_len, C, dtype=batch[0]['signal'].dtype)
    attention_mask = torch.zeros(B, max_len, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    prototypes = torch.zeros(B, dtype=torch.long)

    for i, sample in enumerate(batch):
        signal = sample['signal']
        L = min(signal.shape[0], max_len)

        padded[i, :L] = signal[:L]
        attention_mask[i, :L] = True
        lengths[i] = L
        prototypes[i] = sample.get('prototype_id', -100)

    return {
        'signals': padded,        # (B, T, C)
        'attn_mask': attention_mask,  # (B, T)  True = valid
        'lengths': lengths,             # (B,)
        'prototypes': prototypes        # (B,)
    }


def dict_collate_fn(batch):
    out = {}
    for key in batch[0].keys():
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return out




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
            "imputation_train",
            "imputation_sample"
        ],
        help="what to do"
    )

    """time series general parameters"""
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--feature_size", type=int, required=True)
    parser.add_argument("--one_channel", type=int, required=True)
    parser.add_argument("--use_prototype", type=str, required=True)

    """model parameters"""
    parser.add_argument("--n_layer_enc", type=int, required=True)
    parser.add_argument("--n_layer_dec", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True)
    parser.add_argument("--num_prototypes", type=int, required=True)
    parser.add_argument("--encoder_H", type=int, required=True)
    parser.add_argument("--encoder_d_h", type=int, required=True)


    """data parameters"""
    # parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--raw_data_path_train", type=str, required=True)
    parser.add_argument("--indices_path_train", type=str, required=True)

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

    """save path """
    parser.add_argument("--generated_dir", type=str, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)

    return parser.parse_args()




def imputation_train(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")

    model = MTANDPrototypeFlow(
        encoder_H=args.encoder_H,
        encoder_d_h=args.encoder_d_h,
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
        num_prototypes=args.num_prototypes,
    )

    train_set = ImputationECGDataset(
        raw_data_path=args.raw_data_path_train,
        indices_path=args.indices_path_train,
        seq_len=args.seq_len,
        one_channel=args.one_channel,
        use_prototype=args.use_prototype
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, drop_last=True,
        collate_fn = dict_collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        collate_fn=dict_collate_fn,
    )

    optimizer= torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,  # multiply LR by 0.5
        patience=1,  # wait 3 epochs with no improvement
        threshold=1e-4,  # improvement threshold
        min_lr=1e-5,  # min LR clamp
    )

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    trainer = PrototypeFlowTSTrainer(
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

    trainer.imputation_train(config=vars(args))





@torch.no_grad()
def imputation_sample(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")

    # -----------------------
    # build & load model
    # -----------------------
    model = MTANDPrototypeFlow(
        encoder_H=args.encoder_H,
        encoder_d_h=args.encoder_d_h,
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
        num_prototypes=args.num_prototypes,
    )

    model.load_state_dict(torch.load(f"{args.ckpt_dir}/ckpt.pth", map_location="cpu"))
    model.eval()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_dtype = next(model.parameters()).dtype

    # -----------------------
    # dataset / loader
    # -----------------------
    train_set = ImputationECGDataset(
        raw_data_path=args.raw_data_path_train,
        indices_path=args.indices_path_train,
        seq_len=args.seq_len,
        one_channel=args.one_channel,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        drop_last=True,
        collate_fn=dict_collate_fn,
    )

    # -----------------------
    # sampling
    # -----------------------
    for batch in tqdm(train_loader):
        signals = batch["signals"].to(device=device, dtype=model_dtype)      # (B, T, C)
        attn_mask = batch["attn_mask"].to(device=device, dtype=torch.bool)   # (B, T)
        noise_mask = batch["noise_mask"].to(device=device, dtype=torch.long) # (B, T)

        B, T, C = signals.shape
        n = 20  # 每个 prototype 采样 n 个

        to_save = {
            "signals": signals.detach().cpu(),
            "attn_mask": attn_mask.detach().cpu(),
            "noise_mask": noise_mask.detach().cpu(),
        }

        # -------------------------------------------------
        # loop over prototype types（这个 loop 是必要的）
        # -------------------------------------------------
        for proto_id in range(args.num_prototypes):
            print(f"Sampling prototype {proto_id}")

            # (B,) -> 当前 prototype id
            proto = torch.full(
                (B,),
                proto_id,
                device=device,
                dtype=torch.long,
            )

            # -------------------------------
            # 并行：repeat batch n 次
            # -------------------------------
            signals_rep = signals.repeat_interleave(n, dim=0)        # (B*n, T, C)
            attn_mask_rep = attn_mask.repeat_interleave(n, dim=0)    # (B*n, T)
            noise_mask_rep = noise_mask.repeat_interleave(n, dim=0)  # (B*n, T)
            proto_rep = proto.repeat_interleave(n, dim=0)            # (B*n,)

            # -------------------------------
            # single forward → B*n samples
            # -------------------------------
            samples_rep = model.impute(
                signals_rep,
                proto_rep,
                attn_mask_rep,
                noise_mask_rep,
            )  # (B*n, T, C)

            # reshape -> (B, n, T, C)
            samples = samples_rep.view(B, n, T, C)

            to_save[f"samples_type_{proto_id}"] = samples.detach().cpu()

        break  # 只跑一个 batch（和你原来逻辑一致）

    # -----------------------
    # save
    # -----------------------
    os.makedirs(args.generated_dir, exist_ok=True)
    save_path = os.path.join(args.generated_dir, "samples.pth")
    torch.save(to_save, save_path)

    print(f"[✓] Saved samples to {save_path}")




def main():
    args = get_args()
    if args.what_to_do == "imputation_train":
        imputation_train(args)
    elif args.what_to_do == "imputation_sample":
        imputation_sample(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
