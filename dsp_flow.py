from torch import dtype

from Trainers import DSPFlowTrainer
from generation_models import DSPFlow
from dataset_utils import ImputationNormalECGDataset
from dataset_utils import ImputationECGDataset, NoContextNormalECGDataset
import argparse
import torch
import json
import os




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
            "imputation_pretrain",
            "no_context_pretrain",
            "no_context_sample",
        ],
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
    parser.add_argument("--raw_data_path_train", type=str, required=True)
    parser.add_argument("--indices_path_train", type=str, required=True)
    parser.add_argument("--min_infill_length", type=int, required=True)
    parser.add_argument("--max_infill_length", type=int, required=True)

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
    parser.add_argument("--vqvae_ckpt", type=str, required=True)

    """save path """
    parser.add_argument("--generated_dir", type=str, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)

    return parser.parse_args()





def imputation_pretrain(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")

    model = DSPFlow(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
        vqvae_ckpt=args.vqvae_ckpt
    )

    train_set = ImputationNormalECGDataset(
        raw_data_path=args.raw_data_path_train,
        indices_path=args.indices_path_train,
        seq_len=args.seq_len,
        one_channel=args.one_channel,
        min_infill_length=args.min_infill_length,
        max_infill_length=args.max_infill_length,
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
    trainer = DSPFlowTrainer(
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




def no_context_pretrain(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")

    model = DSPFlow(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
        vqvae_ckpt=args.vqvae_ckpt
    )

    train_set = NoContextNormalECGDataset(
        raw_data_path=args.raw_data_path_train,
        indices_path=args.indices_path_train,
        seq_len=args.seq_len,
        one_channel=args.one_channel,
        min_infill_length=args.min_infill_length,
        max_infill_length=args.max_infill_length,
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
    trainer = DSPFlowTrainer(
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

    trainer.no_context_train(config=vars(args))




def no_context_sample(args):
    model = DSPFlow(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
        vqvae_ckpt=args.vqvae_ckpt
    )
    model.load_state_dict(torch.load(f"{args.ckpt_dir}/ckpt.pth"))
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    model.eval()

    train_set = NoContextNormalECGDataset(
        raw_data_path=args.raw_data_path_train,
        indices_path=args.indices_path_train,
        seq_len=args.seq_len,
        one_channel=args.one_channel,
        min_infill_length=args.min_infill_length,
        max_infill_length=args.max_infill_length,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, drop_last=True,
        collate_fn=dict_collate_fn,
    )

    for batch in train_loader:
        signals = batch['signals'].to(device=device, dtype=torch.float32) #(batch_size, seq_len, ts_dim)
        attn_mask = batch['attn_mask'].to(device=device, dtype=torch.bool) # (batch_size, seq_len)

        repeat_signals = signals.unsqueeze(-1) #(batch_size, 1, seq_len, ts_dim)
        repeat_attn_mask = attn_mask.unsqueeze(-1) #(batch_size, 1, seq_len)

        repeat_signals = repeat_signals.repeat(1, 10, 1, 1) #(batch_size, 10, seq_len, ts_dim)
        repeat_attn_mask = repeat_attn_mask.repeat(1, 10, 1, 1) #(batch_size, 10, seq_len)

        repeat_signals = repeat_signals.reshape(-1, args.max_infill_length, args.feature_size) #(batch_size*10, seq_len, ts_dim)
        repeat_attn_mask = repeat_attn_mask.reshape(-1, args.max_infill_length) # (batch_size*10, seq_len)

        with torch.no_grad():
            samples = model.no_context_generation(repeat_signals, repeat_attn_mask)
        samples = samples.reshape(args.batch_size, -1, args.max_infill_length, args.feature_size)

        result = {
            'reals': signals,
            'samples': samples,
            'attn_mask': attn_mask,
        }
        torch.save(result, f"{args.ckpt_dir}/no_context_samples.pth")
        break




def main():
    args = get_args()
    if args.what_to_do == "imputation_pretrain":
        imputation_pretrain(args)
    elif args.what_to_do == "no_context_pretrain":
        no_context_pretrain(args)
    elif args.what_to_do == "no_context_sample":
        no_context_sample(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
