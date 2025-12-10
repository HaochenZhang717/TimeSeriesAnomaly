#!/bin/bash

export hucfg_t_sampling=logitnorm

# ============================
# Define model size presets
# ============================


MODELS=(
  "VL     384 8 10 10"
#  "LARGE  256 8 8 8"
#  "MEDIUM 128 8 6 6"
)

# ============================
# Loop over model sizes
# ============================

for m in "${MODELS[@]}"; do
  set -- $m
  NAME=$1
  DMODEL=$2
  NHEAD=$3
  NLAY_ENC=$4
  NLAY_DEC=$5

  echo "========================================"
  echo " Running Model: $NAME"
  echo " d_model=$DMODEL  n_heads=$NHEAD"
  echo " enc_layers=$NLAY_ENC  dec_layers=$NLAY_DEC"
  echo "========================================"

  # ----------------------------------------------------
  # 1) Unconditional Training
  # ----------------------------------------------------
  python FlowTwoTogether.py \
    --what_to_do "unconditional_training" \
    \
    --seq_len 1800 \
    --feature_size 1 \
    --one_channel 1 \
    \
    --n_layer_enc $NLAY_ENC \
    --n_layer_dec $NLAY_DEC \
    --d_model $DMODEL \
    --n_heads $NHEAD \
    \
    --dataset_name "ECG" \
    --max_anomaly_length 629 \
    --min_anomaly_length 74 \
    --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
    --raw_data_paths_val "none" \
    --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
    --indices_paths_val "none" \
    \
    --lr 1e-4 \
    --batch_size 64 \
    --max_epochs 1000 \
    --grad_clip_norm 1.0 \
    --early_stop "true" \
    --patience 50 \
    \
    --wandb_project "flow_unconditional" \
    --wandb_run "${NAME}_mitdb1800_unconditional" \
    \
    --ckpt_dir "../TSA-ckpts/flow_two_together_logit_normal/${NAME}/uncondition_ckpt" \
    \
    --cond_eval_model_ckpt "none" \
    --generated_path "none" \
    --normal_data_path "none" \
    \
    --uncond_eval_model_ckpt "none" \
    --uncond_num_samples -1 \
    \
    --eval_train_size -1 \
    --gpu_id 2


  # ----------------------------------------------------
  # 2) Conditional Training
  # ----------------------------------------------------
  python FlowTwoTogether.py \
    --what_to_do "conditional_training" \
    \
    --seq_len 1800 \
    --feature_size 1 \
    --one_channel 1 \
    \
    --n_layer_enc $NLAY_ENC \
    --n_layer_dec $NLAY_DEC \
    --d_model $DMODEL \
    --n_heads $NHEAD \
    \
    --dataset_name "ECG" \
    --max_anomaly_length 629 \
    --min_anomaly_length 74 \
    --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
    --raw_data_paths_val "none" \
    --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
    --indices_paths_val "none" \
    \
    --lr 1e-4 \
    --batch_size 64 \
    --max_epochs 1000 \
    --grad_clip_norm 1.0 \
    --early_stop "true" \
    --patience 50 \
    \
    --wandb_project "flow_conditional" \
    --wandb_run "${NAME}_mitdb1800_conditional" \
    \
    --ckpt_dir "../TSA-ckpts/flow_two_together_logit_normal/${NAME}/conditional_ckpt" \
    \
    --cond_eval_model_ckpt "none" \
    --generated_path "none" \
    --normal_data_path "none" \
    \
    --uncond_eval_model_ckpt "none" \
    --uncond_num_samples -1 \
    \
    --eval_train_size -1 \
    --gpu_id 2

done

echo "======================"
echo "ALL TRAINING FINISHED"
echo "======================"