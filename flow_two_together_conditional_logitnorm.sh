


#export hucfg_attention_rope_use=1
#export hucfg_t_sampling=logitnorm
#python FlowTrainTogether.py \
#  --what_to_do "conditional_training" \
#  \
#  --seq_len 800 \
#  --feature_size 2 \
#  \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  \
#  --dataset_name "ECG" \
#  --max_anomaly_length 160 \
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
#  --indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
#  \
#  --lr 1e-3 \
#  --batch_size 64 \
#  --max_epochs 1000 \
#  --grad_clip_norm 1.0 \
#  --early_stop "true" \
#  --patience 50 \
#  \
#  --wandb_project "flow_imputation" \
#  --wandb_run "mitdb106v_logit_norm" \
#  \
#  --ckpt_dir "../TSA-ckpts/flow_imputation_logit_norm" \
#  --gpu_id 1


python FlowTwoTogether.py \
  --what_to_do "conditional_evaluate" \
  \
  --seq_len 800 \
  --feature_size 2 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  \
  --dataset_name "ECG" \
  --max_anomaly_length 160 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
  --indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
  \
  --lr 1e-3 \
  --batch_size 64 \
  --max_epochs 1000 \
  --grad_clip_norm 1.0 \
  --early_stop "true" \
  --patience 50 \
  \
  --wandb_project "flow_imputation" \
  --wandb_run "mitdb106v_logit_norm" \
  \
  --ckpt_dir "../TSA-ckpts/flow_imputation_logit_norm" \
  --gpu_id 1 \
  \
  --cond_eval_model_ckpt "../TSA-ckpts/flow_imputation_logit_norm/ema_ckpt.pth" \
  --generated_path "../samples_path/flow_imputation_logit_norm"