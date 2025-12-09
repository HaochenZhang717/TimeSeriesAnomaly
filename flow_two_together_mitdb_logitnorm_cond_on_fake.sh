#export hucfg_t_sampling=logitnorm
#python FlowTwoTogether.py \
#  --what_to_do "conditional_sample_on_fake" \
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
#  --raw_data_paths_train "none" \
#  --raw_data_paths_val "none" \
#  --indices_paths_train "none" \
#  --indices_paths_val "none" \
#  \
#  --lr 5e-4 \
#  --batch_size 64 \
#  --max_epochs 1000 \
#  --grad_clip_norm 1.0 \
#  --early_stop "true" \
#  --patience 50 \
#  \
#  --wandb_project "flow_unconditional" \
#  --wandb_run "mitdb_unconditional_logit_norm" \
#  \
#  --ckpt_dir "none" \
#  \
#  --cond_eval_model_ckpt "../TSA-ckpts/flow_imputation_logit_norm/ema_ckpt.pth" \
#  --generated_path "../samples_path/flow_imputation_logit_norm/mitdblen800_a_few_to_look" \
#  --normal_data_path "../samples_path/flow_unconditional/mitdblen800_a_few_to_look/generated_normal.pt" \
#  \
#  --uncond_eval_model_ckpt "none" \
#  --uncond_num_samples -1 \
#  \
#  --eval_train_size -1 \
#  --gpu_id 1



python FlowTwoTogether.py \
  --what_to_do "conditional_sample_on_real" \
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
  --raw_data_paths_val "none" \
  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
  --indices_paths_val "none" \
  \
  --lr 1e-3 \
  --batch_size 64 \
  --max_epochs 1000 \
  --grad_clip_norm 1.0 \
  --early_stop "true" \
  --patience 50 \
  \
  --wandb_project "none" \
  --wandb_run "none" \
  \
  --ckpt_dir "none" \
  \
  --cond_eval_model_ckpt "../TSA-ckpts/flow_imputation_logit_norm/ema_ckpt.pth" \
  --generated_path "../samples_path/flow_imputation_logit_norm/mitdblen800_a_few_to_look" \
  --normal_data_path "none" \
  \
  --uncond_eval_model_ckpt "none" \
  --uncond_num_samples -1 \
  \
  --eval_train_size -1 \
  --gpu_id 1
