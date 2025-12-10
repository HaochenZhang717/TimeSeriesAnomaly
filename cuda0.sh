export hucfg_t_sampling=logitnorm
#python FlowTwoTogether.py \
#  --what_to_do "unconditional_training" \
#  \
#  --seq_len 1800 \
#  --feature_size 1 \
#  --one_channel 1 \
#  \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  \
#  --dataset_name "ECG" \
#  --max_anomaly_length 629 \
#  --min_anomaly_length 74 \
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --raw_data_paths_val "none" \
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
#  --indices_paths_val "none" \
#  \
#  --lr 1e-4 \
#  --batch_size 64 \
#  --max_epochs 1000 \
#  --grad_clip_norm 1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "true" \
#  --patience 50 \
#  \
#  --wandb_project "flow_unconditional" \
#  --wandb_run "mitdb1800_unconditional_logit_norm" \
#  \
#  --ckpt_dir "../TSA-ckpts/flow_two_together_logit_normal/mitdb1800/uncondition_ckpt" \
#  \
#  --cond_eval_model_ckpt "none" \
#  --generated_path "none" \
#  --normal_data_path "none" \
#  \
#  --uncond_eval_model_ckpt "none" \
#  --uncond_num_samples -1 \
#  \
#  --eval_train_size -1 \
#  \
#  --gpu_id 0




#export hucfg_t_sampling=logitnorm
#python FlowTwoTogether.py \
#  --what_to_do "conditional_training" \
#  \
#  --seq_len 1800 \
#  --feature_size 1 \
#  --one_channel 1 \
#  \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  \
#  --dataset_name "ECG" \
#  --max_anomaly_length 629 \
#  --min_anomaly_length 74 \
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --raw_data_paths_val "none" \
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
#  --indices_paths_val "none" \
#  \
#  --lr 1e-4 \
#  --batch_size 64 \
#  --max_epochs 1000 \
#  --grad_clip_norm 1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "true" \
#  --patience 50 \
#  \
#  --wandb_project "flow_conditional" \
#  --wandb_run "mitdb1800_conditional_logit_norm" \
#  \
#  --ckpt_dir "../TSA-ckpts/flow_two_together_logit_normal/mitdb1800/conditional_ckpt" \
#  \
#  --cond_eval_model_ckpt "none" \
#  --generated_path "none" \
#  --normal_data_path "none" \
#  \
#  --uncond_eval_model_ckpt "none" \
#  --uncond_num_samples -1 \
#  \
#  --eval_train_size -1 \
#  \
#  --gpu_id 0


#FileNotFoundError: [Errno 2] No such file or directory:
#'../TSA-ckpts/flow_two_together_logit_normal/mitdb1800/unconditional_ckpt/ema_ckpt.pth'


export hucfg_t_sampling=logitnorm
python FlowTwoTogether.py \
  --what_to_do "unconditional_sample" \
  \
  --seq_len 1800 \
  --feature_size 1 \
  --one_channel 1 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  \
  --dataset_name "ECG" \
  --max_anomaly_length 629 \
  --min_anomaly_length 74 \
  --raw_data_paths_train "none" \
  --raw_data_paths_val "none" \
  --indices_paths_train "none" \
  --indices_paths_val "none" \
  \
  --lr 5e-4 \
  --batch_size 64 \
  --max_epochs -1 \
  --grad_clip_norm -1.0 \
  --grad_accum_steps 1 \
  --early_stop "none" \
  --patience -1 \
  \
  --wandb_project "none" \
  --wandb_run "none" \
  \
  --ckpt_dir "none" \
  \
  --cond_eval_model_ckpt "none" \
  --generated_path "../samples_path/flow_two_together_logit_normal/mitdb1800" \
  --normal_data_path "none" \
  \
  --uncond_eval_model_ckpt "../TSA-ckpts/flow_two_together_logit_normal/mitdb1800/uncondition_ckpt/ema_ckpt.pth" \
  --uncond_num_samples 50000 \
  \
  --eval_train_size -1 \
  \
  --gpu_id 0



export hucfg_t_sampling=logitnorm
python FlowTwoTogether.py \
  --what_to_do "conditional_sample_on_fake" \
  \
  --seq_len 1800 \
  --feature_size 1 \
  --one_channel 1 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  \
  --dataset_name "ECG" \
  --max_anomaly_length 629 \
  --min_anomaly_length 74 \
  --raw_data_paths_train "none" \
  --raw_data_paths_val "none" \
  --indices_paths_train "none" \
  --indices_paths_val "none" \
  \
  --lr 5e-4 \
  --batch_size 64 \
  --max_epochs -1 \
  --grad_clip_norm -1.0 \
  --grad_accum_steps 1 \
  --early_stop "none" \
  --patience -1 \
  \
  --wandb_project "none" \
  --wandb_run "none" \
  \
  --ckpt_dir "none" \
  \
  --cond_eval_model_ckpt "../TSA-ckpts/flow_two_together_logit_normal/mitdb1800/conditional_ckpt/ema_ckpt.pth" \
  --generated_path "../samples_path/flow_two_together_logit_normal/mitdb1800" \
  --normal_data_path "../samples_path/flow_two_together_logit_normal/mitdb1800/generated_normal.pt" \
  \
  --uncond_eval_model_ckpt "none" \
  --uncond_num_samples -1 \
  \
  --eval_train_size -1 \
  \
  --gpu_id 0