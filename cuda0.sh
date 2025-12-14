#export hucfg_t_sampling=logitnorm
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
#  --ckpt_dir "../TSA-ckpts/flow_two_together_logit_normal/mitdb1800_200npz/conditional_ckpt" \
#  \
#  --cond_eval_model_ckpt "none" \
#  --generated_path "none" \
#  --normal_data_path "none" \
#  --generated_file "none" \
#  --cond_num_samples -1 \
#  \
#  --uncond_eval_model_ckpt "none" \
#  --uncond_num_samples -1 \
#  \
#  --eval_train_size -1 \
#  \
#  --gpu_id 0


#export hucfg_t_sampling=logitnorm
#python FlowTwoTogether.py \
#  --what_to_do "unconditional_sample" \
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
#  --raw_data_paths_train "none" \
#  --raw_data_paths_val "none" \
#  --indices_paths_train "none" \
#  --indices_paths_val "none" \
#  \
#  --lr 5e-4 \
#  --batch_size 64 \
#  --max_epochs -1 \
#  --grad_clip_norm -1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "none" \
#  --patience -1 \
#  \
#  --wandb_project "none" \
#  --wandb_run "none" \
#  \
#  --ckpt_dir "none" \
#  \
#  --cond_eval_model_ckpt "none" \
#  --generated_path "../samples_path/flow_two_together_logit_normal/mitdb1800" \
#  --normal_data_path "none" \
#  \
#  --uncond_eval_model_ckpt "../TSA-ckpts/flow_two_together_logit_normal/mitdb1800/uncondition_ckpt/ema_ckpt.pth" \
#  --uncond_num_samples 50000 \
#  \
#  --eval_train_size -1 \
#  \
#  --gpu_id 0


#export hucfg_t_sampling=logitnorm
#python FlowTwoTogether.py \
#  --what_to_do "unconditional_evaluate" \
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
#  --lr 5e-4 \
#  --batch_size 64 \
#  --max_epochs -1 \
#  --grad_clip_norm -1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "none" \
#  --patience -1 \
#  \
#  --wandb_project "none" \
#  --wandb_run "none" \
#  \
#  --ckpt_dir "none" \
#  \
#  --cond_eval_model_ckpt "none" \
#  --generated_path "none" \
#  --normal_data_path "../samples_path/flow_two_together_logit_normal/mitdb1800/generated_normal.pt" \
#  \
#  --uncond_eval_model_ckpt "../TSA-ckpts/flow_two_together_logit_normal/mitdb1800/uncondition_ckpt/ema_ckpt.pth" \
#  --uncond_num_samples 1000 \
#  \
#  --eval_train_size -1 \
#  \
#  --gpu_id 0


#export hucfg_t_sampling=logitnorm
#python FlowTwoTogether.py \
#  --what_to_do "conditional_sample_on_real_anomaly" \
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
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/200.npz" \
#  --raw_data_paths_val "none" \
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_200npz/val/V.jsonl" \
#  --indices_paths_val "none" \
#  \
#  --lr 5e-4 \
#  --batch_size 64 \
#  --max_epochs -1 \
#  --grad_clip_norm -1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "none" \
#  --patience -1 \
#  \
#  --wandb_project "none" \
#  --wandb_run "none" \
#  \
#  --ckpt_dir "none" \
#  \
#  --cond_eval_model_ckpt "../TSA-ckpts/flow_two_together_logit_normal/mitdb1800_200npz/conditional_ckpt/ema_ckpt.pth" \
#  --generated_path "../samples_path/flow_two_together_logit_normal/mitdb1800_200npz" \
#  --generated_file "anomaly_cond_on_anomaly" \
#  --normal_data_path "none" \
#  --cond_num_samples 10000 \
#  \
#  --uncond_eval_model_ckpt "none" \
#  --uncond_num_samples -1 \
#  \
#  --eval_train_size -1 \
#  \
#  --gpu_id 0

#export hucfg_t_sampling=logitnorm
#python FlowTwoTogether.py \
#  --what_to_do "conditional_sample_on_real_normal" \
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
#  --min_anomaly_length 160 \
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/200.npz" \
#  --raw_data_paths_val "none" \
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_200npz/train/normal.jsonl" \
#  --indices_paths_val "none" \
#  \
#  --lr 5e-4 \
#  --batch_size 64 \
#  --max_epochs -1 \
#  --grad_clip_norm -1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "none" \
#  --patience -1 \
#  \
#  --wandb_project "none" \
#  --wandb_run "none" \
#  \
#  --ckpt_dir "none" \
#  \
#  --cond_eval_model_ckpt "../TSA-ckpts/flow_two_together_logit_normal/mitdb1800_200npz/conditional_ckpt/ema_ckpt.pth" \
#  --generated_path "../samples_path/flow_two_together_logit_normal/mitdb1800_200npz" \
#  --generated_file "anomaly_cond_on_normal" \
#  --normal_data_path "none" \
#  --cond_num_samples 10000 \
#  \
#  --uncond_eval_model_ckpt "none" \
#  --uncond_num_samples -1 \
#  \
#  --eval_train_size -1 \
#  \
#  --gpu_id 0

#export hucfg_t_sampling=logitnorm
#python FlowTwoTogether.py \
#  --what_to_do "conditional_sample_on_fake" \
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
#  --raw_data_paths_train "none" \
#  --raw_data_paths_val "none" \
#  --indices_paths_train "none" \
#  --indices_paths_val "none" \
#  \
#  --lr 5e-4 \
#  --batch_size 64 \
#  --max_epochs -1 \
#  --grad_clip_norm -1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "none" \
#  --patience -1 \
#  \
#  --wandb_project "none" \
#  --wandb_run "none" \
#  \
#  --ckpt_dir "none" \
#  \
#  --cond_eval_model_ckpt "../TSA-ckpts/flow_two_together_logit_normal/mitdb1800/conditional_ckpt/ema_ckpt.pth" \
#  --generated_path "../samples_path/flow_two_together_logit_normal/mitdb1800" \
#  --normal_data_path "../samples_path/flow_two_together_logit_normal/mitdb1800/generated_normal.pt" \
#  \
#  --uncond_eval_model_ckpt "none" \
#  --uncond_num_samples -1 \
#  \
#  --eval_train_size -1 \
#  \
#  --gpu_id 0




#python FlowTwoTogether.py \
#  --what_to_do "anomaly_evaluate" \
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
#  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
#  --indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
#  \
#  --lr 5e-4 \
#  --batch_size 64 \
#  --max_epochs 1000 \
#  --grad_clip_norm 1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "true" \
#  --patience 50 \
#  \
#  --wandb_project "none" \
#  --wandb_run "none" \
#  \
#  --ckpt_dir "none" \
#  \
#  --cond_eval_model_ckpt "none" \
#  --generated_path "../samples_path/flow_two_together_logit_normal/mitdb1800" \
#  --generated_file "generated_anomaly_on_real_normal.pt" \
#  --normal_data_path "none" \
#  --cond_num_samples -1 \
#  \
#  --uncond_eval_model_ckpt "none" \
#  --uncond_num_samples -1 \
#  \
#  --eval_train_size 10000 \
#  --gpu_id 0


#python FlowTwoTogether.py \
#  --what_to_do "anomaly_evaluate" \
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
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/200.npz" \
#  --raw_data_paths_val "none" \
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_200npz/train/V.jsonl" \
#  --indices_paths_val "none" \
#  \
#  --lr 5e-4 \
#  --batch_size 64 \
#  --max_epochs 1000 \
#  --grad_clip_norm 1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "true" \
#  --patience 50 \
#  \
#  --wandb_project "none" \
#  --wandb_run "none" \
#  \
#  --ckpt_dir "none" \
#  \
#  --cond_eval_model_ckpt "none" \
#  --generated_path "../samples_path/flow_two_together_logit_normal/mitdb1800_200npz" \
#  --generated_file "generated_anomaly_on_real_anomaly.pt" \
#  --normal_data_path "none" \
#  --cond_num_samples -1 \
#  \
#  --uncond_eval_model_ckpt "none" \
#  --uncond_num_samples -1 \
#  \
#  --eval_train_size 10000 \
#  --gpu_id 0



export hucfg_t_sampling=logitnorm
python VarFlow.py \
  --what_to_do "conditional_training" \
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
  --ve_channels "[16,32,64]" \
  --ve_kernel_size 3 \
  --ve_pool_kernel 4 \
  --ve_pool_stride 4 \
  --ve_z_dim 16 \
  \
  --kl_beta 1e-3 \
  \
  --dataset_name "ECG" \
  --max_anomaly_length 629 \
  --min_anomaly_length 74 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/213.npz" \
  --raw_data_paths_val "none" \
  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_213npz/train/mixed.jsonl" \
  --indices_paths_val "none" \
  --limited_data_size 1000000 \
  \
  --lr 1e-4 \
  --batch_size 64 \
  --max_epochs 1000 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 50 \
  \
  --wandb_project "VRF_conditional" \
  --wandb_run "213npz-stochastic-latent" \
  \
  --ckpt_dir "../TSA-ckpts/VRF/mitdb1800_213mixed/conditional_ckpt" \
  \
  --cond_eval_model_ckpt "none" \
  --generated_path "none" \
  --normal_data_path "none" \
  --generated_file "none" \
  --cond_num_samples -1 \
  \
  --uncond_eval_model_ckpt "none" \
  --uncond_num_samples -1 \
  \
  --eval_train_size -1 \
  \
  --gpu_id 0



#export hucfg_t_sampling=logitnorm
#python VarFlow.py \
#  --what_to_do "conditional_sample_on_real_anomaly" \
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
#  --ve_channels "[16,32,64]" \
#  --ve_kernel_size 3 \
#  --ve_pool_kernel 4 \
#  --ve_pool_stride 4 \
#  --ve_z_dim 16 \
#  \
#  --kl_beta 1e-3 \
#  \
#  --dataset_name "ECG" \
#  --max_anomaly_length 629 \
#  --min_anomaly_length 74 \
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --raw_data_paths_val "none" \
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
#  --indices_paths_val "none" \
#  --limited_data_size 1000000 \
#  \
#  --lr 5e-4 \
#  --batch_size 64 \
#  --max_epochs -1 \
#  --grad_clip_norm -1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "none" \
#  --patience -1 \
#  \
#  --wandb_project "none" \
#  --wandb_run "none" \
#  \
#  --ckpt_dir "none" \
#  \
#  --cond_eval_model_ckpt "../TSA-ckpts/VRF/mitdb1800_106npz/conditional_ckpt/ema_ckpt.pth" \
#  --generated_path "../samples_path/VRF/mitdb1800_106npz" \
#  --generated_file "anomaly_cond_on_anomaly" \
#  --normal_data_path "none" \
#  --cond_num_samples 10 \
#  \
#  --uncond_eval_model_ckpt "none" \
#  --uncond_num_samples -1 \
#  \
#  --eval_train_size -1 \
#  \
#  --gpu_id 0


export hucfg_t_sampling=logitnorm
python VarFlow.py \
  --what_to_do "conditional_sample_on_real_anomaly" \
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
  --ve_channels "[16,32,64]" \
  --ve_kernel_size 3 \
  --ve_pool_kernel 4 \
  --ve_pool_stride 4 \
  --ve_z_dim 16 \
  \
  --kl_beta 1e-3 \
  \
  --dataset_name "ECG" \
  --max_anomaly_length 629 \
  --min_anomaly_length 74 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/213.npz" \
  --raw_data_paths_val "none" \
  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_213npz/train/A.jsonl" \
  --indices_paths_val "none" \
  --limited_data_size 1000000 \
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
  --cond_eval_model_ckpt "../TSA-ckpts/VRF/mitdb1800_213mixed/conditional_ckpt/ema_ckpt.pth" \
  --generated_path "../samples_path/VRF/mitdb1800_213A" \
  --generated_file "anomaly_cond_on_anomaly" \
  --normal_data_path "none" \
  --cond_num_samples 10 \
  \
  --uncond_eval_model_ckpt "none" \
  --uncond_num_samples -1 \
  \
  --eval_train_size -1 \
  \
  --gpu_id 0


export hucfg_t_sampling=logitnorm
python VarFlow.py \
  --what_to_do "conditional_sample_on_real_anomaly" \
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
  --ve_channels "[16,32,64]" \
  --ve_kernel_size 3 \
  --ve_pool_kernel 4 \
  --ve_pool_stride 4 \
  --ve_z_dim 16 \
  \
  --kl_beta 1e-3 \
  \
  --dataset_name "ECG" \
  --max_anomaly_length 629 \
  --min_anomaly_length 74 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/213.npz" \
  --raw_data_paths_val "none" \
  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_213npz/train/V.jsonl" \
  --indices_paths_val "none" \
  --limited_data_size 1000000 \
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
  --cond_eval_model_ckpt "../TSA-ckpts/VRF/mitdb1800_213mixed/conditional_ckpt/ema_ckpt.pth" \
  --generated_path "../samples_path/VRF/mitdb1800_213V" \
  --generated_file "anomaly_cond_on_anomaly" \
  --normal_data_path "none" \
  --cond_num_samples 10 \
  \
  --uncond_eval_model_ckpt "none" \
  --uncond_num_samples -1 \
  \
  --eval_train_size -1 \
  \
  --gpu_id 0
