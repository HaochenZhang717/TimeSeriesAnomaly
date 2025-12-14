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
#  --gpu_id 2
#


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
#  --generated_file "generated_anomaly_on_real_normal.pt" \
#  --normal_data_path "none" \
#  --cond_num_samples -1 \
#  \
#  --uncond_eval_model_ckpt "none" \
#  --uncond_num_samples -1 \
#  \
#  --eval_train_size 10000 \
#  --gpu_id 2


#export hucfg_t_sampling=logitnorm
#python VarFlow.py \
#  --what_to_do "conditional_training" \
#  \
#  --seq_len 1800 \
#  --feature_size 1 \
#  --one_channel 1 \
#  \
#  --model_name "vrf_v2" \
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
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/213.npz" \
#  --raw_data_paths_val "none" \
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_213npz/train/mixed.jsonl" \
#  --indices_paths_val "none" \
#  --limited_data_size 1000000 \
#  \
#  --lr 1e-4 \
#  --batch_size 64 \
#  --max_epochs 1000 \
#  --grad_clip_norm 1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "true" \
#  --patience 50 \
#  \
#  --wandb_project "VRF_conditional" \
#  --wandb_run "v2-213npz-stochastic-latent" \
#  \
#  --ckpt_dir "../TSA-ckpts/VRF_v2/mitdb1800_213mixed/conditional_ckpt" \
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
#  --gpu_id 2





#export hucfg_t_sampling=logitnorm
#python VarFlow.py \
#  --what_to_do "conditional_sample_on_real_anomaly" \
#  \
#  --seq_len 1800 \
#  --feature_size 1 \
#  --one_channel 1 \
#  \
#  --model_name "vrf_v2" \
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
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/213.npz" \
#  --raw_data_paths_val "none" \
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_213npz/train/A.jsonl" \
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
#  --cond_eval_model_ckpt "../TSA-ckpts/VRF_v2/mitdb1800_213mixed/conditional_ckpt/ema_ckpt.pth" \
#  --generated_path "../samples_path/vrf_v2/mitdb1800_213A" \
#  --generated_file "anomaly_cond_on_anomaly" \
#  --normal_data_path "none" \
#  --cond_num_samples 10 \
#  \
#  --uncond_eval_model_ckpt "none" \
#  --uncond_num_samples -1 \
#  \
#  --eval_train_size -1 \
#  \
#  --gpu_id 2
#
#
#export hucfg_t_sampling=logitnorm
#python VarFlow.py \
#  --what_to_do "conditional_sample_on_real_anomaly" \
#  \
#  --seq_len 1800 \
#  --feature_size 1 \
#  --one_channel 1 \
#  \
#  --model_name "vrf_v2" \
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
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/213.npz" \
#  --raw_data_paths_val "none" \
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_213npz/train/V.jsonl" \
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
#  --cond_eval_model_ckpt "../TSA-ckpts/VRF_v2/mitdb1800_213mixed/conditional_ckpt/ema_ckpt.pth" \
#  --generated_path "../samples_path/vrf_v2/mitdb1800_213V" \
#  --generated_file "anomaly_cond_on_anomaly" \
#  --normal_data_path "none" \
#  --cond_num_samples 256 \
#  \
#  --uncond_eval_model_ckpt "none" \
#  --uncond_num_samples -1 \
#  \
#  --eval_train_size -1 \
#  \
#  --gpu_id 2


##########################################
export hucfg_t_sampling=logitnorm
python VarFlow.py \
  --what_to_do "conditional_training" \
  \
  --seq_len 1800 \
  --feature_size 1 \
  --one_channel 1 \
  \
  --model_name "vrf_v2" \
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
  --wandb_run "v2-decoder-rand-213npz-stochastic-latent" \
  \
  --ckpt_dir "../TSA-ckpts/VRF_v2_decoder_rand/mitdb1800_213mixed/conditional_ckpt" \
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
  --gpu_id 2





export hucfg_t_sampling=logitnorm
python VarFlow.py \
  --what_to_do "conditional_sample_on_real_anomaly" \
  \
  --seq_len 1800 \
  --feature_size 1 \
  --one_channel 1 \
  \
  --model_name "vrf_v2" \
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
  --cond_eval_model_ckpt "../TSA-ckpts/VRF_v2_decoder_rand/mitdb1800_213mixed/conditional_ckpt/ema_ckpt.pth" \
  --generated_path "../samples_path/vrf_v2_decoder_rand/mitdb1800_213A" \
  --generated_file "anomaly_cond_on_anomaly" \
  --normal_data_path "none" \
  --cond_num_samples 10 \
  \
  --uncond_eval_model_ckpt "none" \
  --uncond_num_samples -1 \
  \
  --eval_train_size -1 \
  \
  --gpu_id 2


export hucfg_t_sampling=logitnorm
python VarFlow.py \
  --what_to_do "conditional_sample_on_real_anomaly" \
  \
  --seq_len 1800 \
  --feature_size 1 \
  --one_channel 1 \
  \
  --model_name "vrf_v2" \
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
  --cond_eval_model_ckpt "../TSA-ckpts/VRF_v2_decoder_rand/mitdb1800_213mixed/conditional_ckpt/ema_ckpt.pth" \
  --generated_path "../samples_path/vrf_v2_decoder_rand/mitdb1800_213V" \
  --generated_file "anomaly_cond_on_anomaly" \
  --normal_data_path "none" \
  --cond_num_samples 256 \
  \
  --uncond_eval_model_ckpt "none" \
  --uncond_num_samples -1 \
  \
  --eval_train_size -1 \
  \
  --gpu_id 2
