##########################################

#python rain.py \
#  --what_to_do "autoencoder_train" \
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
#  --wandb_project "rain_VAE" \
#  --wandb_run "106npz" \
#  \
#  --ckpt_dir "../TSA-ckpts/rain/mitdb1800_106/autoencoder" \
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
#  --gpu_id 5


#python rain.py \
#  --what_to_do "autoencoder_eval" \
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
#  --wandb_project "rain_VAE" \
#  --wandb_run "106npz" \
#  \
#  --ckpt_dir "../TSA-ckpts/rain/mitdb1800_106/autoencoder" \
#  \
#  --cond_eval_model_ckpt "none" \
#  --generated_path "../samples_path/rain/" \
#  --normal_data_path "none" \
#  --generated_file "none" \
#  --cond_num_samples -1 \
#  \
#  --autoencoder_ckpt "../TSA-ckpts/rain/mitdb1800_106/autoencoder/ckpt.pth" \
#  \
#  --eval_train_size -1 \
#  \
#  --gpu_id 5



#python rain.py \
#  --what_to_do "flow_training" \
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
#  --wandb_project "rain_flow" \
#  --wandb_run "106npz" \
#  \
#  --ckpt_dir "../TSA-ckpts/rain/mitdb1800_106/flow" \
#  \
#  --cond_eval_model_ckpt "none" \
#  --generated_path "none" \
#  --normal_data_path "none" \
#  --generated_file "none" \
#  --cond_num_samples -1 \
#  \
#  --autoencoder_ckpt "../TSA-ckpts/rain/mitdb1800_106/autoencoder/ckpt.pth" \
#  \
#  --eval_train_size -1 \
#  \
#  --gpu_id 5




python rain.py \
  --what_to_do "flow_sample" \
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
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --raw_data_paths_val "none" \
  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
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
  --wandb_project "rain_flow" \
  --wandb_run "106npz" \
  \
  --ckpt_dir "none" \
  \
  --cond_eval_model_ckpt "none" \
  --generated_path "none" \
  --normal_data_path "none" \
  --generated_file "none" \
  --generated_path "../samples_path/rain/" \
  --cond_num_samples -1 \
  \
  --autoencoder_ckpt "../TSA-ckpts/rain/mitdb1800_106/autoencoder/ckpt.pth" \
  --flow_ckpt "../TSA-ckpts/rain/mitdb1800_106/flow/ema_ckpt.pth" \
  \
  --eval_train_size -1 \
  \
  --gpu_id 5