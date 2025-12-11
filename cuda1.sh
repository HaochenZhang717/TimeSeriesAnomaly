python CGATSPretrainPipeline.py \
  --seq_len 1800 \
  --feature_size 1 \
  --one_channel 1 \
  \
  --latent_dim 64 \
  --trend_poly 3 \
  --kl_wt 1e-3 \
  --hidden_layer_sizes "[50,100,200]" \
  --custom_seas "[[10,80],[20,40],[40,20],[80,10]]" \
  \
  --dataset_name "ECG" \
  --max_anomaly_length 629 \
  --min_anomaly_length 74 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/200.npz" \
  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_200npz/train/normal.jsonl" \
  \
  --lr 1e-5 \
  --batch_size 64 \
  --epochs 1000 \
  --grad_clip_norm 1.0 \
  --early_stop "true" \
  \
  --wandb_project cgats_pretrain \
  --wandb_run cgats_pretrain_mitdb1800_200npz \
  \
  --ckpt_dir "../TSA-ckpts/cgats/mitdb1800_200npz/pretrain_ckpt" \
  --gpu_id 1




python CGATSFinetunePipeline.py \
  --what_to_do "finetune" \
  \
  --seq_len 1800 \
  --feature_size 1 \
  --one_channel 1 \
  \
  --latent_dim 64 \
  --trend_poly 3 \
  --kl_wt 1e-3 \
  --hidden_layer_sizes "[50,100,200]" \
  --custom_seas "[[10,80],[20,40],[40,20],[80,10]]" \
  \
  --dataset_name "ECG" \
  --max_anomaly_length 629 \
  --min_anomaly_length 74 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/200.npz" \
  --normal_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_200npz/train/normal.jsonl" \
  --anomaly_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_200npz/train/V.jsonl" \
  \
  --lr 1e-5 \
  --batch_size 64 \
  --epochs 10000 \
  --grad_clip_norm 1.0 \
  --early_stop "true" \
  \
  --wandb_project cgats_finetune \
  --wandb_run cgats_fientune_mitdb1800_200npz \
  \
  --ckpt_dir "../TSA-ckpts/cgats/mitdb1800_200npz/finetune_ckpt" \
  --pretrained_ckpt "../TSA-ckpts/cgats/mitdb1800_200npz/pretrain_ckpt/ckpt.pth" \
  \
  --tuned_ckpt "none" \
  --num_samples -1 \
  --eval_train_size -1 \
  --generated_path "none" \
  \
  --gpu_id 1



python CGATSFinetunePipeline.py \
  --what_to_do "sample_anomaly" \
  \
  --seq_len 1800 \
  --feature_size 1 \
  --one_channel 1 \
  \
  --latent_dim 64 \
  --trend_poly 3 \
  --kl_wt 1e-3 \
  --hidden_layer_sizes "[50,100,200]" \
  --custom_seas "[[10,80],[20,40],[40,20],[80,10]]" \
  \
  --dataset_name "ECG" \
  --max_anomaly_length 629 \
  --min_anomaly_length 74 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/200.npz" \
  --normal_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_200npz/train/normal.jsonl" \
  --anomaly_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_200npz/train/V.jsonl" \
  \
  --lr 1e-5 \
  --batch_size 64 \
  --epochs 10000 \
  --grad_clip_norm 1.0 \
  --early_stop "true" \
  \
  --wandb_project "none" \
  --wandb_run "none" \
  \
  --ckpt_dir "none" \
  --pretrained_ckpt "none" \
  \
  --tuned_ckpt "../TSA-ckpts/cgats/mitdb1800_200npz/finetune_ckpt/ckpt.pth" \
  --num_samples 50000 \
  --eval_train_size 10000 \
  --generated_path "../samples_path/cgats/mitdb1800_200npz" \
  \
  --gpu_id 1