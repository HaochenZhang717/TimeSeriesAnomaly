#python CGATSFinetunePipeline.py \
#  --seq_len 800 \
#  --feature_size 2 \
#  \
#  --latent_dim 64 \
#  --trend_poly 3 \
#  --kl_wt 1e-3 \
#  --hidden_layer_sizes "[50,100,200]" \
#  --custom_seas "[[10,80],[20,40],[40,20],[80,10]]" \
#  \
#  --max_anomaly_ratio 0.2 \
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/100.npz" \
#  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/100.npz" \
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_100npz/train/A.jsonl" \
#  --indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_100npz/validation/A.jsonl" \
#  \
#  --lr 1e-5 \
#  --batch_size 128 \
#  --epochs 100000 \
#  --grad_clip_norm 1.0 \
#  \
#  --wandb_project cgats_finetune \
#  --wandb_run cgats_fientune \
#  \
#  --ckpt_dir "../TSA-ckpts/cgats_finetune_ckpt" \
#  --pretrained_ckpt "../TSA-ckpts/cgats_normal_pretrain_ckpt/2025-11-30-00:35:44/ckpt.pth" \
#  --gpu_id 0


#python CGATSFinetunePipeline.py \
#  --seq_len 800 \
#  --feature_size 2 \
#  \
#  --latent_dim 64 \
#  --trend_poly 3 \
#  --kl_wt 1e-3 \
#  --hidden_layer_sizes "[50,100,200]" \
#  --custom_seas "[[10,80],[20,40],[40,20],[80,10]]" \
#  \
#  --max_anomaly_ratio 0.2 \
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
#  --indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
#  \
#  --lr 1e-5 \
#  --batch_size 128 \
#  --epochs 10000 \
#  --grad_clip_norm 1.0 \
#  \
#  --wandb_project cgats_finetune \
#  --wandb_run cgats_fientune_mitdb106 \
#  \
#  --ckpt_dir "../TSA-ckpts/cgats_finetune_mitdb106_ckpt" \
#  --pretrained_ckpt "../TSA-ckpts/cgats_mitdb106_pretrain_ckpt/2025-11-30-18:19:27/ckpt.pth" \
#  --gpu_id 0


python CGATSFinetuneEvaluate.py \
  --seq_len 800 \
  --feature_size 2 \
  \
  --latent_dim 64 \
  --trend_poly 3 \
  --kl_wt 1e-3 \
  --hidden_layer_sizes "[50,100,200]" \
  --custom_seas "[[10,80],[20,40],[40,20],[80,10]]" \
  \
  --max_anomaly_length 160 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --normal_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
  --normal_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/normal.jsonl" \
  --anomaly_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
  --anomaly_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
  \
  --batch_size 128 \
  \
  --model_ckpt "../TSA-ckpts/cgats_finetune_mitdb106_ckpt/2025-12-01-05:49:18/ckpt.pth" \
  --gpu_id 0 \
  \
  --eval_train_size 10000 \
  --num_samples 50000 \
  --generated_path "../samples_path/cgats_anomaly_mitdb106"