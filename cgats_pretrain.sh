#python CGATSPretrainPipeline.py \
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
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_100npz/train/normal.jsonl" \
#  --indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_100npz/validation/normal.jsonl" \
#  \
#  --lr 1e-5 \
#  --batch_size 128 \
#  --epochs 1000 \
#  --grad_clip_norm 1.0 \
#  \
#  --wandb_project cgats_pretrain \
#  --wandb_run cgats_pretrain \
#  \
#  --ckpt_dir ../TSA-ckpts/cgats_normal_pretrain_ckpt \
#  --gpu_id 0


#python CGATSPretrainPipeline.py \
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
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
#  --indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/normal.jsonl" \
#  \
#  --lr 1e-5 \
#  --batch_size 128 \
#  --epochs 1000 \
#  --grad_clip_norm 1.0 \
#  \
#  --wandb_project cgats_pretrain \
#  --wandb_run cgats_pretrain_mitdb106 \
#  \
#  --ckpt_dir ../TSA-ckpts/cgats_mitdb106_pretrain_ckpt \
#  --gpu_id 0



python CGATSPretrainEvaluate.py \
  --seq_len 800 \
  --feature_size 2 \
  \
  --latent_dim 64 \
  --trend_poly 3 \
  --kl_wt 1e-3 \
  --hidden_layer_sizes "[50,100,200]" \
  --custom_seas "[[10,80],[20,40],[40,20],[80,10]]" \
  \
  --max_anomaly_ratio 0.2 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
  --indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/normal.jsonl" \
  \
  --lr 1e-5 \
  --batch_size 128 \
  --epochs 1000 \
  --grad_clip_norm 1.0 \
  \
  --pretrained_ckpt "../TSA-ckpts/cgats_mitdb106_pretrain_ckpt/2025-11-30-18:19:27/ckpt.pth" \
  --gpu_id 0 \
  \
  --need_to_generate 1 \
  --generated_path "../samples_path/cgats_normal_mitdb106"
