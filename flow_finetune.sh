python FlowFinetunePipeline.py \
  --seq_len 800 \
  --feature_size 2 \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  \
  --max_anomaly_ratio 0.2 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/100.npz" \
  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/100.npz" \
  --normal_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_100npz/train/normal.jsonl" \
  --normal_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_100npz/validation/normal.jsonl" \
  --anomaly_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_100npz/train/A.jsonl" \
  --anomaly_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_100npz/validation/A.jsonl" \
  \
  --lr 1e-5 \
  --batch_size 64 \
  --max_iters 1000000 \
  --grad_clip_norm 1.0 \
  \
  --wandb_project flow_normal_finetune \
  --wandb_run flow_normal_finetune \
  \
  --ckpt_dir ../TSA-ckpts/flow_normal_finetune_ckpt \
  --pretrained_ckpt "/root/tianyi/TSA-ckpts/flow_normal_pretrain_ckpt/2025-11-29-00:08:47/ckpt.pth" \
  --gpu_id 1
