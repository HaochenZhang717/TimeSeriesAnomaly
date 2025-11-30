python FlowEvaluate.py \
  --seq_len 800 \
  --feature_size 2 \
  \
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
  --batch_size 64 \
  \
  --model_ckpt ../TSA-ckpts/flow_normal_finetune_ckpt/2025-11-29-06:31:47/ckpt.pth \
  --gpu_id 1 \
  --num_samples 30524 \
  --generated_path "../samples_path/flow/2025-11-29-06:31:47" \







