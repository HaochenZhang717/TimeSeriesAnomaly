python FlowFinetuneEvaluateAnomaly.py \
  --seq_len 800 \
  --feature_size 2 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  \
  --max_anomaly_ratio 0.2 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --normal_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
  --normal_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/normal.jsonl" \
  --anomaly_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
  --anomaly_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
  \
  --batch_size 64 \
  \
  --model_ckpt "../TSA-ckpts/flow_mitdb106v_finetune_ckpt/2025-11-30-03:01:14/ckpt.pth" \
  --gpu_id 2 \
  \
  --need_to_generate 1 \
  --generated_path "../samples_path/flow/mitdb106v-finetuned" \

