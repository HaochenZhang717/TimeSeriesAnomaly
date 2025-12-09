####################################################################
python FlowFinetuneEvaluateAnomaly.py \
  --seq_len 800 \
  --feature_size 2 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  --version 2 \
  \
  --dataset_name "ECG" \
  --max_anomaly_length 160 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --normal_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
  --normal_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/normal.jsonl" \
  --anomaly_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
  --anomaly_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
  \
  --batch_size 64 \
  \
  --model_ckpt "../TSA-ckpts/flow_mitdb106v_finetune_ckpt_v2_anomaly_only/2025-12-05-12:50:54/ckpt.pth" \
  --gpu_id 2 \
  \
  --need_to_generate 0 \
  --generated_path "../samples_path/flow/mitdb106v-flow-finetuned-v2-anomaly-only" \
  --num_samples -1 \
  --eval_train_size 10000

####################################################################
python FlowFinetuneEvaluateAnomaly.py \
  --seq_len 800 \
  --feature_size 2 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  --version 2 \
  \
  --dataset_name "ECG" \
  --max_anomaly_length 160 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --normal_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
  --normal_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/normal.jsonl" \
  --anomaly_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
  --anomaly_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
  \
  --batch_size 64 \
  \
  --model_ckpt "../TSA-ckpts/flow_mitdb106v_finetune_ckpt_v2_mixed_data/2025-12-05-15:09:46/ckpt.pth" \
  --gpu_id 2 \
  \
  --need_to_generate 0 \
  --generated_path "../samples_path/flow/mitdb106v-flow-finetuned-v2-mixed-data" \
  --num_samples -1 \
  --eval_train_size 10000


####################################################################
python FlowFinetuneEvaluateAnomaly.py \
  --seq_len 800 \
  --feature_size 2 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  --version 3 \
  \
  --dataset_name "ECG" \
  --max_anomaly_length 160 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --normal_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
  --normal_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/normal.jsonl" \
  --anomaly_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
  --anomaly_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
  \
  --batch_size 64 \
  \
  --model_ckpt "../TSA-ckpts/flow_mitdb106v_finetune_ckpt_v3_anomaly_only/2025-12-05-12:50:10/ckpt.pth" \
  --gpu_id 2 \
  \
  --need_to_generate 0 \
  --generated_path "../samples_path/flow/mitdb106v-flow-finetuned-v3-anomaly-only" \
  --num_samples -1 \
  --eval_train_size 10000

####################################################################
python FlowFinetuneEvaluateAnomaly.py \
  --seq_len 800 \
  --feature_size 2 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  --version 3 \
  \
  --dataset_name "ECG" \
  --max_anomaly_length 160 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --normal_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
  --normal_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/normal.jsonl" \
  --anomaly_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
  --anomaly_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
  \
  --batch_size 64 \
  \
  --model_ckpt "../TSA-ckpts/flow_mitdb106v_finetune_ckpt_v3_mixed_data/2025-12-05-15:03:35/ckpt.pth" \
  --gpu_id 2 \
  \
  --need_to_generate 0 \
  --generated_path "../samples_path/flow/mitdb106v-flow-finetuned-v3-mixed-data" \
  --num_samples -1 \
  --eval_train_size 10000
