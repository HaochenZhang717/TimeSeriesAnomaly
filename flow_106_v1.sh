#python FlowFinetuneEvaluateAnomaly.py \
#  --seq_len 800 \
#  --feature_size 2 \
#  \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  --version 1 \
#  \
#  --max_anomaly_ratio 0.2 \
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --normal_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
#  --normal_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/normal.jsonl" \
#  --anomaly_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
#  --anomaly_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
#  \
#  --batch_size 64 \
#  \
#  --model_ckpt "../TSA-ckpts/flow_mitdb106v_finetune_ckpt/2025-11-30-03:01:14/ckpt.pth" \
#  --gpu_id 4 \
#  \
#  --need_to_generate 1 \
#  --generated_path "../samples_path/flow/mitdb106v-finetuned" \


#python FlowFinetunePipeline.py \
#  --seq_len 800 \
#  --feature_size 2 \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  --version 3 \
#  \
#  --dataset_name "ECG" \
#  --max_anomaly_length 160 \
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --normal_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
#  --normal_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/normal.jsonl" \
#  --anomaly_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
#  --anomaly_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
#  \
#  --lr 1e-5 \
#  --batch_size 64 \
#  --max_iters 50000 \
#  --grad_clip_norm 1.0 \
#  --mode "anomaly_only" \
#  \
#  --wandb_project flow_mitdb106v_finetune \
#  --wandb_run flow_mitdb106v_finetune_v3_norope_anomaly_only \
#  \
#  --ckpt_dir "../TSA-ckpts/flow_mitdb106v_finetune_ckpt_v3_anomaly_only" \
#  --pretrained_ckpt "../TSA-ckpts/flow_normal_pretrain_ckpt_mitdb106/2025-11-30-00:52:35/ckpt.pth" \
#  --gpu_id 7
#
#
#python FlowFinetunePipeline.py \
#  --seq_len 800 \
#  --feature_size 2 \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  --version 3 \
#  \
#  --dataset_name "ECG" \
#  --max_anomaly_length 160 \
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --normal_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
#  --normal_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/normal.jsonl" \
#  --anomaly_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
#  --anomaly_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
#  \
#  --lr 1e-5 \
#  --batch_size 64 \
#  --max_iters 50000 \
#  --grad_clip_norm 1.0 \
#  --mode "mixed_data" \
#  \
#  --wandb_project flow_mitdb106v_finetune \
#  --wandb_run flow_mitdb106v_finetune_v3_norope_mixed_data \
#  \
#  --ckpt_dir "../TSA-ckpts/flow_mitdb106v_finetune_ckpt_v3_mixed_data" \
#  --pretrained_ckpt "../TSA-ckpts/flow_normal_pretrain_ckpt_mitdb106/2025-11-30-00:52:35/ckpt.pth" \
#  --gpu_id 7
