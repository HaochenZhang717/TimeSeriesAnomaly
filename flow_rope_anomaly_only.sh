#export hucfg_attention_rope_use=1

#python FlowPretrainPipeline.py \
#  --seq_len 800 \
#  --feature_size 2 \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
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
#  --wandb_project flow_normal_pretrain_mitdb106 \
#  --wandb_run flow_normal_pretrain_mitdb106_rope \
#  \
#  --ckpt_dir ../TSA-ckpts/flow_normal_pretrain_ckpt_mitdb106_rope \
#  --gpu_id 3



#export hucfg_attention_rope_use=1
#python FlowFinetunePipeline.py \
#  --seq_len 800 \
#  --feature_size 2 \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  --version 2 \
#  \
#  --max_anomaly_ratio 0.2 \
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --normal_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
#  --normal_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/normal.jsonl" \
#  --anomaly_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
#  --anomaly_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
#  \
#  --lr 1e-5 \
#  --batch_size 64 \
#  --max_iters 1000000 \
#  --grad_clip_norm 1.0 \
#  --mode "anomaly_only" \
#  \
#  --wandb_project flow_mitdb106v_finetune \
#  --wandb_run flow_mitdb106v_finetune_v2_rope_anomaly_only \
#  \
#  --ckpt_dir "../TSA-ckpts/flow_normal_finetune_ckpt_mitdb106_rope_v2_anomaly_only" \
#  --pretrained_ckpt "../TSA-ckpts/flow_normal_pretrain_ckpt_mitdb106_rope/2025-12-01-14:16:17/ckpt.pth" \
#  --gpu_id 0


#export hucfg_attention_rope_use=1
#python FlowFinetuneEvaluateAnomaly.py \
#  --seq_len 800 \
#  --feature_size 2 \
#  \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  --version 2 \
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
#  --model_ckpt "../TSA-ckpts/flow_normal_finetune_ckpt_mitdb106_rope_v2_anomaly_only/2025-12-01-16:42:26/ckpt.pth" \
#  --gpu_id 7 \
#  \
#  --need_to_generate 1 \
#  --generated_path "../samples_path/flow/mitdb106v-finetuned-rope-v2-anomaly-only" \
