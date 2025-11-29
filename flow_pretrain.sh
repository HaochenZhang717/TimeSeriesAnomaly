python FlowPretrainPipeline.py \
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
  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_100npz/train/normal.jsonl" \
  --indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_100npz/validation/normal.jsonl" \
  \
  --lr 1e-5 \
  --batch_size 128 \
  --epochs 1000 \
  --grad_clip_norm 1.0 \
  \
  --wandb_project flow_normal_pretrain \
  --wandb_run flow_normal_pretrain \
  \
  --ckpt_dir flow_normal_pretrain_ckpt \
  --gpu_id 1
