python FlowPretrainPipeline.py \
  --seq_len 800 \
  --feature_size 2 \
  \
  --model_type  "LastLayerPerturbFlow"\
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  \
  --dataset_name "ECG" \
  --max_anomaly_length 160 \
  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
  --indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/normal.jsonl" \
  \
  --lr 1e-5 \
  --batch_size 128 \
  --epochs 200 \
  --grad_clip_norm 1.0 \
  --early_stop "false" \
  \
  --wandb_project flow_normal_pretrain \
  --wandb_run LastLayerPerturbFlow_pretrain_mitdb106 \
  \
  --ckpt_dir LastLayerPerturbFlow_pretrain_mitdb106_ckpt \
  --gpu_id 6
