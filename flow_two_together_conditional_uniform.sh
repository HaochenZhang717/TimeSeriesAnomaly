


#python FlowTrainTogether.py \
#  --seq_len 800 \
#  --feature_size 2 \
#  \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  \
#  --dataset_name "ECG" \
#  --max_anomaly_length 160 \
#  --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
#  --indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
#  \
#  --lr 1e-3 \
#  --batch_size 64 \
#  --max_epochs 1000 \
#  --grad_clip_norm 1.0 \
#  --early_stop "true" \
#  --patience 50 \
#  \
#  --wandb_project "flow_imputation" \
#  --wandb_run "mitdb106v_uniform" \
#  \
#  --ckpt_dir "../TSA-ckpts/flow_imputation_uniform" \
#  --gpu_id 0


python FlowTwoTogether.py \
  --what_to_do "unconditional_training" \
  \
  --seq_len 800 \
  --feature_size 2 \
  \
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
  --lr 5e-4 \
  --batch_size 64 \
  --max_epochs 1000 \
  --grad_clip_norm 1.0 \
  --early_stop "true" \
  --patience 50 \
  \
  --wandb_project "flow_unconditional" \
  --wandb_run "mitdb_unconditional_uniform" \
  \
  --ckpt_dir "../TSA-ckpts/flow_unconditional_uniform" \
  --gpu_id 0 \
  \
  --cond_eval_model_ckpt "none" \
  --generated_path "none"
