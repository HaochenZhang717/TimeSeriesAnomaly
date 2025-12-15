
export hucfg_t_sampling=logitnorm
python PrototypeFlow.py \
  --what_to_do "no_context_train" \
  \
  --seq_len 100 \
  --feature_size 1 \
  --one_channel 1 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  --num_prototypes 8 \
  \
  --raw_data_path_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --indices_path_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/anomaly_segments_with_prototype.jsonl" \
  \
  --lr 1e-4 \
  --batch_size 64 \
  --max_epochs 1000 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 50 \
  \
  --wandb_project "Prototype Flow" \
  --wandb_run "106npz-no-context" \
  \
  --ckpt_dir "../TSA-ckpts/PrototypeFlow/mitdb1800_106/no_context_ckpt" \
  \
  --generated_dir "none" \
  \
  --gpu_id 0


export hucfg_t_sampling=logitnorm
python PrototypeFlow.py \
  --what_to_do "no_context_sample" \
  \
  --seq_len 100 \
  --feature_size 1 \
  --one_channel 1 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  --num_prototypes 8 \
  \
  --raw_data_path_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --indices_path_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/anomaly_segments_with_prototype.jsonl" \
  \
  --lr 1e-4 \
  --batch_size 64 \
  --max_epochs 1000 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 50 \
  \
  --wandb_project "Prototype Flow" \
  --wandb_run "106npz-no-context" \
  \
  --ckpt_dir "../TSA-ckpts/PrototypeFlow/mitdb1800_106/no_context_ckpt" \
  \
  --generated_dir "../samples_path/PrototypeFlow/mitdb1800_106/no_context" \
  \
  --gpu_id 0

