#export hucfg_t_sampling=logitnorm
#python dsp_flow.py \
#  --what_to_do "no_context_pretrain" \
#  \
#  --seq_len 1800 \
#  --feature_size 1 \
#  --one_channel 1 \
#  \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  \
#  --raw_data_path_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --indices_path_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
#  --min_infill_length 180 \
#  --max_infill_length 800 \
#  \
#  --lr 1e-4 \
#  --batch_size 64 \
#  --max_epochs 1000 \
#  --grad_clip_norm 1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "true" \
#  --patience 50 \
#  \
#  --wandb_project "dsp_flow" \
#  --wandb_run "106npz_no_context_pretrain" \
#  \
#  --ckpt_dir "../TSA-ckpts/dsp_flow/106npz/impute_pretrain_ckpt" \
#  --vqvae_ckpt "/root/tianyi/vqvae_save_path/vqvae_1d.pt" \
#  \
#  --generated_dir "none" \
#  \
#  --gpu_id 0



export hucfg_t_sampling=logitnorm

#python dsp_flow.py \
#  --what_to_do "imputation_finetune" \
#  \
#  --seq_len 1000 \
#  --feature_size 1 \
#  --one_channel 1 \
#  \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  \
#  --raw_data_path_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --indices_path_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
#  --min_infill_length 180 \
#  --max_infill_length 800 \
#  \
#  --lr 1e-4 \
#  --batch_size 64 \
#  --max_epochs 100 \
#  --grad_clip_norm 1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "true" \
#  --patience 50 \
#  \
#  --wandb_project "dsp_flow" \
#  --wandb_run "106npz_impute_from_scratch" \
#  \
#  --ckpt_dir "../TSA-ckpts/dsp_flow/106npz/impute_from_scratch_ckpt" \
#  --pretrained_ckpt "none" \
#  --vqvae_ckpt "/root/tianyi/vqvae_save_path/vqvae_1d.pt" \
#  \
#  --generated_dir "none" \
#  \
#  --gpu_id 0
#
#
#


python dsp_flow.py \
  --what_to_do "anomaly_evaluate" \
  \
  --seq_len 1000 \
  --feature_size 1 \
  --one_channel 1 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  \
  --raw_data_path_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --indices_path_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
  --indices_path_anomaly_for_sample "none" \
  --min_infill_length 180 \
  --max_infill_length 800 \
  \
  --lr 1e-4 \
  --batch_size 64 \
  --max_epochs 2000 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 50 \
  \
  --wandb_project "dsp_flow" \
  --wandb_run "106npz_impute_from_scratch" \
  \
  --ckpt_dir "" \
  --pretrained_ckpt "none" \
  --vqvae_ckpt "" \
  \
  --generated_path "/Users/zhc/Documents/PhD/projects/TimeSeriesAnomaly/samples_path/dsp_flow/impute_from_scratch_ckpt" \
  \
  --gpu_id 0