export hucfg_t_sampling=logitnorm
#python dsp_flow.py \
#  --what_to_do "no_context_no_code_pretrain" \
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
#  --wandb_run "106npz_no_context_no_code_pretrain" \
#  \
#  --ckpt_dir "../TSA-ckpts/dsp_flow/106npz/no_context_no_code_pretrain_ckpt" \
#  --vqvae_ckpt "/root/tianyi/vqvae_save_path/vqvae_1d.pt" \
#  --pretrained_ckpt "none" \
#  \
#  --generated_dir "none" \
#  \
#  --gpu_id 1



python dsp_flow.py \
  --what_to_do "no_code_imputation_finetune" \
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
  --indices_path_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V_train.jsonl" \
  --indices_path_test "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V_test.jsonl" \
  --min_infill_length 180 \
  --max_infill_length 800 \
  \
  --lr 1e-4 \
  --batch_size 64 \
  --max_epochs 500 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 50 \
  \
  --wandb_project "dsp_flow" \
  --wandb_run "106npz_no_code_impute_finetune" \
  \
  --ckpt_dir "../TSA-ckpts/dsp_flow/106npz/no_code_impute_finetune_ckpt" \
  --pretrained_ckpt "../TSA-ckpts/dsp_flow/106npz/no_context_no_code_pretrain_ckpt" \
  --vqvae_ckpt "/root/tianyi/vqvae_save_path/vqvae_1d.pt" \
  \
  --generated_path "none" \
  \
  --gpu_id 1



python dsp_flow.py \
  --what_to_do "no_code_impute_sample" \
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
  --indices_path_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal_1000.jsonl" \
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
  --wandb_project "none" \
  --wandb_run "none" \
  \
  --ckpt_dir "../TSA-ckpts/dsp_flow/106npz/no_code_impute_finetune_ckpt" \
  --pretrained_ckpt "none" \
  --vqvae_ckpt "/root/tianyi/vqvae_save_path/vqvae_1d.pt" \
  \
  --generated_path "" \
  \
  --gpu_id 1


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
  --indices_path_train "none" \
  --indices_path_test "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V_test.jsonl" \
  --indices_path_anomaly_for_sample "" \
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
  --wandb_project "none" \
  --wandb_run "none" \
  \
  --ckpt_dir "" \
  --pretrained_ckpt "none" \
  --vqvae_ckpt "" \
  \
  --generated_path "../TSA-ckpts/dsp_flow/106npz/no_code_impute_finetune_ckpt/no_code_impute_samples.pth" \
  \
  --gpu_id 1