


#export hucfg_attention_rope_use=1
#export hucfg_t_sampling=logitnorm
#python FlowTwoTogether.py \
#  --what_to_do "conditional_training" \
#  \
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
#  --wandb_run "mitdb106v_logit_norm" \
#  \
#  --ckpt_dir "../TSA-ckpts/flow_imputation_logit_norm" \
#  --gpu_id 1
#
#
#python FlowTwoTogether.py \
#  --what_to_do "conditional_evaluate" \
#  \
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
#  --wandb_run "mitdb106v_logit_norm" \
#  \
#  --ckpt_dir "../TSA-ckpts/flow_imputation_logit_norm" \
#  --gpu_id 1 \
#  \
#  --cond_eval_model_ckpt "../TSA-ckpts/flow_imputation_logit_norm/ema_ckpt.pth" \
#  --generated_path "../samples_path/flow_imputation_logit_norm"


export hucfg_t_sampling=logitnorm
#subsets=("041_WSD_id_13_WebService_tr_4296_1st_5196" "128_WSD_id_100_WebService_tr_4068_1st_4168" "178_SMD_id_1_Facility_tr_6000_1st_10609" "182_SMD_id_5_Facility_tr_7174_1st_21230")
#SEQLENS=(100 200 400)
subsets=( "178_SMD_id_1_Facility_tr_6000_1st_10609")
SEQLENS=(400)
for seq_len in "${SEQLENS[@]}"; do
  for subset in "${subsets[@]}"; do
    python FlowTwoTogether.py \
      --what_to_do "conditional_sample" \
      \
      --seq_len ${seq_len} \
      --feature_size 1 \
      \
      --n_layer_enc 4 \
      --n_layer_dec 4 \
      --d_model 64 \
      --n_heads 4 \
      \
      --dataset_name "TSBAD" \
      --max_anomaly_length 10 \
      --raw_data_paths_train "./dataset_utils/TSBAD_datasets/raw_data/selected_uts/${subset}.csv" \
      --raw_data_paths_val "none" \
      --indices_paths_train "./dataset_utils/TSBAD_datasets/indices_len${seq_len}/slide_windows_${subset}/train/anomaly.jsonl" \
      --indices_paths_val "none" \
      \
      --lr 5e-4 \
      --batch_size 64 \
      --max_epochs 1000 \
      --grad_clip_norm 1.0 \
      --early_stop "true" \
      --patience 20 \
      \
      --wandb_project "none" \
      --wandb_run "none" \
      \
      --ckpt_dir "none" \
      --gpu_id 2 \
      \
      --cond_eval_model_ckpt "../TSA-ckpts/flow_conditional_logit_norm/len${seq_len}/${subset}/ema_ckpt.pth" \
      --generated_path "../samples_path/flow_imputation_logit_norm/len${seq_len}/${subset}" \
      --eval_train_size -1
  done
done

