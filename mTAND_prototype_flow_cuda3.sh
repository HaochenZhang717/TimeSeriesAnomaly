SEQLEN=1800
USE_PROTOTPYE="true"
export hucfg_t_sampling=logitnorm
python mTANDPrototypeFlow.py \
  --what_to_do "imputation_train" \
  \
  --seq_len ${SEQLEN} \
  --feature_size 1 \
  --one_channel 1 \
  --use_prototype ${USE_PROTOTPYE} \
  \
  --n_layer_enc 4 \
  --n_layer_dec 6 \
  --d_model 64 \
  --n_heads 4 \
  --num_prototypes 8 \
  --encoder_H 4 \
  --encoder_d_h 16 \
  \
  --raw_data_path_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
  --indices_path_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V_more.jsonl" \
  \
  --lr 1e-4 \
  --batch_size 64 \
  --max_epochs 1000 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 50 \
  \
  --wandb_project "mTAND Prototype Flow" \
  --wandb_run "106npz-imputation" \
  \
  --ckpt_dir "../TSA-ckpts/mTANDPrototypeFlow/mitdb1800_106/imputation_len${SEQLEN}_prototype${USE_PROTOTPYE}_ckpt" \
  \
  --generated_dir "none" \
  \
  --gpu_id 3


#export hucfg_t_sampling=logitnorm
#python PrototypeFlow.py \
#  --what_to_do "imputation_sample" \
#  \
#  --seq_len ${SEQLEN} \
#  --feature_size 1 \
#  --one_channel 1 \
#  \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  --num_prototypes 8 \
#  \
#  --raw_data_path_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
#  --indices_path_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V_more.jsonl" \
#  \
#  --lr 1e-4 \
#  --batch_size 64 \
#  --max_epochs 1000 \
#  --grad_clip_norm 1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "true" \
#  --patience 50 \
#  \
#  --wandb_project "none" \
#  --wandb_run "none" \
#  \
#  --ckpt_dir "../TSA-ckpts/PrototypeFlow/mitdb1800_106/imputation_len${SEQLEN}_ckpt" \
#  \
#  --generated_dir "../samples_path/PrototypeFlow/mitdb1800_106/imputation_len${SEQLEN}" \
#  \
#  --gpu_id 1
