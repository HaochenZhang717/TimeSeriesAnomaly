export hucfg_t_sampling=logitnorm
LR=1e-4
LEN_WHOLE=1000
MAX_LEN_ANOMALY=800
MIN_LEN_ANOMALY=180
GPU_ID=4

DATA_TYPE="ecg"
WANDB_PROJECT="dsp_flow_mitdb_no_code"

VQVAE_CKPT="none"
PRETRAIN_CKPT="/root/tianyi/formal_experiment/mitdb/dsp_flow_no_code/no_context_pretrain_ckpt"
FINETUNE_CKPT="/root/tianyi/formal_experiment/mitdb/dsp_flow_no_code/impute_finetune_ckpt_lr${LR}"


DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data/106.npz"]'
TEST_DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data/106.npz"]'
PRETRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal_800.jsonl"]'
FINETUNE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V_train.jsonl"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V_test.jsonl"]'
ANOMALY_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/anomaly_segments_with_prototype_train.jsonl"]'
NORMAL_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal_1000.jsonl"]'

#VQVAE Train Parameters


python dsp_flow.py \
  --what_to_do "no_context_no_code_pretrain" \
  \
  --seq_len ${LEN_WHOLE} \
  --data_type ${DATA_TYPE} \
  --feature_size 1 \
  --one_channel 1 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  \
  --raw_data_paths_train ${DATA_PATHS} \
  --raw_data_paths_test ${TEST_DATA_PATHS} \
  --indices_paths_train ${PRETRAIN_INDICES_PATHS} \
  --indices_paths_test "[]" \
  --indices_paths_anomaly_for_sample "[]" \
  --min_infill_length ${MIN_LEN_ANOMALY} \
  --max_infill_length ${MAX_LEN_ANOMALY} \
  \
  --lr 1e-4 \
  --batch_size 64 \
  --max_epochs 1000 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 50 \
  \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "no_context_pretrain" \
  \
  --ckpt_dir ${PRETRAIN_CKPT} \
  --pretrained_ckpt "none" \
  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt"\
  \
  --generated_path "none" \
  \
  --gpu_id ${GPU_ID}



python dsp_flow.py \
  --what_to_do "no_code_imputation_finetune" \
  \
  --seq_len ${LEN_WHOLE} \
  --data_type ${DATA_TYPE} \
  --feature_size 1 \
  --one_channel 1 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  \
  --raw_data_paths_train ${DATA_PATHS} \
  --raw_data_paths_test ${TEST_DATA_PATHS} \
  --indices_paths_train ${FINETUNE_TRAIN_INDICES_PATHS} \
  --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
  --indices_paths_anomaly_for_sample "[]" \
  --min_infill_length ${MIN_LEN_ANOMALY} \
  --max_infill_length ${MAX_LEN_ANOMALY} \
  \
  --lr ${LR} \
  --batch_size 64 \
  --max_epochs 500 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 50 \
  \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "impute_finetune_lr${LR}" \
  \
  --ckpt_dir ${FINETUNE_CKPT} \
  --pretrained_ckpt ${PRETRAIN_CKPT} \
  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
  \
  --generated_path "none" \
  \
  --gpu_id ${GPU_ID}


python dsp_flow.py \
  --what_to_do "no_code_impute_sample" \
  \
  --seq_len ${LEN_WHOLE} \
  --data_type ${DATA_TYPE} \
  --feature_size 1 \
  --one_channel 1 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  \
  --raw_data_paths_train ${DATA_PATHS} \
  --raw_data_paths_test ${TEST_DATA_PATHS} \
  --indices_paths_train ${NORMAL_INDICES_FOR_SAMPLE} \
  --indices_paths_test "[]" \
  --indices_paths_anomaly_for_sample ${ANOMALY_INDICES_FOR_SAMPLE} \
  --min_infill_length ${MIN_LEN_ANOMALY} \
  --max_infill_length ${MAX_LEN_ANOMALY} \
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
  --ckpt_dir ${FINETUNE_CKPT} \
  --pretrained_ckpt "none" \
  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
  \
  --generated_path "" \
  \
  --gpu_id ${GPU_ID}



python dsp_flow.py \
  --what_to_do "anomaly_evaluate" \
  \
  --seq_len ${LEN_WHOLE} \
  --data_type ${DATA_TYPE} \
  --feature_size 1 \
  --one_channel 1 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  \
  --raw_data_paths_train ${DATA_PATHS} \
  --raw_data_paths_test ${TEST_DATA_PATHS} \
  --indices_paths_train "[]" \
  --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
  --indices_paths_anomaly_for_sample "[]" \
  --min_infill_length ${MIN_LEN_ANOMALY} \
  --max_infill_length ${MAX_LEN_ANOMALY} \
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
  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
  \
  --generated_path "${FINETUNE_CKPT}/no_code_impute_samples.pth" \
  \
  --gpu_id ${GPU_ID}
