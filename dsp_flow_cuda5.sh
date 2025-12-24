export hucfg_t_sampling=logitnorm
LR=1e-4
LEN_WHOLE=1000
MAX_LEN_ANOMALY=193
MIN_LEN_ANOMALY=190
GPU_ID=5

DATA_TYPE="ercot"
WANDB_PROJECT="dsp_flow_no_code_ercot"

VQVAE_CKPT="none"
PRETRAIN_CKPT="/root/tianyi/formal_experiment/ercot/dsp_flow_no_code/no_context_pretrain_ckpt"
FINETUNE_CKPT="/root/tianyi/formal_experiment/ercot/dsp_flow_no_code/impute_finetune_ckpt_lr${LR}"


DATA_PATHS='["./dataset_utils/ERCOT_datasets/raw_data/coast.npy", "./dataset_utils/ERCOT_datasets/raw_data/east.npy", "./dataset_utils/ERCOT_datasets/raw_data/fwest.npy", "./dataset_utils/ERCOT_datasets/raw_data/ncent.npy"]'
TEST_DATA_PATHS='["./dataset_utils/ERCOT_datasets/raw_data/north.npy", "./dataset_utils/ERCOT_datasets/raw_data/scent.npy", "./dataset_utils/ERCOT_datasets/raw_data/south.npy", "./dataset_utils/ERCOT_datasets/raw_data/west.npy"]'
PRETRAIN_INDICES_PATHS='["./dataset_utils/ERCOT_datasets/indices/coast_normal_200.jsonl", "./dataset_utils/ERCOT_datasets/indices/east_normal_200.jsonl", "./dataset_utils/ERCOT_datasets/indices/fwest_normal_200.jsonl", "./dataset_utils/ERCOT_datasets/indices/ncent_normal_200.jsonl"]'
FINETUNE_TRAIN_INDICES_PATHS='["./dataset_utils/ERCOT_datasets/indices/coast_anomaly.jsonl", "./dataset_utils/ERCOT_datasets/indices/east_anomaly.jsonl", "./dataset_utils/ERCOT_datasets/indices/fwest_anomaly.jsonl", "./dataset_utils/ERCOT_datasets/indices/ncent_anomaly.jsonl"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ERCOT_datasets/indices/north_anomaly.jsonl", "./dataset_utils/ERCOT_datasets/indices/scent_anomaly.jsonl", "./dataset_utils/ERCOT_datasets/indices/south_anomaly.jsonl", "./dataset_utils/ERCOT_datasets/indices/west_anomaly.jsonl"]'


ANOMALY_INDICES_FOR_SAMPLE='["./dataset_utils/ERCOT_datasets/indices/anomaly_segments_coast.jsonl", "./dataset_utils/ERCOT_datasets/indices/anomaly_segments_east.jsonl", "./dataset_utils/ERCOT_datasets/indices/anomaly_segments_fwest.jsonl", "./dataset_utils/ERCOT_datasets/indices/anomaly_segments_ncent.jsonl"]'
NORMAL_INDICES_FOR_SAMPLE='["./dataset_utils/ERCOT_datasets/indices/coast_normal_1000.jsonl", "./dataset_utils/ERCOT_datasets/indices/east_normal_1000.jsonl", "./dataset_utils/ERCOT_datasets/indices/fwest_normal_1000.jsonl", "./dataset_utils/ERCOT_datasets/indices/ncent_normal_1000.jsonl"]'


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
  --raw_data_paths_train "${DATA_PATHS}" \
  --raw_data_paths_test "${TEST_DATA_PATHS}" \
  --indices_paths_train "${PRETRAIN_INDICES_PATHS}" \
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
  --raw_data_paths_train "${DATA_PATHS}" \
  --raw_data_paths_test "${TEST_DATA_PATHS}" \
  --indices_paths_train "${FINETUNE_TRAIN_INDICES_PATHS}" \
  --indices_paths_test "${FINETUNE_TEST_INDICES_PATHS}" \
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
  --raw_data_paths_train "${DATA_PATHS}" \
  --raw_data_paths_test "${TEST_DATA_PATHS}" \
  --indices_paths_train "${NORMAL_INDICES_FOR_SAMPLE}" \
  --indices_paths_test "[]" \
  --indices_paths_anomaly_for_sample "${ANOMALY_INDICES_FOR_SAMPLE}" \
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
  --raw_data_paths_train "${DATA_PATHS}" \
  --raw_data_paths_test "${TEST_DATA_PATHS}" \
  --indices_paths_train "[]" \
  --indices_paths_test "${FINETUNE_TEST_INDICES_PATHS}" \
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
