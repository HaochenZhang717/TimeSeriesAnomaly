versions=(2 3)
modes=(mixed_data anomaly_only)

for version in "${versions[@]}"; do
  for mode in "${modes[@]}"; do
    python FlowFinetunePipeline.py \
      --seq_len 800 \
      --feature_size 2 \
      \
      --model_type "LastLayerPerturbFlow" \
      --n_layer_enc 4 \
      --n_layer_dec 4 \
      --d_model 64 \
      --n_heads 4 \
      --version ${version} \
      --early_stop "false" \
      \
      --dataset_name "ECG" \
      --max_anomaly_length 160 \
      --raw_data_paths_train "./dataset_utils/ECG_datasets/raw_data/106.npz" \
      --raw_data_paths_val "./dataset_utils/ECG_datasets/raw_data/106.npz" \
      --normal_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal.jsonl" \
      --normal_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/normal.jsonl" \
      --anomaly_indices_paths_train "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V.jsonl" \
      --anomaly_indices_paths_val "./dataset_utils/ECG_datasets/indices/slide_windows_106npz/validation/V.jsonl" \
      \
      --lr 1e-5 \
      --batch_size 64 \
      --max_iters 50000 \
      --grad_clip_norm 1.0 \
      --mode ${mode} \
      \
      --wandb_project flow_mitdb106v_finetune \
      --wandb_run "LastLayerPerturbFlow_mitdb106v_finetune_v${version}_norope_${mode}" \
      \
      --ckpt_dir "../TSA-ckpts/lastlayerperturbflow_mitdb106v_finetune_ckpt_v${version}_${mode}" \
      --pretrained_ckpt "LastLayerPerturbFlow_pretrain_mitdb106_ckpt/2025-12-05-17:24:49/ckpt.pth" \
      --generated_path "../samples_path/lastlayerperturbflow_mitdb106v_finetune_ckpt_v${version}_${mode}" \
      --gpu_id 7
  done
done