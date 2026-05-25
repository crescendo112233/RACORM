#!/usr/bin/env bash
set -e

python ./ACORM_QMIX/main.py \
  --algorithm RACORM \
  --env_name MMM2 \
  --cluster_num 3 \
  --max_train_steps 3050000 \
  --evaluate_freq 10000 \
  --tb_log_dir /root/tf-logs \
  --relation_pos_threshold 0.65 \
  --relation_neg_threshold 0.35 \
  --relation_temperature 0.2 \
  --relation_loss_weight 1.0 \
  --kmeans_loss_weight 0.0
