#!/usr/bin/env bash
DATASET="${DATASET:-maze2d-large-v1}"
CONFIG="${CONFIG:-config.maze2d}"
METHOD="${METHOD:-cfm}"
LOGBASE="${LOGBASE:-logs}"
DEVICE="${DEVICE:-cuda:0}"

# Aegis knobs.
AEGIS_GLOBAL_EM="${AEGIS_GLOBAL_EM:-true}"
AEGIS_ITERS="${AEGIS_ITERS:-15}"
AEGIS_LR="${AEGIS_LR:-0.05}"
AEGIS_DECAY="${AEGIS_DECAY:-0.995}"
AEGIS_LAMBDA="${AEGIS_LAMBDA:-1.5}"
AEGIS_BETA="${AEGIS_BETA:-0.01}"
REWARD_SCALE="${REWARD_SCALE:-0.01}"
AEGIS_CURVATURE_WEIGHT="${AEGIS_CURVATURE_WEIGHT:-0.6}"
AEGIS_ACCELERATION_WEIGHT="${AEGIS_ACCELERATION_WEIGHT:-0.4}"
AEGIS_EM_ROLLOUT_STEPS="${AEGIS_EM_ROLLOUT_STEPS:-1}"

AEGIS_SUFFIX="${AEGIS_SUFFIX:-aegis_random_iter10}"

echo "[aegis] running plan_maze2d with AegisFlow"
python scripts/plan_maze2d.py \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --logbase "$LOGBASE" \
  --method "$METHOD" \
  --device "$DEVICE" \
  --safety false \
  --aegis true \
  --suffix "$AEGIS_SUFFIX" \
  --aegis_global_em "$AEGIS_GLOBAL_EM" \
  --aegis_iterations "$AEGIS_ITERS" \
  --aegis_lr "$AEGIS_LR" \
  --aegis_decay "$AEGIS_DECAY" \
  --aegis_lambda "$AEGIS_LAMBDA" \
  --aegis_beta "$AEGIS_BETA" \
  --aegis_em_rollout_steps "$AEGIS_EM_ROLLOUT_STEPS" \
  --aegis_reward_scale "$REWARD_SCALE" \
  --aegis_curvature_weight "$AEGIS_CURVATURE_WEIGHT" \
  --aegis_acceleration_weight "$AEGIS_ACCELERATION_WEIGHT" \
  --aegis_barrier_delta .05 \
  "$@"
