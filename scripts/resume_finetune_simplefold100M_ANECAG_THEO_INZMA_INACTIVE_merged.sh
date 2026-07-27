#!/usr/bin/env bash
set -euo pipefail

# Resume the ANECAG/THEO/INZMA/INACTIVE fine-tuning run from its full
# PyTorch Lightning checkpoint. trainer.max_steps is the total target step,
# not an additional number of steps after resuming.
#
# Usage:
#   bash scripts/resume_finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged.sh
#
# Optional overrides:
#   CUDA_VISIBLE_DEVICES=1 CHECKPOINT_PATH=/path/to/last.ckpt \
#     bash scripts/resume_finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged.sh
#
# Additional arguments are forwarded to Hydra, for example:
#   bash scripts/resume_finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged.sh \
#     trainer.max_steps=950000

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/nobilm@usi.ch/miniconda3/envs/simplefold/bin/python}"
CONDA_ENV="${CONDA_ENV:-simplefold}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/storage_common/nobilm/ml-simplefold/fine_tune_with_clusters/finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged/checkpoints/last.ckpt}"
OUTPUT_DIR="/storage_common/nobilm/ml-simplefold/fine_tune_with_clusters/finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged"
DATA_DIR="/scratch/nobilm/quantum_backmapping/ANECAG_THEO_INZMA_INACTIVE_merged"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Resume checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

if [[ ! -f "${DATA_DIR}/manifest.json" ]]; then
    echo "Dataset manifest not found: ${DATA_DIR}/manifest.json" >&2
    exit 1
fi

cd "${REPO_ROOT}"

echo "Resuming from: ${CHECKPOINT_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}"
echo "Total training target: 900000 steps"

exec env \
    CONDA_DEFAULT_ENV="${CONDA_ENV}" \
    PATH="$(dirname "${PYTHON_BIN}"):${PATH}" \
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    "${PYTHON_BIN}" src/simplefold/train.py \
    experiment=train \
    hydra.job.name=finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged \
    "paths.output_dir=${OUTPUT_DIR}" \
    paths.tmp_dir=/tmp/tmp \
    "+load_ckpt_path=${CHECKPOINT_PATH}" \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    "data.datasets.0.target_dir=${DATA_DIR}" \
    "data.datasets.0.tokenized_dir=${DATA_DIR}" \
    "data.datasets.0.manifest_path=${DATA_DIR}/manifest.json" \
    "callbacks.model_checkpoint.dirpath=${OUTPUT_DIR}/checkpoints" \
    "callbacks.model_checkpoint.filename='finetune_simplefold100M_finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged-quick-ft-ep{epoch:04d}-step{step:08d}'" \
    +callbacks.model_checkpoint.every_n_epochs=1 \
    callbacks.model_checkpoint.save_on_train_epoch_end=true \
    callbacks.model_checkpoint.every_n_train_steps=null \
    +callbacks.lr_monitor._target_=lightning.pytorch.callbacks.LearningRateMonitor \
    +callbacks.lr_monitor.logging_interval=step \
    model.optimizer.lr=1e-4 \
    model.scheduler._target_=utils.lr_scheduler.LinearWarmupCosineAnnealingLR \
    model.scheduler._partial_=true \
    model.scheduler.min_lr=5e-7 \
    model.scheduler.max_lr=1e-4 \
    model.scheduler.warmup_steps=10000 \
    +model.scheduler.T_max=850000 \
    +model.scheduler.eta_min=5e-7 \
    trainer.max_steps=900000 \
    model.cluster_mlm_loss_weight=0.05 \
    model.architecture.use_conditioning_masking=true \
    model.architecture.cluster_mask_prob=0.15 \
    model.architecture.conditioning_dropout_prob=0.1 \
    "$@"
