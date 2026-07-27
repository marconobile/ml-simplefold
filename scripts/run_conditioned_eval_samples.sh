#!/usr/bin/env bash
set -euo pipefail

# Run cluster-conditioned SimpleFold evaluation multiple times for each NPZ type.
#
# Defaults:
#   N=5
#   STRUCTURE_TYPES=(active inactive pas)
#   output root=/storage_common/nobilm/backmapping_pots_model/results
#
# Example:
#   bash scripts/run_conditioned_eval_samples.sh
#
# Optional overrides:
#   N=10 DEVICE=cuda:0 BASE_SEED=1234 bash scripts/run_conditioned_eval_samples.sh

# active: /home/nobilm@usi.ch/ml-simplefold/test_new_data_with_clusters/active_without_hs.npz
# inactive: /home/nobilm@usi.ch/ml-simplefold/test_new_data_with_clusters/inactive_without_hs.npz
# pas: /home/nobilm@usi.ch/ml-simplefold/test_new_data_with_clusters/pas_without_hs.npz

# check for changes
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/storage_common/nobilm/ml-simplefold/fine_tune_with_clusters/finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged/checkpoints/last.ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/storage_common/nobilm/ml-simplefold/fine_tune_with_clusters/finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged/samples}"

# fixed
N="${N:-5}"
BASE_SEED="${BASE_SEED:-123}"
CONDA_ENV="${CONDA_ENV:-simplefold}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


# RAW_NPZ_DIR="${RAW_NPZ_DIR:-${REPO_ROOT}/test_new_data_with_clusters}"
# RAW_NPZ_PATH="/storage_common/nobilm/backmapping_pots_model/datasets/active/with_hs/active_NECA_G_protein_4000_frames_chi2fixed/without_hs/active_NECA_G_protein_4000_frames_chi2fixed_with_globalclusters.npz"
RAW_NPZ_PATH="/storage_common/nobilm/backmapping_pots_model/datasets/active/with_hs/active_NECA_G_protein_4000_frames_chi2fixed/without_hs/active_without_hs.npz" # copy of above


LABELS_NPZ_PATH="${LABELS_NPZ_PATH:-}"


STRUCTURE_TYPES=("active") # "inactive" "pas") # for denovo just need 1 for input processing


#! for denovo
# STRUCTURE_TYPES=("active") # for denovo just need 1 for input processing
# LABELS_NPZ_PATH="/storage_common/nobilm/backmapping_pots_model/pots_samples/sample.npz"
# TYPE_OUTPUT_DIR="${OUTPUT_ROOT}/denovo_samples"


cd "${REPO_ROOT}"
for TYPE in "${STRUCTURE_TYPES[@]}"; do
    # RAW_NPZ_PATH="${RAW_NPZ_DIR}/${TYPE}_without_hs.npz"
    echo "Processing TYPE=${TYPE} with raw NPZ: ${RAW_NPZ_PATH}"    
        
    TYPE_OUTPUT_DIR="${OUTPUT_ROOT}/${TYPE}_samples" #! here for active inactive pas splitting
    mkdir -p "${TYPE_OUTPUT_DIR}"

    echo "Running TYPE=${TYPE} N=${N} BASE_SEED=${BASE_SEED}"
    echo "Output: ${TYPE_OUTPUT_DIR}"
    
    # LABELS_NPZ_PATH: optional. If set, Python samples N label rows.
    # Otherwise, Python samples N random observations from RAW_NPZ_PATH.
    cmd=(
        python scripts/sample_with_conditioning.py
        --seed "${BASE_SEED}" # BASE_SEED + sample_index, so 123, 124, 125, 126, 127 for N=5.
        -N "${N}"
        --checkpoint-path "${CHECKPOINT_PATH}"
        --raw-npz-path "${RAW_NPZ_PATH}"
        --output-dir "${TYPE_OUTPUT_DIR}"
        --device "${DEVICE}"
        --guidance-scale 3.0
    )
    if [[ -n "${LABELS_NPZ_PATH}" ]]; then
        cmd+=(
            --labels-npz-path "${LABELS_NPZ_PATH}"
        )
    fi
    "${cmd[@]}"

    echo "Assigning oracle clusters for TYPE=${TYPE}"
    python scripts/assign_conditioned_eval_sample_clusters.py --base-path "${TYPE_OUTPUT_DIR}"

    echo "Comparing original conditioning vs oracle labels for TYPE=${TYPE}"
    python scripts/compare_conditioning_to_oracle.py \
        --base-path "${TYPE_OUTPUT_DIR}" \
        --out-dir "${TYPE_OUTPUT_DIR}"

    echo "Plotting assigned-cluster evaluation for TYPE=${TYPE}"
    python plot_evaluation.py --base_path "${TYPE_OUTPUT_DIR}" --out_dir "${TYPE_OUTPUT_DIR}"
done

echo "Comparing original conditioning vs oracle labels across all structure types"
python scripts/compare_conditioning_to_oracle.py \
    --base-path "${OUTPUT_ROOT}" \
    --out-dir "${OUTPUT_ROOT}"

echo "Plotting assigned-cluster evaluation across all structure types"
python plot_evaluation.py --base_path "${OUTPUT_ROOT}" --out_dir "${OUTPUT_ROOT}"
