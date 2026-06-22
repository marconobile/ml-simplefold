#!/usr/bin/env bash
set -euo pipefail

# Run cluster-conditioned SimpleFold evaluation multiple times for each NPZ type.
#
# Defaults:
#   N=5
#   t_values=(active inactive pas)
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
DEVICE="${DEVICE:-cuda:3}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/storage_common/nobilm/ml-simplefold/fine_tune_with_clusters/ft_merged_npz_from_simplefold100M_max_step_600k_fix_ref_pos/checkpoints/last.ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/storage_common/nobilm/backmapping_pots_model/ft_merged_npz_from_simplefold100M_max_step_60000_fix_ref_pos_vs_ref_act_inact_pas}"

# fixed
N="${N:-1}" # leave 1 change the N below 
BASE_SEED="${BASE_SEED:-123}"
CONDA_ENV="${CONDA_ENV:-simplefold}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_NPZ_DIR="${RAW_NPZ_DIR:-${REPO_ROOT}/test_new_data_with_clusters}"
t_values=("active" "inactive" "pas") # for denovo just need 1 for input processing

#! for denovo
# t_values=("active") # for denovo just need 1 for input processing
# LABELS_NPZ_PATH="/storage_common/nobilm/backmapping_pots_model/pots_samples/sample.npz"
# TYPE_OUTPUT_DIR="${OUTPUT_ROOT}/denovo_samples"


cd "${REPO_ROOT}"
for TYPE in "${t_values[@]}"; do
    RAW_NPZ_PATH="${RAW_NPZ_DIR}/${TYPE}_without_hs.npz"
    echo "Processing TYPE=${TYPE} with raw NPZ: ${RAW_NPZ_PATH}"    
        
    TYPE_OUTPUT_DIR="${OUTPUT_ROOT}/${TYPE}_samples" #! here for active inactive pas splitting

    for SAMPLE_INDEX in $(seq 1 "${N}"); do
        SAMPLE_OUTPUT_DIR="${TYPE_OUTPUT_DIR}/sample_${SAMPLE_INDEX}"
        SEED=$((BASE_SEED + SAMPLE_INDEX - 1))

        mkdir -p "${SAMPLE_OUTPUT_DIR}"

        echo "Running TYPE=${TYPE} SAMPLE=${SAMPLE_INDEX}/${N} SEED=${SEED}"
        echo "Output: ${SAMPLE_OUTPUT_DIR}"
        
        # --labels-npz-path: optional, if present used for conditioning, if not present uses structures from --raw-npz-path 
        # --ref-pos-mode zero \
        python scripts/sample_with_conditioning.py \
            --seed "${SEED}" \
            -N 5 \
            --checkpoint-path "${CHECKPOINT_PATH}" \
            --raw-npz-path "${RAW_NPZ_PATH}" \
            --output-dir "${SAMPLE_OUTPUT_DIR}" \
            --device "${DEVICE}"
            # --labels-npz-path "${LABELS_NPZ_PATH}" \
    done
done

python scripts/assign_conditioned_eval_sample_clusters.py --base-path "${SAMPLE_OUTPUT_DIR}"
python plot_evaluation.py --base_path "${SAMPLE_OUTPUT_DIR}" --out_dir "${SAMPLE_OUTPUT_DIR}"
