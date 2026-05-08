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


N="${N:-5}"
BASE_SEED="${BASE_SEED:-123}"
DEVICE="${DEVICE:-cuda:1}"
CONDA_ENV="${CONDA_ENV:-simplefold}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/storage_common/nobilm/ml-simplefold/fine_tune_with_clusters/ft_merged_npz_from_simplefold100M/checkpoints/last.ckpt}"
RAW_NPZ_DIR="${RAW_NPZ_DIR:-${REPO_ROOT}/test_new_data_with_clusters}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/storage_common/nobilm/backmapping_pots_model/results}"

t_values=("active" "inactive" "pas")

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Could not find conda. Load conda first or set up your shell initialization." >&2
    exit 1
fi

conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"

for TYPE in "${t_values[@]}"; do
    RAW_NPZ_PATH="${RAW_NPZ_DIR}/${TYPE}_without_hs.npz"
    TYPE_OUTPUT_DIR="${OUTPUT_ROOT}/${TYPE}_samples"

    if [[ ! -f "${RAW_NPZ_PATH}" ]]; then
        echo "Missing raw NPZ: ${RAW_NPZ_PATH}" >&2
        exit 1
    fi

    for SAMPLE_INDEX in $(seq 1 "${N}"); do
        SAMPLE_OUTPUT_DIR="${TYPE_OUTPUT_DIR}/sample_${SAMPLE_INDEX}"
        SEED=$((BASE_SEED + SAMPLE_INDEX - 1))

        mkdir -p "${SAMPLE_OUTPUT_DIR}"

        echo "Running TYPE=${TYPE} SAMPLE=${SAMPLE_INDEX}/${N} SEED=${SEED}"
        echo "Output: ${SAMPLE_OUTPUT_DIR}"

        python scripts/evaluate_active_npz_conditioned_sample.py \
            --seed "${SEED}" \
            --checkpoint-path "${CHECKPOINT_PATH}" \
            --raw-npz-path "${RAW_NPZ_PATH}" \
            --output-dir "${SAMPLE_OUTPUT_DIR}" \
            --device "${DEVICE}"
    done
done
