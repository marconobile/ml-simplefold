#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/run_conditioned_eval_tau_guidance_sweep.sh [options]

Run 100 all-structures samples for every pair in this grid:
  tau:            0.1, 0.3, 0.5, 0.7, 0.9
  guidance-scale: 2, 3, 5, 8, 12

By default, two useful controls are also run:
  tau=0.0, guidance-scale=3  (setting used by the source workflow)
  tau=0.3, guidance-scale=1  (sampler default; no extra unconditional pass)

Each parameter set gets its own sampling, oracle assignment, comparison, and
plot_evaluation outputs. A summary CSV and decision report are written at the
sweep root.
The same base seed is used for every parameter set, so target frames and sample
seeds are paired across the sweep.

Options:
  --analysis-only       Re-run comparisons, plots, and the sweep summary using
                        samples and oracle assignments that already exist.
  --requested-grid-only Skip the two additional control combinations.
  --dry-run             Print commands without executing them.
  -h, --help            Show this help message.

Environment overrides:
  DEVICE, CHECKPOINT_PATH, OUTPUT_ROOT, BASE_SEED, PYTHON_BIN, REFERENCE_PDB

Sampling is resumable: a completed parameter set has a .sampling_complete
marker. Existing oracle assignments are skipped automatically.
EOF
}

ANALYSIS_ONLY=false
INCLUDE_EXTRA_COMBINATIONS=true
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --analysis-only)
            ANALYSIS_ONLY=true
            shift
            ;;
        --requested-grid-only)
            INCLUDE_EXTRA_COMBINATIONS=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:3}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/storage_common/nobilm/ml-simplefold/fine_tune_with_clusters/finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged/checkpoints/last.ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/storage_common/nobilm/ml-simplefold/fine_tune_with_clusters/finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged/postcontinue/tau_guidance_sweep_all_structures_N100}"
BASE_SEED="${BASE_SEED:-123}"
REFERENCE_PDB="${REFERENCE_PDB:-${REPO_ROOT}/data/pdb_for_sampling_jupyter/INApo_no_caps.pdb}"

readonly TYPE="all_structures"
readonly RAW_NPZ_PATH="/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/ANECAG_THEO_INZMA_INACTIVE_merged_with_globalclusters.npz"
readonly N=100

TAU_VALUES=(0.1 0.3 0.5 0.7 0.9)
GUIDANCE_SCALE_VALUES=(2 3 5 8 12)
EXTRA_COMBINATIONS=(
    "0.0 3"
    "0.3 1"
)

run_command() {
    printf ' +'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == false ]]; then
        "$@"
    fi
}

count_outputs() {
    local directory="$1"
    local pattern="$2"
    find "${directory}" -maxdepth 1 -type f -name "${pattern}" -printf '\n' | wc -l
}

require_count() {
    local directory="$1"
    local pattern="$2"
    local description="$3"
    local observed
    observed="$(count_outputs "${directory}" "${pattern}")"
    if [[ "${observed}" -ne "${N}" ]]; then
        echo "Expected ${N} ${description} under ${directory}, found ${observed}." >&2
        return 1
    fi
}

parameter_slug() {
    local value="$1"
    printf '%s' "${value//./p}"
}

cd "${REPO_ROOT}"

if [[ "${DRY_RUN}" == false ]]; then
    mkdir -p "${OUTPUT_ROOT}"
    if [[ "${ANALYSIS_ONLY}" == false ]]; then
        [[ -f "${CHECKPOINT_PATH}" ]] || {
            echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
            exit 1
        }
        [[ -f "${RAW_NPZ_PATH}" ]] || {
            echo "Raw NPZ not found: ${RAW_NPZ_PATH}" >&2
            exit 1
        }
    fi
    [[ -f "${REFERENCE_PDB}" ]] || {
        echo "Reference PDB not found: ${REFERENCE_PDB}" >&2
        exit 1
    }
fi

MANIFEST_PATH="${OUTPUT_ROOT}/tau_guidance_sweep_manifest.tsv"
if [[ "${DRY_RUN}" == false ]]; then
    printf 'tau\tguidance_scale\tis_extra\toutput_dir\n' > "${MANIFEST_PATH}"
fi

run_parameter_set() {
    local tau="$1"
    local guidance_scale="$2"
    local is_extra="$3"
    local tau_slug
    local guidance_slug
    local parameter_dir
    local type_output_dir
    local sampling_marker

    tau_slug="$(parameter_slug "${tau}")"
    guidance_slug="$(parameter_slug "${guidance_scale}")"
    parameter_dir="${OUTPUT_ROOT}/tau_${tau_slug}_guidance_${guidance_slug}"
    type_output_dir="${parameter_dir}/${TYPE}_samples"
    sampling_marker="${parameter_dir}/.sampling_complete"

    echo
    echo "=== tau=${tau}, guidance-scale=${guidance_scale}, N=${N} ==="

    if [[ "${DRY_RUN}" == false ]]; then
        mkdir -p "${type_output_dir}"
        printf '%s\t%s\t%s\t%s\n' \
            "${tau}" "${guidance_scale}" "${is_extra}" "${type_output_dir}" \
            >> "${MANIFEST_PATH}"
    fi

    if [[ "${ANALYSIS_ONLY}" == false ]]; then
        if [[ -f "${sampling_marker}" ]]; then
            echo "Sampling marker found; validating and reusing completed samples."
            require_count \
                "${type_output_dir}" \
                '*_conditioned_eval.npz' \
                "conditioned-evaluation NPZ files"
        else
            run_command \
                "${PYTHON_BIN}" scripts/sample_with_conditioning.py \
                --seed "${BASE_SEED}" \
                -N "${N}" \
                --checkpoint-path "${CHECKPOINT_PATH}" \
                --raw-npz-path "${RAW_NPZ_PATH}" \
                --output-dir "${type_output_dir}" \
                --device "${DEVICE}" \
                --guidance-scale "${guidance_scale}" \
                --tau "${tau}"

            if [[ "${DRY_RUN}" == false ]]; then
                require_count \
                    "${type_output_dir}" \
                    '*_conditioned_eval.npz' \
                    "conditioned-evaluation NPZ files"
                printf 'tau=%s\nguidance_scale=%s\nN=%s\nbase_seed=%s\n' \
                    "${tau}" "${guidance_scale}" "${N}" "${BASE_SEED}" \
                    > "${sampling_marker}"
            fi
        fi

        run_command \
            "${PYTHON_BIN}" scripts/assign_conditioned_eval_sample_clusters.py \
            --base-path "${type_output_dir}" \
            --skip-existing
    fi

    if [[ "${DRY_RUN}" == false ]]; then
        require_count \
            "${type_output_dir}" \
            '*_assigned_clusters.npz' \
            "oracle-assigned cluster NPZ files"
    fi

    run_command \
        "${PYTHON_BIN}" scripts/compare_conditioning_to_oracle.py \
        --base-path "${type_output_dir}" \
        --out-dir "${type_output_dir}" \
        --reference-pdb "${REFERENCE_PDB}"

    run_command \
        "${PYTHON_BIN}" plot_evaluation.py \
        --base_path "${type_output_dir}" \
        --out_dir "${type_output_dir}"
}

for tau in "${TAU_VALUES[@]}"; do
    for guidance_scale in "${GUIDANCE_SCALE_VALUES[@]}"; do
        run_parameter_set "${tau}" "${guidance_scale}" false
    done
done

if [[ "${INCLUDE_EXTRA_COMBINATIONS}" == true ]]; then
    for combination in "${EXTRA_COMBINATIONS[@]}"; do
        read -r tau guidance_scale <<< "${combination}"
        run_parameter_set "${tau}" "${guidance_scale}" true
    done
fi

if [[ "${DRY_RUN}" == false ]]; then
    run_command \
        "${PYTHON_BIN}" scripts/summarize_tau_guidance_sweep.py \
        --manifest "${MANIFEST_PATH}" \
        --out-dir "${OUTPUT_ROOT}"

    echo
    echo "Sweep complete."
    echo "Summary CSV: ${OUTPUT_ROOT}/tau_guidance_sweep_summary.csv"
    echo "Decision report: ${OUTPUT_ROOT}/tau_guidance_sweep_report.png"
else
    echo
    echo "Dry run complete."
fi
