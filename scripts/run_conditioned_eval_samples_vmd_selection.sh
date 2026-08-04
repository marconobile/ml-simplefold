#!/usr/bin/env bash
set -euo pipefail

# Run the same workflow as run_conditioned_eval_samples.sh. The primary
# conditioning-vs-oracle comparison uses this VMD selection by default:
#
#   name CA and resid 2 to 30 35 to 65 69 to 104 113 to 138 169 to 209 215 to 255 261 to 287 288 to 300
#
# Every run also evaluates and plots sampled-PDB-vs-target-PDB RMSD histograms for the
# strict VMD CA selection, all CA atoms, backbone atoms, protein non-backbone
# atoms, and all atoms. Pass --all-res to use every residue for the primary
# cluster-label comparison.

ALL_RES=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all-res)
            ALL_RES=true
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/run_conditioned_eval_samples_vmd_selection.sh [--all-res]

By default, conditioning-vs-oracle evaluation is restricted exactly to:
  name CA and resid 2 to 30 35 to 65 69 to 104 113 to 138 169 to 209 215 to 255 261 to 287 288 to 300

Every run computes sampled-PDB-vs-target-PDB RMSD after an independent fit and
writes a separate histogram for each of these selections:
  1. The strict VMD CA/resid selection above
  2. All CA atoms
  3. Backbone atoms (N, CA, C, O)
  4. Protein and not backbone
  5. All ATOM/HETATM records

Options:
  --all-res  Use all residues for the primary cluster-label comparison.
  -h, --help Show this help message.
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Use --help for usage." >&2
            exit 2
            ;;
    esac
done

# check for changes
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/storage_common/nobilm/ml-simplefold/fine_tune_with_clusters/finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged/checkpoints/last.ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/storage_common/nobilm/ml-simplefold/fine_tune_with_clusters/finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged/postcontinue/deleteme}" # samples_cfg3_aip_selection_v2

# fixed
N="${N:-5}"
BASE_SEED="${BASE_SEED:-123}"
CONDA_ENV="${CONDA_ENV:-simplefold}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -----

#! COMPARING AGAINST REFERENCE LABELS

LABELS_NPZ_PATH="${LABELS_NPZ_PATH:-}" # none if comparing against reference
echo "LABELS_NPZ_PATH=${LABELS_NPZ_PATH}"

#! aggregated data

# TYPE="all_structures"
# RAW_NPZ_PATH="/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/ANECAG_THEO_INZMA_INACTIVE_merged_with_globalclusters.npz"

#! splitted data
TYPE='anecag'
RAW_NPZ_PATH='/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/split_per_simulation_type/anecag_4000_noh_global_clusters.npz'

# TYPE='inactive'
# RAW_NPZ_PATH='/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/split_per_simulation_type/inactive_4000_noh_global_clusters.npz'

# TYPE='inzma'
# RAW_NPZ_PATH='/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/split_per_simulation_type/inzma_4000_noh_global_clusters.npz'

# TYPE='theo'
# RAW_NPZ_PATH='/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/split_per_simulation_type/theo_4000_noh_global_clusters.npz'

# -----

#! FOR DENOVO
# TYPE='denovo'
#* here we can use directly:
# RAW_NPZ_PATH="/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/ANECAG_THEO_INZMA_INACTIVE_merged_with_globalclusters.npz"
# LABELS_NPZ_PATH="/storage_common/nobilm/backmapping_pots_model/pots_samples/sample.npz"

# -----

comparison_scope_args=()
if [[ "${ALL_RES}" == true ]]; then
    comparison_scope_args+=(--all-res)
fi
cd "${REPO_ROOT}"

# -----

TYPE_OUTPUT_DIR="${OUTPUT_ROOT}/${TYPE}_samples"
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
    # --tau 0.1
)
if [[ -n "${LABELS_NPZ_PATH}" ]]; then
    cmd+=(
        --labels-npz-path "${LABELS_NPZ_PATH}"
    )
fi
"${cmd[@]}"

echo "Assigning oracle clusters for TYPE=${TYPE}"
python scripts/assign_conditioned_eval_sample_clusters.py --base-path "${TYPE_OUTPUT_DIR}" #! assign with oracle 

echo "Comparing original conditioning vs oracle labels for TYPE=${TYPE}"
python scripts/compare_conditioning_to_oracle.py \
    --base-path "${TYPE_OUTPUT_DIR}" \
    --out-dir "${TYPE_OUTPUT_DIR}" \
    "${comparison_scope_args[@]}"

echo "Plotting assigned-cluster evaluation across all structure types"
python plot_evaluation.py --base_path "${TYPE_OUTPUT_DIR}" --out_dir "${TYPE_OUTPUT_DIR}"
