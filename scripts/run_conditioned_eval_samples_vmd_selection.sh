#!/usr/bin/env bash
set -euo pipefail

# Run the same workflow as run_conditioned_eval_samples.sh. The primary
# conditioning-vs-oracle comparison uses this VMD selection by default:
#
#   name CA and resid 2 to 30 35 to 65 69 to 104 113 to 138 169 to 209 215 to 255 261 to 287 288 to 300
#
# Every run also evaluates and plots sampled-PDB-vs-target-PDB RMSD violins for the
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
writes a separate violin subplot for each of these selections:
  1. The strict VMD CA/resid selection above
  2. All CA atoms
  3. Backbone atoms (N, CA, C, O)
  4. Protein and not backbone
  5. All ATOM/HETATM records

Each violin title also reports the average sampled-structure RMSD against:
  /home/nobilm@usi.ch/ml-simplefold/data/pdb_for_sampling_jupyter/INApo_no_caps.pdb

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

# -----
BASE_SEED="${BASE_SEED:-123}"
CONDA_ENV="${CONDA_ENV:-simplefold}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABELS_NPZ_PATH="${LABELS_NPZ_PATH:-}" # none if comparing against reference
echo "LABELS_NPZ_PATH=${LABELS_NPZ_PATH}"
# -----
#! ##################
#! --- START HERE ---
#! ##################

#! 1 -> (LEAVE AS IS) select Checkpoint path
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/storage_common/nobilm/ml-simplefold/fine_tune_with_clusters/finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged/checkpoints/last.ckpt}"

#! 1.1 -> select device
DEVICE="${DEVICE:-cuda:2}"

#! 2 -> select OUTPUT_ROOT
OUTPUT_ROOT="${OUTPUT_ROOT:-/storage_common/nobilm/ml-simplefold/fine_tune_with_clusters/finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged/TEST_DELETE_ME}"

#! 3 -> select TYPE / RAW_NPZ_PATH / N
#* aggregated data
# TYPE="all_structures"
# RAW_NPZ_PATH="/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/ANECAG_THEO_INZMA_INACTIVE_merged_with_globalclusters.npz"
# N="${N:-200}"

#* splitted TRAIN data
# TYPE='anecag'
# RAW_NPZ_PATH='/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/split_per_simulation_type/anecag_4000_noh_global_clusters.npz'
# N="${N:-200}"

# TYPE='inactive'
# RAW_NPZ_PATH='/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/split_per_simulation_type/inactive_4000_noh_global_clusters.npz'
# N="${N:-200}"

# TYPE='inzma'
# RAW_NPZ_PATH='/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/split_per_simulation_type/inzma_4000_noh_global_clusters.npz'
# N="${N:-200}"

# TYPE='theo'
# RAW_NPZ_PATH='/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/split_per_simulation_type/theo_4000_noh_global_clusters.npz'
# N="${N:-200}"

#*******

#* TEST data
# TYPE='ANECAG_test'
# RAW_NPZ_PATH='/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/split_per_simulation_type/test_sets_v0/ANECAG/without_hs/backmapping_dataset.npz'
# N="${N:-1000}"

# TYPE='INACTIVE_test'
# RAW_NPZ_PATH='/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/split_per_simulation_type/test_sets_v0/INACTIVE/without_hs/backmapping_dataset.npz'
# N="${N:--1}"

# TYPE='INZMA_test'
# RAW_NPZ_PATH='/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/split_per_simulation_type/test_sets_v0/INZMA/without_hs/backmapping_dataset.npz'
# N="${N:--1}"

# TYPE='THEO_test'
# RAW_NPZ_PATH='/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/split_per_simulation_type/test_sets_v0/THEO/without_hs/backmapping_dataset.npz'
# N="${N:--1}"

# TYPE='PAS_test'
# RAW_NPZ_PATH='/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/split_per_simulation_type/test_sets_v0/PAS/without_hs/backmapping_dataset.npz'
# N="${N:--1}"

#*******

#* IF FOR DENOVO
RAW_NPZ_PATH="/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/ANECAG_THEO_INZMA_INACTIVE_merged_with_globalclusters.npz" # anything is ok
TYPE='TEST_DELETE_ME'
# LABELS_NPZ_PATH="/storage_common/nobilm/backmapping_pots_model/pots_samples/sample.npz" # this must have global clusters
# INECA LABELS_NPZ_PATH="/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/split_per_simulation_type/test_sets_v0/INeca/without_hs/backmapping_dataset.npz" # this must have global clusters
LABELS_NPZ_PATH="/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/split_per_simulation_type/sample_from_pots/INzma/without_hs/backmapping_dataset.npz"
N="${N:-100}" # if -1 then comment -N in scripts/sample_with_conditioning.py


#! ###################
#! ------- END -------
#! ###################

































#! #################
#! # ACTUAL  LOGIC #
#! # DO NOT CHANGE #
#! #################

comparison_scope_args=()
if [[ "${ALL_RES}" == true ]]; then
    comparison_scope_args+=(--all-res)
fi
cd "${REPO_ROOT}"

# -----

TYPE_OUTPUT_DIR="${OUTPUT_ROOT}/${TYPE}_samples"
mkdir -p "${TYPE_OUTPUT_DIR}"
CHIRALITY_LOG_PATH="${OUTPUT_ROOT}/ensure_l_chirality.tsv"

echo "Running TYPE=${TYPE} N=${N} BASE_SEED=${BASE_SEED}"
echo "Output: ${TYPE_OUTPUT_DIR}"
echo "Chirality log: ${CHIRALITY_LOG_PATH}"

# LABELS_NPZ_PATH: optional. If set, Python samples N label rows.
# Otherwise, Python samples N random observations from RAW_NPZ_PATH.
cmd=(
    python scripts/sample_with_conditioning.py
    --seed "${BASE_SEED}" # BASE_SEED + sample_index, so 123, 124, 125, 126, 127 for N=5.
    -N "${N}"
    --checkpoint-path "${CHECKPOINT_PATH}"
    --raw-npz-path "${RAW_NPZ_PATH}"
    --output-dir "${TYPE_OUTPUT_DIR}"
    --chirality-log-path "${CHIRALITY_LOG_PATH}"
    --device "${DEVICE}"
    --guidance-scale 1.0
    --tau 0.01
)
if [[ -n "${LABELS_NPZ_PATH}" ]]; then
    cmd+=(
        --labels-npz-path "${LABELS_NPZ_PATH}"
    )
fi
"${cmd[@]}"

echo "Assigning oracle clusters_conditioned_eval_target_reference.pdb for TYPE=${TYPE}"
python scripts/assign_conditioned_eval_sample_clusters.py --base-path "${TYPE_OUTPUT_DIR}" #! assign with oracle

if [[ -z "${LABELS_NPZ_PATH}" ]]; then
    echo "Plotting dihedral errors for residues with wrong oracle clusters for TYPE=${TYPE}"
    python scripts/plot_wrong_cluster_dihedral_errors.py --base-path "${TYPE_OUTPUT_DIR}"
else
    echo "Skipping mismatch-only dihedral errors: labels-NPZ samples have no target dihedrals"
fi

echo "Comparing original conditioning vs oracle labels for TYPE=${TYPE}"
python scripts/compare_conditioning_to_oracle.py \
    --base-path "${TYPE_OUTPUT_DIR}" \
    --out-dir "${TYPE_OUTPUT_DIR}" \
    "${comparison_scope_args[@]}"

# this can be executed only for the last
# echo "Comparing original conditioning vs oracle labels across all available structure types"
# python scripts/compare_conditioning_to_oracle.py \
#     --base-path "${OUTPUT_ROOT}" \
#     --out-dir "${OUTPUT_ROOT}" \
#     "${comparison_scope_args[@]}"

echo "Plotting assigned-cluster evaluation across all available structure types"
python plot_evaluation.py --base_path "${OUTPUT_ROOT}" --out_dir "${OUTPUT_ROOT}"