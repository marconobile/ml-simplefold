#!/usr/bin/env python3
"""Shared constants and import-path bootstrap for conditioned sampling."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
SIMPLEFOLD_ROOT = SRC_ROOT / "simplefold"

for import_path in (SRC_ROOT, SIMPLEFOLD_ROOT):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

DEFAULT_DATA_PATH = Path("/scratch/nobilm/quantum_backmapping/training_data_active_npz")
DEFAULT_RAW_NPZ_CANDIDATES = (
    REPO_ROOT / "test_new_data_with_clusters" / "active_without_hs.npz",
    REPO_ROOT / "traj_with_cluster_labels.npz",
)
DEFAULT_CHECKPOINT_DIR = Path(
    "/storage_common/nobilm/ml-simplefold/"
    "fine_tune_with_clusters/inapo_ft_active_npz_from_simplefold100M_gpu0/checkpoints"
)
DEFAULT_CONDITIONED_EVAL_PDB_BASE_PATH = Path(
    "/storage_common/nobilm/backmapping_pots_model/results"
)
CLUSTER_KEY = "atom_idx_and_glob_cluster_id_per_frame"
CONDITIONED_EVAL_SAMPLED_CIF_TOKEN = "_conditioned_eval_sampled.cif"
PDB_ATOM_RECORDS = ("ATOM", "HETATM")
