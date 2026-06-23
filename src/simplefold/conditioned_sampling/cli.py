#!/usr/bin/env python3
"""Utilities extracted from scripts/sample_with_conditioning.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CONDITIONED_EVAL_PDB_BASE_PATH,
    REPO_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample one or more cluster-conditioned SimpleFold structures with "
            "the fine-tuned checkpoint. Without --labels-npz-path, each sample "
            "uses cluster labels from a raw or processed observation. With "
            "--labels-npz-path, each sample uses one provided label row and "
            "skips original-frame coordinate/dihedral evaluation."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "Raw trajectory NPZ, or the processed SimpleFold directory. The requested "
            "default is the processed directory at /scratch/nobilm/quantum_backmapping/"
            "training_data_active_npz."
        ),
    )
    parser.add_argument(
        "--raw-npz-path",
        type=Path,
        default=None,
        help=(
            "Raw trajectory NPZ containing `trajectory`, `dihedrals`, and cluster labels. "
            "Required only when --data-path points to a processed SimpleFold directory "
            "and auto-discovery does not find the raw NPZ."
        ),
    )
    parser.add_argument(
        "--labels-npz-path",
        type=Path,
        default=None,
        help=(
            "Optional NPZ containing `atom_idx_and_glob_cluster_id_per_frame` "
            "with shape (n_samples, n_model_atoms). When provided, one conditioned "
            "sample is generated for each row using the row exactly, and "
            "original-structure coordinate/dihedral evaluation is skipped."
        ),
    )
    parser.add_argument(
        "-N",
        "--num-samples",
        "--num-label-samples",
        dest="num_samples",
        type=int,
        default=None,
        help=(
            "Number of conditioned structures to generate. Without --labels-npz-path, "
            "this samples N random observations from the raw NPZ or processed input. "
            "With --labels-npz-path, this samples the first N label rows. Defaults "
            "to 1 without labels and all rows with labels."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help=(
            "Processed SimpleFold directory containing structures/, records/, and "
            "optionally tokens/. Defaults to --data-path when it is a directory."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing last.ckpt.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to --checkpoint-dir/last.ckpt.",
    )
    parser.add_argument(
        "--architecture-config",
        type=Path,
        default=REPO_ROOT / "configs/model/architecture/foldingdit_100M.yaml",
        help="Hydra YAML config for the FoldingDiT architecture.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/active_npz_conditioned_eval",
        help="Directory for JSON, NPZ, and CSV evaluation outputs.",
    )
    parser.add_argument(
        "--conditioned-eval-pdb-base-path",
        type=Path,
        default=DEFAULT_CONDITIONED_EVAL_PDB_BASE_PATH,
        help=(
            "Base path passed to scripts/convert_conditioned_eval_cifs_to_pdb.py "
            "before dihedral evaluation when --output-dir is inside this tree. "
            "Otherwise, only --output-dir is converted."
        ),
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help=(
            "Trajectory frame position to evaluate. If omitted, a random frame is "
            "sampled from the raw NPZ or processed manifest."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for frame selection and sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to the visible CUDA device with most free memory, otherwise cpu.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=500,
        help="Number of Euler-Maruyama sampling steps.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.3,
        help="EMSampler tau value.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=16.0,
        help="Coordinate scale used by ProteinDataProcessor.",
    )
    parser.add_argument(
        "--ref-scale",
        type=float,
        default=5.0,
        help="Reference-position scale used by ProteinDataProcessor.",
    )
    parser.add_argument(
        "--esm-model",
        type=str,
        default="esm2_3B",
        help="ESM model name used by the trained SimpleFold model.",
    )
    parser.add_argument(
        "--ref-pos-mode",
        choices=("input", "zero"),
        default="input",
        help=(
            "How to pass ref_pos into FoldingDiT. Use `zero` with checkpoints "
            "fine-tuned using model.processor.ref_pos_mode=zero to avoid leaking "
            "template-frame geometry into label-conditioned sampling."
        ),
    )
    parser.add_argument(
        "--use-non-ema-weights",
        action="store_true",
        help="Load `model.` weights instead of `model_ema.module.` weights when both exist.",
    )
    parser.add_argument(
        "--no-mmap-checkpoint",
        action="store_true",
        help="Disable torch.load(..., mmap=True) for the checkpoint.",
    )
    parser.add_argument(
        "--dihedral-angle-bins",
        type=int,
        default=72,
        help="Number of bins for dihedral angle histograms (degrees).",
    )
    parser.add_argument(
        "--dihedral-error-bins",
        type=int,
        default=72,
        help="Number of bins for dihedral error histograms (degrees).",
    )
    return parser.parse_args()
