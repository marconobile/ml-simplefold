#!/usr/bin/env python3
"""Preprocess POTs sampling output for conditioning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_BACKMAPPING_NPZ = Path(
    "/storage_common/nobilm/backmapping_pots_model/datasets/pas/with_hs/"
    "without_hs/backmapping_dataset.npz"
)
DEFAULT_POTS_SAMPLE_NPZ = Path(
    "/storage_common/angiod/phase-data/projects/a2a/systems/a2a/clusters/"
    "cb3c3cae-5316-47db-8fbb-0567d5f0f75b/samples/"
    "a1d26e1b-cf9e-43af-a7f6-b66bd026505a/sample.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add global cluster IDs to a POTs sample NPZ so it can be used for "
            "conditioning."
        )
    )
    parser.add_argument(
        "--npz-path-pots-samples",
        "--npz_path_pots_samples",
        dest="npz_path_pots_samples",
        type=Path,
        default=DEFAULT_POTS_SAMPLE_NPZ,
        help="Path to the POTs sample NPZ containing labels_assigned.",
    )
    parser.add_argument(
        "--save-path",
        "--save_path",
        dest="save_path",
        type=Path,
        required=True,
        help="Path where the preprocessed NPZ should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # A .npz with local clustering id and residue_cluster_counts.
    data = dict(np.load(DEFAULT_BACKMAPPING_NPZ))

    # Fix residue indices.
    new_atom_resids = data["atom_resids"] - 2
    data["atom_resids"] = new_atom_resids

    res_idx = list(set(data["atom_residue_index"].tolist()))
    res_idx.sort()
    data["res_idx"] = res_idx

    # Build the mapping to global_id.
    residue_cluster_counts = list(zip(data["res_idx"], data["residue_cluster_counts"]))
    res_idx_to_glob_cluster_ids = {}
    global_id = 0
    for res_id, clu_count in residue_cluster_counts:
        for clu_global_idx in range(clu_count):
            res_idx_to_glob_cluster_ids[(res_id, clu_global_idx)] = global_id
            global_id += 1

    new_samples = dict(np.load(args.npz_path_pots_samples, allow_pickle=True))
    
    _labels_key = None
    if new_samples.get('labels_assigned') is not None:
        _labels_key = 'labels_assigned'
    elif new_samples.get('labels') is not None:
        if _labels_key is not None: raise ValueError("Both 'labels_assigned' and 'labels' found in the POTs sample NPZ. This is unexpected.")
        _labels_key = 'labels'
    else:
        raise ValueError("Neither 'labels_assigned' nor 'labels' found in the POTs sample NPZ.")
    
    res_idx_and_glob_cluster_id = []
    for frame in new_samples[_labels_key]:
        for_this_frame = []
        for res_idx, cluster_id in enumerate(frame):
            glob_cluster_id = res_idx_to_glob_cluster_ids[(res_idx, cluster_id)]
            for_this_frame.append(glob_cluster_id)
        res_idx_and_glob_cluster_id.append(for_this_frame)

    new_samples["res_idx_and_glob_cluster_id_per_frame"] = np.array(
        res_idx_and_glob_cluster_id
    )
    new_samples["atom_idx_and_glob_cluster_id_per_frame"] = new_samples[
        "res_idx_and_glob_cluster_id_per_frame"
    ][:, new_atom_resids]
    new_samples["og_npz_path"] = str(args.npz_path_pots_samples)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.save_path, **new_samples)


if __name__ == "__main__":
    main()
