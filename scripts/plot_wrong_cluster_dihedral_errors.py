#!/usr/bin/env python3
"""Plot dihedral errors only at residues with an oracle cluster mismatch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simplefold.conditioned_sampling.outputs import (  # noqa: E402
    write_dihedral_error_histogram_png,
)
from simplefold.conditioned_sampling.pdb_conversion import (  # noqa: E402
    read_pdb_residue_names,
)


ASSIGNED_TOKEN = "_assigned_clusters.npz"
CONDITIONED_EVAL_TOKEN = "_conditioned_eval.npz"
METRICS_TOKEN = "_conditioned_eval.json"
TARGET_PDB_TOKEN = "_conditioned_eval_target_reference.pdb"
OUTPUT_TOKEN = "_conditioned_eval_wrong_cluster_dihedral_error_histograms.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find assigned conditioned-evaluation samples and plot symmetry-corrected "
            "dihedral errors only for residues whose conditioning and oracle cluster "
            "labels differ."
        )
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Directory or *_assigned_clusters.npz file to process.",
    )
    parser.add_argument(
        "--error-bins",
        type=int,
        default=72,
        help="Number of histogram bins. Defaults to 72.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not replace an existing mismatch-only PNG.",
    )
    return parser.parse_args()


def one_dimensional(array: np.ndarray, name: str, path: Path) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"{name} for {path} must be 1D, got shape {array.shape}.")
    return array


def optional_npz_array(path: Path, *keys: str) -> np.ndarray | None:
    with np.load(path, allow_pickle=False) as data:
        for key in keys:
            if key in data.files:
                return np.asarray(data[key])
    return None


def required_npz_array(path: Path, *keys: str) -> np.ndarray:
    value = optional_npz_array(path, *keys)
    if value is None:
        raise KeyError(f"{path} is missing all expected keys: {', '.join(keys)}.")
    return value


def companion_path(assigned_path: Path, token: str) -> Path:
    if not assigned_path.name.endswith(ASSIGNED_TOKEN):
        raise ValueError(f"Unexpected assigned-clusters filename: {assigned_path}")
    return assigned_path.with_name(
        assigned_path.name.replace(ASSIGNED_TOKEN, token, 1)
    )


def find_assigned_paths(base_path: Path) -> list[Path]:
    if base_path.is_file():
        return [base_path] if base_path.name.endswith(ASSIGNED_TOKEN) else []
    return sorted(base_path.rglob(f"*{ASSIGNED_TOKEN}"))


def load_atom_resids(conditioned_eval_path: Path, metrics_path: Path) -> np.ndarray:
    atom_resids = optional_npz_array(conditioned_eval_path, "atom_resids")
    if atom_resids is not None:
        return one_dimensional(atom_resids, "atom_resids", conditioned_eval_path).astype(
            np.int64,
            copy=False,
        )

    if metrics_path.is_file():
        with metrics_path.open() as handle:
            raw_npz_path = json.load(handle).get("raw_npz_path")
        if raw_npz_path:
            raw_npz_path = Path(raw_npz_path).expanduser()
            if raw_npz_path.is_file():
                atom_resids = optional_npz_array(raw_npz_path, "atom_resids")
                if atom_resids is not None:
                    return one_dimensional(
                        atom_resids,
                        "atom_resids",
                        raw_npz_path,
                    ).astype(np.int64, copy=False)

    raise KeyError(
        f"Could not load atom_resids from {conditioned_eval_path} or its raw NPZ."
    )


def labels_to_residue_global(
    atom_global_labels: np.ndarray,
    atom_resids: np.ndarray,
    n_residues: int,
    path: Path,
) -> np.ndarray:
    atom_global_labels = one_dimensional(
        atom_global_labels,
        "conditioning labels",
        path,
    ).astype(np.int64, copy=False)
    if atom_global_labels.shape != atom_resids.shape:
        raise ValueError(
            f"Conditioning label/atom_resids shape mismatch for {path}: "
            f"{atom_global_labels.shape} vs {atom_resids.shape}."
        )
    observed_residues = np.unique(atom_resids)
    expected_residues = np.arange(n_residues, dtype=np.int64)
    if not np.array_equal(observed_residues, expected_residues):
        raise ValueError(
            f"atom_resids for {path} must cover exactly 0..{n_residues - 1}; "
            f"observed {observed_residues[:20].tolist()}."
        )

    residue_global = np.full(n_residues, -1, dtype=np.int64)
    for residue_idx in range(n_residues):
        labels = atom_global_labels[atom_resids == residue_idx]
        labels = labels[labels >= 0]
        if labels.size == 0:
            continue
        unique = np.unique(labels)
        if unique.size != 1:
            raise ValueError(
                f"Conditioning labels are inconsistent at residue {residue_idx} "
                f"in {path}: {unique.tolist()}."
            )
        residue_global[residue_idx] = int(unique[0])
    return residue_global


def global_to_local_labels(
    residue_global: np.ndarray,
    cluster_counts: np.ndarray,
    path: Path,
) -> np.ndarray:
    cluster_counts = one_dimensional(cluster_counts, "cluster_counts", path).astype(
        np.int64,
        copy=False,
    )
    if residue_global.shape != cluster_counts.shape:
        raise ValueError(
            f"Residue label/cluster-count shape mismatch for {path}: "
            f"{residue_global.shape} vs {cluster_counts.shape}."
        )
    if np.any(cluster_counts <= 0):
        raise ValueError(f"Cluster counts must all be positive in {path}.")

    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(cluster_counts[:-1], dtype=np.int64))
    )
    local = residue_global - offsets
    valid = residue_global >= 0
    invalid = valid & ((local < 0) | (local >= cluster_counts))
    if np.any(invalid):
        raise ValueError(
            f"Invalid global-to-local labels in {path} at residues "
            f"{np.flatnonzero(invalid)[:20].tolist()}."
        )
    local[~valid] = -1
    return local


def load_residue_names(conditioned_eval_path: Path, target_pdb_path: Path) -> np.ndarray:
    residue_names = optional_npz_array(conditioned_eval_path, "residue_names")
    if residue_names is not None:
        return one_dimensional(
            residue_names,
            "residue_names",
            conditioned_eval_path,
        ).astype(str)
    if not target_pdb_path.is_file():
        raise FileNotFoundError(
            f"Neither stored residue_names nor target-reference PDB is available: "
            f"{target_pdb_path}"
        )
    return read_pdb_residue_names(target_pdb_path)


def plot_assigned_sample(assigned_path: Path, error_bins: int) -> tuple[Path, int]:
    conditioned_eval_path = companion_path(assigned_path, CONDITIONED_EVAL_TOKEN)
    metrics_path = companion_path(assigned_path, METRICS_TOKEN)
    target_pdb_path = companion_path(assigned_path, TARGET_PDB_TOKEN)
    output_path = companion_path(assigned_path, OUTPUT_TOKEN)
    if not conditioned_eval_path.is_file():
        raise FileNotFoundError(
            f"Conditioned-evaluation arrays not found for {assigned_path}: "
            f"{conditioned_eval_path}"
        )

    dihedral_diff_rad = required_npz_array(conditioned_eval_path, "dihedral_diff_rad")
    dihedral_mask = required_npz_array(conditioned_eval_path, "dihedral_mask").astype(
        bool,
        copy=False,
    )
    dihedral_keys = one_dimensional(
        required_npz_array(conditioned_eval_path, "dihedral_keys"),
        "dihedral_keys",
        conditioned_eval_path,
    ).astype(str).tolist()
    if dihedral_diff_rad.ndim != 2 or dihedral_mask.shape != dihedral_diff_rad.shape:
        raise ValueError(
            f"Invalid dihedral array shapes in {conditioned_eval_path}: "
            f"diff={dihedral_diff_rad.shape}, mask={dihedral_mask.shape}."
        )
    n_residues = dihedral_diff_rad.shape[0]

    atom_resids = load_atom_resids(conditioned_eval_path, metrics_path)
    conditioning_global = required_npz_array(
        conditioned_eval_path,
        "conditioning_cluster_labels",
        "original_cluster_labels",
    )
    expected_local = global_to_local_labels(
        labels_to_residue_global(
            conditioning_global,
            atom_resids,
            n_residues,
            conditioned_eval_path,
        ),
        required_npz_array(assigned_path, "cluster_counts", "merged__cluster_counts"),
        assigned_path,
    )
    oracle_labels = one_dimensional(
        required_npz_array(assigned_path, "labels_assigned", "merged__labels_assigned"),
        "labels_assigned",
        assigned_path,
    ).astype(np.int64, copy=False)
    if oracle_labels.shape != expected_local.shape:
        raise ValueError(
            f"Conditioning/oracle label shape mismatch for {assigned_path}: "
            f"{expected_local.shape} vs {oracle_labels.shape}."
        )

    comparable = (expected_local >= 0) & (oracle_labels >= 0)
    mismatches = comparable & (expected_local != oracle_labels)
    residue_names = load_residue_names(conditioned_eval_path, target_pdb_path)
    if residue_names.shape != (n_residues,):
        raise ValueError(
            f"Residue-name shape mismatch for {conditioned_eval_path}: "
            f"{residue_names.shape} vs {(n_residues,)}."
        )

    write_dihedral_error_histogram_png(
        output_path=output_path,
        dihedral_diff_rad=dihedral_diff_rad,
        dihedral_mask=dihedral_mask,
        dihedral_keys=dihedral_keys,
        residue_names=residue_names,
        error_bins=error_bins,
        residue_selection_mask=mismatches,
        figure_title=(
            "Dihedral errors at residues with wrong assigned clusters "
            f"({int(mismatches.sum())}/{int(comparable.sum())} residues)"
        ),
    )
    return output_path, int(mismatches.sum())


def main() -> None:
    args = parse_args()
    if args.error_bins <= 0:
        raise ValueError("--error-bins must be positive.")
    base_path = args.base_path.expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base path not found: {base_path}")
    assigned_paths = find_assigned_paths(base_path)
    if not assigned_paths:
        print(f"No *{ASSIGNED_TOKEN} files found under {base_path}.")
        return

    written = 0
    skipped = 0
    for assigned_path in assigned_paths:
        output_path = companion_path(assigned_path, OUTPUT_TOKEN)
        if args.skip_existing and output_path.is_file():
            skipped += 1
            print(f"Skipping existing output: {output_path}")
            continue
        output_path, mismatch_count = plot_assigned_sample(
            assigned_path,
            args.error_bins,
        )
        written += 1
        print(
            f"Wrote mismatch-only dihedral error histogram "
            f"({mismatch_count} residues): {output_path}"
        )

    print(f"Done. Wrote {written} image(s), skipped {skipped} existing image(s).")


if __name__ == "__main__":
    main()
