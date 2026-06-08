#!/usr/bin/env python3
"""Compare oracle-assigned clusters against the conditioning used for sampling."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DEFAULT_OG_LABELS = Path(
    "/storage_common/nobilm/backmapping_pots_model/pots_samples/sample.npz"
)
MATCH_PATTERN = "*_assigned_clusters.npz"
IDX_RE = re.compile(r"(?:^|_)(\d+)_assigned_clusters\.npz$")
ASSIGNED_TOKEN = "_assigned_clusters.npz"
CONDITIONED_EVAL_TOKEN = "_conditioned_eval.npz"
CONDITIONED_EVAL_JSON_TOKEN = "_conditioned_eval.json"
CLUSTER_KEY = "atom_idx_and_glob_cluster_id_per_frame"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find *_assigned_clusters.npz files under a base directory, compare "
            "their labels_assigned rows to the labels that were actually used to "
            "condition the matching *_conditioned_eval.npz sample, and save a "
            "histogram of per-file label differences."
        )
    )
    parser.add_argument(
        "--base-path",
        "--base_path",
        dest="base_path",
        type=Path,
        required=True,
        help="Directory to recursively search for *_assigned_clusters.npz files.",
    )
    parser.add_argument(
        "--og-labels",
        "--og_labels",
        dest="og_labels",
        type=Path,
        default=DEFAULT_OG_LABELS,
        help=(
            "Original NPZ containing atom-level global conditioning labels under "
            f"{CLUSTER_KEY!r} and residue-level local labels under 'labels'."
        ),
    )
    parser.add_argument(
        "--output-name",
        "--output_name",
        dest="output_name",
        default="assigned_cluster_label_error_histogram.png",
        help="Histogram PNG filename written inside --base-path.",
    )
    return parser.parse_args()


def find_assigned_cluster_files(base_path: Path) -> list[Path]:
    return sorted(path for path in base_path.rglob(MATCH_PATTERN) if path.is_file())


def extract_index(path: Path) -> int:
    match = IDX_RE.search(path.name)
    if match is None:
        raise ValueError(
            f"Could not extract index from {path.name!r}. Expected a filename "
            "ending like 'following_label_0_assigned_clusters.npz'."
        )
    return int(match.group(1))


def conditioned_eval_path_for_assigned(path: Path) -> Path:
    if path.name.endswith(ASSIGNED_TOKEN):
        return path.with_name(path.name.replace(ASSIGNED_TOKEN, CONDITIONED_EVAL_TOKEN, 1))
    raise ValueError(f"Unexpected assigned-clusters filename: {path}")


def conditioned_eval_json_path_for_assigned(path: Path) -> Path:
    if path.name.endswith(ASSIGNED_TOKEN):
        return path.with_name(
            path.name.replace(ASSIGNED_TOKEN, CONDITIONED_EVAL_JSON_TOKEN, 1)
        )
    raise ValueError(f"Unexpected assigned-clusters filename: {path}")


def load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path) as data:
        if key not in data.files:
            available = ", ".join(data.files)
            raise KeyError(
                f"{path} is missing required key {key!r}. Available keys: {available}"
            )
        return np.asarray(data[key])


def one_dimensional_labels(labels: np.ndarray, label_name: str, path: Path) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[0] == 1:
        labels = labels[0]
    if labels.ndim != 1:
        raise ValueError(
            f"{label_name} for {path} must be 1D or a single-row 2D array, "
            f"got shape {labels.shape}."
        )
    return labels


def load_label_sample_index(json_path: Path) -> int | None:
    if not json_path.is_file():
        return None

    import json

    with json_path.open() as f:
        payload = json.load(f)
    value = payload.get("label_sample_index")
    return None if value is None else int(value)


def verified_conditioning_row_index(
    *,
    assigned_path: Path,
    conditioning_labels: np.ndarray,
    reference_atom_labels: np.ndarray,
) -> int:
    json_index = load_label_sample_index(conditioned_eval_json_path_for_assigned(assigned_path))
    file_index = extract_index(assigned_path)

    candidate_indices = []
    if json_index is not None:
        candidate_indices.append(("json label_sample_index", json_index))
    candidate_indices.append(("filename index", file_index))

    for source, index in candidate_indices:
        if 0 <= index < reference_atom_labels.shape[0]:
            row = one_dimensional_labels(
                reference_atom_labels[index],
                CLUSTER_KEY,
                assigned_path,
            )
            if row.shape == conditioning_labels.shape and np.array_equal(
                row,
                conditioning_labels,
            ):
                return index

    if reference_atom_labels.ndim != 2:
        raise ValueError(
            f"{CLUSTER_KEY!r} must be a 2D array indexed by sample/frame first, "
            f"got shape {reference_atom_labels.shape}."
        )
    if reference_atom_labels.shape[1] != conditioning_labels.shape[0]:
        raise ValueError(
            f"Cannot match conditioning labels from {conditioned_eval_path_for_assigned(assigned_path)}: "
            f"conditioning has shape {conditioning_labels.shape}, but "
            f"{CLUSTER_KEY!r} in --og-labels has row shape "
            f"{reference_atom_labels.shape[1:]}."
        )

    matches = np.flatnonzero(np.all(reference_atom_labels == conditioning_labels[None, :], axis=1))
    if matches.size == 1:
        return int(matches[0])
    if matches.size > 1:
        raise ValueError(
            f"Conditioning labels for {assigned_path} match multiple rows in --og-labels: "
            f"{matches[:20].tolist()}."
        )

    raise ValueError(
        f"Could not match saved conditioning labels from "
        f"{conditioned_eval_path_for_assigned(assigned_path)} to any row in --og-labels."
    )


def reference_local_labels_for_row(
    *,
    row_index: int,
    assigned_path: Path,
    reference_labels: np.ndarray | None,
    reference_residue_global_labels: np.ndarray | None,
) -> np.ndarray:
    labels = None
    if reference_labels is not None:
        labels = one_dimensional_labels(
            reference_labels[row_index],
            "labels",
            assigned_path,
        ).astype(np.int64, copy=False)

    if reference_residue_global_labels is not None:
        residue_global = one_dimensional_labels(
            reference_residue_global_labels[row_index],
            "res_idx_and_glob_cluster_id_per_frame",
            assigned_path,
        ).astype(np.int64, copy=False)
        cluster_counts = one_dimensional_labels(
            load_npz_array(assigned_path, "cluster_counts"),
            "cluster_counts",
            assigned_path,
        ).astype(np.int64, copy=False)
        if residue_global.shape == cluster_counts.shape:
            offsets = np.concatenate(
                [
                    np.zeros((1,), dtype=np.int64),
                    np.cumsum(cluster_counts[:-1], dtype=np.int64),
                ]
            )
            derived_labels = residue_global - offsets
            invalid = (derived_labels < 0) | (derived_labels >= cluster_counts)
            if np.any(invalid):
                bad = np.flatnonzero(invalid)[:10].tolist()
                raise ValueError(
                    f"Global-to-local label conversion produced invalid labels for "
                    f"{assigned_path} at residues {bad}."
                )
            if labels is None:
                labels = derived_labels
            elif not np.array_equal(labels, derived_labels):
                mismatch = np.flatnonzero(labels != derived_labels)[:10].tolist()
                raise ValueError(
                    f"'labels' and global-to-local labels disagree for row {row_index} "
                    f"at residues {mismatch}."
                )

    if labels is None:
        raise KeyError(
            "--og-labels must contain either 'labels' or "
            "'res_idx_and_glob_cluster_id_per_frame' plus assigned-file cluster_counts."
        )
    return labels


def integer_bins(values: list[int]) -> np.ndarray:
    min_value = min(values)
    max_value = max(values)
    return np.arange(min_value - 0.5, max_value + 1.5, 1)


def save_error_histogram(error_counts: list[int], output_path: Path) -> None:
    fig_width = max(8.0, (max(error_counts) - min(error_counts) + 1) * 0.18)
    fig, ax = plt.subplots(figsize=(fig_width, 5), constrained_layout=True)
    ax.hist(error_counts, bins=integer_bins(error_counts), edgecolor="black")
    ax.set_title("Assigned Cluster Label Errors")
    ax.set_xlabel("Number of differing labels")
    ax.set_ylabel("Number of files")
    ax.set_xlim(min(error_counts) - 0.5, max(error_counts) + 0.5)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_path = args.base_path.expanduser().resolve()
    og_labels_path = args.og_labels.expanduser().resolve()

    if not base_path.is_dir():
        raise NotADirectoryError(f"Base path is not a directory: {base_path}")
    if not og_labels_path.is_file():
        raise FileNotFoundError(f"Original labels NPZ not found: {og_labels_path}")

    assigned_cluster_files = find_assigned_cluster_files(base_path)
    if not assigned_cluster_files:
        raise FileNotFoundError(
            f"No files matching {MATCH_PATTERN!r} were found under {base_path}"
        )

    with np.load(og_labels_path) as og_data:
        if CLUSTER_KEY not in og_data.files:
            available = ", ".join(og_data.files)
            raise KeyError(
                f"{og_labels_path} is missing required key {CLUSTER_KEY!r}. "
                f"Available keys: {available}"
            )
        reference_atom_labels = np.asarray(og_data[CLUSTER_KEY])
        reference_labels = (
            np.asarray(og_data["labels"]) if "labels" in og_data.files else None
        )
        reference_residue_global_labels = (
            np.asarray(og_data["res_idx_and_glob_cluster_id_per_frame"])
            if "res_idx_and_glob_cluster_id_per_frame" in og_data.files
            else None
        )

    if reference_atom_labels.ndim != 2:
        raise ValueError(
            f"{CLUSTER_KEY!r} in {og_labels_path} must be indexed by sample/frame first, "
            f"got shape {reference_atom_labels.shape}."
        )

    rows: list[tuple[Path, int, int, int]] = []
    error_counts: list[int] = []
    for assigned_path in assigned_cluster_files:
        labels_assigned = one_dimensional_labels(
            load_npz_array(assigned_path, "labels_assigned"),
            "labels_assigned",
            assigned_path,
        )

        conditioned_eval_path = conditioned_eval_path_for_assigned(assigned_path)
        conditioning_labels = one_dimensional_labels(
            load_npz_array(conditioned_eval_path, "conditioning_cluster_labels"),
            "conditioning_cluster_labels",
            conditioned_eval_path,
        ).astype(np.int64, copy=False)
        row_index = verified_conditioning_row_index(
            assigned_path=assigned_path,
            conditioning_labels=conditioning_labels,
            reference_atom_labels=reference_atom_labels,
        )
        labels = reference_local_labels_for_row(
            row_index=row_index,
            assigned_path=assigned_path,
            reference_labels=reference_labels,
            reference_residue_global_labels=reference_residue_global_labels,
        )

        if labels_assigned.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch for {assigned_path}: labels_assigned has shape "
                f"{labels_assigned.shape}, but conditioning-derived labels row {row_index} from "
                f"{og_labels_path} has shape {labels.shape}."
            )

        matches = labels_assigned == labels
        difference_count = int(matches.size - np.count_nonzero(matches))
        rows.append((assigned_path, extract_index(assigned_path), row_index, difference_count))
        error_counts.append(difference_count)

    output_path = base_path / args.output_name
    save_error_histogram(error_counts, output_path)

    print(f"Processed {len(rows)} file(s).")
    print(f"Saved error histogram to: {output_path}")
    for assigned_path, file_index, row_index, difference_count in rows:
        print(
            f"{assigned_path}\tfilename_index={file_index}"
            f"\tconditioning_row={row_index}\tdifferences={difference_count}"
        )


if __name__ == "__main__":
    main()
