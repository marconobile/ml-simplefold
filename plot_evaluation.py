#!/usr/bin/env python3
"""Plot assigned-cluster label matches against the reference cluster labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
import numpy as np  # noqa: E402


DEFAULT_REF_NPZ = Path(
    "/storage_common/angiod/phase-data/projects/a2a/systems/a2a/clusters/"
    "cb3c3cae-5316-47db-8fbb-0567d5f0f75b/cluster.npz"
)
MATCH_TOKEN = "_assigned_clusters.npz"
IDX_RE = re.compile(r"(?:^|_)(\d+)_assigned_clusters\.npz$")
FOLLOWING_LABEL_IDX_RE = re.compile(
    r"(?:^|_)following_label_(\d+)_assigned_clusters\.npz$"
)
DEFAULT_LABELS_KEY = "labels"
SOURCE_LABELS = {
    "active_samples": "active",
    "inactive_samples": "inactive",
    "pas_samples": "pas",
}
SOURCE_ORDER = ["active", "inactive", "pas"]
SOURCE_COLORS = {
    "active": "#1f77b4",
    "inactive": "#d62728",
    "pas": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find *_assigned_clusters.npz files, compare labels_assigned to "
            "the corresponding reference labels, and plot histograms of exact "
            "matches and differences. following_label_* samples are compared "
            "to rows from the labels NPZ; other samples are compared to "
            "merged__labels_assigned from the reference cluster NPZ."
        )
    )
    parser.add_argument(
        "--base_path",
        type=Path,
        required=True,
        help="Directory to recursively search for files containing '_assigned_clusters.npz'.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory where the output PNG will be written.",
    )
    parser.add_argument(
        "--ref_npz",
        type=Path,
        default=DEFAULT_REF_NPZ,
        help="Reference cluster NPZ containing merged__labels_assigned.",
    )
    parser.add_argument(
        "--output_name",
        default="assigned_cluster_match_histograms.png",
        help="Output PNG filename.",
    )
    parser.add_argument(
        "--labels_npz",
        "--labels-npz",
        dest="labels_npz",
        type=Path,
        default=None,
        help=(
            "Optional labels NPZ used for following_label_* samples. If omitted, "
            "the script reads labels_npz_path from the companion JSON file."
        ),
    )
    parser.add_argument(
        "--labels_key",
        "--labels-key",
        dest="labels_key",
        default=DEFAULT_LABELS_KEY,
        help="Key in --labels_npz containing residue-level labels.",
    )
    return parser.parse_args()


def find_sampled_structures(base_path: Path) -> list[Path]:
    return sorted(
        path
        for path in base_path.rglob("*")
        if path.is_file() and MATCH_TOKEN in path.name
    )


def extract_idx(sampled_npz: Path) -> int:
    match = IDX_RE.search(sampled_npz.name)
    if match is None:
        raise ValueError(
            f"Could not extract frame index from {sampled_npz}. Expected a name like "
            "active_without_hs_000771_assigned_clusters.npz."
        )
    return int(match.group(1))


def extract_following_label_idx(sampled_npz: Path) -> int | None:
    match = FOLLOWING_LABEL_IDX_RE.search(sampled_npz.name)
    if match is None:
        return None
    return int(match.group(1))


def extract_source_type(sampled_npz: Path, base_path: Path) -> str:
    try:
        relative_parts = sampled_npz.relative_to(base_path).parts[:-1]
    except ValueError:
        relative_parts = ()

    for part in relative_parts:
        if part in SOURCE_LABELS:
            return SOURCE_LABELS[part]

    for parent in sampled_npz.parents:
        if parent.name in SOURCE_LABELS:
            return SOURCE_LABELS[parent.name]

    expected_dirs = ", ".join(sorted(SOURCE_LABELS))
    raise ValueError(
        f"Could not determine source type for {sampled_npz}. Expected the file "
        f"or one of its parent directories to be under one of: {expected_dirs}."
    )


def companion_metrics_path(sampled_npz: Path) -> Path:
    return sampled_npz.with_name(
        sampled_npz.name.replace(MATCH_TOKEN, "_conditioned_eval.json", 1)
    )


def labels_npz_path_for_sample(
    sampled_npz: Path,
    label_idx: int,
    labels_npz_arg: Path | None,
) -> Path:
    if labels_npz_arg is not None:
        return labels_npz_arg

    metrics_path = companion_metrics_path(sampled_npz)
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Could not find companion metrics JSON for {sampled_npz}: "
            f"{metrics_path}. Pass --labels_npz explicitly."
        )

    with metrics_path.open() as f:
        metrics = json.load(f)

    metrics_label_idx = metrics.get("label_sample_index")
    if metrics_label_idx is not None and int(metrics_label_idx) != label_idx:
        raise ValueError(
            f"Filename label index {label_idx} does not match "
            f"label_sample_index={metrics_label_idx} in {metrics_path}."
        )

    labels_npz_path = metrics.get("labels_npz_path")
    if not labels_npz_path:
        raise KeyError(
            f"{metrics_path} does not contain labels_npz_path. Pass --labels_npz "
            "explicitly."
        )

    return Path(labels_npz_path).expanduser().resolve()


def load_labels_rows(labels_npz: Path, labels_key: str) -> np.ndarray:
    if not labels_npz.is_file():
        raise FileNotFoundError(f"Labels NPZ not found: {labels_npz}")

    with np.load(labels_npz, allow_pickle=False) as labels_data:
        if labels_key not in labels_data.files:
            available = ", ".join(labels_data.files)
            raise KeyError(
                f"Labels NPZ {labels_npz} is missing key {labels_key!r}. "
                f"Available keys: {available}"
            )
        labels = np.asarray(labels_data[labels_key], dtype=np.int64)

    if labels.ndim != 2:
        raise ValueError(
            f"{labels_key!r} in {labels_npz} must be 2D "
            f"(n_samples, n_residues), got shape {labels.shape}."
        )
    if labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError(f"{labels_key!r} in {labels_npz} has empty shape {labels.shape}.")
    return labels


def labels_row_for_sample(
    sampled_npz: Path,
    label_idx: int,
    labels_npz_arg: Path | None,
    labels_key: str,
    labels_cache: dict[tuple[Path, str], np.ndarray],
) -> np.ndarray:
    labels_npz = labels_npz_path_for_sample(sampled_npz, label_idx, labels_npz_arg)
    cache_key = (labels_npz, labels_key)
    if cache_key not in labels_cache:
        labels_cache[cache_key] = load_labels_rows(labels_npz, labels_key)

    labels = labels_cache[cache_key]
    if label_idx >= labels.shape[0]:
        raise IndexError(
            f"Extracted label index {label_idx} from {sampled_npz}, but "
            f"{labels_npz}:{labels_key} only contains {labels.shape[0]} row(s)."
        )
    return labels[label_idx]


def build_reference_row_lookup(ref_data: np.lib.npyio.NpzFile) -> dict[tuple[str, int], int]:
    required = {"merged__frame_state_ids", "merged__frame_indices"}
    missing = sorted(required - set(ref_data.files))
    if missing:
        raise KeyError(
            "Reference NPZ is missing key(s) needed to match samples by source "
            f"type and frame index: {', '.join(missing)}"
        )

    state_ids = np.asarray(ref_data["merged__frame_state_ids"]).astype(str)
    frame_indices = np.asarray(ref_data["merged__frame_indices"], dtype=np.int64)
    if state_ids.shape[0] != frame_indices.shape[0]:
        raise ValueError(
            "`merged__frame_state_ids` and `merged__frame_indices` must have the "
            f"same length, got {state_ids.shape[0]} and {frame_indices.shape[0]}."
        )

    lookup: dict[tuple[str, int], int] = {}
    for row_idx, (state_id, frame_idx) in enumerate(zip(state_ids, frame_indices, strict=True)):
        key = (str(state_id), int(frame_idx))
        if key in lookup:
            raise ValueError(
                f"Reference NPZ contains duplicate rows for state={state_id!r}, "
                f"frame_index={int(frame_idx)}."
            )
        lookup[key] = int(row_idx)
    return lookup


def reference_row_for_sample(
    source_type: str,
    frame_index: int,
    reference_lookup: dict[tuple[str, int], int],
) -> int:
    key = (source_type, int(frame_index))
    try:
        return reference_lookup[key]
    except KeyError as exc:
        raise KeyError(
            f"Could not find reference labels for source={source_type!r}, "
            f"frame_index={int(frame_index)}."
        ) from exc


def load_sampled_labels(sampled_npz: Path) -> np.ndarray:
    with np.load(sampled_npz) as sampled_obs:
        if "labels_assigned" not in sampled_obs.files:
            raise KeyError(f"{sampled_npz} is missing required key 'labels_assigned'.")
        return np.asarray(sampled_obs["labels_assigned"], dtype=np.int64)


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


def count_label_matches(
    reference_labels: np.ndarray,
    sampled_labels: np.ndarray,
    sampled_npz: Path,
) -> tuple[int, int]:
    reference_labels = one_dimensional_labels(reference_labels, "Reference labels", sampled_npz)
    sampled_labels = one_dimensional_labels(sampled_labels, "Sampled labels", sampled_npz)

    if reference_labels.shape != sampled_labels.shape:
        raise ValueError(
            f"Label shape mismatch for {sampled_npz}: reference labels have shape "
            f"{reference_labels.shape}, sampled labels have shape {sampled_labels.shape}."
        )

    exact_match_count = int(np.count_nonzero(reference_labels == sampled_labels))
    difference_count = int(sampled_labels.size - exact_match_count)
    return exact_match_count, difference_count


def integer_bins(values: list[int]) -> np.ndarray:
    min_value = min(values)
    max_value = max(values)
    return np.arange(min_value - 0.5, max_value + 1.5, 1)


def plot_integer_histogram(
    ax: plt.Axes,
    values_by_source: dict[str, list[int]],
    title: str,
    xlabel: str,
) -> None:
    values = [
        value
        for source in SOURCE_ORDER
        for value in values_by_source[source]
    ]
    min_value = min(values)
    max_value = max(values)
    sources_with_values = [
        source for source in SOURCE_ORDER if values_by_source[source]
    ]

    ax.hist(
        [values_by_source[source] for source in sources_with_values],
        bins=integer_bins(values),
        color=[SOURCE_COLORS[source] for source in sources_with_values],
        edgecolor="black",
        label=sources_with_values,
        stacked=True,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_xlim(min_value - 0.5, max_value + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis="x", labelrotation=90)


def main() -> None:
    args = parse_args()
    base_path = args.base_path.resolve()
    out_dir = args.out_dir.resolve()
    ref_npz = args.ref_npz.resolve()
    labels_npz_arg = (
        args.labels_npz.expanduser().resolve()
        if args.labels_npz is not None
        else None
    )

    if not base_path.is_dir():
        raise NotADirectoryError(f"Base path is not a directory: {base_path}")

    sampled_structures = find_sampled_structures(base_path)
    if not sampled_structures:
        raise FileNotFoundError(
            f"No files containing {MATCH_TOKEN!r} were found under {base_path}"
        )

    exact_matches_by_source: dict[str, list[int]] = {
        source: [] for source in SOURCE_ORDER
    }
    differences_by_source: dict[str, list[int]] = {
        source: [] for source in SOURCE_ORDER
    }
    labels_cache: dict[tuple[Path, str], np.ndarray] = {}

    needs_reference_npz = any(
        extract_following_label_idx(sampled_npz) is None
        for sampled_npz in sampled_structures
    )
    ref_data = None
    ref_labels_all = None
    reference_lookup = None
    if needs_reference_npz:
        if not ref_npz.is_file():
            raise FileNotFoundError(f"Reference NPZ not found: {ref_npz}")
        ref_data = np.load(ref_npz)
        if "merged__labels_assigned" not in ref_data.files:
            raise KeyError(
                f"Reference NPZ is missing key 'merged__labels_assigned': {ref_npz}"
            )
        ref_labels_all = ref_data["merged__labels_assigned"]
        reference_lookup = build_reference_row_lookup(ref_data)

    try:
        for sampled_npz in sampled_structures:
            source_type = extract_source_type(sampled_npz, base_path)
            sampled_labels = load_sampled_labels(sampled_npz)
            label_idx = extract_following_label_idx(sampled_npz)
            if label_idx is None:
                idx = extract_idx(sampled_npz)
                if ref_labels_all is None or reference_lookup is None:
                    raise RuntimeError("Reference NPZ was not loaded.")
                ref_row_idx = reference_row_for_sample(source_type, idx, reference_lookup)
                if ref_row_idx >= ref_labels_all.shape[0]:
                    raise IndexError(
                        f"Matched reference row {ref_row_idx} for {sampled_npz}, but "
                        f"reference labels only contain {ref_labels_all.shape[0]} frame(s)."
                    )
                reference_labels = ref_labels_all[ref_row_idx]
            else:
                reference_labels = labels_row_for_sample(
                    sampled_npz,
                    label_idx,
                    labels_npz_arg,
                    args.labels_key,
                    labels_cache,
                )

            exact_match_count, difference_count = count_label_matches(
                reference_labels,
                sampled_labels,
                sampled_npz,
            )
            exact_matches_by_source[source_type].append(int(exact_match_count))
            differences_by_source[source_type].append(int(difference_count))
    finally:
        if ref_data is not None:
            ref_data.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / args.output_name

    exact_matches = [
        value
        for source in SOURCE_ORDER
        for value in exact_matches_by_source[source]
    ]
    differences = [
        value
        for source in SOURCE_ORDER
        for value in differences_by_source[source]
    ]
    max_integer_range = max(
        max(exact_matches) - min(exact_matches) + 1,
        max(differences) - min(differences) + 1,
    )
    fig_width = max(12, max_integer_range * 0.18)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5), constrained_layout=True)

    plot_integer_histogram(
        axes[0],
        exact_matches_by_source,
        "Exact Matches",
        "Number of matching labels",
    )
    plot_integer_histogram(
        axes[1],
        differences_by_source,
        "Differences",
        "Number of differing labels",
    )
    axes[1].legend(title="Source")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Processed {len(sampled_structures)} file(s).")
    print(f"Saved histogram plot to: {output_path}")


if __name__ == "__main__":
    main()
