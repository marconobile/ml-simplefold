#!/usr/bin/env python3
"""Plot assigned-cluster label matches against the reference cluster labels."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find *_assigned_clusters.npz files, compare labels_assigned to "
            "merged__labels_assigned from the reference cluster NPZ, and plot "
            "histograms of exact matches and differences."
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


def integer_bins(values: list[int]) -> np.ndarray:
    min_value = min(values)
    max_value = max(values)
    return np.arange(min_value - 0.5, max_value + 1.5, 1)


def plot_integer_histogram(
    ax: plt.Axes,
    values: list[int],
    title: str,
    xlabel: str,
) -> None:
    min_value = min(values)
    max_value = max(values)

    ax.hist(values, bins=integer_bins(values), edgecolor="black")
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

    if not base_path.is_dir():
        raise NotADirectoryError(f"Base path is not a directory: {base_path}")

    if not ref_npz.is_file():
        raise FileNotFoundError(f"Reference NPZ not found: {ref_npz}")

    sampled_structures = find_sampled_structures(base_path)
    if not sampled_structures:
        raise FileNotFoundError(
            f"No files containing {MATCH_TOKEN!r} were found under {base_path}"
        )

    exact_matches: list[int] = []
    differences: list[int] = []

    with np.load(ref_npz) as ref_data:
        ref_labels_all = ref_data["merged__labels_assigned"]

        for sampled_npz in sampled_structures:
            idx = extract_idx(sampled_npz)
            with np.load(sampled_npz) as sampled_obs:
                sampled_labels = sampled_obs["labels_assigned"]

            if idx >= ref_labels_all.shape[0]:
                raise IndexError(
                    f"Extracted index {idx} from {sampled_npz}, but reference labels "
                    f"only contain {ref_labels_all.shape[0]} frame(s)."
                )

            matches = ref_labels_all[idx] == sampled_labels
            exact_match_count = matches.sum().item()
            exact_matches.append(int(exact_match_count))
            differences.append(int(sampled_labels.shape[-1] - exact_match_count))

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / args.output_name

    max_integer_range = max(
        max(exact_matches) - min(exact_matches) + 1,
        max(differences) - min(differences) + 1,
    )
    fig_width = max(12, max_integer_range * 0.18)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5), constrained_layout=True)

    plot_integer_histogram(
        axes[0],
        exact_matches,
        "Exact Matches",
        "Number of matching labels",
    )
    plot_integer_histogram(
        axes[1],
        differences,
        "Differences",
        "Number of differing labels",
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Processed {len(sampled_structures)} file(s).")
    print(f"Saved histogram plot to: {output_path}")


if __name__ == "__main__":
    main()
