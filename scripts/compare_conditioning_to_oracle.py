#!/usr/bin/env python3
"""Compare sampling conditioning labels to oracle-assigned sampled labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ASSIGNED_TOKEN = "_assigned_clusters.npz"
CONDITIONED_EVAL_TOKEN = "_conditioned_eval.npz"
CONDITIONED_EVAL_JSON_TOKEN = "_conditioned_eval.json"
CLUSTER_KEY = "atom_idx_and_glob_cluster_id_per_frame"
SOURCE_SUFFIX = "_samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the conditioning vector saved by sample_with_conditioning.py "
            "against oracle labels written by assign_conditioned_eval_sample_clusters.py."
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
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to --base-path.",
    )
    parser.add_argument(
        "--output-prefix",
        default="conditioning_vs_oracle",
        help="Prefix for CSV, TXT, and PNG outputs.",
    )
    return parser.parse_args()


def find_assigned_cluster_files(base_path: Path) -> list[Path]:
    return sorted(path for path in base_path.rglob(f"*{ASSIGNED_TOKEN}") if path.is_file())


def conditioned_eval_path_for_assigned(path: Path) -> Path:
    if not path.name.endswith(ASSIGNED_TOKEN):
        raise ValueError(f"Unexpected assigned-clusters filename: {path}")
    return path.with_name(path.name.replace(ASSIGNED_TOKEN, CONDITIONED_EVAL_TOKEN, 1))


def conditioned_eval_json_path_for_assigned(path: Path) -> Path:
    if not path.name.endswith(ASSIGNED_TOKEN):
        raise ValueError(f"Unexpected assigned-clusters filename: {path}")
    return path.with_name(path.name.replace(ASSIGNED_TOKEN, CONDITIONED_EVAL_JSON_TOKEN, 1))


def load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            available = ", ".join(data.files)
            raise KeyError(f"{path} is missing {key!r}. Available keys: {available}")
        return np.asarray(data[key])


def optional_npz_array(path: Path, key: str) -> np.ndarray | None:
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            return None
        return np.asarray(data[key])


def one_dimensional(array: np.ndarray, name: str, path: Path) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"{name} for {path} must be 1D, got shape {array.shape}.")
    return array


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open() as f:
        return json.load(f)


def source_type_for_path(path: Path, base_path: Path) -> str:
    candidates = []
    try:
        candidates.extend(path.relative_to(base_path).parts[:-1])
    except ValueError:
        pass
    candidates.extend(parent.name for parent in path.parents)
    candidates.append(base_path.name)

    for part in candidates:
        if part.endswith(SOURCE_SUFFIX):
            return part[: -len(SOURCE_SUFFIX)]

    name = path.name.lower()
    for source in ("active", "inactive", "pas"):
        if source in name:
            return source
    return "unknown"


def sample_name_for_assigned(path: Path) -> str:
    return path.name.replace(ASSIGNED_TOKEN, "")


def load_conditioning_labels(conditioned_eval_path: Path) -> np.ndarray:
    labels = optional_npz_array(conditioned_eval_path, "conditioning_cluster_labels")
    if labels is None:
        labels = optional_npz_array(conditioned_eval_path, "original_cluster_labels")
    if labels is None:
        raise KeyError(
            f"{conditioned_eval_path} is missing both 'conditioning_cluster_labels' "
            "and 'original_cluster_labels'."
        )
    return one_dimensional(labels, "conditioning labels", conditioned_eval_path).astype(
        np.int64,
        copy=False,
    )


def load_cluster_counts(assigned_path: Path) -> np.ndarray:
    cluster_counts = optional_npz_array(assigned_path, "cluster_counts")
    if cluster_counts is None:
        cluster_counts = optional_npz_array(assigned_path, "merged__cluster_counts")
    if cluster_counts is None:
        raise KeyError(
            f"{assigned_path} is missing both 'cluster_counts' and "
            "'merged__cluster_counts'."
        )
    return one_dimensional(cluster_counts, "cluster_counts", assigned_path).astype(
        np.int64,
        copy=False,
    )


def load_atom_resids(conditioned_eval_path: Path, metrics: dict[str, Any]) -> np.ndarray:
    atom_resids = optional_npz_array(conditioned_eval_path, "atom_resids")
    if atom_resids is not None:
        return one_dimensional(atom_resids, "atom_resids", conditioned_eval_path).astype(
            np.int64,
            copy=False,
        )

    raw_npz = metrics.get("raw_npz_path")
    if raw_npz:
        raw_npz_path = Path(raw_npz).expanduser()
        if raw_npz_path.is_file():
            raw_atom_resids = optional_npz_array(raw_npz_path, "atom_resids")
            if raw_atom_resids is not None:
                return one_dimensional(raw_atom_resids, "atom_resids", raw_npz_path).astype(
                    np.int64,
                    copy=False,
                )

    raise KeyError(
        f"Could not load atom_resids for {conditioned_eval_path}. "
        "Expected atom_resids in the conditioned NPZ or in metrics['raw_npz_path']."
    )


def labels_to_residue_global(
    atom_global_labels: np.ndarray,
    atom_resids: np.ndarray,
    path: Path,
) -> np.ndarray:
    if atom_global_labels.shape != atom_resids.shape:
        raise ValueError(
            f"Atom label/residue shape mismatch for {path}: "
            f"{atom_global_labels.shape} vs {atom_resids.shape}."
        )
    if atom_resids.size == 0:
        raise ValueError(f"No atom_resids for {path}.")
    if int(atom_resids.min()) < 0:
        raise ValueError(f"atom_resids contains negative residue indices for {path}.")

    n_residues = int(atom_resids.max()) + 1
    residue_global = np.full(n_residues, -1, dtype=np.int64)
    for residue_idx in range(n_residues):
        atom_mask = atom_resids == residue_idx
        if not np.any(atom_mask):
            raise ValueError(f"Residue {residue_idx} has no atoms for {path}.")
        labels = atom_global_labels[atom_mask]
        labels = labels[labels >= 0]
        if labels.size == 0:
            continue
        unique = np.unique(labels)
        if unique.size != 1:
            raise ValueError(
                f"Conditioning labels are not residue-consistent for {path} "
                f"at residue {residue_idx}: {unique[:20].tolist()}."
            )
        residue_global[residue_idx] = int(unique[0])
    return residue_global


def global_to_local_labels(
    residue_global_labels: np.ndarray,
    cluster_counts: np.ndarray,
    path: Path,
) -> np.ndarray:
    cluster_counts = one_dimensional(cluster_counts, "cluster_counts", path).astype(
        np.int64,
        copy=False,
    )
    if residue_global_labels.shape != cluster_counts.shape:
        raise ValueError(
            f"Residue global label shape {residue_global_labels.shape} does not match "
            f"cluster_counts shape {cluster_counts.shape} for {path}."
        )
    offsets = np.concatenate(
        [np.zeros((1,), dtype=np.int64), np.cumsum(cluster_counts[:-1], dtype=np.int64)]
    )
    local = residue_global_labels - offsets
    valid = residue_global_labels >= 0
    invalid = valid & ((local < 0) | (local >= cluster_counts))
    if np.any(invalid):
        bad = np.flatnonzero(invalid)[:20].tolist()
        raise ValueError(
            f"Global-to-local conversion is invalid for {path} at residues {bad}."
        )
    local[~valid] = -1
    return local.astype(np.int64, copy=False)


def compare_labels(expected: np.ndarray, oracle: np.ndarray, path: Path) -> dict[str, Any]:
    expected = one_dimensional(expected, "expected labels", path).astype(np.int64, copy=False)
    oracle = one_dimensional(oracle, "oracle labels", path).astype(np.int64, copy=False)
    if expected.shape != oracle.shape:
        raise ValueError(
            f"Expected/oracle label shape mismatch for {path}: "
            f"{expected.shape} vs {oracle.shape}."
        )
    valid = (expected >= 0) & (oracle >= 0)
    if not np.any(valid):
        raise ValueError(f"No comparable residue labels for {path}.")
    diff = oracle - expected
    mismatches = valid & (diff != 0)
    abs_diff = np.abs(diff[valid])
    n_compared = int(valid.sum())
    mismatch_count = int(mismatches.sum())
    return {
        "n_residues": int(expected.shape[0]),
        "n_compared": n_compared,
        "match_count": int(n_compared - mismatch_count),
        "mismatch_count": mismatch_count,
        "accuracy": float((n_compared - mismatch_count) / n_compared),
        "mismatch_fraction": float(mismatch_count / n_compared),
        "mean_abs_error": float(np.mean(abs_diff)),
        "max_abs_error": int(np.max(abs_diff)),
        "diff": diff,
        "valid": valid,
        "mismatches": mismatches,
    }


def write_per_sample_report(path: Path, row: dict[str, Any]) -> None:
    lines = [
        "Conditioning vs oracle comparison",
        "",
        "Sample",
        f"  sample_name: {row['sample_name']}",
        f"  source_type: {row['source_type']}",
        f"  assigned_npz: {row['assigned_npz']}",
        f"  conditioned_eval_npz: {row['conditioned_eval_npz']}",
        f"  raw_npz_path: {row.get('raw_npz_path')}",
        f"  frame_index: {row.get('frame_index')}",
        f"  sample_index: {row.get('sample_index')}",
        f"  seed: {row.get('seed')}",
        "",
        "Metrics",
        f"  residues: {row['n_residues']}",
        f"  compared: {row['n_compared']}",
        f"  matches: {row['match_count']}",
        f"  mismatches: {row['mismatch_count']}",
        f"  accuracy: {row['accuracy']:.6f}",
        f"  mismatch_fraction: {row['mismatch_fraction']:.6f}",
        f"  mean_abs_error: {row['mean_abs_error']:.6f}",
        f"  max_abs_error: {row['max_abs_error']}",
    ]
    path.write_text("\n".join(lines) + "\n")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {"overall": rows}
    for row in rows:
        groups.setdefault(row["source_type"], []).append(row)

    summary = {}
    for name, group_rows in groups.items():
        n_compared = sum(int(row["n_compared"]) for row in group_rows)
        mismatches = sum(int(row["mismatch_count"]) for row in group_rows)
        matches = sum(int(row["match_count"]) for row in group_rows)
        perfect_samples = sum(1 for row in group_rows if int(row["mismatch_count"]) == 0)
        summary[name] = {
            "sample_count": len(group_rows),
            "perfect_sample_count": perfect_samples,
            "perfect_sample_fraction": float(perfect_samples / len(group_rows)),
            "n_compared": n_compared,
            "match_count": matches,
            "mismatch_count": mismatches,
            "accuracy": float(matches / n_compared) if n_compared else None,
            "mismatch_fraction": float(mismatches / n_compared) if n_compared else None,
            "mean_sample_mismatch_fraction": float(
                np.mean([row["mismatch_fraction"] for row in group_rows])
            ),
            "max_sample_mismatches": int(max(row["mismatch_count"] for row in group_rows)),
        }
    return summary


def write_summary_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, dict[str, Any]]) -> None:
    lines = [
        "Conditioning vs oracle aggregate report",
        "",
        "Groups",
    ]
    for group_name in sorted(summary, key=lambda name: (name != "overall", name)):
        stats = summary[group_name]
        accuracy = stats["accuracy"]
        mismatch_fraction = stats["mismatch_fraction"]
        lines.extend(
            [
                f"  {group_name}",
                f"    sample_count: {stats['sample_count']}",
                f"    perfect_sample_count: {stats['perfect_sample_count']}",
                f"    perfect_sample_fraction: {stats['perfect_sample_fraction']:.6f}",
                f"    compared: {stats['n_compared']}",
                f"    matches: {stats['match_count']}",
                f"    mismatches: {stats['mismatch_count']}",
                f"    accuracy: {accuracy:.6f}" if accuracy is not None else "    accuracy: n/a",
                (
                    f"    mismatch_fraction: {mismatch_fraction:.6f}"
                    if mismatch_fraction is not None
                    else "    mismatch_fraction: n/a"
                ),
                f"    mean_sample_mismatch_fraction: {stats['mean_sample_mismatch_fraction']:.6f}",
                f"    max_sample_mismatches: {stats['max_sample_mismatches']}",
                "",
            ]
        )

    lines.append("Per Sample")
    for row in rows:
        lines.append(
            "  "
            f"{row['sample_name']}: source={row['source_type']} "
            f"mismatches={row['mismatch_count']}/{row['n_compared']} "
            f"accuracy={row['accuracy']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n")


def write_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "source_type",
        "sample_name",
        "assigned_npz",
        "conditioned_eval_npz",
        "raw_npz_path",
        "frame_index",
        "sample_index",
        "seed",
        "n_residues",
        "n_compared",
        "match_count",
        "mismatch_count",
        "accuracy",
        "mismatch_fraction",
        "mean_abs_error",
        "max_abs_error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_residue_csv(path: Path, residue_rows: list[dict[str, Any]]) -> None:
    columns = [
        "source_type",
        "sample_name",
        "residue_index",
        "expected_local_label",
        "oracle_label",
        "match",
        "label_error",
        "abs_label_error",
        "expected_global_label",
        "cluster_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in residue_rows:
            writer.writerow({column: row.get(column) for column in columns})


def plot_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    source_types = sorted({row["source_type"] for row in rows})
    colors = plt.get_cmap("tab10")
    color_by_source = {
        source_type: colors(idx % 10) for idx, source_type in enumerate(source_types)
    }

    fig, axes = plt.subplots(2, 1, figsize=(max(10.0, 0.45 * len(rows)), 9.0), constrained_layout=True)
    x = np.arange(len(rows))
    mismatch_counts = np.asarray([row["mismatch_count"] for row in rows], dtype=np.int64)
    mismatch_fracs = np.asarray([row["mismatch_fraction"] for row in rows], dtype=np.float64)
    bar_colors = [color_by_source[row["source_type"]] for row in rows]

    axes[0].bar(x, mismatch_counts, color=bar_colors, edgecolor="black", linewidth=0.4)
    axes[0].set_title("Conditioning vs Oracle Mismatches Per Structure")
    axes[0].set_ylabel("Mismatching residues")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([row["sample_name"] for row in rows], rotation=90, fontsize=7)

    bins = np.linspace(0.0, max(1e-9, float(mismatch_fracs.max())), 20)
    if bins[-1] < 1.0:
        bins = np.linspace(0.0, 1.0, 21)
    for source_type in source_types:
        values = [
            row["mismatch_fraction"]
            for row in rows
            if row["source_type"] == source_type
        ]
        axes[1].hist(
            values,
            bins=bins,
            alpha=0.65,
            label=source_type,
            color=color_by_source[source_type],
            edgecolor="black",
        )
    axes[1].set_title("Mismatch Fraction Distribution")
    axes[1].set_xlabel("Mismatching residue fraction")
    axes[1].set_ylabel("Structures")
    axes[1].legend(title="Type")

    fig.savefig(path, dpi=220)
    plt.close(fig)


def compare_file(assigned_path: Path, base_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    conditioned_eval_path = conditioned_eval_path_for_assigned(assigned_path)
    metrics_path = conditioned_eval_json_path_for_assigned(assigned_path)
    metrics = load_metrics(metrics_path)

    if not conditioned_eval_path.is_file():
        raise FileNotFoundError(f"Conditioned eval NPZ not found: {conditioned_eval_path}")

    oracle_labels = one_dimensional(
        load_npz_array(assigned_path, "labels_assigned"),
        "labels_assigned",
        assigned_path,
    ).astype(np.int64, copy=False)
    cluster_counts = load_cluster_counts(assigned_path)
    atom_conditioning = load_conditioning_labels(conditioned_eval_path)
    atom_resids = load_atom_resids(conditioned_eval_path, metrics)

    residue_global = labels_to_residue_global(atom_conditioning, atom_resids, conditioned_eval_path)
    expected_local = global_to_local_labels(residue_global, cluster_counts, assigned_path)
    comparison = compare_labels(expected_local, oracle_labels, assigned_path)

    sample_name = sample_name_for_assigned(assigned_path)
    source_type = source_type_for_path(assigned_path, base_path)
    row = {
        "source_type": source_type,
        "sample_name": sample_name,
        "assigned_npz": str(assigned_path),
        "conditioned_eval_npz": str(conditioned_eval_path),
        "raw_npz_path": metrics.get("raw_npz_path"),
        "frame_index": metrics.get("frame_index"),
        "sample_index": metrics.get("sample_index"),
        "seed": metrics.get("seed"),
        "n_residues": comparison["n_residues"],
        "n_compared": comparison["n_compared"],
        "match_count": comparison["match_count"],
        "mismatch_count": comparison["mismatch_count"],
        "accuracy": comparison["accuracy"],
        "mismatch_fraction": comparison["mismatch_fraction"],
        "mean_abs_error": comparison["mean_abs_error"],
        "max_abs_error": comparison["max_abs_error"],
    }

    residue_rows = []
    diff = comparison["diff"]
    valid = comparison["valid"]
    mismatches = comparison["mismatches"]
    for residue_idx in range(expected_local.shape[0]):
        if not bool(valid[residue_idx]):
            continue
        residue_rows.append(
            {
                "source_type": source_type,
                "sample_name": sample_name,
                "residue_index": residue_idx,
                "expected_local_label": int(expected_local[residue_idx]),
                "oracle_label": int(oracle_labels[residue_idx]),
                "match": int(not bool(mismatches[residue_idx])),
                "label_error": int(diff[residue_idx]),
                "abs_label_error": int(abs(diff[residue_idx])),
                "expected_global_label": int(residue_global[residue_idx]),
                "cluster_count": int(cluster_counts[residue_idx]),
            }
        )

    report_path = assigned_path.with_name(
        assigned_path.name.replace(ASSIGNED_TOKEN, "_conditioning_vs_oracle_report.txt", 1)
    )
    write_per_sample_report(report_path, row)
    return row, residue_rows


def main() -> None:
    args = parse_args()
    base_path = args.base_path.expanduser().resolve()
    out_dir = (args.out_dir or base_path).expanduser().resolve()
    if not base_path.is_dir():
        raise NotADirectoryError(f"Base path is not a directory: {base_path}")

    assigned_files = find_assigned_cluster_files(base_path)
    if not assigned_files:
        raise FileNotFoundError(f"No files matching '*{ASSIGNED_TOKEN}' under {base_path}.")

    rows = []
    residue_rows = []
    for assigned_path in assigned_files:
        row, per_residue = compare_file(assigned_path, base_path)
        rows.append(row)
        residue_rows.extend(per_residue)

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    sample_csv = out_dir / f"{prefix}_per_structure.csv"
    residue_csv = out_dir / f"{prefix}_per_residue.csv"
    report_path = out_dir / f"{prefix}_report.txt"
    plot_path = out_dir / f"{prefix}_errors.png"

    rows = sorted(rows, key=lambda row: (row["source_type"], row["sample_name"]))
    residue_rows = sorted(
        residue_rows,
        key=lambda row: (row["source_type"], row["sample_name"], row["residue_index"]),
    )
    summary = summarize_rows(rows)
    write_sample_csv(sample_csv, rows)
    write_residue_csv(residue_csv, residue_rows)
    write_summary_report(report_path, rows, summary)
    plot_summary(plot_path, rows)

    print(f"Processed {len(rows)} assigned cluster file(s).")
    print(f"Wrote per-structure CSV: {sample_csv}")
    print(f"Wrote per-residue CSV: {residue_csv}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote error plot: {plot_path}")


if __name__ == "__main__":
    main()
