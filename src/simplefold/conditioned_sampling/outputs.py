#!/usr/bin/env python3
"""Utilities extracted from scripts/sample_with_conditioning.py."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def _histogram_counts(
    values: np.ndarray,
    bins: int,
    value_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(value_range[0], value_range[1], bins + 1, dtype=np.float32)
    if values.size == 0:
        return np.zeros(bins, dtype=np.int64), edges
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    return counts.astype(np.int64, copy=False), edges

def write_dihedral_histograms(
    output_dir: Path,
    output_stem: str,
    original_dihedrals: np.ndarray,
    sampled_dihedrals: np.ndarray,
    dihedral_diff_rad: np.ndarray,
    dihedral_mask: np.ndarray,
    dihedral_keys: list[str],
    angle_bins: int,
    error_bins: int,
) -> dict[str, Any]:
    valid = (
        dihedral_mask.astype(bool)
        & np.isfinite(original_dihedrals)
        & np.isfinite(sampled_dihedrals)
        & np.isfinite(dihedral_diff_rad)
    )

    original_deg = np.degrees(original_dihedrals)
    sampled_deg = np.degrees(sampled_dihedrals)
    error_deg = np.degrees(dihedral_diff_rad)
    abs_error_deg = np.abs(error_deg)

    angle_csv_path = output_dir / f"{output_stem}_dihedral_angle_histograms.csv"
    error_csv_path = output_dir / f"{output_stem}_dihedral_error_histograms.csv"
    angle_png_path = output_dir / f"{output_stem}_dihedral_angle_histograms.png"
    error_png_path = output_dir / f"{output_stem}_dihedral_error_histograms.png"

    with angle_csv_path.open("w", newline="") as f_angle, error_csv_path.open("w", newline="") as f_error:
        angle_writer = csv.writer(f_angle)
        error_writer = csv.writer(f_error)

        angle_writer.writerow(
            [
                "dihedral_key",
                "series",
                "bin_left_deg",
                "bin_right_deg",
                "count",
            ]
        )
        error_writer.writerow(
            [
                "dihedral_key",
                "series",
                "bin_left_deg",
                "bin_right_deg",
                "count",
            ]
        )

        all_valid = valid.reshape(-1)
        all_orig = original_deg.reshape(-1)[all_valid]
        all_sampled = sampled_deg.reshape(-1)[all_valid]
        all_err = error_deg.reshape(-1)[all_valid]
        all_abs_err = abs_error_deg.reshape(-1)[all_valid]

        angle_specs = [
            ("all", "original_deg", all_orig),
            ("all", "sampled_deg", all_sampled),
        ]
        error_specs = [
            ("all", "signed_error_deg", all_err),
            ("all", "abs_error_deg", all_abs_err),
        ]

        for key_idx, key in enumerate(dihedral_keys):
            key_valid = valid[:, key_idx]
            angle_specs.extend(
                [
                    (key, "original_deg", original_deg[:, key_idx][key_valid]),
                    (key, "sampled_deg", sampled_deg[:, key_idx][key_valid]),
                ]
            )
            error_specs.extend(
                [
                    (key, "signed_error_deg", error_deg[:, key_idx][key_valid]),
                    (key, "abs_error_deg", abs_error_deg[:, key_idx][key_valid]),
                ]
            )

        for key, series, values in angle_specs:
            counts, edges = _histogram_counts(values, angle_bins, (-180.0, 180.0))
            for bin_idx, count in enumerate(counts):
                angle_writer.writerow([key, series, float(edges[bin_idx]), float(edges[bin_idx + 1]), int(count)])

        for key, series, values in error_specs:
            if series == "abs_error_deg":
                hist_range = (0.0, 180.0)
            else:
                hist_range = (-180.0, 180.0)
            counts, edges = _histogram_counts(values, error_bins, hist_range)
            for bin_idx, count in enumerate(counts):
                error_writer.writerow([key, series, float(edges[bin_idx]), float(edges[bin_idx + 1]), int(count)])

    histogram_artifacts: dict[str, Any] = {
        "angle_histogram_csv": str(angle_csv_path),
        "error_histogram_csv": str(error_csv_path),
        "angle_histogram_png": None,
        "error_histogram_png": None,
        "angle_bins": int(angle_bins),
        "error_bins": int(error_bins),
    }

    try:
        import matplotlib.pyplot as plt

        row_labels = ["all", *dihedral_keys]

        angle_fig, angle_axes = plt.subplots(
            nrows=len(row_labels),
            ncols=1,
            figsize=(9.0, 2.4 * len(row_labels)),
            constrained_layout=True,
        )
        if len(row_labels) == 1:
            angle_axes = [angle_axes]
        for row_idx, label in enumerate(row_labels):
            if label == "all":
                row_valid = all_valid
                orig_vals = all_orig
                samp_vals = all_sampled
            else:
                key_idx = dihedral_keys.index(label)
                row_valid = valid[:, key_idx]
                orig_vals = original_deg[:, key_idx][row_valid]
                samp_vals = sampled_deg[:, key_idx][row_valid]
            ax = angle_axes[row_idx]
            ax.hist(orig_vals, bins=angle_bins, range=(-180.0, 180.0), alpha=0.5, label="original")
            ax.hist(samp_vals, bins=angle_bins, range=(-180.0, 180.0), alpha=0.5, label="sampled")
            ax.set_xlim(-180.0, 180.0)
            ax.set_ylabel("count")
            ax.set_title(f"{label} angle distribution (n={int(row_valid.sum())})")
            if row_idx == 0:
                ax.legend(loc="upper right")
        angle_axes[-1].set_xlabel("dihedral angle (deg)")
        angle_fig.savefig(angle_png_path, dpi=180)
        plt.close(angle_fig)
        histogram_artifacts["angle_histogram_png"] = str(angle_png_path)

        error_fig, error_axes = plt.subplots(
            nrows=len(row_labels),
            ncols=2,
            figsize=(12.0, 2.6 * len(row_labels)),
            constrained_layout=True,
        )
        if len(row_labels) == 1:
            error_axes = np.asarray([error_axes])
        for row_idx, label in enumerate(row_labels):
            if label == "all":
                row_valid = all_valid
                signed_vals = all_err
                abs_vals = all_abs_err
            else:
                key_idx = dihedral_keys.index(label)
                row_valid = valid[:, key_idx]
                signed_vals = error_deg[:, key_idx][row_valid]
                abs_vals = abs_error_deg[:, key_idx][row_valid]
            signed_ax = error_axes[row_idx, 0]
            abs_ax = error_axes[row_idx, 1]
            signed_ax.hist(signed_vals, bins=error_bins, range=(-180.0, 180.0), color="tab:orange", alpha=0.8)
            abs_ax.hist(abs_vals, bins=error_bins, range=(0.0, 180.0), color="tab:red", alpha=0.8)
            signed_ax.set_xlim(-180.0, 180.0)
            abs_ax.set_xlim(0.0, 180.0)
            signed_ax.set_ylabel("count")
            signed_ax.set_title(f"{label} signed error (n={int(row_valid.sum())})")
            abs_ax.set_title(f"{label} absolute error")
        error_axes[-1, 0].set_xlabel("signed error (deg)")
        error_axes[-1, 1].set_xlabel("absolute error (deg)")
        error_fig.savefig(error_png_path, dpi=180)
        plt.close(error_fig)
        histogram_artifacts["error_histogram_png"] = str(error_png_path)
    except Exception as exc:
        print(f"Warning: failed to render dihedral histogram PNG files: {exc}")

    return histogram_artifacts

def write_atomwise_csv(
    path: Path,
    atomwise_rmsd: np.ndarray,
    atom_names: np.ndarray | None,
    atom_resids: np.ndarray | None,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["atom_index", "atom_name", "residue_index", "atomwise_rmsd"])
        for atom_idx, value in enumerate(atomwise_rmsd):
            atom_name = "" if atom_names is None else str(atom_names[atom_idx])
            residue_index = "" if atom_resids is None else int(atom_resids[atom_idx])
            writer.writerow([atom_idx, atom_name, residue_index, value])

def write_dihedral_csv(
    path: Path,
    original_dihedrals: np.ndarray,
    sampled_dihedrals: np.ndarray,
    dihedral_diff_rad: np.ndarray,
    dihedral_keys: list[str],
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "residue_index",
                "dihedral_key",
                "original_rad",
                "sampled_rad",
                "diff_rad",
                "abs_diff_deg",
            ]
        )
        for res_idx in range(original_dihedrals.shape[0]):
            for dih_idx, key in enumerate(dihedral_keys):
                diff_rad = dihedral_diff_rad[res_idx, dih_idx]
                writer.writerow(
                    [
                        res_idx,
                        key,
                        original_dihedrals[res_idx, dih_idx],
                        sampled_dihedrals[res_idx, dih_idx],
                        diff_rad,
                        np.abs(np.degrees(diff_rad)) if np.isfinite(diff_rad) else np.nan,
                    ]
                )

def format_optional_float(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}{suffix}"

def write_report(
    path: Path,
    metrics: dict[str, Any],
    sampled_cif_path: Path,
    sampled_pdb_path: Path | None,
    target_cif_path: Path | None,
    arrays_path: Path,
    atom_csv_path: Path | None,
    dihedral_csv_path: Path | None,
    raw_sampled_cif_path: Path | None = None,
) -> None:
    dihedrals = metrics.get("dihedrals") or {}
    sample_header = "Template sample" if metrics.get("labels_npz_path") else "Selected sample"
    lines = [
        "SimpleFold conditioned sampling evaluation",
        "",
        sample_header,
        f"  record_id: {metrics['record_id']}",
        f"  raw_record_id: {metrics.get('raw_record_id')}",
        f"  sample_id: {metrics['sample_id']}",
        f"  frame_position: {metrics['frame_position']}",
        f"  frame_index: {metrics['frame_index']}",
        f"  atoms: {metrics['num_atoms']}",
        "",
        "Inputs",
        f"  raw_npz_path: {metrics['raw_npz_path']}",
        f"  labels_npz_path: {metrics.get('labels_npz_path')}",
        f"  processed_dir: {metrics['processed_dir']}",
        f"  checkpoint_path: {metrics['checkpoint_path']}",
        f"  seed: {metrics['seed']}",
        f"  sampler: EMSampler, num_steps={metrics['num_steps']}, tau={metrics['tau']}",
    ]
    if metrics.get("label_sample_index") is not None:
        lines.append(f"  label_sample_index: {metrics['label_sample_index']}")
    if metrics.get("sample_index") is not None:
        lines.append(f"  sample_index: {metrics['sample_index']}")

    if metrics.get("global_rmsd") is not None:
        lines.extend(
            [
                "",
                "Coordinate comparison after Kabsch alignment",
                f"  global_rmsd_A: {metrics['global_rmsd']:.6f}",
                f"  atomwise_rmsd_mean_A: {metrics['atomwise_rmsd_mean']:.6f}",
                f"  atomwise_rmsd_median_A: {metrics['atomwise_rmsd_median']:.6f}",
                f"  atomwise_rmsd_max_A: {metrics['atomwise_rmsd_max']:.6f}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Coordinate comparison",
                "  skipped: no original structure was used for this labels-NPZ sample",
            ]
        )

    lines.extend(["", "Dihedral comparison"])
    if dihedrals:
        lines.extend(
            [
                f"  compared_angles: {dihedrals.get('count', 'n/a')}",
                f"  mae_deg: {format_optional_float(dihedrals.get('mae_deg'))}",
                f"  rmse_deg: {format_optional_float(dihedrals.get('rmse_deg'))}",
                f"  max_abs_error_deg: {format_optional_float(dihedrals.get('max_abs_error_deg'))}",
                "  stored_vs_recomputed_original_mae_deg: "
                f"{format_optional_float(dihedrals.get('stored_vs_recomputed_original_mae_deg'))}",
            ]
        )
    else:
        lines.append("  skipped: no original structure dihedrals were evaluated")

    if dihedrals.get("sampled_dihedral_source") is not None:
        lines.append(f"  sampled_dihedral_source: {dihedrals.get('sampled_dihedral_source')}")
    if dihedrals.get("sampled_pdb_path") is not None:
        lines.append(f"  sampled_pdb_path: {dihedrals.get('sampled_pdb_path')}")

    raw_vs_pdb = dihedrals.get("sampled_raw_vs_pdb")
    if raw_vs_pdb is not None:
        lines.extend(
            [
                "  sampled_raw_vs_pdb_count: "
                f"{raw_vs_pdb.get('count', 'n/a')}",
                "  sampled_raw_vs_pdb_mae_deg: "
                f"{format_optional_float(raw_vs_pdb.get('mae_deg'))}",
                "  sampled_raw_vs_pdb_max_abs_error_deg: "
                f"{format_optional_float(raw_vs_pdb.get('max_abs_error_deg'))}",
            ]
        )
    raw_vs_aligned = dihedrals.get("sampled_raw_vs_aligned")
    if raw_vs_aligned is not None:
        lines.extend(
            [
                "  sampled_raw_vs_aligned_count: "
                f"{raw_vs_aligned.get('count', 'n/a')}",
                "  sampled_raw_vs_aligned_mae_deg: "
                f"{format_optional_float(raw_vs_aligned.get('mae_deg'))}",
                "  sampled_raw_vs_aligned_max_abs_error_deg: "
                f"{format_optional_float(raw_vs_aligned.get('max_abs_error_deg'))}",
            ]
        )
    aligned_vs_pdb = dihedrals.get("sampled_aligned_vs_pdb")
    if aligned_vs_pdb is not None:
        lines.extend(
            [
                "  sampled_aligned_vs_pdb_count: "
                f"{aligned_vs_pdb.get('count', 'n/a')}",
                "  sampled_aligned_vs_pdb_mae_deg: "
                f"{format_optional_float(aligned_vs_pdb.get('mae_deg'))}",
                "  sampled_aligned_vs_pdb_max_abs_error_deg: "
                f"{format_optional_float(aligned_vs_pdb.get('max_abs_error_deg'))}",
            ]
        )

    by_key = dihedrals.get("by_key") or {}
    if by_key:
        lines.extend(["", "Dihedral breakdown"])
        for key, stats in by_key.items():
            lines.append(
                "  "
                f"{key}: count={stats['count']}, "
                f"mae_deg={format_optional_float(stats['mae_deg'])}, "
                f"rmse_deg={format_optional_float(stats['rmse_deg'])}, "
                f"max_abs_error_deg={format_optional_float(stats['max_abs_error_deg'])}"
            )

    histograms = dihedrals.get("histograms")
    if histograms:
        lines.extend(
            [
                "",
                "Dihedral histograms",
                f"  angle_bins: {histograms.get('angle_bins', 'n/a')}",
                f"  error_bins: {histograms.get('error_bins', 'n/a')}",
                f"  angle_histogram_csv: {histograms.get('angle_histogram_csv')}",
                f"  error_histogram_csv: {histograms.get('error_histogram_csv')}",
            ]
        )
        if histograms.get("angle_histogram_png") is not None:
            lines.append(f"  angle_histogram_png: {histograms.get('angle_histogram_png')}")
        if histograms.get("error_histogram_png") is not None:
            lines.append(f"  error_histogram_png: {histograms.get('error_histogram_png')}")

    lines.extend(
        [
            "",
            "Main artifacts",
            f"  sampled_model_cif: {sampled_cif_path}",
            f"  detailed_arrays_npz: {arrays_path}",
        ]
    )
    if target_cif_path is not None:
        lines.append(f"  target_reference_cif: {target_cif_path}")
    if atom_csv_path is not None:
        lines.append(f"  atomwise_rmsd_csv: {atom_csv_path}")
    if sampled_pdb_path is not None:
        lines.append(f"  sampled_model_pdb_for_dihedrals: {sampled_pdb_path}")
    if raw_sampled_cif_path is not None:
        lines.append(f"  sampled_model_raw_unaligned_cif: {raw_sampled_cif_path}")
    if dihedral_csv_path is not None:
        lines.append(f"  dihedral_csv: {dihedral_csv_path}")

    path.write_text("\n".join(lines) + "\n")

def output_stem_for_sample(
    record_id: str,
    label_sample_index: int | None,
    sample_index: int | None = None,
) -> str:
    if label_sample_index is None:
        if sample_index is not None:
            return f"sample_{sample_index + 1:04d}_{record_id}_conditioned_eval"
        return f"{record_id}_conditioned_eval"
    return f"following_label_{label_sample_index}_conditioned_eval"
