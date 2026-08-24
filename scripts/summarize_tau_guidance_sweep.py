#!/usr/bin/env python3
"""Create a decision report for a tau/guidance conditioned-sampling sweep."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


COMPARISON_CSV = "conditioning_vs_oracle_per_structure.csv"
SELECTION_COMPARISON_CSV = "conditioning_vs_oracle_per_selection.csv"
RMSD_SELECTIONS = (
    "vmd_ca_residues",
    "ca",
    "backbone",
    "protein_not_backbone",
    "all_atoms",
)
MISMATCH_SELECTIONS = (
    "vmd_ca_residues",
    "all_atoms",
)
MISMATCH_METRICS = tuple(
    f"mean_mismatch_{selection}_percent" for selection in MISMATCH_SELECTIONS
)
RMSD_METRICS = tuple(
    f"mean_rmsd_{selection}_angstrom" for selection in RMSD_SELECTIONS
)
PANEL_TITLE_FONTSIZE = 17.0
AXIS_LABEL_FONTSIZE = 15.0
TICK_LABEL_FONTSIZE = 15.0
COMBINED_ANNOTATION_FONTSIZE = 14.0
STANDALONE_ANNOTATION_FONTSIZE = 16.0
COMBINED_RANKING_FONTSIZE = 16.0
STANDALONE_RANKING_FONTSIZE = 18.0
REPORT_TITLE_FONTSIZE = 20.0
PLOT_SPECS = (
    (
        "overall_score",
        "Balanced decision score (0–100)\nhigher is better",
        "RdYlGn",
        "01_overall_score.png",
    ),
    (
        "mean_mismatch_vmd_ca_residues_percent",
        "Mean cluster mismatch: residues in structured regions (%)\nlower is better",
        "RdYlGn_r",
        "02_cluster_mismatch_strict_vmd.png",
    ),
    (
        "mean_mismatch_all_atoms_percent",
        "Mean cluster mismatch: all residues (%)\nlower is better",
        "RdYlGn_r",
        "03_cluster_mismatch_all_residues.png",
    ),
    (
        "mean_rmsd_vmd_ca_residues_angstrom",
        "Mean $\\mathrm{C}_{\\alpha}$ in structured regions RMSD (Å)\n"
        "lower is better",
        "RdYlGn_r",
        "04_rmsd_strict_vmd_ca.png",
    ),
    (
        "mean_rmsd_ca_angstrom",
        "Mean all $\\mathrm{C}_{\\alpha}$ RMSD (Å)\nlower is better",
        "RdYlGn_r",
        "05_rmsd_all_ca.png",
    ),
    (
        "mean_rmsd_backbone_angstrom",
        "Mean backbone heavy-atoms RMSD (Å)\nlower is better",
        "RdYlGn_r",
        "06_rmsd_backbone.png",
    ),
    (
        "mean_rmsd_protein_not_backbone_angstrom",
        "Mean side-chain heavy-atoms RMSD (Å)\nlower is better",
        "RdYlGn_r",
        "07_rmsd_protein_non_backbone.png",
    ),
    (
        "mean_rmsd_all_atoms_angstrom",
        "Mean all heavy-atoms RMSD (Å)\nlower is better",
        "RdYlGn_r",
        "08_rmsd_all_atoms.png",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read compare_conditioning_to_oracle.py CSVs listed in a sweep "
            "manifest and write a parameter-level summary, combined decision "
            "report, and one standalone PNG for each report panel."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Tab-separated manifest written by the tau/guidance sweep script.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for the summary CSV and combined/standalone report PNGs.",
    )
    return parser.parse_args()


def required_float(row: dict[str, str], column: str, path: Path) -> float:
    value = row.get(column)
    if value is None or value == "":
        raise ValueError(f"{path} is missing a value for column {column!r}.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{path} contains a non-finite {column!r} value: {value!r}")
    return result


def required_int(row: dict[str, str], column: str, path: Path) -> int:
    value = row.get(column)
    if value is None or value == "":
        raise ValueError(f"{path} is missing a value for column {column!r}.")
    return int(value)


def metric_summary(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        raise ValueError(f"Cannot summarize empty metric {prefix!r}.")
    return {
        f"mean_{prefix}": statistics.fmean(values),
        f"median_{prefix}": statistics.median(values),
        f"std_{prefix}": statistics.pstdev(values),
    }


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Sweep manifest not found: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Sweep manifest contains no parameter sets: {path}")
    return rows


def load_comparison_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Comparison CSV not found: {path}. Run the sweep comparison stage first."
        )
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Comparison CSV contains no samples: {path}")
    return rows


def input_signature(rows: list[dict[str, str]], path: Path) -> tuple[tuple[int, int, int], ...]:
    signature = [
        (
            required_int(row, "sample_index", path),
            required_int(row, "seed", path),
            required_int(row, "frame_index", path),
        )
        for row in rows
    ]
    return tuple(sorted(signature))


def summarize_parameter_set(
    manifest_row: dict[str, str],
) -> tuple[dict[str, Any], tuple[tuple[int, int, int], ...]]:
    tau = float(manifest_row["tau"])
    guidance_scale = float(manifest_row["guidance_scale"])
    output_dir = Path(manifest_row["output_dir"]).expanduser().resolve()
    comparison_path = output_dir / COMPARISON_CSV
    selection_comparison_path = output_dir / SELECTION_COMPARISON_CSV
    rows = load_comparison_rows(comparison_path)
    selection_rows = load_comparison_rows(selection_comparison_path)

    selections = {row.get("selection", "") for row in rows}
    if len(selections) != 1:
        raise ValueError(
            f"Expected one comparison selection in {comparison_path}, found {selections}."
        )

    mismatch_percentages = [
        100.0 * required_float(row, "mismatch_fraction", comparison_path)
        for row in rows
    ]
    summary: dict[str, Any] = {
        "tau": tau,
        "guidance_scale": guidance_scale,
        "sample_count": len(rows),
        "comparison_selection": next(iter(selections)),
        "output_dir": str(output_dir),
        **metric_summary(mismatch_percentages, "mismatch_percent"),
        "perfect_structure_percent": 100.0
        * sum(value == 0.0 for value in mismatch_percentages)
        / len(mismatch_percentages),
    }
    expected_sample_names = sorted(row.get("sample_name", "") for row in rows)

    for selection in MISMATCH_SELECTIONS:
        selected_rows = [
            row
            for row in selection_rows
            if row.get("selection_name") == selection
        ]
        if len(selected_rows) != len(rows):
            raise ValueError(
                f"Expected {len(rows)} {selection!r} rows in "
                f"{selection_comparison_path}, found {len(selected_rows)}."
            )
        selected_sample_names = sorted(
            row.get("sample_name", "") for row in selected_rows
        )
        if selected_sample_names != expected_sample_names:
            raise ValueError(
                f"Sample names for selection {selection!r} in "
                f"{selection_comparison_path} do not match {comparison_path}."
            )
        selected_mismatch_percentages = [
            100.0
            * required_float(
                row,
                "mismatch_fraction",
                selection_comparison_path,
            )
            for row in selected_rows
        ]
        summary.update(
            metric_summary(
                selected_mismatch_percentages,
                f"mismatch_{selection}_percent",
            )
        )
        summary[f"perfect_{selection}_structure_percent"] = (
            100.0
            * sum(value == 0.0 for value in selected_mismatch_percentages)
            / len(selected_mismatch_percentages)
        )

    for selection in RMSD_SELECTIONS:
        rmsd_column = f"rmsd_{selection}_angstrom"
        reference_column = f"reference_rmsd_{selection}_angstrom"
        rmsds = [required_float(row, rmsd_column, comparison_path) for row in rows]
        reference_rmsds = [
            required_float(row, reference_column, comparison_path) for row in rows
        ]
        summary.update(metric_summary(rmsds, f"rmsd_{selection}_angstrom"))
        summary.update(
            metric_summary(
                reference_rmsds,
                f"reference_rmsd_{selection}_angstrom",
            )
        )

    return summary, input_signature(rows, comparison_path)


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = list(summaries[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def normalized_losses(
    summaries: list[dict[str, Any]],
    metric: str,
) -> np.ndarray:
    """Min-max normalize one lower-is-better metric over the tested settings."""
    values = np.asarray([float(summary[metric]) for summary in summaries])
    value_range = float(np.ptp(values))
    if value_range == 0.0:
        return np.zeros(values.shape, dtype=np.float64)
    return (values - float(np.min(values))) / value_range


def add_decision_scores(summaries: list[dict[str, Any]]) -> None:
    """Add a balanced cluster/RMSD score and deterministic rank in place."""
    mismatch_loss = np.mean(
        np.vstack(
            [normalized_losses(summaries, metric) for metric in MISMATCH_METRICS]
        ),
        axis=0,
    )
    rmsd_loss = np.mean(
        np.vstack([normalized_losses(summaries, metric) for metric in RMSD_METRICS]),
        axis=0,
    )
    mismatch_scores = 100.0 * (1.0 - mismatch_loss)
    rmsd_scores = 100.0 * (1.0 - rmsd_loss)
    overall_scores = 0.5 * mismatch_scores + 0.5 * rmsd_scores

    for index, summary in enumerate(summaries):
        summary["cluster_match_component_score"] = float(mismatch_scores[index])
        summary["rmsd_component_score"] = float(rmsd_scores[index])
        summary["overall_score"] = float(overall_scores[index])

    ranked_indices = sorted(
        range(len(summaries)),
        key=lambda index: (
            -float(summaries[index]["overall_score"]),
            float(summaries[index]["tau"]),
            float(summaries[index]["guidance_scale"]),
        ),
    )
    for rank, index in enumerate(ranked_indices, start=1):
        summaries[index]["rank"] = rank


def display_number(value: float) -> str:
    return f"{value:g}"


def metric_grid(
    summaries: list[dict[str, Any]],
    metric: str,
    tau_values: list[float],
    guidance_values: list[float],
) -> np.ndarray:
    grid = np.full((len(tau_values), len(guidance_values)), np.nan, dtype=np.float64)
    tau_indices = {value: index for index, value in enumerate(tau_values)}
    guidance_indices = {value: index for index, value in enumerate(guidance_values)}

    for summary in summaries:
        row = tau_indices[float(summary["tau"])]
        column = guidance_indices[float(summary["guidance_scale"])]
        if np.isfinite(grid[row, column]):
            raise ValueError(
                "Duplicate sweep parameter set: "
                f"tau={summary['tau']}, guidance-scale={summary['guidance_scale']}"
            )
        grid[row, column] = float(summary[metric])
    return grid


def draw_metric_heatmap(
    fig: Any,
    axis: Any,
    summaries: list[dict[str, Any]],
    metric: str,
    title: str,
    color_map_name: str,
    tau_values: list[float],
    guidance_values: list[float],
    annotation_fontsize: float = COMBINED_ANNOTATION_FONTSIZE,
) -> None:
    grid = metric_grid(
        summaries,
        metric,
        tau_values,
        guidance_values,
    )
    color_map = plt.get_cmap(color_map_name).copy()
    color_map.set_bad("#dddddd")
    image = axis.imshow(np.ma.masked_invalid(grid), cmap=color_map, aspect="auto")
    axis.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    axis.set_xlabel(r"$\text{Guidance scale } \gamma$", fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel(r"$\tau$", fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_xticks(range(len(guidance_values)))
    axis.set_xticklabels(
        [display_number(value) for value in guidance_values],
        fontsize=TICK_LABEL_FONTSIZE,
    )
    axis.set_yticks(range(len(tau_values)))
    axis.set_yticklabels(
        [display_number(value) for value in tau_values],
        fontsize=TICK_LABEL_FONTSIZE,
    )

    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            value = grid[row, column]
            if not np.isfinite(value):
                continue
            if metric == "overall_score":
                matching_summary = next(
                    summary
                    for summary in summaries
                    if float(summary["tau"]) == tau_values[row]
                    and float(summary["guidance_scale"])
                    == guidance_values[column]
                )
                best_suffix = "\nBEST" if int(matching_summary["rank"]) == 1 else ""
                label = f"{value:.1f}{best_suffix}"
            else:
                label = f"{value:.2f}"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=annotation_fontsize,
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.65,
                },
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    colorbar.ax.xaxis.get_offset_text().set_fontsize(TICK_LABEL_FONTSIZE)
    colorbar.ax.yaxis.get_offset_text().set_fontsize(TICK_LABEL_FONTSIZE)


def ranking_panel_lines(summaries: list[dict[str, Any]]) -> list[str]:
    ranked = sorted(summaries, key=lambda summary: int(summary["rank"]))
    lines = ["Top parameter combinations", ""]
    for summary in ranked[:5]:
        lines.append(
            f"{int(summary['rank'])}. τ={display_number(float(summary['tau']))}, "
            f"guidance={display_number(float(summary['guidance_scale']))}"
        )
        lines.append(
            f"    overall={float(summary['overall_score']):.1f}  "
            f"clusters={float(summary['cluster_match_component_score']):.1f}  "
            f"RMSD={float(summary['rmsd_component_score']):.1f}"
        )
    lines.extend(
        [
            "",
            "Score construction",
            "• 50% cluster agreement component",
            "  (2 mismatch selections, equally weighted)",
            "• 50% structural RMSD component",
            "  (5 atom selections, equally weighted)",
            "• Each raw metric is min–max normalized",
            "  across the tested parameter combinations.",
            "• Lower raw mismatch and RMSD are better.",
        ]
    )
    return lines


def draw_ranking_panel(
    ranking_axis: Any,
    summaries: list[dict[str, Any]],
    fontsize: float = COMBINED_RANKING_FONTSIZE,
) -> None:
    ranking_axis.axis("off")
    ranking_axis.text(
        0.02,
        0.98,
        "\n".join(ranking_panel_lines(summaries)),
        transform=ranking_axis.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        linespacing=1.25,
    )


def write_decision_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    tau_values = sorted({float(summary["tau"]) for summary in summaries})
    guidance_values = sorted(
        {float(summary["guidance_scale"]) for summary in summaries}
    )
    fig, axes = plt.subplots(3, 3, figsize=(26, 21), constrained_layout=True)
    for axis, (metric, title, color_map_name, _) in zip(
        axes.flat[: len(PLOT_SPECS)],
        PLOT_SPECS,
        strict=True,
    ):
        draw_metric_heatmap(
            fig,
            axis,
            summaries,
            metric,
            title,
            color_map_name,
            tau_values,
            guidance_values,
        )

    draw_ranking_panel(axes.flat[-1], summaries)

    sample_counts = sorted({int(summary["sample_count"]) for summary in summaries})
    count_text = ", ".join(str(count) for count in sample_counts)
    fig.suptitle(
        "Tau / guidance-scale sweep — conditioning fidelity and structural quality "
        f"(sample counts per cell: {count_text})",
        fontsize=REPORT_TITLE_FONTSIZE,
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_individual_report_panels(
    output_dir: Path,
    summaries: list[dict[str, Any]],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tau_values = sorted({float(summary["tau"]) for summary in summaries})
    guidance_values = sorted(
        {float(summary["guidance_scale"]) for summary in summaries}
    )
    output_paths = []
    for metric, title, color_map_name, filename in PLOT_SPECS:
        output_path = output_dir / filename
        fig, axis = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
        draw_metric_heatmap(
            fig,
            axis,
            summaries,
            metric,
            title,
            color_map_name,
            tau_values,
            guidance_values,
            annotation_fontsize=STANDALONE_ANNOTATION_FONTSIZE,
        )
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        output_paths.append(output_path)

    ranking_path = output_dir / "09_top_parameter_combinations.png"
    fig, axis = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    draw_ranking_panel(axis, summaries, fontsize=STANDALONE_RANKING_FONTSIZE)
    fig.savefig(ranking_path, dpi=220)
    plt.close(fig)
    output_paths.append(ranking_path)
    return output_paths


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    manifest_rows = [
        row
        for row in load_manifest(manifest_path)
        if float(row["tau"]) != 0.0
    ]
    if not manifest_rows:
        raise ValueError(
            f"Sweep manifest contains no nonzero-tau parameter sets: {manifest_path}"
        )

    summaries = []
    reference_signature = None
    reference_parameters = None
    for manifest_row in manifest_rows:
        summary, signature = summarize_parameter_set(manifest_row)
        parameters = (summary["tau"], summary["guidance_scale"])
        if reference_signature is None:
            reference_signature = signature
            reference_parameters = parameters
        elif signature != reference_signature:
            raise ValueError(
                "Sweep inputs are not paired: sample indices, seeds, or frame "
                f"indices for {parameters} differ from {reference_parameters}."
            )
        summaries.append(summary)

    summaries.sort(key=lambda row: (float(row["tau"]), float(row["guidance_scale"])))
    add_decision_scores(summaries)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "tau_guidance_sweep_summary.csv"
    report_path = out_dir / "tau_guidance_sweep_report.png"
    panel_dir = out_dir / "tau_guidance_sweep_report_panels"
    write_summary_csv(summary_path, summaries)
    write_decision_report(report_path, summaries)
    panel_paths = write_individual_report_panels(panel_dir, summaries)

    print(
        f"Verified paired sample indices, seeds, and frames across "
        f"{len(summaries)} parameter sets."
    )
    print(f"Wrote sweep summary CSV: {summary_path}")
    best = min(summaries, key=lambda summary: int(summary["rank"]))
    print(
        "Best balanced parameter combination: "
        f"tau={display_number(float(best['tau']))}, "
        f"guidance-scale={display_number(float(best['guidance_scale']))}, "
        f"score={float(best['overall_score']):.2f}"
    )
    print(f"Wrote sweep decision report: {report_path}")
    print(f"Wrote {len(panel_paths)} standalone report panels under: {panel_dir}")


if __name__ == "__main__":
    main()
