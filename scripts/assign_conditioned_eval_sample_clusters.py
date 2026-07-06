#!/usr/bin/env python3
"""Assign custom clusters to conditioned evaluation sampled structures."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE_ROOT = Path("/home/nobilm@usi.ch/PHASE")
DEFAULT_PHASE_PYTHON = DEFAULT_PHASE_ROOT / ".venv-phase" / "bin" / "python"
DEFAULT_ASSIGN_SCRIPT = DEFAULT_PHASE_ROOT / "scripts" / "assign_custom_structure_clusters.py"
DEFAULT_CLUSTER_DIR = Path(
    # "/storage_common/angiod/phase-data/projects/a2a/systems/a2a/clusters/"
    # "cb3c3cae-5316-47db-8fbb-0567d5f0f75b/"
    "/storage_common/angiod/phase-data/projects/a2a/systems/a2a_small/clusters/6d4a7baa-c096-494e-b417-c8014437d37d/"
)
DEFAULT_MATCH_TOKENS = (
    "_conditioned_eval_sampled.npz",
    "_conditioned_eval_sampled.pdb",
)
OUTPUT_TOKEN = "_assigned_clusters.npz"
REPORT_TOKEN = "_assigned_clusters_report.txt"
SAMPLED_CIF_TOKEN = "_conditioned_eval_sampled.cif"
SAMPLED_NPZ_TOKEN = "_conditioned_eval_sampled.npz"
SAMPLED_PDB_TOKEN = "_conditioned_eval_sampled.pdb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find conditioned evaluation sampled structures under a base path and "
            "run PHASE scripts/assign_custom_structure_clusters.py for each match. "
            "Each successful assignment also writes a sidecar report containing "
            "PDB-vs-CIF coordinate consistency metrics."
        )
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Directory or single file to search for conditioned evaluation samples.",
    )
    parser.add_argument(
        "--cluster-dir",
        type=Path,
        default=DEFAULT_CLUSTER_DIR,
        help="Cluster NPZ passed to --cluster-dir.",
    )
    parser.add_argument(
        "--assign-script",
        type=Path,
        default=DEFAULT_ASSIGN_SCRIPT,
        help="Path to PHASE assign_custom_structure_clusters.py.",
    )
    parser.add_argument(
        "--phase-python",
        type=Path,
        default=DEFAULT_PHASE_PYTHON,
        help="Python executable from the PHASE virtualenv.",
    )
    parser.add_argument(
        "--match-token",
        dest="match_tokens",
        action="append",
        default=None,
        help=(
            "Filename token to match. Can be passed multiple times. Defaults to "
            "matching both '_conditioned_eval_sampled.npz' and "
            "'_conditioned_eval_sampled.pdb'."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose output *_assigned_clusters.npz already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def find_sampled_structures(base_path: Path, match_tokens: tuple[str, ...]) -> list[Path]:
    def is_match(path: Path) -> bool:
        return path.is_file() and any(token in path.name for token in match_tokens)

    if base_path.is_file():
        return [base_path] if is_match(base_path) else []

    return sorted(path for path in base_path.rglob("*") if is_match(path))


def assigned_clusters_path(structure_path: Path, match_tokens: tuple[str, ...]) -> Path:
    for token in match_tokens:
        if token in structure_path.name:
            return structure_path.with_name(structure_path.name.replace(token, OUTPUT_TOKEN, 1))

    return structure_path.with_name(f"{structure_path.stem}{OUTPUT_TOKEN}")


def assignment_report_path(output_path: Path) -> Path:
    if output_path.name.endswith(OUTPUT_TOKEN):
        return output_path.with_name(output_path.name.replace(OUTPUT_TOKEN, REPORT_TOKEN, 1))
    return output_path.with_suffix(output_path.suffix + ".report.txt")


def paired_sampled_pdb_cif_paths(sampled_structure: Path) -> tuple[Path | None, Path | None]:
    name = sampled_structure.name
    if SAMPLED_PDB_TOKEN in name:
        return sampled_structure, sampled_structure.with_name(
            name.replace(SAMPLED_PDB_TOKEN, SAMPLED_CIF_TOKEN, 1)
        )
    if SAMPLED_CIF_TOKEN in name:
        return sampled_structure.with_name(
            name.replace(SAMPLED_CIF_TOKEN, SAMPLED_PDB_TOKEN, 1)
        ), sampled_structure
    if SAMPLED_NPZ_TOKEN in name:
        return (
            sampled_structure.with_name(name.replace(SAMPLED_NPZ_TOKEN, SAMPLED_PDB_TOKEN, 1)),
            sampled_structure.with_name(name.replace(SAMPLED_NPZ_TOKEN, SAMPLED_CIF_TOKEN, 1)),
        )
    if sampled_structure.suffix.lower() == ".pdb":
        return sampled_structure, sampled_structure.with_suffix(".cif")
    if sampled_structure.suffix.lower() == ".cif":
        return sampled_structure.with_suffix(".pdb"), sampled_structure
    return None, None


def read_pdb_atom_coordinates(path: Path) -> np.ndarray:
    coords: list[list[float]] = []
    with path.open() as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if len(line) < 54:
                raise ValueError(f"{path}: malformed ATOM/HETATM line at {line_num}.")
            try:
                coords.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
            except ValueError as exc:
                raise ValueError(
                    f"{path}: malformed coordinates in ATOM/HETATM line {line_num}."
                ) from exc

    if not coords:
        raise ValueError(f"{path}: no ATOM/HETATM records found.")
    return np.asarray(coords, dtype=np.float32)


def read_mmcif_atom_coordinates(path: Path) -> np.ndarray:
    coords: list[list[float]] = []
    atom_site_fields: list[str] = []
    cartn_indices: tuple[int, int, int] | None = None

    with path.open() as handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped == "loop_":
                atom_site_fields = []
                cartn_indices = None
                continue

            if stripped.startswith("_atom_site."):
                atom_site_fields.append(stripped.split()[0])
                cartn_indices = None
                continue

            if stripped.startswith("_"):
                atom_site_fields = []
                cartn_indices = None
                continue

            if not stripped.startswith(("ATOM", "HETATM")):
                continue

            tokens = shlex.split(stripped)
            if atom_site_fields:
                if cartn_indices is None:
                    try:
                        cartn_indices = (
                            atom_site_fields.index("_atom_site.Cartn_x"),
                            atom_site_fields.index("_atom_site.Cartn_y"),
                            atom_site_fields.index("_atom_site.Cartn_z"),
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"{path}: atom_site loop does not define Cartn_x/y/z fields."
                        ) from exc
                coordinate_indices = cartn_indices
            else:
                coordinate_indices = (10, 11, 12)

            if len(tokens) <= max(coordinate_indices):
                raise ValueError(
                    f"{path}: malformed atom_site row at line {line_num}; "
                    f"expected coordinate columns {coordinate_indices}."
                )

            try:
                coords.append([float(tokens[idx]) for idx in coordinate_indices])
            except ValueError as exc:
                raise ValueError(
                    f"{path}: malformed coordinates in atom_site row {line_num}."
                ) from exc

    if not coords:
        raise ValueError(f"{path}: no atom_site ATOM/HETATM rows found.")
    return np.asarray(coords, dtype=np.float32)


def evaluate_pdb_vs_cif_coordinates(sampled_structure: Path) -> dict[str, Any]:
    pdb_path, cif_path = paired_sampled_pdb_cif_paths(sampled_structure)
    if pdb_path is None or cif_path is None:
        return {
            "status": "skipped",
            "reason": f"Could not infer paired PDB/CIF paths from {sampled_structure}.",
        }
    if not pdb_path.exists():
        return {
            "status": "skipped",
            "pdb_path": str(pdb_path),
            "cif_path": str(cif_path),
            "reason": f"Paired PDB does not exist: {pdb_path}",
        }
    if not cif_path.exists():
        return {
            "status": "skipped",
            "pdb_path": str(pdb_path),
            "cif_path": str(cif_path),
            "reason": f"Paired CIF does not exist: {cif_path}",
        }

    try:
        pdb_coords = read_pdb_atom_coordinates(pdb_path)
        cif_coords = read_mmcif_atom_coordinates(cif_path)
        if pdb_coords.shape != cif_coords.shape:
            return {
                "status": "failed",
                "pdb_path": str(pdb_path),
                "cif_path": str(cif_path),
                "pdb_shape": tuple(int(v) for v in pdb_coords.shape),
                "cif_shape": tuple(int(v) for v in cif_coords.shape),
                "error": (
                    f"PDB coordinate shape {pdb_coords.shape} does not match "
                    f"CIF coordinate shape {cif_coords.shape}."
                ),
            }

        deltas = np.abs(pdb_coords - cif_coords)
        return {
            "status": "ok",
            "pdb_path": str(pdb_path),
            "cif_path": str(cif_path),
            "atom_count": int(pdb_coords.shape[0]),
            "max_abs_delta_A": float(np.max(deltas)),
            "rmsd_A": float(np.sqrt(np.mean(np.sum(deltas * deltas, axis=-1)))),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "pdb_path": str(pdb_path),
            "cif_path": str(cif_path),
            "error": str(exc),
        }


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def write_assignment_report(
    report_path: Path,
    *,
    sampled_structure: Path,
    output_path: Path,
    cluster_dir: Path,
    assign_script: Path,
    phase_python: Path,
    coordinate_metrics: dict[str, Any],
) -> None:
    lines = [
        "Conditioned evaluation sample cluster assignment",
        "",
        "Inputs",
        f"  sampled_structure: {sampled_structure}",
        f"  cluster_dir: {cluster_dir}",
        f"  assign_script: {assign_script}",
        f"  phase_python: {phase_python}",
        "",
        "Outputs",
        f"  assigned_clusters_npz: {output_path}",
        "",
        "PDB vs sampled CIF coordinate consistency",
    ]

    status = coordinate_metrics.get("status")
    if status == "ok":
        lines.extend(
            [
                f"  status: ok",
                f"  pdb_path: {coordinate_metrics.get('pdb_path')}",
                f"  cif_path: {coordinate_metrics.get('cif_path')}",
                f"  atom_count: {coordinate_metrics.get('atom_count')}",
                "  max_abs_delta_A: "
                f"{format_optional_float(coordinate_metrics.get('max_abs_delta_A'))}",
                f"  rmsd_A: {format_optional_float(coordinate_metrics.get('rmsd_A'))}",
            ]
        )
    elif status == "skipped":
        lines.extend(
            [
                "  status: skipped",
                f"  pdb_path: {coordinate_metrics.get('pdb_path')}",
                f"  cif_path: {coordinate_metrics.get('cif_path')}",
                f"  reason: {coordinate_metrics.get('reason')}",
            ]
        )
    else:
        lines.extend(
            [
                "  status: failed",
                f"  pdb_path: {coordinate_metrics.get('pdb_path')}",
                f"  cif_path: {coordinate_metrics.get('cif_path')}",
                f"  error: {coordinate_metrics.get('error')}",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    base_path = args.base_path.resolve()
    cluster_dir = args.cluster_dir.resolve()
    assign_script = args.assign_script.resolve()
    phase_python = args.phase_python.expanduser()
    if not phase_python.is_absolute():
        phase_python = Path.cwd() / phase_python
    match_tokens = tuple(args.match_tokens or DEFAULT_MATCH_TOKENS)

    if not base_path.exists():
        raise FileNotFoundError(f"Base path not found: {base_path}")

    if not assign_script.exists() and not args.dry_run:
        raise FileNotFoundError(f"Assignment script not found: {assign_script}")

    if not phase_python.exists() and not args.dry_run:
        raise FileNotFoundError(f"PHASE Python not found: {phase_python}")

    sampled_structures = find_sampled_structures(base_path, match_tokens)
    if not sampled_structures:
        tokens = ", ".join(repr(token) for token in match_tokens)
        print(f"No sampled structures matching {tokens} found under {base_path}.")
        return

    print(f"Found {len(sampled_structures)} sampled structure(s).")

    completed = 0
    skipped = 0
    failures: list[tuple[Path, subprocess.CalledProcessError]] = []

    for sampled_structure in sampled_structures:
        output_path = assigned_clusters_path(sampled_structure, match_tokens)

        if output_path.exists() and args.skip_existing:
            skipped += 1
            print(f"Skipping existing output: {output_path}")
            continue

        command = [
            str(phase_python),
            str(assign_script),
            "--cluster-dir",
            str(cluster_dir),
            "--structure",
            str(sampled_structure),
            "--output",
            str(output_path),
        ]

        if args.dry_run:
            print(" ".join(command))
            continue

        print(f"Assigning clusters: {sampled_structure} -> {output_path}")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append((sampled_structure, exc))
            print(f"Failed: {sampled_structure} (exit code {exc.returncode})")
            continue

        coordinate_metrics = evaluate_pdb_vs_cif_coordinates(sampled_structure)
        report_path = assignment_report_path(output_path)
        write_assignment_report(
            report_path,
            sampled_structure=sampled_structure,
            output_path=output_path,
            cluster_dir=cluster_dir,
            assign_script=assign_script,
            phase_python=phase_python,
            coordinate_metrics=coordinate_metrics,
        )
        if coordinate_metrics.get("status") == "ok":
            print(
                "PDB vs sampled CIF coordinate consistency: "
                f"max_abs_delta_A={coordinate_metrics['max_abs_delta_A']:.6f}, "
                f"rmsd_A={coordinate_metrics['rmsd_A']:.6f}"
            )
        else:
            print(
                "Warning: PDB vs sampled CIF coordinate consistency "
                f"{coordinate_metrics.get('status')}: "
                f"{coordinate_metrics.get('reason') or coordinate_metrics.get('error')}"
            )
        print(f"Wrote report: {report_path}")

        completed += 1

    if args.dry_run:
        print(f"Dry run complete. {len(sampled_structures)} command(s) prepared.")
        return

    print(
        f"Done. Assigned {completed} file(s), skipped {skipped} existing file(s), "
        f"failed {len(failures)} file(s)."
    )

    if failures:
        failed_paths = "\n".join(f"  {path}: exit code {exc.returncode}" for path, exc in failures)
        raise RuntimeError(f"Failed to assign clusters for {len(failures)} file(s):\n{failed_paths}")


if __name__ == "__main__":
    main()
