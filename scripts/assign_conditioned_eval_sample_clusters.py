#!/usr/bin/env python3
"""Assign custom clusters to conditioned evaluation sampled structures."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE_ROOT = Path("/home/nobilm@usi.ch/PHASE")
DEFAULT_PHASE_PYTHON = DEFAULT_PHASE_ROOT / ".venv-phase" / "bin" / "python"
DEFAULT_ASSIGN_SCRIPT = DEFAULT_PHASE_ROOT / "scripts" / "assign_custom_structure_clusters.py"
DEFAULT_CLUSTER_DIR = Path(
    "/storage_common/angiod/phase-data/projects/a2a/systems/a2a/clusters/"
    "cb3c3cae-5316-47db-8fbb-0567d5f0f75b/"
)
DEFAULT_MATCH_TOKENS = (
    "_conditioned_eval_sampled.npz",
    "_conditioned_eval_sampled.pdb",
)
OUTPUT_TOKEN = "_assigned_clusters.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find conditioned evaluation sampled structures under a base path and "
            "run PHASE scripts/assign_custom_structure_clusters.py for each match."
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
