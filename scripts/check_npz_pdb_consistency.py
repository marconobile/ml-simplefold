#!/usr/bin/env python3
"""Check 1-to-1 consistency between NPZ frames and exported PDB files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


ATOM_RECORDS = ("ATOM", "HETATM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that NPZ frames and PDB files have a strict 1-to-1 mapping "
            "and matching coordinates."
        )
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=Path("traj_with_cluster_labels.npz"),
        help="Path to NPZ file containing `xyz`.",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=Path("data/pdb_multi_struct"),
        help="Directory containing per-frame PDBs.",
    )
    parser.add_argument(
        "--filename-prefix",
        type=str,
        default="conformation",
        help="Expected file prefix, e.g. conformation_00000.pdb.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First frame index to check (inclusive).",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Last frame index to check (exclusive). Defaults to all frames.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=6e-4,
        help="Maximum allowed absolute coordinate error in Angstrom.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N checked frames.",
    )
    parser.add_argument(
        "--max-reported",
        type=int,
        default=10,
        help="Max number of examples printed per error class.",
    )
    parser.add_argument(
        "--allow-extra-files",
        action="store_true",
        help="Do not fail when PDB directory has files outside checked frame range.",
    )
    return parser.parse_args()


def choose_atom_indices(
    xyz_atom_count: int,
    pdb_atom_count: int,
    atom_residue_index: np.ndarray | None,
) -> np.ndarray | None:
    if xyz_atom_count == pdb_atom_count:
        return None

    if atom_residue_index is None:
        raise ValueError(
            "Atom count mismatch between NPZ xyz and PDB files, and "
            "`atom_residue_index` is missing."
        )

    if atom_residue_index.shape[0] != xyz_atom_count:
        raise ValueError(
            "`atom_residue_index` length does not match xyz atom dimension."
        )

    no_cap_mask = atom_residue_index >= 0
    if int(no_cap_mask.sum()) != pdb_atom_count:
        raise ValueError(
            "Atom count mismatch cannot be resolved automatically. "
            f"xyz atoms={xyz_atom_count}, pdb atoms={pdb_atom_count}, "
            f"atoms with residue_index>=0={int(no_cap_mask.sum())}."
        )

    return np.flatnonzero(no_cap_mask)


def extract_frame_index(path: Path, prefix: str) -> Optional[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.pdb$")
    match = pattern.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def collect_pdb_mapping(
    pdb_dir: Path, prefix: str
) -> Tuple[Dict[int, Path], List[Tuple[int, Path, Path]]]:
    mapping: Dict[int, Path] = {}
    duplicates: List[Tuple[int, Path, Path]] = []

    for path in sorted(pdb_dir.glob(f"{prefix}_*.pdb")):
        idx = extract_frame_index(path, prefix)
        if idx is None:
            continue
        prev = mapping.get(idx)
        if prev is not None:
            duplicates.append((idx, prev, path))
        else:
            mapping[idx] = path
    return mapping, duplicates


def parse_pdb_coordinates(path: Path) -> np.ndarray:
    coords: List[Tuple[float, float, float]] = []
    with path.open() as fh:
        for line_num, line in enumerate(fh, start=1):
            if not line.startswith(ATOM_RECORDS):
                continue
            if len(line) < 54:
                raise ValueError(
                    f"{path}: malformed ATOM/HETATM line at {line_num} (too short)."
                )
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError as exc:
                raise ValueError(
                    f"{path}: failed parsing coordinates at line {line_num}."
                ) from exc
            coords.append((x, y, z))

    if not coords:
        raise ValueError(f"{path}: no ATOM/HETATM records found.")
    return np.asarray(coords, dtype=np.float32)


def main() -> None:
    args = parse_args()

    if args.progress_every <= 0:
        raise ValueError("--progress-every must be > 0")
    if args.max_reported <= 0:
        raise ValueError("--max-reported must be > 0")
    if args.tolerance < 0:
        raise ValueError("--tolerance must be >= 0")

    npz_path = args.npz_path.resolve()
    pdb_dir = args.pdb_dir.resolve()

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    if not pdb_dir.exists():
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")

    with np.load(npz_path, allow_pickle=False) as data:
        if "xyz" not in data:
            raise KeyError("NPZ file does not contain required key: `xyz`.")
        xyz = data["xyz"]
        atom_residue_index = (
            data["atom_residue_index"] if "atom_residue_index" in data else None
        )

    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"`xyz` must have shape (n_frames, n_atoms, 3), got {xyz.shape}.")

    total_frames, xyz_atoms, _ = xyz.shape
    start = args.start
    stop = total_frames if args.stop is None else args.stop

    if start < 0 or start >= total_frames:
        raise ValueError(f"--start must be in [0, {total_frames - 1}], got {start}.")
    if stop <= start or stop > total_frames:
        raise ValueError(f"--stop must be in [{start + 1}, {total_frames}], got {stop}.")

    expected_indices = set(range(start, stop))
    frames_to_check = stop - start

    index_to_path, duplicates = collect_pdb_mapping(pdb_dir, args.filename_prefix)
    if not index_to_path:
        raise FileNotFoundError(
            f"No files matched {args.filename_prefix}_*.pdb in {pdb_dir}."
        )
    if duplicates:
        details = "\n".join(
            f"  frame {idx}: {a.name} and {b.name}" for idx, a, b in duplicates[: args.max_reported]
        )
        raise ValueError(f"Duplicate frame-index filenames detected:\n{details}")

    found_indices = set(index_to_path.keys())
    missing_indices = sorted(expected_indices - found_indices)
    extra_indices = sorted(found_indices - expected_indices)

    # Use one representative file to infer atom count and cap-handling policy.
    common_indices = sorted(expected_indices & found_indices)
    if not common_indices:
        raise ValueError(
            "No overlap between expected frame indices and found PDB filenames."
        )
    ref_idx = common_indices[0]
    ref_path = index_to_path[ref_idx]
    ref_coords = parse_pdb_coordinates(ref_path)
    pdb_atoms = ref_coords.shape[0]

    atom_indices = choose_atom_indices(
        xyz_atom_count=xyz_atoms,
        pdb_atom_count=pdb_atoms,
        atom_residue_index=atom_residue_index,
    )

    expected_atoms = pdb_atoms
    if atom_indices is None:
        print(
            f"Using all xyz atoms ({xyz_atoms}) for comparison with PDB atoms ({pdb_atoms})."
        )
    else:
        print(
            f"Detected cap atoms in xyz. Using {len(atom_indices)} no-cap atoms for comparison."
        )

    print(
        f"Checking frames [{start}, {stop}) from {npz_path} against files in {pdb_dir}."
    )

    parse_errors = 0
    parse_error_examples: List[str] = []
    atom_count_errors = 0
    atom_count_examples: List[str] = []
    coord_errors = 0
    coord_error_examples: List[str] = []
    worst_error = -1.0
    worst_error_idx: Optional[int] = None

    checked = 0
    for idx in range(start, stop):
        path = index_to_path.get(idx)
        if path is None:
            continue

        try:
            coords = ref_coords if idx == ref_idx else parse_pdb_coordinates(path)
        except Exception as exc:  # noqa: BLE001
            parse_errors += 1
            if len(parse_error_examples) < args.max_reported:
                parse_error_examples.append(f"frame {idx}: {exc}")
            continue

        if coords.shape[0] != expected_atoms:
            atom_count_errors += 1
            if len(atom_count_examples) < args.max_reported:
                atom_count_examples.append(
                    f"frame {idx}: PDB has {coords.shape[0]} atoms; expected {expected_atoms}"
                )
            continue

        expected_xyz = xyz[idx] if atom_indices is None else xyz[idx, atom_indices, :]
        if expected_xyz.shape != coords.shape:
            atom_count_errors += 1
            if len(atom_count_examples) < args.max_reported:
                atom_count_examples.append(
                    f"frame {idx}: shape mismatch PDB {coords.shape} vs NPZ {expected_xyz.shape}"
                )
            continue

        max_abs_err = float(np.max(np.abs(coords - expected_xyz)))
        if max_abs_err > args.tolerance:
            coord_errors += 1
            if len(coord_error_examples) < args.max_reported:
                coord_error_examples.append(
                    f"frame {idx}: max |delta|={max_abs_err:.6f} A ({path.name})"
                )
        if max_abs_err > worst_error:
            worst_error = max_abs_err
            worst_error_idx = idx

        checked += 1
        if checked % args.progress_every == 0 or checked == frames_to_check:
            print(f"Checked {checked}/{frames_to_check} frame(s)...")

    failed = False

    print("\nSummary")
    print(f"- Expected frame files: {frames_to_check}")
    print(f"- Found matching prefix files: {len(found_indices)}")
    print(f"- Missing expected frame files: {len(missing_indices)}")
    print(f"- Extra frame files outside range: {len(extra_indices)}")
    print(f"- Parsed/compared frames: {checked}")
    print(f"- Parse errors: {parse_errors}")
    print(f"- Atom-count/shape mismatches: {atom_count_errors}")
    print(f"- Coordinate mismatches (> {args.tolerance} A): {coord_errors}")
    if worst_error_idx is not None:
        print(f"- Worst max |delta|: {worst_error:.6f} A at frame {worst_error_idx}")

    if missing_indices:
        failed = True
        print("Missing frame examples:", ", ".join(map(str, missing_indices[: args.max_reported])))

    if extra_indices and not args.allow_extra_files:
        failed = True
        print("Extra frame examples:", ", ".join(map(str, extra_indices[: args.max_reported])))

    if parse_error_examples:
        failed = True
        print("Parse error examples:")
        for msg in parse_error_examples:
            print(f"  {msg}")

    if atom_count_examples:
        failed = True
        print("Atom-count mismatch examples:")
        for msg in atom_count_examples:
            print(f"  {msg}")

    if coord_error_examples:
        failed = True
        print("Coordinate mismatch examples:")
        for msg in coord_error_examples:
            print(f"  {msg}")

    if failed:
        raise SystemExit(1)

    print("PASS: strict 1-to-1 frame mapping and coordinates are consistent.")


if __name__ == "__main__":
    main()
