#!/usr/bin/env python3
"""Export conformations stored in an NPZ trajectory to PDB files.

This script uses a template PDB for atom/residue metadata and replaces only the
XYZ coordinates with each frame from the NPZ `xyz` array.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


ATOM_RECORDS = ("ATOM", "HETATM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert NPZ trajectory coordinates into per-frame PDB files "
            "using a template PDB."
        )
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=Path("traj_with_cluster_labels.npz"),
        help="Path to the NPZ file containing an `xyz` array.",
    )
    parser.add_argument(
        "--template-pdb",
        type=Path,
        default=Path("data/pdb_for_train/INApo_no_caps.pdb"),
        help="Template PDB path (metadata source for ATOM/HETATM lines).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/pdb_multi_struct"),
        help="Directory where per-frame PDB files will be written.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First frame index to export (inclusive).",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Last frame index to export (exclusive). Defaults to all frames.",
    )
    parser.add_argument(
        "--filename-prefix",
        type=str,
        default="conformation",
        help="Prefix used for output files, e.g. conformation_00000.pdb.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files. By default, existing files raise an error.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N frames.",
    )
    return parser.parse_args()


def split_pdb_template(
    pdb_lines: Sequence[str],
) -> Tuple[List[str], List[str], List[str]]:
    atom_indices = [
        i for i, line in enumerate(pdb_lines) if line.startswith(ATOM_RECORDS)
    ]
    if not atom_indices:
        raise ValueError("Template PDB contains no ATOM/HETATM records.")

    first_atom = atom_indices[0]
    last_atom = atom_indices[-1]
    header = list(pdb_lines[:first_atom])
    atom_lines = list(pdb_lines[first_atom : last_atom + 1])
    footer = list(pdb_lines[last_atom + 1 :])
    return header, atom_lines, footer


def atom_line_fields(atom_lines: Sequence[str]) -> List[Tuple[str, str]]:
    fields: List[Tuple[str, str]] = []
    for line in atom_lines:
        padded = line if len(line) >= 54 else line.ljust(54)
        prefix = padded[:30]
        suffix = padded[54:]
        fields.append((prefix, suffix))
    return fields


def choose_atom_indices(
    xyz_atom_count: int,
    template_atom_count: int,
    atom_residue_index: np.ndarray | None,
) -> np.ndarray | None:
    if xyz_atom_count == template_atom_count:
        return None

    if atom_residue_index is None:
        raise ValueError(
            "Atom count mismatch between NPZ xyz and template PDB, and "
            "`atom_residue_index` is missing."
        )

    if atom_residue_index.shape[0] != xyz_atom_count:
        raise ValueError(
            "`atom_residue_index` length does not match xyz atom dimension."
        )

    no_cap_mask = atom_residue_index >= 0
    if int(no_cap_mask.sum()) != template_atom_count:
        raise ValueError(
            "Atom count mismatch cannot be resolved automatically. "
            f"xyz atoms={xyz_atom_count}, template atoms={template_atom_count}, "
            f"atoms with residue_index>=0={int(no_cap_mask.sum())}."
        )

    return np.flatnonzero(no_cap_mask)


def format_atom_line(prefix: str, suffix: str, xyz: np.ndarray) -> str:
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    return f"{prefix}{x:8.3f}{y:8.3f}{z:8.3f}{suffix}\n"


def main() -> None:
    args = parse_args()

    if args.progress_every <= 0:
        raise ValueError("--progress-every must be > 0")

    npz_path = args.npz_path.resolve()
    template_path = args.template_pdb.resolve()
    out_dir = args.out_dir.resolve()

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template PDB not found: {template_path}")

    with np.load(npz_path) as data:
        if "xyz" not in data:
            raise KeyError("NPZ file does not contain required key: `xyz`.")
        xyz = data["xyz"]
        atom_residue_index = (
            data["atom_residue_index"] if "atom_residue_index" in data else None
        )

        if xyz.ndim != 3 or xyz.shape[2] != 3:
            raise ValueError(
                f"`xyz` must have shape (n_frames, n_atoms, 3), got {xyz.shape}."
            )

        total_frames, xyz_atoms, _ = xyz.shape
        start = args.start
        stop = total_frames if args.stop is None else args.stop

        if start < 0 or start >= total_frames:
            raise ValueError(
                f"--start must be in [0, {total_frames - 1}], got {start}."
            )
        if stop <= start or stop > total_frames:
            raise ValueError(
                f"--stop must be in [{start + 1}, {total_frames}], got {stop}."
            )

        template_lines = template_path.read_text().splitlines()
        header_lines, template_atom_lines, footer_lines = split_pdb_template(template_lines)
        template_atoms = len(template_atom_lines)
        selected_atom_indices = choose_atom_indices(
            xyz_atom_count=xyz_atoms,
            template_atom_count=template_atoms,
            atom_residue_index=atom_residue_index,
        )

        atom_fields = atom_line_fields(template_atom_lines)
        header_text = "".join(f"{line}\n" for line in header_lines)
        footer_text = "".join(f"{line}\n" for line in footer_lines)

        out_dir.mkdir(parents=True, exist_ok=True)
        width = max(5, len(str(stop - 1)))
        frames_to_export = stop - start

        if selected_atom_indices is None:
            print(
                f"Using all atoms from xyz ({xyz_atoms}) to match template atoms ({template_atoms})."
            )
        else:
            print(
                "Detected cap atoms in xyz. "
                f"Using {len(selected_atom_indices)} no-cap atoms to match template."
            )

        print(
            f"Exporting {frames_to_export} frame(s) from {npz_path} "
            f"to {out_dir} using template {template_path}."
        )

        for frame_idx in range(start, stop):
            out_path = out_dir / f"{args.filename_prefix}_{frame_idx:0{width}d}.pdb"
            if out_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Output file exists: {out_path}. "
                    "Use --overwrite to replace existing files."
                )

            frame_xyz = (
                xyz[frame_idx]
                if selected_atom_indices is None
                else xyz[frame_idx, selected_atom_indices, :]
            )
            if frame_xyz.shape[0] != template_atoms:
                raise ValueError(
                    "Frame atom count does not match template atom count: "
                    f"frame atoms={frame_xyz.shape[0]}, template atoms={template_atoms}."
                )

            with out_path.open("w") as fh:
                if header_text:
                    fh.write(header_text)
                for (prefix, suffix), atom_xyz in zip(atom_fields, frame_xyz, strict=True):
                    fh.write(format_atom_line(prefix, suffix, atom_xyz))
                if footer_text:
                    fh.write(footer_text)

            done = frame_idx - start + 1
            if done % args.progress_every == 0 or done == frames_to_export:
                print(f"Wrote {done}/{frames_to_export} PDB files...")

        print("Export complete.")


if __name__ == "__main__":
    main()
