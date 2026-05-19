#!/usr/bin/env python3
"""Export one frame from an NPZ `trajectory` array to a PDB file.

The PDB template supplies atom/residue metadata. Coordinates are replaced with
the selected frame from the NPZ trajectory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


ATOM_RECORDS = ("ATOM", "HETATM")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ_PATH = REPO_ROOT / "test_new_data_with_clusters" / "pas_without_hs.npz"
DEFAULT_TEMPLATE_PDB = REPO_ROOT / "data" / "pdb_for_train" / "INApo_no_caps.pdb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write one PDB by taking coordinates from NPZ key `trajectory` and "
            "metadata from a template PDB."
        )
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=DEFAULT_NPZ_PATH,
        help=f"Input NPZ path. Default: {DEFAULT_NPZ_PATH}",
    )
    parser.add_argument(
        "--template-pdb",
        type=Path,
        default=DEFAULT_TEMPLATE_PDB,
        help=f"Template PDB path. Default: {DEFAULT_TEMPLATE_PDB}",
    )
    parser.add_argument(
        "--output-pdb",
        type=Path,
        default=None,
        help=(
            "Output PDB path. Default: next to the NPZ as "
            "<npz-stem>_frame_<index>.pdb."
        ),
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Index into trajectory's first dimension. Negative indices are allowed.",
    )
    parser.add_argument(
        "--template-atom-mode",
        choices=("auto", "all", "heavy"),
        default="auto",
        help=(
            "Which template atom records to use. `auto` uses all atoms if counts "
            "match, otherwise non-hydrogen atoms if that count matches."
        ),
    )
    parser.add_argument(
        "--no-validate-atom-names",
        action="store_true",
        help="Skip validation of NPZ atom_names against selected template atom names.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output PDB if it already exists.",
    )
    return parser.parse_args()


def pdb_atom_name(line: str) -> str:
    return line[12:16].strip()


def is_hydrogen_atom(line: str) -> bool:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element in {"H", "D"}

    name = pdb_atom_name(line).upper()
    return name.startswith(("H", "D"))


def split_template_lines(
    pdb_lines: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    atom_indices = [
        i for i, line in enumerate(pdb_lines) if line.startswith(ATOM_RECORDS)
    ]
    if not atom_indices:
        raise ValueError("Template PDB contains no ATOM/HETATM records.")

    first_atom = atom_indices[0]
    last_atom = atom_indices[-1]
    header = list(pdb_lines[:first_atom])
    atom_lines = [
        line
        for line in pdb_lines[first_atom : last_atom + 1]
        if line.startswith(ATOM_RECORDS)
    ]
    footer = list(pdb_lines[last_atom + 1 :])
    return header, atom_lines, footer


def select_template_atom_lines(
    atom_lines: Sequence[str], frame_atom_count: int, atom_mode: str
) -> tuple[list[str], str]:
    all_atom_lines = list(atom_lines)
    heavy_atom_lines = [line for line in atom_lines if not is_hydrogen_atom(line)]

    if atom_mode == "all":
        selected_atom_lines = all_atom_lines
    elif atom_mode == "heavy":
        selected_atom_lines = heavy_atom_lines
    else:
        if frame_atom_count == len(all_atom_lines):
            return all_atom_lines, "all"
        if frame_atom_count == len(heavy_atom_lines):
            return heavy_atom_lines, "heavy"
        raise ValueError(
            "Could not match trajectory atom count to template atoms: "
            f"trajectory has {frame_atom_count}, template has "
            f"{len(all_atom_lines)} total atoms and {len(heavy_atom_lines)} "
            "non-hydrogen atoms."
        )

    if frame_atom_count != len(selected_atom_lines):
        raise ValueError(
            f"Trajectory atom count ({frame_atom_count}) does not match selected "
            f"template atom count ({len(selected_atom_lines)}) for mode `{atom_mode}`."
        )
    return selected_atom_lines, atom_mode


def normalize_frame_index(frame_index: int, total_frames: int) -> int:
    if -total_frames <= frame_index < total_frames:
        return frame_index % total_frames
    raise IndexError(
        f"--frame-index must be in [{-total_frames}, {total_frames - 1}], "
        f"got {frame_index}."
    )


def validate_atom_names(npz_atom_names: np.ndarray, template_atom_lines: Sequence[str]) -> None:
    template_names = [pdb_atom_name(line) for line in template_atom_lines]
    npz_names = [str(name).strip() for name in npz_atom_names]

    if len(npz_names) != len(template_names):
        raise ValueError(
            f"NPZ atom_names length ({len(npz_names)}) does not match selected "
            f"template atom count ({len(template_names)})."
        )

    mismatches = [
        (i, npz_name, template_name)
        for i, (npz_name, template_name) in enumerate(zip(npz_names, template_names))
        if npz_name != template_name
    ]
    if mismatches:
        preview = ", ".join(
            f"index {i}: npz={npz_name!r}, template={template_name!r}"
            for i, npz_name, template_name in mismatches[:10]
        )
        raise ValueError(
            "NPZ atom_names do not match selected template atom names. "
            f"First mismatches: {preview}"
        )


def format_atom_line(template_line: str, xyz: np.ndarray) -> str:
    padded = template_line if len(template_line) >= 54 else template_line.ljust(54)
    prefix = padded[:30]
    suffix = padded[54:]
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    return f"{prefix}{x:8.3f}{y:8.3f}{z:8.3f}{suffix}\n"


def default_output_path(npz_path: Path, frame_index: int) -> Path:
    return npz_path.with_name(f"{npz_path.stem}_frame_{frame_index:05d}.pdb")


def main() -> None:
    args = parse_args()

    npz_path = args.npz_path.expanduser().resolve()
    template_path = args.template_pdb.expanduser().resolve()

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template PDB not found: {template_path}")

    template_lines = template_path.read_text().splitlines()
    header_lines, template_atom_lines, footer_lines = split_template_lines(template_lines)

    with np.load(npz_path) as data:
        if "trajectory" not in data:
            raise KeyError("NPZ file does not contain required key `trajectory`.")

        trajectory = data["trajectory"]
        if trajectory.ndim != 3 or trajectory.shape[2] != 3:
            raise ValueError(
                "`trajectory` must have shape (n_frames, n_atoms, 3), "
                f"got {trajectory.shape}."
            )

        total_frames, frame_atom_count, _ = trajectory.shape
        frame_index = normalize_frame_index(args.frame_index, total_frames)
        frame_xyz = np.asarray(trajectory[frame_index], dtype=np.float64)

        selected_atom_lines, selected_atom_mode = select_template_atom_lines(
            template_atom_lines, frame_atom_count, args.template_atom_mode
        )

        if not args.no_validate_atom_names and "atom_names" in data:
            validate_atom_names(data["atom_names"], selected_atom_lines)

    if not np.isfinite(frame_xyz).all():
        raise ValueError(f"Selected frame {frame_index} contains non-finite coordinates.")

    output_path = (
        args.output_pdb.expanduser().resolve()
        if args.output_pdb is not None
        else default_output_path(npz_path, frame_index)
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output PDB already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as fh:
        for line in header_lines:
            fh.write(f"{line}\n")
        for template_line, atom_xyz in zip(selected_atom_lines, frame_xyz, strict=True):
            fh.write(format_atom_line(template_line, atom_xyz))
        for line in footer_lines:
            fh.write(f"{line}\n")

    print(
        f"Wrote frame {frame_index} from {npz_path} to {output_path} "
        f"using {selected_atom_mode} template atoms ({len(selected_atom_lines)} records)."
    )


if __name__ == "__main__":
    main()
