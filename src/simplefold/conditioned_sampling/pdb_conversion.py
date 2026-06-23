#!/usr/bin/env python3
"""Utilities extracted from scripts/sample_with_conditioning.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from .constants import CONDITIONED_EVAL_SAMPLED_CIF_TOKEN, PDB_ATOM_RECORDS, REPO_ROOT


def conditioned_eval_sampled_pdb_path(cif_path: Path) -> Path:
    if cif_path.name.endswith(CONDITIONED_EVAL_SAMPLED_CIF_TOKEN):
        return cif_path.with_suffix(".pdb")

    pdb_name = cif_path.name.replace(
        CONDITIONED_EVAL_SAMPLED_CIF_TOKEN,
        "_conditioned_eval_sampled.pdb",
        1,
    )
    return cif_path.with_name(pdb_name)

def resolve_conditioned_eval_converter_base_path(
    configured_base_path: Path,
    output_dir: Path,
) -> Path:
    configured_base_path = configured_base_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if configured_base_path.exists() and output_dir.is_relative_to(configured_base_path):
        return configured_base_path
    return output_dir

def run_conditioned_eval_cif_to_pdb_converter(base_path: Path) -> None:
    converter_path = REPO_ROOT / "scripts" / "convert_conditioned_eval_cifs_to_pdb.py"
    command = [
        sys.executable,
        str(converter_path),
        "--base-path",
        str(base_path),
    ]
    print("Running CIF-to-PDB converter: " + " ".join(command))
    subprocess.run(command, check=True)

def ensure_conditioned_eval_sampled_pdb(
    sampled_cif_path: Path,
    configured_base_path: Path,
    output_dir: Path,
    current_file_only: bool = False,
) -> Path:
    sampled_pdb_path = conditioned_eval_sampled_pdb_path(sampled_cif_path)
    converter_base_path = (
        sampled_cif_path
        if current_file_only
        else resolve_conditioned_eval_converter_base_path(
            configured_base_path,
            output_dir,
        )
    )

    try:
        run_conditioned_eval_cif_to_pdb_converter(converter_base_path)
    except subprocess.CalledProcessError:
        if converter_base_path == sampled_cif_path:
            raise
        print(
            "Warning: base-path conversion failed; retrying only the current "
            f"sampled CIF: {sampled_cif_path}"
        )
        run_conditioned_eval_cif_to_pdb_converter(sampled_cif_path)

    if not sampled_pdb_path.exists():
        run_conditioned_eval_cif_to_pdb_converter(sampled_cif_path)

    if not sampled_pdb_path.exists():
        raise FileNotFoundError(
            "CIF-to-PDB conversion did not produce the expected sampled PDB: "
            f"{sampled_pdb_path}"
        )
    return sampled_pdb_path

def read_pdb_atom_coordinates(path: Path) -> np.ndarray:
    coords: list[list[float]] = []
    with path.open() as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.startswith(PDB_ATOM_RECORDS):
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

def load_sampled_pdb_dihedral_coords(
    sampled_pdb_path: Path,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    coords = read_pdb_atom_coordinates(sampled_pdb_path)
    if coords.shape != expected_shape:
        raise ValueError(
            f"Converted sampled PDB has coordinates with shape {coords.shape}, "
            f"but expected {expected_shape}: {sampled_pdb_path}"
        )
    return coords

def validate_sampled_pdb_matches_coords(
    sampled_pdb_path: Path,
    sampled_coords: np.ndarray,
    *,
    atol: float = 1e-2,
    allow_global_inversion: bool = True,
) -> np.ndarray:
    pdb_coords = load_sampled_pdb_dihedral_coords(
        sampled_pdb_path,
        expected_shape=sampled_coords.shape,
    )
    sampled_coords = sampled_coords.astype(np.float32, copy=False)
    pdb_coords_f32 = pdb_coords.astype(np.float32, copy=False)
    deltas = np.abs(pdb_coords_f32 - sampled_coords)
    max_abs_delta = float(np.max(deltas))
    if max_abs_delta <= float(atol):
        return pdb_coords

    if allow_global_inversion:
        inverted_deltas = np.abs(pdb_coords_f32 + sampled_coords)
        inverted_max_abs_delta = float(np.max(inverted_deltas))
        if inverted_max_abs_delta <= float(atol):
            print(
                "Converted sampled PDB coordinates match the sampled arrays after "
                f"global chirality inversion: {sampled_pdb_path}"
            )
            return pdb_coords

    rmsd = float(np.sqrt(np.mean(np.sum(deltas * deltas, axis=-1))))
    message = (
        f"Converted sampled PDB coordinates do not match the sampled arrays "
        f"for {sampled_pdb_path}: max_abs_delta={max_abs_delta:.4f}, "
        f"rmsd={rmsd:.4f}."
    )
    if allow_global_inversion:
        inverted_deltas = np.abs(pdb_coords_f32 + sampled_coords)
        inverted_rmsd = float(
            np.sqrt(np.mean(np.sum(inverted_deltas * inverted_deltas, axis=-1)))
        )
        message += (
            f" Also checked global inversion: "
            f"max_abs_delta={float(np.max(inverted_deltas)):.4f}, "
            f"rmsd={inverted_rmsd:.4f}."
        )
    raise ValueError(message)
