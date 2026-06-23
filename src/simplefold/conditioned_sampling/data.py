#!/usr/bin/env python3
"""Utilities extracted from scripts/sample_with_conditioning.py."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .constants import CLUSTER_KEY, DEFAULT_DATA_PATH, DEFAULT_RAW_NPZ_CANDIDATES

from boltz_data_pipeline import const  # noqa: E402
from boltz_data_pipeline.types import ChainInfo, Record, Structure, StructureInfo  # noqa: E402
from utils.trajectory_npz_utils import (  # noqa: E402
    assert_no_conformer_coordinate_leak,
    build_static_topology,
    sanitize_record_prefix,
)


def resolve_raw_npz_path(data_path: Path, raw_npz_path: Path | None) -> Path | None:
    if raw_npz_path is not None:
        raw_npz_path = raw_npz_path.expanduser().resolve()
        if not raw_npz_path.exists():
            raise FileNotFoundError(f"Raw NPZ not found: {raw_npz_path}")
        return raw_npz_path

    data_path = data_path.expanduser().resolve()
    if data_path.is_file():
        return data_path

    candidates = [
        data_path.with_suffix(".npz"),
        data_path / "trajectory.npz",
        data_path / "active_without_hs.npz",
        *DEFAULT_RAW_NPZ_CANDIDATES,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None

def resolve_labels_npz_path(labels_npz_path: Path | None) -> Path | None:
    if labels_npz_path is None:
        return None
    labels_npz_path = labels_npz_path.expanduser().resolve()
    if not labels_npz_path.exists():
        raise FileNotFoundError(f"Labels NPZ not found: {labels_npz_path}")
    return labels_npz_path

def load_conditioning_label_rows(labels_npz_path: Path) -> np.ndarray:
    with np.load(labels_npz_path, allow_pickle=False) as data:
        if CLUSTER_KEY not in data.files:
            raise KeyError(f"Labels NPZ is missing required key `{CLUSTER_KEY}`.")
        label_rows = np.asarray(data[CLUSTER_KEY], dtype=np.int64)

    if label_rows.ndim != 2:
        raise ValueError(
            f"`{CLUSTER_KEY}` in --labels-npz-path must have shape "
            f"(n_samples, n_model_atoms), got {label_rows.shape}."
        )
    if label_rows.shape[0] == 0:
        raise ValueError("--labels-npz-path contains zero label rows.")
    if label_rows.shape[1] == 0:
        raise ValueError("--labels-npz-path contains zero labels per row.")
    if label_rows.min(initial=0) < -1:
        raise ValueError(f"`{CLUSTER_KEY}` in --labels-npz-path contains labels below -1.")
    return label_rows

def resolve_data_path(data_path: Path | None) -> Path:
    return (data_path or DEFAULT_DATA_PATH).expanduser().resolve()

def resolve_processed_dir(
    data_path: Path,
    processed_dir: Path | None,
    *,
    allow_data_path_dir: bool = True,
    allow_default_fallback: bool = True,
) -> Path | None:
    if processed_dir is not None:
        processed_dir = processed_dir.expanduser().resolve()
        if not processed_dir.exists():
            raise FileNotFoundError(f"Processed SimpleFold directory not found: {processed_dir}")
        return processed_dir

    data_path = data_path.expanduser().resolve()
    if allow_data_path_dir and data_path.is_dir():
        return data_path
    if allow_default_fallback and DEFAULT_DATA_PATH.exists() and DEFAULT_DATA_PATH.is_dir():
        return DEFAULT_DATA_PATH.resolve()
    return None

def np_scalar_to_string(data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in data.files:
        return default
    value = data[key]
    if value.shape == ():
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(-1)[0])
    return default

def choose_frame_position(
    nframes: int,
    frame_index: int | None,
    rng: np.random.Generator,
) -> int:
    if frame_index is None:
        return int(rng.integers(0, nframes))
    if frame_index < 0 or frame_index >= nframes:
        raise ValueError(f"--frame-index must be in [0, {nframes - 1}], got {frame_index}.")
    return int(frame_index)

def load_raw_frame(
    raw_npz_path: Path,
    frame_index: int | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    with np.load(raw_npz_path, allow_pickle=False) as data:
        required = {
            "trajectory",
            CLUSTER_KEY,
            "dihedrals",
            "dihedral_atom_indices",
            "dihedral_mask",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"Raw NPZ is missing required key(s): {', '.join(missing)}")

        trajectory = data["trajectory"]
        if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
            raise ValueError(
                f"`trajectory` must have shape (n_frames, n_atoms, 3), got {trajectory.shape}."
            )

        nframes, natoms, _ = trajectory.shape
        frame_position = choose_frame_position(nframes, frame_index, rng)

        frame_indices = (
            data["frame_indices"].astype(np.int64, copy=False)
            if "frame_indices" in data.files
            else np.arange(nframes, dtype=np.int64)
        )
        if frame_indices.shape[0] != nframes:
            frame_indices = np.arange(nframes, dtype=np.int64)
        frame_number = int(frame_indices[frame_position])

        sample_id = np_scalar_to_string(data, "sample_id", raw_npz_path.stem)
        record_prefix = sanitize_record_prefix(raw_npz_path.stem)
        record_id = f"{record_prefix}_{frame_number:06d}"

        cluster_labels = data[CLUSTER_KEY]
        dihedrals = data["dihedrals"]
        selected_sample = np.asarray(trajectory[frame_position], dtype=np.float32)
        original_cluster_labels = np.asarray(cluster_labels[frame_position], dtype=np.int64)
        original_dihedrals = np.asarray(dihedrals[frame_position], dtype=np.float32)

        if original_cluster_labels.shape != (natoms,):
            raise ValueError(
                f"`{CLUSTER_KEY}` frame must have shape ({natoms},), "
                f"got {original_cluster_labels.shape}."
            )
        if original_cluster_labels.min(initial=0) < -1:
            raise ValueError(f"`{CLUSTER_KEY}` contains labels below -1.")

        payload = {
            "selected_sample": selected_sample.copy(),
            "original_coords": selected_sample.copy(),
            "original_cluster_labels": original_cluster_labels.copy(),
            "original_dihedrals": original_dihedrals.copy(),
            "dihedral_atom_indices": np.asarray(data["dihedral_atom_indices"], dtype=np.int64),
            "dihedral_mask": np.asarray(data["dihedral_mask"], dtype=bool),
            "dihedral_keys": (
                np.asarray(data["dihedral_keys"]).astype(str).tolist()
                if "dihedral_keys" in data.files
                else [f"dihedral_{i}" for i in range(original_dihedrals.shape[-1])]
            ),
            "atom_names": (
                np.asarray(data["atom_names"]).astype(str)
                if "atom_names" in data.files
                else np.asarray([f"A{i}" for i in range(natoms)], dtype="U8")
            ),
            "atom_resids": (
                np.asarray(data["atom_resids"], dtype=np.int64)
                if "atom_resids" in data.files
                else None
            ),
            "frame_position": frame_position,
            "frame_index": frame_number,
            "record_id": record_id,
            "sample_id": sample_id,
        }
    return payload

def load_processed_frame(
    processed_dir: Path,
    frame_index: int | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Processed manifest not found: {manifest_path}")

    with manifest_path.open() as f:
        manifest = json.load(f)
    if not manifest:
        raise ValueError(f"Processed manifest is empty: {manifest_path}")

    if frame_index is None:
        entry = manifest[int(rng.integers(0, len(manifest)))]
    else:
        suffix = f"_{frame_index:06d}"
        matches = [record for record in manifest if str(record["id"]).endswith(suffix)]
        if not matches:
            raise ValueError(
                f"No processed record ending in {suffix} found in {manifest_path}."
            )
        entry = matches[0]

    record_id = str(entry["id"])
    struct_path = processed_dir / "structures" / f"{record_id}.npz"
    if not struct_path.exists():
        raise FileNotFoundError(f"Processed structure not found: {struct_path}")

    with np.load(struct_path, allow_pickle=False) as data:
        atoms = data["atoms"]
        assert_no_conformer_coordinate_leak(atoms, str(struct_path))
        selected_sample = np.asarray(atoms["coords"], dtype=np.float32)
        if CLUSTER_KEY not in data.files:
            raise KeyError(f"Processed structure is missing `{CLUSTER_KEY}`: {struct_path}")
        original_cluster_labels = np.asarray(data[CLUSTER_KEY], dtype=np.int64)
        frame_number = int(data["frame_index"]) if "frame_index" in data.files else frame_index

    return {
        "selected_sample": selected_sample.copy(),
        "original_coords": selected_sample.copy(),
        "original_cluster_labels": original_cluster_labels.copy(),
        "original_dihedrals": None,
        "dihedral_atom_indices": None,
        "dihedral_mask": None,
        "dihedral_keys": [],
        "atom_names": None,
        "atom_resids": None,
        "frame_position": frame_number,
        "frame_index": frame_number,
        "record_id": record_id,
        "sample_id": processed_dir.name,
    }

def match_processed_record_id(
    processed_dir: Path | None,
    record_id: str,
    frame_index: int,
) -> str:
    if processed_dir is None:
        return record_id

    structures_dir = processed_dir / "structures"
    exact_path = structures_dir / f"{record_id}.npz"
    if exact_path.exists():
        return record_id

    suffix = f"_{frame_index:06d}"
    matches = sorted(structures_dir.glob(f"*{suffix}.npz"))
    if len(matches) == 1:
        return matches[0].stem

    manifest_path = processed_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        manifest_matches = [str(record["id"]) for record in manifest if str(record["id"]).endswith(suffix)]
        if len(manifest_matches) == 1:
            return manifest_matches[0]

    return record_id

def find_processed_paths(
    processed_dir: Path | None,
    record_id: str,
    frame_index: int,
) -> tuple[Path | None, Path | None, Path | None]:
    if processed_dir is None:
        return None, None, None

    structures_dir = processed_dir / "structures"
    records_dir = processed_dir / "records"
    tokens_dir = processed_dir / "tokens"

    structure_path = structures_dir / f"{record_id}.npz"
    record_path = records_dir / f"{record_id}.json"
    tokenized_path = tokens_dir / f"{record_id}.pkl"

    if not structure_path.exists():
        matches = sorted(structures_dir.glob(f"*_{frame_index:06d}.npz"))
        if matches:
            structure_path = matches[0]
            record_id = structure_path.stem
            record_path = records_dir / f"{record_id}.json"
            tokenized_path = tokens_dir / f"{record_id}.pkl"

    if not structure_path.exists() or not record_path.exists():
        return None, None, None
    return structure_path, record_path, tokenized_path if tokenized_path.exists() else None

def build_structure_and_record_from_raw(frame_data: dict[str, Any]) -> tuple[Structure, dict[str, Any]]:
    atom_names = frame_data["atom_names"]
    atom_resids = frame_data["atom_resids"]
    if atom_names is None or atom_resids is None:
        raise ValueError(
            "Cannot build a SimpleFold structure from raw data without `atom_names` "
            "and `atom_resids`."
        )

    atoms_template, bonds, residues, chains, connections, interfaces, mask = build_static_topology(
        atom_names,
        atom_resids,
    )
    atoms = atoms_template.copy()
    coords = frame_data["original_coords"].astype(np.float32, copy=False)
    atoms["coords"] = coords
    assert_no_conformer_coordinate_leak(atoms, frame_data["record_id"])

    structure = Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        connections=connections,
        interfaces=interfaces,
        mask=mask,
    )
    record = Record(
        id=frame_data["record_id"],
        structure=StructureInfo(
            resolution=0.0,
            method="simulation",
            deposited="",
            released="",
            revised="",
            num_chains=1,
            num_interfaces=0,
        ),
        chains=[
            ChainInfo(
                chain_id=0,
                chain_name="A",
                mol_type=const.chain_type_ids["PROTEIN"],
                cluster_id=-1,
                msa_id="",
                num_residues=int(residues.shape[0]),
                valid=True,
                entity_id=0,
            )
        ],
        interfaces=[],
    )
    return structure, asdict(record)
