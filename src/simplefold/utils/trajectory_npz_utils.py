#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from boltz_data_pipeline import const
from boltz_data_pipeline.types import (
    Atom,
    Bond,
    Chain,
    ChainInfo,
    Connection,
    Interface,
    Record,
    Residue,
    StructureInfo,
)


PROTEIN_RESIDUES = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)

ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "K": 19,
    "I": 53,
}


def is_trajectory_npz_input(data_path: Path) -> bool:
    return data_path.is_file() and data_path.suffix.lower() == ".npz"


def sanitize_record_prefix(prefix: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", prefix).strip("_").lower()
    return value or "sample"


def encode_atom_name(atom_name: str) -> np.ndarray:
    encoded = np.zeros(4, dtype=np.int8)
    for i, char in enumerate(atom_name[:4].upper()):
        encoded[i] = ord(char) - 32
    return encoded


def guess_atomic_number(atom_name: str) -> int:
    for char in atom_name.upper():
        if char.isalpha():
            return ATOMIC_NUMBERS.get(char, 0)
    return 0


def infer_residue_name(atom_names: np.ndarray) -> str:
    observed = {name.strip().upper() for name in atom_names.tolist() if name}
    observed.discard("OXT")

    if not observed:
        return "UNK"

    exact = [
        residue_name
        for residue_name in PROTEIN_RESIDUES
        if observed == set(const.ref_atoms[residue_name])
    ]
    if len(exact) == 1:
        return exact[0]

    supersets = [
        residue_name
        for residue_name in PROTEIN_RESIDUES
        if observed.issubset(set(const.ref_atoms[residue_name]))
    ]
    if len(supersets) == 1:
        return supersets[0]

    if supersets:
        shortest = min(len(const.ref_atoms[name]) for name in supersets)
        shortest_candidates = sorted(
            [name for name in supersets if len(const.ref_atoms[name]) == shortest]
        )
        if len(shortest_candidates) == 1:
            return shortest_candidates[0]

    return "UNK"


def find_local_atom_index(atom_names: list[str], preferred_names: tuple[str, ...]) -> int:
    index_by_name = {name: idx for idx, name in enumerate(atom_names)}
    for atom_name in preferred_names:
        if atom_name in index_by_name:
            return index_by_name[atom_name]
    return 0


def build_static_topology(
    atom_names: np.ndarray,
    atom_resids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    natoms = int(atom_names.shape[0])
    if natoms != int(atom_resids.shape[0]):
        raise ValueError(
            "`atom_names` and `atom_resids` must have the same length."
        )

    atom_resids = atom_resids.astype(np.int64, copy=False)

    if np.any(np.diff(atom_resids) < 0):
        raise ValueError(
            "`atom_resids` must be monotonic non-decreasing so residues are contiguous."
        )

    nresidues = int(atom_resids.max()) + 1
    expected = set(range(nresidues))
    observed = set(atom_resids.tolist())
    if expected != observed:
        raise ValueError(
            "`atom_resids` must cover residue indices from 0..N-1 without gaps."
        )

    atoms = np.zeros(natoms, dtype=Atom)
    atoms["name"] = np.stack([encode_atom_name(name) for name in atom_names], axis=0)
    atoms["element"] = np.array(
        [guess_atomic_number(name) for name in atom_names],
        dtype=np.int8,
    )
    atoms["charge"] = np.zeros(natoms, dtype=np.int8)
    atoms["is_present"] = np.ones(natoms, dtype=bool)
    atoms["chirality"] = np.zeros(natoms, dtype=np.int8)

    residues = np.zeros(nresidues, dtype=Residue)
    for residue_index in range(nresidues):
        atom_indices = np.where(atom_resids == residue_index)[0]
        if atom_indices.size == 0:
            raise ValueError(f"Residue index {residue_index} has no atoms.")

        expected_slice = np.arange(atom_indices[0], atom_indices[0] + atom_indices.size)
        if not np.array_equal(atom_indices, expected_slice):
            raise ValueError(
                "Atoms for each residue must form one contiguous block."
            )

        residue_atom_names = atom_names[atom_indices]
        residue_name = infer_residue_name(residue_atom_names)
        residue_token_id = const.token_ids.get(residue_name, const.token_ids["UNK"])

        atom_name_list = [name.strip().upper() for name in residue_atom_names.tolist()]
        center_name = const.res_to_center_atom.get(residue_name, "CA")
        disto_name = const.res_to_disto_atom.get(residue_name, "CA")

        center_local = find_local_atom_index(
            atom_name_list,
            (center_name, "CA", "C", "N"),
        )
        disto_local = find_local_atom_index(
            atom_name_list,
            (disto_name, "CB", "CA", "C", "N"),
        )

        residues[residue_index] = (
            residue_name,
            residue_token_id,
            residue_index,
            int(atom_indices[0]),
            int(atom_indices.size),
            int(atom_indices[center_local]),
            int(atom_indices[disto_local]),
            residue_name in PROTEIN_RESIDUES,
            True,
        )

    chains = np.array(
        [
            (
                "A",
                const.chain_type_ids["PROTEIN"],
                0,
                0,
                0,
                0,
                natoms,
                0,
                nresidues,
            )
        ],
        dtype=Chain,
    )

    bonds = np.array([], dtype=Bond)
    connections = np.array([], dtype=Connection)
    interfaces = np.array([], dtype=Interface)
    mask = np.array([True], dtype=bool)

    return atoms, bonds, residues, chains, connections, interfaces, mask


def extract_scalar_string(data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in data.files:
        return default
    value = data[key]
    if value.shape == ():
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(-1)[0])
    return default


def convert_trajectory_npz_to_simplefold(
    npz_path: Path,
    out_dir: Path,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    record_prefix: Optional[str] = None,
) -> None:
    if frame_stride <= 0:
        raise ValueError("`frame_stride` must be >= 1.")
    if start_frame < 0:
        raise ValueError("`start_frame` must be >= 0.")

    out_dir.mkdir(parents=True, exist_ok=True)
    records_dir = out_dir / "records"
    structures_dir = out_dir / "structures"
    records_dir.mkdir(parents=True, exist_ok=True)
    structures_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=False) as data:
        required_keys = {
            "trajectory",
            "atom_names",
            "atom_resids",
        }
        missing = sorted(required_keys - set(data.files))
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                f"Missing required keys in {npz_path}: {missing_str}."
            )

        trajectory = data["trajectory"]
        atom_names = data["atom_names"].astype("U4", copy=False)
        atom_resids = data["atom_resids"].astype(np.int64, copy=False)

        if trajectory.ndim != 3:
            raise ValueError(
                "`trajectory` must have shape [num_frames, num_atoms, 3]."
            )

        nframes, natoms, xyz = trajectory.shape
        if xyz != 3:
            raise ValueError("Last `trajectory` dimension must be 3.")
        if natoms != atom_names.shape[0] or natoms != atom_resids.shape[0]:
            raise ValueError(
                "`trajectory`, `atom_names`, and `atom_resids` must agree on num_atoms."
            )

        frame_indices = (
            data["frame_indices"].astype(np.int64, copy=False)
            if "frame_indices" in data.files
            else np.arange(nframes, dtype=np.int64)
        )
        if frame_indices.shape[0] != nframes:
            frame_indices = np.arange(nframes, dtype=np.int64)

        frame_state_ids = (
            data["frame_state_ids"].astype("U32", copy=False)
            if "frame_state_ids" in data.files
            else np.full(nframes, "", dtype="U32")
        )
        if frame_state_ids.shape[0] != nframes:
            frame_state_ids = np.full(nframes, "", dtype="U32")

        atom_cluster_ids = (
            data["atom_idx_and_glob_cluster_id_per_frame"].astype(np.int64, copy=False)
            if "atom_idx_and_glob_cluster_id_per_frame" in data.files
            else None
        )
        if atom_cluster_ids is not None and atom_cluster_ids.shape != (nframes, natoms):
            raise ValueError(
                "`atom_idx_and_glob_cluster_id_per_frame` must have shape [num_frames, num_atoms]."
            )

        residue_cluster_ids = (
            data["res_idx_and_glob_cluster_id_per_frame"].astype(np.int64, copy=False)
            if "res_idx_and_glob_cluster_id_per_frame" in data.files
            else None
        )

        atoms_template, bonds, residues, chains, connections, interfaces, mask = build_static_topology(
            atom_names,
            atom_resids,
        )

        sample_id = extract_scalar_string(data, "sample_id", npz_path.stem)
        state_id = extract_scalar_string(data, "state_id", "")

        record_prefix = sanitize_record_prefix(record_prefix or sample_id)

        frame_positions = np.arange(start_frame, nframes, frame_stride)
        if max_frames is not None:
            frame_positions = frame_positions[:max_frames]

        chain_info = [
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
        ]

        structure_info = StructureInfo(
            resolution=0.0,
            method="simulation",
            deposited="",
            released="",
            revised="",
            num_chains=1,
            num_interfaces=0,
        )

        records = []
        for frame_position in tqdm(frame_positions, desc="Converting frames"):
            frame_number = int(frame_indices[frame_position])
            record_id = f"{record_prefix}_{frame_number:06d}"

            coords = trajectory[frame_position].astype(np.float32, copy=False)
            atoms = atoms_template.copy()
            atoms["coords"] = coords
            atoms["conformer"] = coords

            payload = {
                "atoms": atoms,
                "bonds": bonds,
                "residues": residues,
                "chains": chains,
                "connections": connections,
                "interfaces": interfaces,
                "mask": mask,
            }
            if atom_cluster_ids is not None:
                payload["atom_idx_and_glob_cluster_id_per_frame"] = atom_cluster_ids[
                    frame_position
                ]
            if residue_cluster_ids is not None:
                payload["res_idx_and_glob_cluster_id_per_frame"] = residue_cluster_ids[
                    frame_position
                ]

            payload["frame_index"] = np.int64(frame_number)
            payload["frame_state_id"] = np.array(
                frame_state_ids[frame_position], dtype="U32"
            )

            struct_path = structures_dir / f"{record_id}.npz"
            np.savez_compressed(struct_path, **payload)

            record = Record(
                id=record_id,
                structure=structure_info,
                chains=chain_info,
                interfaces=[],
            )
            with (records_dir / f"{record_id}.json").open("w") as f:
                json.dump(asdict(record), f)
            records.append(asdict(record))

    with (out_dir / "manifest.json").open("w") as f:
        json.dump(records, f)

    print(
        f"Wrote {len(records)} records to {out_dir} "
        f"(sample_id={sample_id}, state_id={state_id or 'n/a'})."
    )
