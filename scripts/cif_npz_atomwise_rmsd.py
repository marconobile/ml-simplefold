#!/usr/bin/env python3
"""Align a CIF structure to an NPZ trajectory frame and compute atomwise RMSD.

The script performs rigid-body alignment (Kabsch, no scaling) and reports:
- global post-alignment RMSD
- per-atom distances after alignment (optionally written to CSV)

It includes strict atom-pairing checks and will fail if it cannot establish a
1:1 mapping between CIF atoms and NPZ atoms using reliable keys.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import gemmi
import numpy as np


def _normalize_atom_name(name: object) -> str:
    if isinstance(name, (bytes, np.bytes_)):
        return name.decode("utf-8").strip().upper()
    return str(name).strip().upper()


def _decode_atom_name_code(code_row: np.ndarray) -> str:
    # SimpleFold encoding: (ord(char) - 32) with zero-padding.
    chars = [chr(int(c) + 32) for c in code_row if int(c) != 0]
    return "".join(chars).strip().upper()


@dataclass(frozen=True)
class CifAtoms:
    coords: np.ndarray
    atom_names: np.ndarray
    residue_ids: np.ndarray
    residue_indices: np.ndarray
    residue_names: np.ndarray
    chain_ids: np.ndarray
    insertion_codes: np.ndarray


@dataclass(frozen=True)
class NpzAtoms:
    coords: np.ndarray
    atom_names: np.ndarray
    atom_resids: Optional[np.ndarray]
    atom_residue_index: Optional[np.ndarray]


@dataclass(frozen=True)
class MappingResult:
    npz_indices_for_cif_order: np.ndarray
    method: str
    subset_name: str
    residue_offset: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align atoms from a CIF structure and one NPZ frame, then compute "
            "atomwise RMSD with strict atom-pairing checks."
        )
    )
    parser.add_argument(
        "--cif-path",
        type=Path,
        required=True,
        help="Path to input CIF/mmCIF file.",
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        required=True,
        help="Path to NPZ file containing trajectory + atom metadata.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index in NPZ trajectory (default: 0).",
    )
    parser.add_argument(
        "--model-index",
        type=int,
        default=0,
        help="Model index in CIF (default: 0).",
    )
    parser.add_argument(
        "--chain-ids",
        type=str,
        default=None,
        help="Comma-separated chain IDs to include (default: all chains).",
    )
    parser.add_argument(
        "--include-hydrogens",
        action="store_true",
        help="Include hydrogen atoms (default: excluded).",
    )
    parser.add_argument(
        "--include-non-polymer",
        action="store_true",
        help="Include non-polymer residues (default: polymer only).",
    )
    parser.add_argument(
        "--allow-name-only-mapping",
        action="store_true",
        help=(
            "Allow fallback mapping using only atom names if residue-based mapping "
            "is unavailable. This is less safe and disabled by default."
        ),
    )
    parser.add_argument(
        "--per-atom-out",
        type=Path,
        default=None,
        help="Optional CSV output path for per-atom distances.",
    )
    return parser.parse_args()


def _parse_chain_filter(chain_ids_arg: Optional[str]) -> Optional[set[str]]:
    if chain_ids_arg is None:
        return None
    chains = {part.strip() for part in chain_ids_arg.split(",") if part.strip()}
    if not chains:
        raise ValueError("--chain-ids provided but no valid chain IDs were parsed.")
    return chains


def _atom_altloc_priority(atom: gemmi.Atom) -> tuple[int, float]:
    altloc = str(atom.altloc).strip()
    if altloc in ("", ".", "?"):
        alt_rank = 0
    elif altloc == "A":
        alt_rank = 1
    else:
        alt_rank = 2
    # Prefer higher occupancy for same altloc rank.
    return alt_rank, -float(atom.occ)


def load_cif_atoms(
    cif_path: Path,
    model_index: int,
    chain_filter: Optional[set[str]],
    include_hydrogens: bool,
    include_non_polymer: bool,
) -> CifAtoms:
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    structure = gemmi.read_structure(str(cif_path))
    if len(structure) == 0:
        raise ValueError(f"CIF contains no models: {cif_path}")
    if model_index < 0 or model_index >= len(structure):
        raise ValueError(
            f"--model-index out of bounds: {model_index}. "
            f"Available models: 0..{len(structure) - 1}."
        )

    model = structure[model_index]

    coords: list[list[float]] = []
    atom_names: list[str] = []
    residue_ids: list[int] = []
    residue_indices: list[int] = []
    residue_names: list[str] = []
    chain_ids: list[str] = []
    insertion_codes: list[str] = []

    residue_index_counter = 0

    for chain in model:
        chain_id = str(chain.name)
        if chain_filter is not None and chain_id not in chain_filter:
            continue

        for residue in chain:
            is_polymer = residue.entity_type.name == "Polymer"
            if not include_non_polymer and not is_polymer:
                continue

            # Some CIFs may have alternate locations. Select one conformer per atom name.
            selected_by_name: dict[str, tuple[tuple[int, float], gemmi.Atom]] = {}
            for atom in residue:
                if not include_hydrogens and atom.element.name == "H":
                    continue

                atom_name = _normalize_atom_name(atom.name)
                rank = _atom_altloc_priority(atom)
                prev = selected_by_name.get(atom_name)
                if prev is None or rank < prev[0]:
                    selected_by_name[atom_name] = (rank, atom)

            if not selected_by_name:
                residue_index_counter += 1
                continue

            resid = int(residue.seqid.num)
            icode = str(residue.seqid.icode).strip()
            resname = str(residue.name).strip().upper()

            for atom_name, (_, atom) in selected_by_name.items():
                coords.append([float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)])
                atom_names.append(atom_name)
                residue_ids.append(resid)
                residue_indices.append(residue_index_counter)
                residue_names.append(resname)
                chain_ids.append(chain_id)
                insertion_codes.append(icode)

            residue_index_counter += 1

    if not coords:
        raise ValueError("No atoms selected from CIF. Check chain/filter options.")

    return CifAtoms(
        coords=np.asarray(coords, dtype=np.float64),
        atom_names=np.asarray(atom_names, dtype=np.str_),
        residue_ids=np.asarray(residue_ids, dtype=np.int64),
        residue_indices=np.asarray(residue_indices, dtype=np.int64),
        residue_names=np.asarray(residue_names, dtype=np.str_),
        chain_ids=np.asarray(chain_ids, dtype=np.str_),
        insertion_codes=np.asarray(insertion_codes, dtype=np.str_),
    )


def _load_coords(npz_data: np.lib.npyio.NpzFile, frame_index: int) -> tuple[np.ndarray, str]:
    coord_key_candidates = ("trajectory", "xyz", "coords", "target_atom_coords")
    coord_key = next((k for k in coord_key_candidates if k in npz_data), None)
    if coord_key is None:
        raise ValueError(
            "NPZ does not contain coordinates. "
            f"Tried keys: {coord_key_candidates}."
        )

    coords = np.asarray(npz_data[coord_key], dtype=np.float64)

    if coords.ndim == 3:
        if frame_index < 0 or frame_index >= coords.shape[0]:
            raise ValueError(
                f"--frame-index out of bounds: {frame_index}. "
                f"Coordinates shape is {coords.shape}."
            )
        coords = coords[frame_index]
    elif coords.ndim == 2:
        if frame_index != 0:
            raise ValueError(
                f"NPZ coordinates are 2D ({coords.shape}), so only frame index 0 is valid; "
                f"got {frame_index}."
            )
    else:
        raise ValueError(
            f"Coordinates must be shape (T, N, 3) or (N, 3), got {coords.shape}."
        )

    if coords.shape[1] != 3:
        raise ValueError(
            f"Coordinates last dimension must be 3, got shape {coords.shape}."
        )

    return coords, coord_key


def _load_atom_names(npz_data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, str]:
    key_candidates = ("atom_names", "target_atom_names", "names")
    key = next((k for k in key_candidates if k in npz_data), None)
    if key is None:
        raise ValueError(
            "NPZ does not contain atom names. "
            f"Tried keys: {key_candidates}."
        )

    raw = npz_data[key]
    if raw.ndim == 2 and raw.shape[-1] == 4 and np.issubdtype(raw.dtype, np.integer):
        decoded = np.asarray([_decode_atom_name_code(row) for row in raw], dtype=np.str_)
        return decoded, key

    if raw.ndim != 1:
        raise ValueError(
            f"Atom names must be 1D (or Nx4 int encoded), got shape {raw.shape}."
        )

    decoded = np.asarray([_normalize_atom_name(x) for x in raw], dtype=np.str_)
    return decoded, key


def _parse_residue_key_to_int(value: object) -> Optional[int]:
    text = str(value)
    match = re.search(r"(-?\d+)\s*$", text)
    if not match:
        return None
    return int(match.group(1))


def _load_atom_residue_index(npz_data: np.lib.npyio.NpzFile) -> tuple[Optional[np.ndarray], Optional[str]]:
    key_candidates = ("atom_residue_index", "residue_index", "target_atom_residue_index")
    key = next((k for k in key_candidates if k in npz_data), None)
    if key is None:
        return None, None

    values = np.asarray(npz_data[key], dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"{key} must be 1D, got shape {values.shape}.")
    return values, key


def _load_atom_resids(
    npz_data: np.lib.npyio.NpzFile,
    atom_residue_index: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], Optional[str]]:
    atom_resids_key_candidates = ("atom_resids", "target_atom_resids")
    key = next((k for k in atom_resids_key_candidates if k in npz_data), None)
    if key is not None:
        values = np.asarray(npz_data[key], dtype=np.int64)
        if values.ndim != 1:
            raise ValueError(f"{key} must be 1D, got shape {values.shape}.")
        return values, key

    if atom_residue_index is None:
        return None, None

    if "residue_resids" in npz_data:
        residue_resids = np.asarray(npz_data["residue_resids"], dtype=np.int64)
        if residue_resids.ndim != 1:
            raise ValueError(
                f"residue_resids must be 1D, got shape {residue_resids.shape}."
            )

        derived = np.full(atom_residue_index.shape[0], fill_value=np.iinfo(np.int64).min, dtype=np.int64)
        valid = (atom_residue_index >= 0) & (atom_residue_index < residue_resids.shape[0])
        derived[valid] = residue_resids[atom_residue_index[valid]]
        return derived, "derived_from_residue_resids"

    if "residue_keys" in npz_data:
        residue_keys = npz_data["residue_keys"]
        if residue_keys.ndim != 1:
            raise ValueError(f"residue_keys must be 1D, got shape {residue_keys.shape}.")
        parsed = [_parse_residue_key_to_int(v) for v in residue_keys]
        if all(v is not None for v in parsed):
            residue_ids = np.asarray(parsed, dtype=np.int64)
            derived = np.full(atom_residue_index.shape[0], fill_value=np.iinfo(np.int64).min, dtype=np.int64)
            valid = (atom_residue_index >= 0) & (atom_residue_index < residue_ids.shape[0])
            derived[valid] = residue_ids[atom_residue_index[valid]]
            return derived, "derived_from_residue_keys"

    return None, None


def load_npz_atoms(npz_path: Path, frame_index: int) -> tuple[NpzAtoms, dict[str, Optional[str]]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as npz_data:
        coords, coords_key = _load_coords(npz_data, frame_index)
        atom_names, names_key = _load_atom_names(npz_data)
        atom_residue_index, atom_residue_index_key = _load_atom_residue_index(npz_data)
        atom_resids, atom_resids_key = _load_atom_resids(npz_data, atom_residue_index)

    if coords.shape[0] != atom_names.shape[0]:
        raise ValueError(
            "Coordinates/atom_names length mismatch: "
            f"{coords.shape[0]} vs {atom_names.shape[0]}."
        )

    if atom_resids is not None and atom_resids.shape[0] != atom_names.shape[0]:
        raise ValueError(
            "atom_resids length mismatch with atom_names: "
            f"{atom_resids.shape[0]} vs {atom_names.shape[0]}."
        )

    if atom_residue_index is not None and atom_residue_index.shape[0] != atom_names.shape[0]:
        raise ValueError(
            "atom_residue_index length mismatch with atom_names: "
            f"{atom_residue_index.shape[0]} vs {atom_names.shape[0]}."
        )

    return (
        NpzAtoms(
            coords=np.asarray(coords, dtype=np.float64),
            atom_names=np.asarray(atom_names, dtype=np.str_),
            atom_resids=np.asarray(atom_resids, dtype=np.int64)
            if atom_resids is not None
            else None,
            atom_residue_index=np.asarray(atom_residue_index, dtype=np.int64)
            if atom_residue_index is not None
            else None,
        ),
        {
            "coords_key": coords_key,
            "names_key": names_key,
            "atom_resids_key": atom_resids_key,
            "atom_residue_index_key": atom_residue_index_key,
        },
    )


def _build_index_mapping(
    reference_keys: Iterable[tuple[object, ...]],
    target_keys: Iterable[tuple[object, ...]],
    key_label: str,
) -> np.ndarray:
    reference_keys = list(reference_keys)
    target_keys = list(target_keys)

    if len(reference_keys) != len(target_keys):
        raise ValueError(
            f"Atom count mismatch for mapping {key_label}: "
            f"reference={len(reference_keys)}, target={len(target_keys)}."
        )

    ref_idx_by_key: dict[tuple[object, ...], list[int]] = defaultdict(list)
    tgt_idx_by_key: dict[tuple[object, ...], list[int]] = defaultdict(list)

    for i, key in enumerate(reference_keys):
        ref_idx_by_key[key].append(i)
    for i, key in enumerate(target_keys):
        tgt_idx_by_key[key].append(i)

    mismatches: list[tuple[tuple[object, ...], int, int]] = []
    for key in set(ref_idx_by_key) | set(tgt_idx_by_key):
        a = len(ref_idx_by_key.get(key, []))
        b = len(tgt_idx_by_key.get(key, []))
        if a != b:
            mismatches.append((key, a, b))

    if mismatches:
        sample = ", ".join(
            f"{key}: ref={a}, tgt={b}" for key, a, b in mismatches[:10]
        )
        raise ValueError(
            f"Key multiplicities mismatch for {key_label}. Examples: {sample}."
        )

    mapping = np.empty(len(reference_keys), dtype=np.int64)
    for key, ref_indices in ref_idx_by_key.items():
        tgt_indices = tgt_idx_by_key[key]
        for occurrence, ref_idx in enumerate(ref_indices):
            mapping[ref_idx] = tgt_indices[occurrence]

    return mapping


def _npz_subset_candidates(npz_atoms: NpzAtoms) -> list[tuple[str, np.ndarray]]:
    n = npz_atoms.coords.shape[0]
    candidates: list[tuple[str, np.ndarray]] = [("all_atoms", np.arange(n, dtype=np.int64))]

    if npz_atoms.atom_residue_index is not None:
        no_caps = np.flatnonzero(npz_atoms.atom_residue_index >= 0)
        if 0 < no_caps.shape[0] < n:
            candidates.append(("atom_residue_index>=0", no_caps))

    # Deduplicate candidates by exact index sequence.
    deduped: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    for name, idx in candidates:
        key = tuple(idx.tolist())
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, idx))
    return deduped


def build_atom_mapping(
    cif_atoms: CifAtoms,
    npz_atoms: NpzAtoms,
    allow_name_only_mapping: bool,
) -> MappingResult:
    n_cif = cif_atoms.coords.shape[0]
    errors: list[str] = []

    subset_candidates = _npz_subset_candidates(npz_atoms)

    for subset_name, subset_idx in subset_candidates:
        if subset_idx.shape[0] != n_cif:
            errors.append(
                f"subset `{subset_name}` atom count {subset_idx.shape[0]} != CIF atom count {n_cif}"
            )
            continue

        npz_names = npz_atoms.atom_names[subset_idx]

        if npz_atoms.atom_resids is not None:
            npz_resids = npz_atoms.atom_resids[subset_idx]
            for offset in (0, -1, 1):
                ref_keys = zip(
                    (cif_atoms.residue_ids + offset).tolist(),
                    cif_atoms.atom_names.tolist(),
                    strict=True,
                )
                tgt_keys = zip(
                    npz_resids.tolist(),
                    npz_names.tolist(),
                    strict=True,
                )
                try:
                    local_map = _build_index_mapping(
                        reference_keys=ref_keys,
                        target_keys=tgt_keys,
                        key_label="(resid, atom_name)",
                    )
                    return MappingResult(
                        npz_indices_for_cif_order=subset_idx[local_map],
                        method="(resid, atom_name)",
                        subset_name=subset_name,
                        residue_offset=offset,
                    )
                except ValueError as exc:
                    errors.append(
                        f"mapping failed for subset `{subset_name}` with (resid, atom_name), offset={offset}: {exc}"
                    )

        if npz_atoms.atom_residue_index is not None:
            npz_residx = npz_atoms.atom_residue_index[subset_idx]
            for offset in (0, 1, -1):
                ref_keys = zip(
                    (cif_atoms.residue_indices + offset).tolist(),
                    cif_atoms.atom_names.tolist(),
                    strict=True,
                )
                tgt_keys = zip(
                    npz_residx.tolist(),
                    npz_names.tolist(),
                    strict=True,
                )
                try:
                    local_map = _build_index_mapping(
                        reference_keys=ref_keys,
                        target_keys=tgt_keys,
                        key_label="(residue_index, atom_name)",
                    )
                    return MappingResult(
                        npz_indices_for_cif_order=subset_idx[local_map],
                        method="(residue_index, atom_name)",
                        subset_name=subset_name,
                        residue_offset=offset,
                    )
                except ValueError as exc:
                    errors.append(
                        f"mapping failed for subset `{subset_name}` with (residue_index, atom_name), "
                        f"offset={offset}: {exc}"
                    )

        if allow_name_only_mapping:
            try:
                local_map = _build_index_mapping(
                    reference_keys=((name,) for name in cif_atoms.atom_names.tolist()),
                    target_keys=((name,) for name in npz_names.tolist()),
                    key_label="(atom_name)",
                )
                return MappingResult(
                    npz_indices_for_cif_order=subset_idx[local_map],
                    method="(atom_name)",
                    subset_name=subset_name,
                    residue_offset=None,
                )
            except ValueError as exc:
                errors.append(
                    f"mapping failed for subset `{subset_name}` with (atom_name): {exc}"
                )

    suffix = "\n".join(f"  - {e}" for e in errors[:20])
    hint = (
        "No reliable atom mapping found. Ensure NPZ contains atom-resolved metadata "
        "(atom_names + atom_resids or atom_residue_index), and that CIF/NPZ represent "
        "the same atom set."
    )
    if not allow_name_only_mapping:
        hint += " Use --allow-name-only-mapping only as a last resort."

    raise ValueError(f"{hint}\nTried:\n{suffix}")


def kabsch_align(reference_xyz: np.ndarray, mobile_xyz: np.ndarray) -> np.ndarray:
    if reference_xyz.shape != mobile_xyz.shape:
        raise ValueError(
            f"Kabsch input shape mismatch: {reference_xyz.shape} vs {mobile_xyz.shape}."
        )
    if reference_xyz.ndim != 2 or reference_xyz.shape[1] != 3:
        raise ValueError(
            f"Kabsch expects shape (N, 3), got {reference_xyz.shape}."
        )

    ref_centroid = reference_xyz.mean(axis=0)
    mob_centroid = mobile_xyz.mean(axis=0)

    ref_centered = reference_xyz - ref_centroid
    mob_centered = mobile_xyz - mob_centroid

    covariance = mob_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(covariance)
    # Row-vector convention: mobile_centered @ R ~= reference_centered.
    rotation = u @ vt

    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt

    return mob_centered @ rotation + ref_centroid


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"RMSD shape mismatch: {a.shape} vs {b.shape}.")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def maybe_warn_scale_mismatch(reference_xyz: np.ndarray, mobile_xyz: np.ndarray) -> None:
    ref_span = float(np.max(np.ptp(reference_xyz, axis=0)))
    mob_span = float(np.max(np.ptp(mobile_xyz, axis=0)))
    min_span = max(min(ref_span, mob_span), 1e-12)
    ratio = max(ref_span, mob_span) / min_span
    if ratio > 5.0:
        print(
            "WARNING: Large coordinate-span mismatch detected between CIF and NPZ "
            f"(CIF span={ref_span:.3f}, NPZ span={mob_span:.3f}, ratio={ratio:.1f}). "
            "RMSD uses rigid alignment without scaling, so this may indicate unit/scale mismatch."
        )


def write_per_atom_csv(
    out_path: Path,
    cif_atoms: CifAtoms,
    npz_indices: np.ndarray,
    distances: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "cif_atom_index",
                "npz_atom_index",
                "chain_id",
                "resid",
                "icode",
                "resname",
                "atom_name",
                "distance_after_alignment",
            ]
        )
        for i in range(cif_atoms.coords.shape[0]):
            writer.writerow(
                [
                    i,
                    int(npz_indices[i]),
                    str(cif_atoms.chain_ids[i]),
                    int(cif_atoms.residue_ids[i]),
                    str(cif_atoms.insertion_codes[i]),
                    str(cif_atoms.residue_names[i]),
                    str(cif_atoms.atom_names[i]),
                    float(distances[i]),
                ]
            )


def main() -> None:
    args = parse_args()

    chain_filter = _parse_chain_filter(args.chain_ids)

    cif_atoms = load_cif_atoms(
        cif_path=args.cif_path.resolve(),
        model_index=args.model_index,
        chain_filter=chain_filter,
        include_hydrogens=args.include_hydrogens,
        include_non_polymer=args.include_non_polymer,
    )

    npz_atoms, npz_info = load_npz_atoms(
        npz_path=args.npz_path.resolve(),
        frame_index=args.frame_index,
    )

    mapping = build_atom_mapping(
        cif_atoms=cif_atoms,
        npz_atoms=npz_atoms,
        allow_name_only_mapping=args.allow_name_only_mapping,
    )

    reference_xyz = cif_atoms.coords
    mobile_xyz = npz_atoms.coords[mapping.npz_indices_for_cif_order]

    if np.isnan(reference_xyz).any() or np.isnan(mobile_xyz).any():
        raise ValueError("NaN coordinates detected in CIF or NPZ frame.")

    maybe_warn_scale_mismatch(reference_xyz, mobile_xyz)

    pre_rmsd = _rmsd(reference_xyz, mobile_xyz)
    mobile_aligned = kabsch_align(reference_xyz, mobile_xyz)
    post_rmsd = _rmsd(reference_xyz, mobile_aligned)

    distances = np.linalg.norm(reference_xyz - mobile_aligned, axis=1)

    print("=== Input Summary ===")
    print(f"CIF path            : {args.cif_path.resolve()}")
    print(f"NPZ path            : {args.npz_path.resolve()}")
    print(f"NPZ frame index     : {args.frame_index}")
    print(f"CIF atoms selected  : {reference_xyz.shape[0]}")
    print(f"NPZ coords key      : {npz_info['coords_key']}")
    print(f"NPZ names key       : {npz_info['names_key']}")
    print(f"NPZ atom_resids key : {npz_info['atom_resids_key']}")
    print(f"NPZ residue_idx key : {npz_info['atom_residue_index_key']}")
    print("=== Mapping ===")
    print(f"Method              : {mapping.method}")
    print(f"NPZ subset          : {mapping.subset_name}")
    if mapping.residue_offset is not None:
        print(f"Residue offset      : {mapping.residue_offset:+d}")
    print("=== RMSD ===")
    print(f"Pre-alignment RMSD  : {pre_rmsd:.6f}")
    print(f"Post-alignment RMSD : {post_rmsd:.6f}")
    print(f"Per-atom distance min/median/max: {distances.min():.6f} / {np.median(distances):.6f} / {distances.max():.6f}")

    if args.per_atom_out is not None:
        out_path = args.per_atom_out.resolve()
        write_per_atom_csv(
            out_path=out_path,
            cif_atoms=cif_atoms,
            npz_indices=mapping.npz_indices_for_cif_order,
            distances=distances,
        )
        print(f"Per-atom distances written to: {out_path}")


if __name__ == "__main__":
    main()
