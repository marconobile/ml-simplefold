#!/usr/bin/env python3
"""Compare generated CIF(s) against target-conditioning PDB references.

This script mirrors the atomwise RMSD workflow from `cif_npz_atomwise_rmsd.py`,
but instead of comparing CIF vs NPZ frame, it compares each generated CIF in a
run directory against both conditioning-reference PDBs written by inference:

- target_conditioning_target_frame_<idx>.pdb
- target_conditioning_random_coords_frame_<idx>.pdb

Only the run directory is required as input. Outputs are written into the same
directory:
- target_conditioning_atomwise_rmsd_summary.csv
- target_conditioning_atomwise_rmsd_report.txt
- one per-atom CSV per comparison pair
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


@dataclass(frozen=True)
class AtomTable:
    coords: np.ndarray
    atom_names: np.ndarray
    residue_ids: np.ndarray
    residue_indices: np.ndarray
    residue_names: np.ndarray
    chain_ids: np.ndarray
    insertion_codes: np.ndarray


@dataclass(frozen=True)
class MappingResult:
    target_indices_for_reference_order: np.ndarray
    method: str
    residue_offset: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align generated CIF structure(s) against target-conditioning PDB "
            "references and compute atomwise RMSD."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="High-level inference output directory (e.g. dbg_output_rand_coords_as_targ).",
    )
    parser.add_argument(
        "--model-index",
        type=int,
        default=0,
        help="Model index in each CIF/PDB (default: 0).",
    )
    parser.add_argument(
        "--chain-ids",
        type=str,
        default=None,
        help="Optional comma-separated chain IDs to include from generated CIF (default: all).",
    )
    parser.add_argument(
        "--include-hydrogens",
        action="store_true",
        help="Include hydrogen atoms (default: excluded).",
    )
    parser.add_argument(
        "--include-non-polymer",
        action="store_true",
        help="Include non-polymer residues from generated CIF (default: polymer only).",
    )
    parser.add_argument(
        "--allow-name-only-mapping",
        action="store_true",
        help=(
            "Allow fallback mapping by atom name only (less strict). "
            "Disabled by default."
        ),
    )
    return parser.parse_args()


def _parse_chain_filter(chain_ids_arg: Optional[str]) -> Optional[set[str]]:
    if chain_ids_arg is None:
        return None
    chain_filter = {token.strip() for token in chain_ids_arg.split(",") if token.strip()}
    if not chain_filter:
        raise ValueError("--chain-ids provided but no valid chain IDs were parsed.")
    return chain_filter


def _atom_altloc_priority(atom: gemmi.Atom) -> tuple[int, float]:
    altloc = str(atom.altloc).strip()
    if altloc in ("", ".", "?"):
        alt_rank = 0
    elif altloc == "A":
        alt_rank = 1
    else:
        alt_rank = 2
    return alt_rank, -float(atom.occ)


def load_structure_atoms(
    structure_path: Path,
    model_index: int,
    chain_filter: Optional[set[str]],
    include_hydrogens: bool,
    include_non_polymer: bool,
) -> AtomTable:
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    structure = gemmi.read_structure(str(structure_path))
    if len(structure) == 0:
        raise ValueError(f"Structure contains no models: {structure_path}")
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

            selected_by_atom_name: dict[str, tuple[tuple[int, float], gemmi.Atom]] = {}
            for atom in residue:
                if not include_hydrogens and atom.element.name == "H":
                    continue
                atom_name = _normalize_atom_name(atom.name)
                rank = _atom_altloc_priority(atom)
                previous = selected_by_atom_name.get(atom_name)
                if previous is None or rank < previous[0]:
                    selected_by_atom_name[atom_name] = (rank, atom)

            if not selected_by_atom_name:
                residue_index_counter += 1
                continue

            resid = int(residue.seqid.num)
            icode = str(residue.seqid.icode).strip()
            resname = str(residue.name).strip().upper()

            for atom_name, (_, atom) in selected_by_atom_name.items():
                coords.append([float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)])
                atom_names.append(atom_name)
                residue_ids.append(resid)
                residue_indices.append(residue_index_counter)
                residue_names.append(resname)
                chain_ids.append(chain_id)
                insertion_codes.append(icode)

            residue_index_counter += 1

    if not coords:
        raise ValueError(f"No atoms selected from structure: {structure_path}")

    return AtomTable(
        coords=np.asarray(coords, dtype=np.float64),
        atom_names=np.asarray(atom_names, dtype=np.str_),
        residue_ids=np.asarray(residue_ids, dtype=np.int64),
        residue_indices=np.asarray(residue_indices, dtype=np.int64),
        residue_names=np.asarray(residue_names, dtype=np.str_),
        chain_ids=np.asarray(chain_ids, dtype=np.str_),
        insertion_codes=np.asarray(insertion_codes, dtype=np.str_),
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
    for idx, key in enumerate(reference_keys):
        ref_idx_by_key[key].append(idx)
    for idx, key in enumerate(target_keys):
        tgt_idx_by_key[key].append(idx)

    mismatches: list[tuple[tuple[object, ...], int, int]] = []
    for key in set(ref_idx_by_key.keys()) | set(tgt_idx_by_key.keys()):
        n_ref = len(ref_idx_by_key.get(key, []))
        n_tgt = len(tgt_idx_by_key.get(key, []))
        if n_ref != n_tgt:
            mismatches.append((key, n_ref, n_tgt))
    if mismatches:
        sample = ", ".join(
            f"{key}: ref={n_ref}, tgt={n_tgt}"
            for key, n_ref, n_tgt in mismatches[:10]
        )
        raise ValueError(
            f"Key multiplicities mismatch for {key_label}. Examples: {sample}."
        )

    mapping = np.empty(len(reference_keys), dtype=np.int64)
    for key, ref_indices in ref_idx_by_key.items():
        tgt_indices = tgt_idx_by_key[key]
        for occurrence_idx, ref_idx in enumerate(ref_indices):
            mapping[ref_idx] = tgt_indices[occurrence_idx]
    return mapping


def build_atom_mapping(
    reference_atoms: AtomTable,
    target_atoms: AtomTable,
    allow_name_only_mapping: bool,
) -> MappingResult:
    if reference_atoms.coords.shape[0] != target_atoms.coords.shape[0]:
        raise ValueError(
            "Atom count mismatch between reference CIF and target PDB: "
            f"{reference_atoms.coords.shape[0]} vs {target_atoms.coords.shape[0]}."
        )

    errors: list[str] = []
    for offset in (0, -1, 1):
        ref_keys = zip(
            (reference_atoms.residue_ids + offset).tolist(),
            reference_atoms.atom_names.tolist(),
            strict=True,
        )
        tgt_keys = zip(
            target_atoms.residue_ids.tolist(),
            target_atoms.atom_names.tolist(),
            strict=True,
        )
        try:
            mapping = _build_index_mapping(
                reference_keys=ref_keys,
                target_keys=tgt_keys,
                key_label="(resid, atom_name)",
            )
            return MappingResult(
                target_indices_for_reference_order=mapping,
                method="(resid, atom_name)",
                residue_offset=offset,
            )
        except ValueError as exc:
            errors.append(f"(resid, atom_name), offset={offset}: {exc}")

    for offset in (0, 1, -1):
        ref_keys = zip(
            (reference_atoms.residue_indices + offset).tolist(),
            reference_atoms.atom_names.tolist(),
            strict=True,
        )
        tgt_keys = zip(
            target_atoms.residue_indices.tolist(),
            target_atoms.atom_names.tolist(),
            strict=True,
        )
        try:
            mapping = _build_index_mapping(
                reference_keys=ref_keys,
                target_keys=tgt_keys,
                key_label="(residue_index, atom_name)",
            )
            return MappingResult(
                target_indices_for_reference_order=mapping,
                method="(residue_index, atom_name)",
                residue_offset=offset,
            )
        except ValueError as exc:
            errors.append(f"(residue_index, atom_name), offset={offset}: {exc}")

    if allow_name_only_mapping:
        try:
            mapping = _build_index_mapping(
                reference_keys=((name,) for name in reference_atoms.atom_names.tolist()),
                target_keys=((name,) for name in target_atoms.atom_names.tolist()),
                key_label="(atom_name)",
            )
            return MappingResult(
                target_indices_for_reference_order=mapping,
                method="(atom_name)",
                residue_offset=None,
            )
        except ValueError as exc:
            errors.append(f"(atom_name): {exc}")

    detail = "\n".join(f"  - {msg}" for msg in errors[:20])
    hint = (
        "No reliable mapping found. Ensure CIF and target-conditioning PDB were "
        "generated from the same atom set."
    )
    if not allow_name_only_mapping:
        hint += " Use --allow-name-only-mapping only as a last resort."
    raise ValueError(f"{hint}\nTried:\n{detail}")


def kabsch_align(reference_xyz: np.ndarray, mobile_xyz: np.ndarray) -> np.ndarray:
    if reference_xyz.shape != mobile_xyz.shape:
        raise ValueError(
            f"Kabsch input shape mismatch: {reference_xyz.shape} vs {mobile_xyz.shape}."
        )
    if reference_xyz.ndim != 2 or reference_xyz.shape[1] != 3:
        raise ValueError(f"Kabsch expects shape (N, 3), got {reference_xyz.shape}.")

    ref_centroid = reference_xyz.mean(axis=0)
    mobile_centroid = mobile_xyz.mean(axis=0)
    ref_centered = reference_xyz - ref_centroid
    mobile_centered = mobile_xyz - mobile_centroid

    covariance = mobile_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt

    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt

    return mobile_centered @ rotation + ref_centroid


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"RMSD shape mismatch: {a.shape} vs {b.shape}.")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def maybe_warn_scale_mismatch(reference_xyz: np.ndarray, mobile_xyz: np.ndarray) -> Optional[str]:
    ref_span = float(np.max(np.ptp(reference_xyz, axis=0)))
    mobile_span = float(np.max(np.ptp(mobile_xyz, axis=0)))
    min_span = max(min(ref_span, mobile_span), 1e-12)
    ratio = max(ref_span, mobile_span) / min_span
    if ratio > 5.0:
        return (
            "WARNING: Large coordinate-span mismatch detected "
            f"(CIF span={ref_span:.3f}, target span={mobile_span:.3f}, ratio={ratio:.1f})."
        )
    return None


def _safe_filename_component(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def write_per_atom_csv(
    out_path: Path,
    reference_atoms: AtomTable,
    target_indices: np.ndarray,
    distances: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "reference_atom_index",
                "target_atom_index",
                "chain_id",
                "resid",
                "icode",
                "resname",
                "atom_name",
                "distance_after_alignment",
            ]
        )
        for idx in range(reference_atoms.coords.shape[0]):
            writer.writerow(
                [
                    idx,
                    int(target_indices[idx]),
                    str(reference_atoms.chain_ids[idx]),
                    int(reference_atoms.residue_ids[idx]),
                    str(reference_atoms.insertion_codes[idx]),
                    str(reference_atoms.residue_names[idx]),
                    str(reference_atoms.atom_names[idx]),
                    float(distances[idx]),
                ]
            )


def _extract_singleton(
    run_dir: Path,
    glob_pattern: str,
    label: str,
) -> Path:
    matches = sorted(run_dir.glob(glob_pattern))
    if not matches:
        raise FileNotFoundError(
            f"No {label} file found with pattern `{glob_pattern}` in {run_dir}."
        )
    if len(matches) > 1:
        joined = "\n".join(f"  - {path}" for path in matches)
        raise ValueError(
            f"Expected exactly one {label} file but found {len(matches)}:\n{joined}"
        )
    return matches[0]


def _discover_generated_cifs(run_dir: Path) -> list[Path]:
    sampled0 = sorted(run_dir.glob("predictions_*/*_sampled_0.cif"))
    if sampled0:
        return sampled0
    all_cifs = sorted(run_dir.glob("predictions_*/*.cif"))
    if all_cifs:
        return all_cifs
    raise FileNotFoundError(
        f"No generated CIF found under {run_dir}/predictions_*."
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    chain_filter = _parse_chain_filter(args.chain_ids)
    cif_paths = _discover_generated_cifs(run_dir)
    random_target_pdb = _extract_singleton(
        run_dir,
        "target_conditioning_random_coords_frame_*.pdb",
        label="random target-conditioning PDB",
    )
    frame_target_pdb = _extract_singleton(
        run_dir,
        "target_conditioning_target_frame_*.pdb",
        label="target-frame conditioning PDB",
    )

    # PDB references are exported from NPZ coordinates and should be loaded as-is.
    random_target_atoms = load_structure_atoms(
        structure_path=random_target_pdb,
        model_index=args.model_index,
        chain_filter=None,
        include_hydrogens=args.include_hydrogens,
        include_non_polymer=True,
    )
    frame_target_atoms = load_structure_atoms(
        structure_path=frame_target_pdb,
        model_index=args.model_index,
        chain_filter=None,
        include_hydrogens=args.include_hydrogens,
        include_non_polymer=True,
    )

    target_refs = [
        ("random_coords", random_target_pdb, random_target_atoms),
        ("target_frame", frame_target_pdb, frame_target_atoms),
    ]

    summary_rows: list[dict[str, object]] = []
    report_lines: list[str] = []
    report_lines.append("=== Target Conditioning Atomwise RMSD Analysis ===")
    report_lines.append(f"Run directory: {run_dir}")
    report_lines.append(f"Generated CIF files analyzed: {len(cif_paths)}")
    report_lines.append(f"Random-coords target PDB: {random_target_pdb}")
    report_lines.append(f"Target-frame PDB: {frame_target_pdb}")
    report_lines.append("")

    for cif_path in cif_paths:
        cif_atoms = load_structure_atoms(
            structure_path=cif_path,
            model_index=args.model_index,
            chain_filter=chain_filter,
            include_hydrogens=args.include_hydrogens,
            include_non_polymer=args.include_non_polymer,
        )
        report_lines.append(f"CIF: {cif_path}")

        for target_kind, target_pdb_path, target_atoms in target_refs:
            row: dict[str, object] = {
                "cif_path": str(cif_path),
                "target_kind": target_kind,
                "target_pdb_path": str(target_pdb_path),
                "status": "failure",
                "error": "",
                "num_atoms": "",
                "mapping_method": "",
                "residue_offset": "",
                "pre_rmsd": "",
                "post_rmsd": "",
                "distance_min": "",
                "distance_median": "",
                "distance_max": "",
                "per_atom_csv": "",
            }
            try:
                mapping = build_atom_mapping(
                    reference_atoms=cif_atoms,
                    target_atoms=target_atoms,
                    allow_name_only_mapping=args.allow_name_only_mapping,
                )
                reference_xyz = cif_atoms.coords
                mobile_xyz = target_atoms.coords[mapping.target_indices_for_reference_order]
                if np.isnan(reference_xyz).any() or np.isnan(mobile_xyz).any():
                    raise ValueError("NaN coordinates detected in CIF or target PDB.")

                scale_warning = maybe_warn_scale_mismatch(reference_xyz, mobile_xyz)
                pre_rmsd = _rmsd(reference_xyz, mobile_xyz)
                mobile_aligned = kabsch_align(reference_xyz, mobile_xyz)
                post_rmsd = _rmsd(reference_xyz, mobile_aligned)
                distances = np.linalg.norm(reference_xyz - mobile_aligned, axis=1)

                per_atom_name = (
                    "target_conditioning_atomwise_rmsd__"
                    f"{_safe_filename_component(cif_path.stem)}__vs__"
                    f"{_safe_filename_component(target_pdb_path.stem)}.csv"
                )
                per_atom_path = run_dir / per_atom_name
                write_per_atom_csv(
                    out_path=per_atom_path,
                    reference_atoms=cif_atoms,
                    target_indices=mapping.target_indices_for_reference_order,
                    distances=distances,
                )

                row.update(
                    {
                        "status": "success",
                        "num_atoms": int(reference_xyz.shape[0]),
                        "mapping_method": mapping.method,
                        "residue_offset": (
                            ""
                            if mapping.residue_offset is None
                            else f"{mapping.residue_offset:+d}"
                        ),
                        "pre_rmsd": f"{pre_rmsd:.6f}",
                        "post_rmsd": f"{post_rmsd:.6f}",
                        "distance_min": f"{float(distances.min()):.6f}",
                        "distance_median": f"{float(np.median(distances)):.6f}",
                        "distance_max": f"{float(distances.max()):.6f}",
                        "per_atom_csv": str(per_atom_path),
                    }
                )

                report_lines.append(f"  - Target: {target_kind}")
                report_lines.append(f"    path               : {target_pdb_path}")
                report_lines.append(f"    status             : success")
                report_lines.append(f"    atoms              : {reference_xyz.shape[0]}")
                report_lines.append(f"    mapping            : {mapping.method}")
                if mapping.residue_offset is not None:
                    report_lines.append(f"    residue offset     : {mapping.residue_offset:+d}")
                report_lines.append(f"    pre-alignment RMSD : {pre_rmsd:.6f}")
                report_lines.append(f"    post-alignment RMSD: {post_rmsd:.6f}")
                report_lines.append(
                    "    per-atom min/med/max: "
                    f"{float(distances.min()):.6f} / "
                    f"{float(np.median(distances)):.6f} / "
                    f"{float(distances.max()):.6f}"
                )
                if scale_warning is not None:
                    report_lines.append(f"    {scale_warning}")
                report_lines.append(f"    per-atom csv       : {per_atom_path}")
            except Exception as exc:  # noqa: BLE001
                row["error"] = str(exc)
                report_lines.append(f"  - Target: {target_kind}")
                report_lines.append(f"    path   : {target_pdb_path}")
                report_lines.append("    status : failure")
                report_lines.append(f"    error  : {exc}")
            summary_rows.append(row)

        report_lines.append("")

    summary_path = run_dir / "target_conditioning_atomwise_rmsd_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cif_path",
                "target_kind",
                "target_pdb_path",
                "status",
                "error",
                "num_atoms",
                "mapping_method",
                "residue_offset",
                "pre_rmsd",
                "post_rmsd",
                "distance_min",
                "distance_median",
                "distance_max",
                "per_atom_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    report_path = run_dir / "target_conditioning_atomwise_rmsd_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n")

    print(f"Summary CSV written to: {summary_path}")
    print(f"Report written to     : {report_path}")


if __name__ == "__main__":
    main()
