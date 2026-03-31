#!/usr/bin/env python3
"""Drop hydrogen atoms from a backmapping dataset NPZ.

Hydrogen detection is based on `atom_names`:
- names beginning with `H`
- names beginning with digits followed by `H` (e.g. `1H`, `2HB`)

All arrays that carry an atom axis are filtered with the same atom mask to keep
the NPZ internally consistent.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format


DEFAULT_INPUT = Path(
    "/storage_common/angiod/phase-data/projects/a2a/systems/a2a/clusters/"
    "cb3c3cae-5316-47db-8fbb-0567d5f0f75b/samples/"
    "e98051c1-744f-4522-bafd-2bfdeea9788b/backmapping_dataset.npz"
)

# Canonical atom-axis mapping for known keys.
KNOWN_ATOM_AXES = {
    "trajectory": 1,
    "atom_resids": 0,
    "atom_names": 0,
    "atom_residue_index": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop hydrogen atoms from backmapping_dataset.npz while keeping all "
            "atom-indexed arrays aligned."
        )
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to input backmapping_dataset.npz.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help=(
            "Path to output NPZ. Default: <input_dir>/without_hs/backmapping_dataset.npz."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser.parse_args()


def is_hydrogen_atom_name(name: str) -> bool:
    token = name.strip().upper()
    if not token:
        return False
    idx = 0
    while idx < len(token) and token[idx].isdigit():
        idx += 1
    return idx < len(token) and token[idx] == "H"


def build_atom_mask(atom_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if atom_names.ndim != 1:
        raise ValueError(f"`atom_names` must be 1D, got shape {atom_names.shape}.")
    is_h = np.fromiter(
        (is_hydrogen_atom_name(name) for name in atom_names.tolist()),
        dtype=bool,
        count=atom_names.shape[0],
    )
    keep = ~is_h
    return keep, is_h


def filter_atom_axis(
    array: np.ndarray,
    key: str,
    atom_count: int,
    keep_indices: np.ndarray,
) -> tuple[np.ndarray, int | None]:
    if array.ndim == 0:
        return array, None

    candidate_axes = [axis for axis, size in enumerate(array.shape) if size == atom_count]
    if not candidate_axes:
        return array, None

    if key in KNOWN_ATOM_AXES:
        atom_axis = KNOWN_ATOM_AXES[key]
        if array.shape[atom_axis] != atom_count:
            raise ValueError(
                f"Key `{key}` expected atom axis {atom_axis}, "
                f"but shape is {array.shape} and atom_count is {atom_count}."
            )
    else:
        if len(candidate_axes) != 1:
            raise ValueError(
                f"Key `{key}` has ambiguous atom axis candidates {candidate_axes} "
                f"for shape {array.shape}. Please add explicit handling."
            )
        atom_axis = candidate_axes[0]

    filtered = np.take(array, keep_indices, axis=atom_axis)
    return filtered, atom_axis


def infer_default_output(input_npz: Path) -> Path:
    return input_npz.parent / "without_hs" / "backmapping_dataset.npz"


def main() -> None:
    args = parse_args()
    input_npz = args.input_npz.resolve()
    output_npz = (
        args.output_npz.resolve()
        if args.output_npz is not None
        else infer_default_output(input_npz)
    )

    if not input_npz.exists():
        raise FileNotFoundError(f"Input NPZ not found: {input_npz}")
    if output_npz.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {output_npz}. Use --overwrite to replace it."
        )

    with np.load(input_npz, allow_pickle=False) as data:
        keys = list(data.files)
        if "atom_names" not in data:
            raise KeyError("Missing required key `atom_names` in input NPZ.")
        atom_names = data["atom_names"]
        atom_count = atom_names.shape[0]

        for required in ("atom_resids", "atom_residue_index", "trajectory"):
            if required not in data:
                raise KeyError(f"Missing required key `{required}` in input NPZ.")
        if data["atom_resids"].shape[0] != atom_count:
            raise ValueError("`atom_resids` length does not match `atom_names` length.")
        if data["atom_residue_index"].shape[0] != atom_count:
            raise ValueError(
                "`atom_residue_index` length does not match `atom_names` length."
            )
        if data["trajectory"].shape[1] != atom_count:
            raise ValueError("`trajectory` atom axis does not match `atom_names` length.")

        keep_mask, h_mask = build_atom_mask(atom_names)
        keep_indices = np.flatnonzero(keep_mask)
        dropped = int(h_mask.sum())

        changed: list[tuple[str, int, tuple[int, ...], tuple[int, ...]]] = []
        output_shapes: dict[str, tuple[int, ...]] = {}

        output_npz.parent.mkdir(parents=True, exist_ok=True)
        tmp_output = output_npz.with_suffix(output_npz.suffix + ".tmp")
        if tmp_output.exists():
            tmp_output.unlink()

        try:
            with zipfile.ZipFile(
                tmp_output,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=4,
                allowZip64=True,
            ) as zf:
                for key in keys:
                    array = data[key]
                    filtered, atom_axis = filter_atom_axis(
                        array=array,
                        key=key,
                        atom_count=atom_count,
                        keep_indices=keep_indices,
                    )
                    output_shapes[key] = filtered.shape
                    if atom_axis is not None:
                        changed.append((key, atom_axis, array.shape, filtered.shape))

                    with zf.open(f"{key}.npy", mode="w", force_zip64=True) as fh:
                        npy_format.write_array(fh, filtered, allow_pickle=False)

            # Post-filter consistency checks.
            new_atom_count = keep_indices.shape[0]
            if output_shapes["atom_names"][0] != new_atom_count:
                raise AssertionError("Filtered `atom_names` has inconsistent length.")
            if output_shapes["atom_resids"][0] != new_atom_count:
                raise AssertionError("Filtered `atom_resids` has inconsistent length.")
            if output_shapes["atom_residue_index"][0] != new_atom_count:
                raise AssertionError(
                    "Filtered `atom_residue_index` has inconsistent length."
                )
            if output_shapes["trajectory"][1] != new_atom_count:
                raise AssertionError("Filtered `trajectory` has inconsistent atom axis.")

            if output_npz.exists() and args.overwrite:
                output_npz.unlink()
            tmp_output.replace(output_npz)
        except Exception:
            if tmp_output.exists():
                tmp_output.unlink()
            raise

    print(f"Input:   {input_npz}")
    print(f"Output:  {output_npz}")
    print(f"Atoms:   {atom_count} -> {keep_indices.shape[0]} (dropped {dropped} hydrogens)")
    print("Filtered keys (axis old_shape -> new_shape):")
    for key, axis, old_shape, new_shape in changed:
        print(f"  {key} (axis {axis}) {old_shape} -> {new_shape}")


if __name__ == "__main__":
    main()
