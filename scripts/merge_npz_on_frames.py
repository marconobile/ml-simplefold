#!/usr/bin/env python3
"""Merge trajectory NPZ files by concatenating only frame-indexed arrays."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.lib import format as npy_format


DEFAULT_INPUT_DIR = Path("~/ml-simplefold/test_new_data_with_clusters").expanduser()
DEFAULT_INPUT_NAMES = (
    "active_without_hs.npz",
    "inactive_without_hs.npz",
    "pas_without_hs.npz",
)
DEFAULT_OUTPUT_NAME = "active_inactive_pas_without_hs.npz"
FRAME_COUNT_KEY = "trajectory"
SPECIAL_SCALARS = {"state_id", "sample_id"}


@dataclass(frozen=True)
class NpyHeader:
    shape: tuple[int, ...]
    fortran_order: bool
    dtype: np.dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge NPZ files by concatenating arrays whose first axis is the "
            "number of frames. Non-frame arrays must be identical across inputs; "
            "`state_id` and `sample_id` are rewritten as merged metadata."
        )
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[DEFAULT_INPUT_DIR / name for name in DEFAULT_INPUT_NAMES],
        help="Input NPZ files. Defaults to the active/inactive/pas files with hydrogens removed.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=DEFAULT_INPUT_DIR / DEFAULT_OUTPUT_NAME,
        help=f"Output NPZ path. Default: {DEFAULT_INPUT_DIR / DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--merged-state-id",
        default="merged",
        help="Scalar `state_id` value to write when input scalar values differ.",
    )
    parser.add_argument(
        "--merged-sample-id",
        default=None,
        help="Scalar `sample_id` value to write. Default: output file stem.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument(
        "--store",
        action="store_true",
        help="Store arrays without ZIP compression. Faster and larger than the default compressed output.",
    )
    parser.add_argument(
        "--compresslevel",
        type=int,
        default=4,
        help="ZIP deflate compression level when not using --store.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the merge plan without writing the output NPZ.",
    )
    return parser.parse_args()


def npz_member_name(key: str) -> str:
    return f"{key}.npy"


def read_npy_header(fh: BinaryIO) -> NpyHeader:
    version = npy_format.read_magic(fh)
    if version == (1, 0):
        shape, fortran_order, dtype = npy_format.read_array_header_1_0(fh)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = npy_format.read_array_header_2_0(fh)
    else:
        raise ValueError(f"Unsupported NPY format version: {version}")
    return NpyHeader(tuple(int(dim) for dim in shape), bool(fortran_order), np.dtype(dtype))


def read_npz_headers(npz_path: Path) -> tuple[list[str], dict[str, NpyHeader]]:
    keys: list[str] = []
    headers: dict[str, NpyHeader] = {}
    with zipfile.ZipFile(npz_path, mode="r") as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            key = name[:-4]
            with zf.open(name, mode="r") as fh:
                headers[key] = read_npy_header(fh)
            keys.append(key)
    if not keys:
        raise ValueError(f"No .npy members found in {npz_path}")
    return keys, headers


def load_array(npz_path: Path, key: str) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        return data[key]


def scalar_string(value: np.ndarray) -> str:
    if value.shape != ():
        raise ValueError(f"Expected scalar array, got shape {value.shape}")
    return str(value.item())


def resolve_paths(args: argparse.Namespace) -> tuple[list[Path], Path]:
    input_paths = [path.expanduser().resolve() for path in args.inputs]
    output_npz = args.output_npz.expanduser().resolve()

    if len(input_paths) < 2:
        raise ValueError("Provide at least two input NPZ files.")
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input NPZ not found: {path}")
        if not path.is_file():
            raise ValueError(f"Input path is not a file: {path}")
        if path == output_npz:
            raise ValueError(f"Output path must differ from input path: {path}")
    if output_npz.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output exists: {output_npz}. Use --overwrite to replace it.")
    return input_paths, output_npz


def validate_key_sets(input_paths: list[Path], key_lists: list[list[str]]) -> list[str]:
    first_keys = key_lists[0]
    first_set = set(first_keys)
    for path, keys in zip(input_paths[1:], key_lists[1:]):
        if set(keys) != first_set:
            missing = sorted(first_set - set(keys))
            extra = sorted(set(keys) - first_set)
            raise ValueError(
                f"Key mismatch in {path}: missing={missing or 'none'}, extra={extra or 'none'}"
            )
    return first_keys


def find_frame_keys(
    keys: list[str],
    headers_by_input: list[dict[str, NpyHeader]],
    frame_counts: list[int],
) -> list[str]:
    frame_keys: list[str] = []
    for key in keys:
        headers = [headers[key] for headers in headers_by_input]
        if not all(header.shape for header in headers):
            continue
        if not all(header.shape[0] == frame_count for header, frame_count in zip(headers, frame_counts)):
            continue
        trailing_shapes = [header.shape[1:] for header in headers]
        if len(set(trailing_shapes)) != 1:
            raise ValueError(f"Frame key `{key}` has inconsistent trailing shapes: {trailing_shapes}")
        frame_keys.append(key)
    return frame_keys


def output_dtype(headers: list[NpyHeader]) -> np.dtype:
    dtypes = [header.dtype for header in headers]
    if len(set(dtypes)) == 1:
        return dtypes[0]
    return np.result_type(*dtypes)


def write_npy_header(fh: BinaryIO, shape: tuple[int, ...], dtype: np.dtype) -> None:
    header = {
        "descr": npy_format.dtype_to_descr(np.dtype(dtype)),
        "fortran_order": False,
        "shape": tuple(shape),
    }
    npy_format.write_array_header_2_0(fh, header)


def can_raw_concatenate(headers: list[NpyHeader], dtype: np.dtype) -> bool:
    return all(
        not header.fortran_order and header.dtype == dtype
        for header in headers
    )


def copy_npy_payload(npz_path: Path, key: str, out_fh: BinaryIO) -> None:
    with zipfile.ZipFile(npz_path, mode="r") as zf:
        with zf.open(npz_member_name(key), mode="r") as in_fh:
            read_npy_header(in_fh)
            shutil.copyfileobj(in_fh, out_fh, length=16 * 1024 * 1024)


def write_frame_key(
    zf: zipfile.ZipFile,
    key: str,
    input_paths: list[Path],
    headers_by_input: list[dict[str, NpyHeader]],
    output_shape: tuple[int, ...],
) -> None:
    headers = [headers[key] for headers in headers_by_input]
    dtype = output_dtype(headers)

    with zf.open(npz_member_name(key), mode="w", force_zip64=True) as out_fh:
        if can_raw_concatenate(headers, dtype):
            write_npy_header(out_fh, output_shape, dtype)
            for path in input_paths:
                copy_npy_payload(path, key, out_fh)
        else:
            arrays = [load_array(path, key).astype(dtype, copy=False) for path in input_paths]
            merged = np.concatenate(arrays, axis=0)
            npy_format.write_array(out_fh, merged, allow_pickle=False)


def merged_scalar_value(key: str, args: argparse.Namespace, output_npz: Path) -> np.ndarray:
    if key == "state_id":
        return np.array(args.merged_state_id)
    if key == "sample_id":
        return np.array(args.merged_sample_id or output_npz.stem)
    raise KeyError(key)


def resolve_static_key(
    key: str,
    input_paths: list[Path],
    args: argparse.Namespace,
    output_npz: Path,
) -> tuple[np.ndarray, bool, str]:
    arrays = [load_array(path, key) for path in input_paths]
    first = arrays[0]
    all_equal = all(
        array.shape == first.shape
        and array.dtype == first.dtype
        and np.array_equal(array, first)
        for array in arrays[1:]
    )

    if all_equal:
        output = first
        message = "kept identical static value"
    elif key in SPECIAL_SCALARS and all(array.shape == () for array in arrays):
        output = merged_scalar_value(key, args, output_npz)
        old_values = [scalar_string(array) for array in arrays]
        message = f"rewrote differing scalar values {old_values} -> {scalar_string(output)}"
    else:
        shapes = [array.shape for array in arrays]
        dtypes = [str(array.dtype) for array in arrays]
        raise ValueError(
            f"Non-frame key `{key}` differs across inputs; shapes={shapes}, dtypes={dtypes}"
        )

    return output, all_equal, message


def write_static_key(
    zf: zipfile.ZipFile,
    key: str,
    input_paths: list[Path],
    args: argparse.Namespace,
    output_npz: Path,
) -> tuple[bool, str]:
    output, all_equal, message = resolve_static_key(key, input_paths, args, output_npz)
    with zf.open(npz_member_name(key), mode="w", force_zip64=True) as out_fh:
        npy_format.write_array(out_fh, output, allow_pickle=False)
    return all_equal, message


def format_shape(shape: tuple[int, ...]) -> str:
    if not shape:
        return "()"
    if len(shape) == 1:
        return f"({shape[0]},)"
    return f"({', '.join(map(str, shape))})"


def print_plan(
    input_paths: list[Path],
    output_npz: Path,
    keys: list[str],
    frame_keys: list[str],
    headers_by_input: list[dict[str, NpyHeader]],
    frame_counts: list[int],
) -> None:
    total_frames = sum(frame_counts)
    print("Inputs:")
    for path, frame_count in zip(input_paths, frame_counts):
        print(f"  {path} ({frame_count} frames)")
    print(f"Output: {output_npz}")
    print(f"Total frames: {total_frames}")
    print("Frame-concatenated keys:")
    for key in frame_keys:
        shape = (total_frames, *headers_by_input[0][key].shape[1:])
        dtype = output_dtype([headers[key] for headers in headers_by_input])
        print(f"  {key}: {format_shape(shape)} dtype={dtype}")
    static_keys = [key for key in keys if key not in set(frame_keys)]
    print("Static keys:")
    for key in static_keys:
        header = headers_by_input[0][key]
        print(f"  {key}: shape={format_shape(header.shape)} dtype={header.dtype}")


def validate_static_keys(
    keys: list[str],
    frame_keys: list[str],
    input_paths: list[Path],
    args: argparse.Namespace,
    output_npz: Path,
) -> list[tuple[str, str]]:
    frame_key_set = set(frame_keys)
    messages: list[tuple[str, str]] = []
    for key in keys:
        if key in frame_key_set:
            continue
        _, _, message = resolve_static_key(key, input_paths, args, output_npz)
        messages.append((key, message))
    return messages


def main() -> None:
    args = parse_args()
    input_paths, output_npz = resolve_paths(args)

    key_lists: list[list[str]] = []
    headers_by_input: list[dict[str, NpyHeader]] = []
    for path in input_paths:
        keys, headers = read_npz_headers(path)
        if FRAME_COUNT_KEY not in headers:
            raise KeyError(f"Missing required key `{FRAME_COUNT_KEY}` in {path}")
        if not headers[FRAME_COUNT_KEY].shape:
            raise ValueError(f"`{FRAME_COUNT_KEY}` must have a frame axis in {path}")
        key_lists.append(keys)
        headers_by_input.append(headers)

    keys = validate_key_sets(input_paths, key_lists)
    frame_counts = [headers[FRAME_COUNT_KEY].shape[0] for headers in headers_by_input]
    frame_keys = find_frame_keys(keys, headers_by_input, frame_counts)
    if FRAME_COUNT_KEY not in frame_keys:
        raise AssertionError(f"`{FRAME_COUNT_KEY}` was not identified as frame-indexed.")

    total_frames = sum(frame_counts)
    print_plan(input_paths, output_npz, keys, frame_keys, headers_by_input, frame_counts)
    static_messages = validate_static_keys(keys, frame_keys, input_paths, args, output_npz)
    if args.dry_run:
        print("Static validation:")
        for key, message in static_messages:
            print(f"  {key}: {message}")
        return

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_npz.with_suffix(output_npz.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    compression = zipfile.ZIP_STORED if args.store else zipfile.ZIP_DEFLATED
    compresslevel = None if args.store else args.compresslevel

    try:
        with zipfile.ZipFile(
            tmp_output,
            mode="w",
            compression=compression,
            compresslevel=compresslevel,
            allowZip64=True,
        ) as zf:
            frame_key_set = set(frame_keys)
            for key in keys:
                if key in frame_key_set:
                    output_shape = (total_frames, *headers_by_input[0][key].shape[1:])
                    write_frame_key(zf, key, input_paths, headers_by_input, output_shape)
                else:
                    write_static_key(zf, key, input_paths, args, output_npz)

        if output_npz.exists() and args.overwrite:
            output_npz.unlink()
        tmp_output.replace(output_npz)
    except Exception:
        if tmp_output.exists():
            tmp_output.unlink()
        raise

    print(f"Wrote merged NPZ: {output_npz}")
    for key, message in static_messages:
        if key in SPECIAL_SCALARS:
            print(f"  {key}: {message}")


if __name__ == "__main__":
    main()
