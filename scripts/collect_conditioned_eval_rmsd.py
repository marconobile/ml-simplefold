#!/usr/bin/env python3
"""
Collect global RMSD values from conditioned-evaluation reports.
Eg usage:
python scripts/collect_conditioned_eval_rmsd.py \
  /storage_common/nobilm/ml-simplefold/fine_tune_with_clusters/finetune_simplefold100M_ANECAG_THEO_INZMA_INACTIVE_merged/official_denovo_with_reference_tests/PAS_test_samples

"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


REPORT_SUFFIX = "_conditioned_eval_report.txt"
DEFAULT_OUTPUT_NAME = "conditioned_eval_global_rmsd.csv"
SAMPLE_ID_PATTERN = re.compile(r"(?:^|_)sample_(\d+)(?:_|$)")
GLOBAL_RMSD_PATTERN = re.compile(r"^\s*global_rmsd_A\s*:\s*(\S+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find *_conditioned_eval_report.txt files recursively, extract each "
            "sample ID and global_rmsd_A value, and write them to a CSV."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        metavar="INPUT_DIR",
        help="Directory to search recursively for conditioned evaluation reports.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=(
            "Name of the CSV created inside INPUT_DIR "
            f"(default: {DEFAULT_OUTPUT_NAME})."
        ),
    )
    return parser.parse_args()


def sample_id_from_filename(path: Path) -> str:
    match = SAMPLE_ID_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not find a sample_<digits> ID in filename: {path}")
    return match.group(1)


def global_rmsd_from_report(path: Path) -> float:
    values: list[float] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = GLOBAL_RMSD_PATTERN.match(line)
            if match is None:
                continue

            try:
                value = float(match.group(1))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid global_rmsd_A value in {path} at line {line_number}: "
                    f"{match.group(1)!r}"
                ) from exc

            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite global_rmsd_A value in {path} at line "
                    f"{line_number}: {match.group(1)!r}"
                )
            values.append(value)

    if not values:
        raise ValueError(f"No global_rmsd_A line found in report: {path}")
    if len(values) > 1:
        raise ValueError(f"Multiple global_rmsd_A lines found in report: {path}")
    return values[0]


def collect_rows(input_dir: Path) -> list[tuple[str, float]]:
    report_paths = sorted(
        path
        for path in input_dir.rglob(f"*{REPORT_SUFFIX}")
        if path.is_file()
    )
    rows = [
        (sample_id_from_filename(path), global_rmsd_from_report(path))
        for path in report_paths
    ]
    return sorted(rows, key=lambda row: (row[1], int(row[0]), row[0]))


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"INPUT_DIR is not a directory: {input_dir}")

    output_name = Path(args.output_name)
    if output_name.name != args.output_name or output_name.suffix.lower() != ".csv":
        raise ValueError("--output-name must be a .csv filename without directories.")

    rows = collect_rows(input_dir)
    output_path = input_dir / output_name
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "global_rmsd_A"))
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
