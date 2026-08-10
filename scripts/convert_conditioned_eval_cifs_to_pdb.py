#!/usr/bin/env python3
"""Convert conditioned-evaluation CIF files to sibling PDB files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gemmi


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval_utils import detect_chirality, flip_pdb_coordinates  # noqa: E402


MATCH_TOKEN = ".cif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find conditioned-evaluation CIF files under a base path and convert "
            "each one to a PDB in the same directory."
        )
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help=(
            "Directory to search recursively for conditioned evaluation CIF files, "
            "or one CIF file to convert."
        ),
    )
    parser.add_argument(
        "--additional-cif-path",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional CIF file to convert. May be repeated. This lets sampling "
            "convert a sampled CIF and its target-reference CIF in one invocation."
        ),
    )
    parser.add_argument(
        "--name-suffix",
        default=MATCH_TOKEN,
        help=(
            "Only recursively discovered files with this suffix are converted. "
            f"Defaults to {MATCH_TOKEN!r}. Explicit --additional-cif-path files "
            "are always included."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversion when the target PDB already exists.",
    )
    parser.set_defaults(fix_chirality=True)
    parser.add_argument(
        "--fix-chirality",
        dest="fix_chirality",
        action="store_true",
        help=(
            "After conversion, invert all coordinates when more D- than "
            "L-amino acids are detected. This changes the sampled coordinates "
            "and is enabled by default."
        ),
    )
    parser.add_argument(
        "--no-fix-chirality",
        dest="fix_chirality",
        action="store_false",
        help=(
            "Convert CIF to PDB without chirality-based coordinate inversion."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the CIF-to-PDB conversions that would be performed.",
    )
    return parser.parse_args()


def find_conditioned_eval_cifs(
    base_path: Path,
    name_suffix: str = MATCH_TOKEN,
) -> list[Path]:
    if base_path.is_file():
        return [base_path] if base_path.name.endswith(name_suffix) else []

    return sorted(
        path
        for path in base_path.rglob("*")
        if path.is_file() and path.name.endswith(name_suffix)
    )


def target_pdb_path(cif_path: Path) -> Path:
    if cif_path.name.endswith(MATCH_TOKEN):
        return cif_path.with_suffix(".pdb")

    pdb_name = cif_path.name.replace(
        MATCH_TOKEN,
        "_conditioned_eval_sampled.pdb",
        1,
    )
    return cif_path.with_name(pdb_name)


def convert_cif_to_pdb(cif_path: Path, pdb_path: Path) -> None:
    structure = gemmi.read_structure(str(cif_path))
    structure.write_pdb(str(pdb_path))


def ensure_l_chirality(pdb_path: Path) -> None:
    print(f"Analyzing chirality for: {pdb_path}\n")
    data = detect_chirality(str(pdb_path)) or []
    l_count = sum(1 for residue in data if residue["chirality"] == "L")
    d_count = sum(1 for residue in data if residue["chirality"] == "D")

    if d_count > l_count:
        print("More D-amino acids detected than L-amino acids. Flipping coordinates...")
        flip_pdb_coordinates(str(pdb_path), str(pdb_path))
        print(f"Flipped PDB saved to: {pdb_path}\n")
        data = detect_chirality(str(pdb_path)) or []
        l_count = sum(1 for residue in data if residue["chirality"] == "L")
        d_count = sum(1 for residue in data if residue["chirality"] == "D")
        assert (
            d_count < l_count
        ), (
            "Flipping did not result in more L-amino acids than D-amino acids. "
            "Please check the input PDB and flipping logic."
        )


def main() -> None:
    args = parse_args()
    base_path = args.base_path.resolve()

    if not base_path.exists():
        raise FileNotFoundError(f"Base path not found: {base_path}")

    additional_cif_paths = [path.expanduser().resolve() for path in args.additional_cif_path]
    missing_additional_paths = [path for path in additional_cif_paths if not path.is_file()]
    if missing_additional_paths:
        missing_paths = "\n".join(f"  {path}" for path in missing_additional_paths)
        raise FileNotFoundError(f"Additional CIF path(s) not found:\n{missing_paths}")

    cif_paths = sorted(
        set(find_conditioned_eval_cifs(base_path, args.name_suffix)).union(
            additional_cif_paths
        )
    )
    if not cif_paths:
        print(f"No files ending with {args.name_suffix!r} found under {base_path}.")
        return

    converted = 0
    skipped = 0
    failures: list[tuple[Path, Exception]] = []

    for cif_path in cif_paths:
        pdb_path = target_pdb_path(cif_path)

        if pdb_path.exists() and args.skip_existing:
            skipped += 1
            print(f"Skipping existing PDB: {pdb_path}")
            continue

        if args.dry_run:
            print(f"Would convert: {cif_path} -> {pdb_path}")
            continue

        try:
            convert_cif_to_pdb(cif_path, pdb_path)
            if args.fix_chirality:
                ensure_l_chirality(pdb_path)
        except Exception as exc:  # Keep converting independent files.
            failures.append((cif_path, exc))
            print(f"Failed: {cif_path} ({exc})")
            continue

        converted += 1
        print(f"Converted: {cif_path} -> {pdb_path}")

    if args.dry_run:
        print(f"Dry run complete. {len(cif_paths)} file(s) matched.")
        return

    print(
        f"Done. Converted {converted} file(s), skipped {skipped} existing file(s), "
        f"failed {len(failures)} file(s)."
    )

    if failures:
        failed_paths = "\n".join(f"  {path}: {exc}" for path, exc in failures)
        raise RuntimeError(f"Failed to convert {len(failures)} file(s):\n{failed_paths}")


if __name__ == "__main__":
    main()
