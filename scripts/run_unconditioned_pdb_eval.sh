#!/usr/bin/env bash
set -euo pipefail
# eg: N=50 bash scripts/run_unconditioned_pdb_eval.sh /home/nobilm@usi.ch/ml-simplefold/artifacts/predictions_simplefold_100M_torch_BASELINE_og

# Evaluation-only wrapper for generated PDBs produced without conditioning.
#
# This intentionally skips:
#   - scripts/sample_with_conditioning.py, because the structure already exists
#   - scripts/compare_conditioning_to_oracle.py, because there is no input
#     conditioning vector to compare against oracle labels
#
# It does run the usable post-sampling pieces:
#   1. assign oracle residue clusters to the generated PDB(s)
#   2. compare those labels to N sampled active/inactive/pas reference structures
#   3. call plot_evaluation.py on a staging tree with the naming/layout it expects
#
# Usage:
#   bash scripts/run_unconditioned_pdb_eval.sh /path/to/folder_or_file.pdb
#
# Common overrides:
#   N=10 BASE_SEED=123 EVAL_ROOT=/path/to/out \
#     bash scripts/run_unconditioned_pdb_eval.sh /path/to/pdb_folder

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT_PATH="${1:-${PDB_INPUT_PATH:-${BASE_PATH:-}}}"
if [[ -z "${INPUT_PATH}" ]]; then
    echo "Usage: bash scripts/run_unconditioned_pdb_eval.sh /path/to/folder_or_file.pdb" >&2
    echo "Or set PDB_INPUT_PATH=/path/to/folder_or_file.pdb" >&2
    exit 2
fi
if [[ ! -e "${INPUT_PATH}" ]]; then
    echo "Input path does not exist: ${INPUT_PATH}" >&2
    exit 2
fi

CLUSTER_DIR="${CLUSTER_DIR:-/storage_common/angiod/phase-data/projects/a2a/systems/a2a_small/clusters/6d4a7baa-c096-494e-b417-c8014437d37d}"
REF_NPZ="${REF_NPZ:-${CLUSTER_DIR}/cluster.npz}"
N="${N:-5}"
BASE_SEED="${BASE_SEED:-123}"
STRUCTURE_TYPES="${STRUCTURE_TYPES:-active inactive pas}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

if [[ -d "${INPUT_PATH}" ]]; then
    DEFAULT_EVAL_ROOT="${INPUT_PATH%/}/unconditioned_cluster_eval"
else
    DEFAULT_EVAL_ROOT="$(dirname "${INPUT_PATH}")/unconditioned_cluster_eval"
fi
EVAL_ROOT="${EVAL_ROOT:-${DEFAULT_EVAL_ROOT}}"
PDB_STAGE_DIR="${EVAL_ROOT}/pdb_inputs"
PLOT_STAGE_DIR="${EVAL_ROOT}/plot_evaluation_inputs"
STAGED_PDB_LIST="${EVAL_ROOT}/staged_pdb_inputs.txt"
SUMMARY_CSV="${EVAL_ROOT}/unconditioned_cluster_reference_matches.csv"
SUMMARY_TXT="${EVAL_ROOT}/unconditioned_cluster_reference_summary.txt"

mkdir -p "${PDB_STAGE_DIR}" "${PLOT_STAGE_DIR}"

cd "${REPO_ROOT}"

echo "Staging generated PDB input(s)"
python - \
    "${INPUT_PATH}" \
    "${PDB_STAGE_DIR}" \
    "${EVAL_ROOT}" \
    "${STAGED_PDB_LIST}" <<'PY'
import hashlib
import re
import shutil
import sys
from pathlib import Path

input_path = Path(sys.argv[1]).expanduser().resolve()
pdb_stage_dir = Path(sys.argv[2]).expanduser().resolve()
eval_root = Path(sys.argv[3]).expanduser().resolve()
staged_pdb_list = Path(sys.argv[4]).expanduser().resolve()
sampled_token = "_conditioned_eval_sampled"

def inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

def safe_stem(path: Path, seen: set[str]) -> str:
    stem = path.stem
    if stem.endswith(sampled_token):
        stem = stem[: -len(sampled_token)]
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "sample"
    if cleaned not in seen:
        seen.add(cleaned)
        return cleaned
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    unique = f"{cleaned}_{digest}"
    seen.add(unique)
    return unique

if input_path.is_file():
    if input_path.suffix.lower() != ".pdb":
        raise SystemExit(f"Input file is not a PDB: {input_path}")
    pdb_paths = [input_path]
else:
    pdb_paths = sorted(
        path
        for path in input_path.rglob("*.pdb")
        if path.is_file() and not inside(path, eval_root)
    )

if not pdb_paths:
    raise SystemExit(f"No .pdb files found in: {input_path}")

seen_names: set[str] = set()
staged_paths: list[Path] = []
pdb_stage_dir.mkdir(parents=True, exist_ok=True)
for pdb_path in pdb_paths:
    stem = safe_stem(pdb_path, seen_names)
    staged_pdb = pdb_stage_dir / f"{stem}{sampled_token}.pdb"
    if staged_pdb.exists() or staged_pdb.is_symlink():
        staged_pdb.unlink()
    shutil.copy2(pdb_path, staged_pdb)

    paired_cif = pdb_path.with_suffix(".cif")
    staged_cif = pdb_stage_dir / f"{stem}{sampled_token}.cif"
    if staged_cif.exists() or staged_cif.is_symlink():
        staged_cif.unlink()
    if paired_cif.is_file():
        shutil.copy2(paired_cif, staged_cif)

    print(f"  {pdb_path} -> {staged_pdb}")
    staged_paths.append(staged_pdb)

staged_pdb_list.parent.mkdir(parents=True, exist_ok=True)
staged_pdb_list.write_text(
    "\n".join(str(path) for path in staged_paths) + "\n",
    encoding="utf-8",
)
PY

assign_cmd=(
    python scripts/assign_conditioned_eval_sample_clusters.py
    --cluster-dir "${CLUSTER_DIR}"
)
if [[ "${SKIP_EXISTING}" == "1" ]]; then
    assign_cmd+=(--skip-existing)
fi

echo "Assigning oracle clusters"
while IFS= read -r staged_pdb; do
    [[ -n "${staged_pdb}" ]] || continue
    "${assign_cmd[@]}" --base-path "${staged_pdb}"
done < "${STAGED_PDB_LIST}"

echo "Preparing unconditioned reference-label evaluation"
python - \
    "${PDB_STAGE_DIR}" \
    "${PLOT_STAGE_DIR}" \
    "${STAGED_PDB_LIST}" \
    "${REF_NPZ}" \
    "${SUMMARY_CSV}" \
    "${SUMMARY_TXT}" \
    "${N}" \
    "${BASE_SEED}" \
    "${STRUCTURE_TYPES}" <<'PY'
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

pdb_stage_dir = Path(sys.argv[1]).expanduser().resolve()
plot_stage_dir = Path(sys.argv[2]).expanduser().resolve()
staged_pdb_list = Path(sys.argv[3]).expanduser().resolve()
ref_npz = Path(sys.argv[4]).expanduser().resolve()
summary_csv = Path(sys.argv[5]).expanduser().resolve()
summary_txt = Path(sys.argv[6]).expanduser().resolve()
n_per_type = int(sys.argv[7])
base_seed = int(sys.argv[8])
structure_types = [token for token in sys.argv[9].split() if token]

if n_per_type <= 0:
    raise SystemExit("N must be > 0")
if not structure_types:
    raise SystemExit("STRUCTURE_TYPES must contain at least one source type")

sampled_token = "_conditioned_eval_sampled.pdb"
assigned_token = "_assigned_clusters.npz"
with staged_pdb_list.open(encoding="utf-8") as handle:
    staged_pdb_paths = [Path(line.strip()).expanduser().resolve() for line in handle if line.strip()]

assigned_paths = []
for staged_pdb in staged_pdb_paths:
    if sampled_token not in staged_pdb.name:
        raise SystemExit(f"Unexpected staged PDB name: {staged_pdb}")
    assigned_paths.append(
        staged_pdb.with_name(staged_pdb.name.replace(sampled_token, assigned_token, 1))
    )
if not assigned_paths:
    raise SystemExit(f"No staged PDBs listed in: {staged_pdb_list}")
missing_assigned = [path for path in assigned_paths if not path.is_file()]
if missing_assigned:
    preview = "\n".join(f"  {path}" for path in missing_assigned[:10])
    raise SystemExit(f"Missing assigned-cluster output(s):\n{preview}")

for old_path in plot_stage_dir.rglob("*_assigned_clusters.npz"):
    if old_path.is_file() or old_path.is_symlink():
        old_path.unlink()

def normalize_source_type(source_type: str) -> str:
    return str(source_type).split("_", 1)[0]

def one_dimensional(labels: np.ndarray, path: Path) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim == 2 and labels.shape[0] == 1:
        labels = labels[0]
    if labels.ndim != 1:
        raise ValueError(f"{path}: expected 1D labels or one 2D row, got {labels.shape}")
    return labels

def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)

with np.load(ref_npz, allow_pickle=False) as ref_data:
    required = {
        "merged__labels_assigned",
        "merged__frame_state_ids",
        "merged__frame_indices",
    }
    missing = sorted(required - set(ref_data.files))
    if missing:
        raise SystemExit(f"{ref_npz} is missing required key(s): {', '.join(missing)}")

    ref_labels_all = np.asarray(ref_data["merged__labels_assigned"], dtype=np.int64)
    state_ids = np.asarray(ref_data["merged__frame_state_ids"]).astype(str)
    frame_indices = np.asarray(ref_data["merged__frame_indices"], dtype=np.int64)

    if ref_labels_all.shape[0] != state_ids.shape[0] or state_ids.shape[0] != frame_indices.shape[0]:
        raise SystemExit("Reference labels, state IDs, and frame indices have inconsistent lengths")

    normalized_state_ids = np.asarray([normalize_source_type(value) for value in state_ids])
    rng = np.random.default_rng(base_seed)
    chosen_rows_by_source: dict[str, np.ndarray] = {}
    for source in structure_types:
        rows = np.flatnonzero(normalized_state_ids == source)
        if rows.size == 0:
            raise SystemExit(f"No reference rows found for source type {source!r} in {ref_npz}")
        take = min(n_per_type, int(rows.size))
        if take < n_per_type:
            print(f"Warning: requested N={n_per_type} for {source}, but only {take} rows exist")
        chosen = rng.choice(rows, size=take, replace=False)
        chosen_rows_by_source[source] = np.asarray(chosen, dtype=np.int64)

rows_out: list[dict[str, object]] = []
for assigned_path in assigned_paths:
    with np.load(assigned_path, allow_pickle=False) as assigned_data:
        if "labels_assigned" not in assigned_data.files:
            raise SystemExit(f"{assigned_path} is missing required key 'labels_assigned'")
        assigned_labels = one_dimensional(assigned_data["labels_assigned"], assigned_path)

    sample_name = assigned_path.name.replace("_assigned_clusters.npz", "")
    for source in structure_types:
        source_rows = chosen_rows_by_source[source]
        for ref_row in source_rows:
            reference_labels = one_dimensional(ref_labels_all[int(ref_row)], ref_npz)
            if reference_labels.shape != assigned_labels.shape:
                raise SystemExit(
                    f"Label shape mismatch for {assigned_path} vs reference row {int(ref_row)}: "
                    f"{assigned_labels.shape} vs {reference_labels.shape}"
                )

            exact = int(np.count_nonzero(reference_labels == assigned_labels))
            total = int(reference_labels.size)
            diff = int(total - exact)
            fraction = float(exact / total) if total else 0.0
            frame_idx = int(frame_indices[int(ref_row)])
            staged_name = f"{sample_name}__vs_{source}_{frame_idx:06d}_assigned_clusters.npz"
            staged_path = plot_stage_dir / f"{source}_samples" / staged_name
            link_or_copy(assigned_path, staged_path)

            rows_out.append(
                {
                    "sample": sample_name,
                    "assigned_clusters_npz": str(assigned_path),
                    "reference_source": source,
                    "reference_state_id": str(state_ids[int(ref_row)]),
                    "reference_row_index": int(ref_row),
                    "reference_frame_index": frame_idx,
                    "matching_residues": exact,
                    "differing_residues": diff,
                    "total_residues": total,
                    "match_fraction": fraction,
                    "staged_npz": str(staged_path),
                }
            )

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="") as handle:
    fieldnames = [
        "sample",
        "assigned_clusters_npz",
        "reference_source",
        "reference_state_id",
        "reference_row_index",
        "reference_frame_index",
        "matching_residues",
        "differing_residues",
        "total_residues",
        "match_fraction",
        "staged_npz",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
for row in rows_out:
    grouped[(str(row["sample"]), str(row["reference_source"]))].append(row)

lines = [
    "Unconditioned PDB oracle-cluster reference comparison",
    "",
    f"reference_npz: {ref_npz}",
    f"N_per_source_requested: {n_per_type}",
    f"base_seed: {base_seed}",
    f"staged_plot_input_dir: {plot_stage_dir}",
    f"per_pair_csv: {summary_csv}",
    "",
]
for sample in sorted({str(row["sample"]) for row in rows_out}):
    lines.append(f"Sample: {sample}")
    sample_rows = [row for row in rows_out if str(row["sample"]) == sample]
    for source in structure_types:
        values = grouped.get((sample, source), [])
        if not values:
            continue
        best = max(values, key=lambda row: (int(row["matching_residues"]), -int(row["reference_row_index"])))
        mean_fraction = float(np.mean([float(row["match_fraction"]) for row in values]))
        mean_matches = float(np.mean([int(row["matching_residues"]) for row in values]))
        lines.extend(
            [
                f"  {source}: n={len(values)}, mean_matches={mean_matches:.2f}, "
                f"mean_match_fraction={mean_fraction:.4f}",
                f"    best_frame={best['reference_frame_index']} "
                f"best_state={best['reference_state_id']} "
                f"matches={best['matching_residues']}/{best['total_residues']} "
                f"fraction={float(best['match_fraction']):.4f}",
            ]
        )
    best_overall = max(sample_rows, key=lambda row: (int(row["matching_residues"]), -int(row["reference_row_index"])))
    lines.append(
        f"  best_overall: source={best_overall['reference_source']} "
        f"frame={best_overall['reference_frame_index']} "
        f"matches={best_overall['matching_residues']}/{best_overall['total_residues']} "
        f"fraction={float(best_overall['match_fraction']):.4f}"
    )
    lines.append("")

summary_txt.write_text("\n".join(lines), encoding="utf-8")

print(f"Wrote per-pair CSV: {summary_csv}")
print(f"Wrote summary: {summary_txt}")
print(f"Prepared plot_evaluation input tree: {plot_stage_dir}")
PY

echo "Plotting assigned-cluster reference matches"
python plot_evaluation.py \
    --base_path "${PLOT_STAGE_DIR}" \
    --out_dir "${EVAL_ROOT}" \
    --ref_npz "${REF_NPZ}"

echo "Done"
echo "Summary: ${SUMMARY_TXT}"
echo "Per-pair CSV: ${SUMMARY_CSV}"
echo "Plot: ${EVAL_ROOT}/assigned_cluster_match_histograms.png"
