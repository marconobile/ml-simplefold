[scripts/evaluate_active_npz_conditioned_sample.py](/home/nobilm@usi.ch/ml-simplefold/scripts/evaluate_active_npz_conditioned_sample.py:1) evaluates a fine-tuned SimpleFold model on one trajectory frame.

In plain terms: it picks one NPZ trajectory frame, feeds the frame’s original atom cluster labels into SimpleFold as conditioning, samples a new structure, then compares the sampled coordinates and dihedral angles against the original frame.

**Main Flow**
1. **Parse CLI args**  
   Defined around [line 53](/home/nobilm@usi.ch/ml-simplefold/scripts/evaluate_active_npz_conditioned_sample.py:53). Important args:
   - `--data-path`: raw `.npz` or processed SimpleFold directory.
   - `--raw-npz-path`: explicit raw NPZ if auto-discovery fails.
   - `--processed-dir`: processed `structures/`, `records/`, optionally `tokens/`.
   - `--checkpoint-path` / `--checkpoint-dir`: fine-tuned checkpoint.
   - `--frame-index`: specific frame; otherwise random.
   - `--seed`: controls random frame and sampling noise.
   - `--num-steps`, `--tau`: sampler settings.

2. **Resolve inputs**
   - `resolve_raw_npz_path()` finds the raw trajectory NPZ.
   - `resolve_processed_dir()` finds the processed SimpleFold dataset.
   - `resolve_checkpoint_path()` picks `last.ckpt` unless overridden.

3. **Load one frame**
   If raw NPZ exists, [load_raw_frame()](/home/nobilm@usi.ch/ml-simplefold/scripts/evaluate_active_npz_conditioned_sample.py:276) reads:
   - `trajectory`: coordinates, shape `(frames, atoms, 3)`
   - `atom_idx_and_glob_cluster_id_per_frame`: atom cluster conditioning labels
   - `dihedrals`
   - `dihedral_atom_indices`
   - `dihedral_mask`

   If raw NPZ is unavailable, [load_processed_frame()](/home/nobilm@usi.ch/ml-simplefold/scripts/evaluate_active_npz_conditioned_sample.py:359) loads coordinates and cluster labels from processed SimpleFold structures, but dihedral comparison is skipped.

4. **Build the model input batch**
   [prepare_conditioned_batch()](/home/nobilm@usi.ch/ml-simplefold/scripts/evaluate_active_npz_conditioned_sample.py:527) either:
   - loads processed `structure`, `record`, and cached tokens, or
   - builds a SimpleFold-compatible structure directly from raw atom names/residue ids.

   Then it tokenizes, featurizes, adds ESM features, and injects the atom cluster labels into the batch under:

   ```python
   atom_idx_and_glob_cluster_id_per_frame
   ```

   Labels are padded with `-1` if the model atom array is larger than the raw atom array.

5. **Load models**
   It first loads the ESM model to compute sequence features, then frees it from memory. After that, [instantiate_and_load_model()](/home/nobilm@usi.ch/ml-simplefold/scripts/evaluate_active_npz_conditioned_sample.py:612) loads the FoldingDiT model from the checkpoint. By default it prefers EMA weights with prefix:

   ```python
   model_ema.module.
   ```

   `--use-non-ema-weights` switches preference to `model.` weights.

6. **Sample a conditioned structure**
   Around [line 1295](/home/nobilm@usi.ch/ml-simplefold/scripts/evaluate_active_npz_conditioned_sample.py:1295), it creates random coordinate noise and runs:

   ```python
   sampler.sample(model, flow, noise, batch)
   ```

   using `EMSampler` and `LinearPath`. The output is postprocessed back into real coordinates.

7. **Compare coordinates**
   [kabsch_align()](/home/nobilm@usi.ch/ml-simplefold/scripts/evaluate_active_npz_conditioned_sample.py:669) aligns sampled coordinates to the original frame and computes:
   - global RMSD
   - per-atom RMSD
   - mean / median / max atom RMSD

8. **Compare dihedrals**
   If raw dihedrals exist, [compute_dihedral_angles()](/home/nobilm@usi.ch/ml-simplefold/scripts/evaluate_active_npz_conditioned_sample.py:698) recomputes sampled dihedrals from coordinates. Then [summarize_dihedrals()](/home/nobilm@usi.ch/ml-simplefold/scripts/evaluate_active_npz_conditioned_sample.py:748) compares sampled vs original using circular angle differences, so wraparound at `-180/180` degrees is handled correctly.

   It reports:
   - dihedral MAE in degrees
   - RMSE
   - max absolute error
   - breakdown by dihedral key
   - histograms as CSV and optionally PNG

9. **Write artifacts**
   Outputs go to `artifacts/active_npz_conditioned_eval` by default:
   - `*_report.txt`: human-readable summary
   - `*.json`: metrics
   - `*.npz`: detailed arrays
   - `*_atomwise_rmsd.csv`
   - `*_dihedrals.csv`
   - `*_dihedral_*_histograms.csv`
   - optional histogram PNGs
   - sampled aligned mmCIF
   - sampled raw unaligned mmCIF
   - target reference mmCIF

**Typical Command**
```bash
python scripts/evaluate_active_npz_conditioned_sample.py --frame-index 0 --seed 123
```

The key idea is: this script checks whether the fine-tuned cluster-conditioned SimpleFold model can regenerate a structure close to a specific simulation frame when given that frame’s original atom-level cluster labels.