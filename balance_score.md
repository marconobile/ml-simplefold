I’ll trace the shell script into the evaluator that defines “Balanced decision score,” then express the exact computation mathematically, including aggregation and edge cases.

• The shell script itself delegates the score to scripts/summarize_tau_guidance_sweep.py. I found the scoring function and am now checking how each input statistic is derived, so the explanation
  distinguishes sample-level quantities, per-grid-cell summaries, and sweep-wide normalization.

• The shell script does not compute the score directly. After evaluating all (6\times6=36) ((\tau,\text{guidance})) combinations, it calls scripts/summarize_tau_guidance_sweep.py:287, where the score is
  defined.

  ## 1. Raw metrics for each parameter setting

  Let (p=(\tau,g)) denote one parameter combination and (i=1,\ldots,N) its samples, normally (N=100).

  ### Cluster-mismatch metrics

  For sample (i) and residue selection (a),

  # [
  q_{p,i,a}

  \frac{
  #{\text{comparable residues whose conditioning and oracle labels differ}}
  }{
  #{\text{comparable residues}}
  }.
  ]

  Only residues for which both labels are non-negative are comparable. This is implemented in scripts/compare_conditioning_to_oracle.py:1083.

  The parameter-level metric is the equally weighted mean over samples:

  # [
  M_{p,a}

  \frac{100}{N}\sum_{i=1}^{N}q_{p,i,a}.
  ]

  Thus it is a mean of per-sample percentages, not a pooled residue-level mismatch percentage.

  Two selections contribute:

  1. (a=\text{VMD}): CA residues in the configured VMD residue ranges.
  2. (a=\text{all}): all residues represented by the all-atom selection.

  These are defined in scripts/summarize_tau_guidance_sweep.py:29.

  ### RMSD metrics

  For sample (i) and atom selection (b), the code computes a rigidly fitted sample-to-target RMSD:

  # [
  d_{p,i,b}

  \min_{R\in SO(3),,t}
  \sqrt{
  \frac{1}{A_b}
  \sum_{j\in b}
  \left|Rx_{p,i,j}+t-y_{i,j}\right|^2
  }.
  ]

  The minimization is performed using an SVD/Kabsch alignment in scripts/compare_conditioning_to_oracle.py:873.

  The parameter-level RMSD metric is

  # [
  R_{p,b}

  \frac{1}{N}\sum_{i=1}^{N}d_{p,i,b}.
  ]

  Five atom selections contribute:

  - strict-VMD CA residues
  - all CA atoms
  - backbone atoms: N, CA, C, O
  - protein non-backbone atoms
  - all atoms

  These are listed in scripts/summarize_tau_guidance_sweep.py:22.

  The separately calculated RMSDs against REFERENCE_PDB do not contribute to the score. Only sampled-PDB versus its paired target-PDB RMSDs are used.

  ## 2. Sweep-wide normalization

  Each of the seven metrics is independently min–max normalized over all 36 parameter combinations.

  For any lower-is-better metric (x),

  # [
  L_{p,x}

  \frac{x_p-\min_{q\in P}x_q}
  {\max_{q\in P}x_q-\min_{q\in P}x_q}.
  ]

  Therefore:

  - (L=0): best observed value in the sweep
  - (L=1): worst observed value
  - intermediate values are linearly interpolated

  If every parameter combination has exactly the same value for a metric, the implementation sets (L_{p,x}=0) for every setting. See scripts/summarize_tau_guidance_sweep.py:275.

  ## 3. Component scores

  The cluster component averages the two normalized mismatch losses:

  # [
  C_p

  100\left[
  1-\frac{
  L_{p,M_{\mathrm{VMD}}}
  +
  L_{p,M_{\mathrm{all}}}
  }{2}
  \right].
  ]

  The structural component averages the five normalized RMSD losses:

  # [
  D_p

  100\left[
  1-\frac{1}{5}
  \sum_{b=1}^{5}L_{p,R_b}
  \right].
  ]

  ## 4. Balanced decision score

  The final score is an equal combination of the two components:

  [
  \boxed{
  B_p=\frac{1}{2}C_p+\frac{1}{2}D_p
  }
  ]

  Equivalently,

  # [
  \boxed{
  B_p

  100
  -25\left(
  L_{p,M_{\mathrm{VMD}}}
  +
  L_{p,M_{\mathrm{all}}}
  \right)
  -10\sum_{b=1}^{5}L_{p,R_b}
  }
  ]

  Consequently, the effective weights are:

  - strict-VMD mismatch: 25%
  - all-residue mismatch: 25%
  - each of the five RMSDs: 10%

  So “balanced” means 50% cluster-label agreement and 50% structural RMSD—not equal weighting among all seven raw metrics.

  ## Interpretation

  The score is relative to this particular sweep, not an absolute physical-quality score. Changing the set of tested ((\tau,g)) combinations can change every score because it changes the per-metric
  minima and maxima. A score of 100 requires being best or tied-best on every nonconstant metric; the best-ranked setting does not necessarily score 100.

  Settings are ranked by decreasing score. Exact score ties are resolved by smaller (\tau), then smaller guidance scale, as defined in scripts/summarize_tau_guidance_sweep.py:308. Medians, standard
  deviations, perfect-structure percentages, and reference-PDB RMSDs are reported but do not affect the decision score.

