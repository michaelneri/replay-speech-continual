"""
Continual learning evaluation metrics from:
  Wang et al. (2024), "A Comprehensive Survey of Continual Learning:
  Theory, Method and Application", IEEE TPAMI, Section II-C.

Setup
-----
We assume Domain-Incremental Learning (DIL): same label space (genuine / replay)
across all tasks, tasks = environments 1-4, device fixed to Device 3.

Performance matrix
------------------
    perf[k, j]  =  performance on task j's test set
                   after training on tasks 0 .. k  (all 0-indexed)

Only the lower triangle (j <= k) is meaningful; upper entries are ignored.

Sign conventions (this module)
-------------------------------
Metrics are returned so that the following universally holds regardless
of whether the raw metric is EER (lower=better) or accuracy (higher=better):

    Scalar metrics  (AA, AIA)
        → in original units (EER % or accuracy %)
        → lower AA is better when lower_is_better=True

    Difference metrics  (FM, BWT, IM, FWT)
        → always returned in "score-space" where higher score = better.
          score = -EER  for EER,  score = accuracy  for accuracy.
        → sign interpretation is CONSISTENT:
              FM  > 0   →  forgetting occurred
              BWT < 0   →  forgetting (backward interference)
              IM  > 0   →  intransient (continual is worse than joint)
              FWT > 0   →  positive forward transfer

This avoids confusion from opposite sign conventions for EER vs accuracy.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_cl_metrics(
    perf_matrix: np.ndarray,
    joint_ref: Optional[np.ndarray] = None,
    single_ref: Optional[np.ndarray] = None,
    lower_is_better: bool = True,
) -> dict:
    """
    Compute all CL evaluation metrics at each step, returning curves and
    a final scalar for each metric.

    Parameters
    ----------
    perf_matrix : ndarray, shape (K, K)
        perf_matrix[k, j] = performance (e.g. EER or accuracy) on task j
        after training on tasks 0..k.  Only lower-triangle (j <= k) is used.
    joint_ref : ndarray, shape (K,), optional
        joint_ref[k] = performance of a JOINTLY trained reference model on
        tasks 0..k, evaluated on task k.  Required for IM.
    single_ref : ndarray, shape (K,), optional
        single_ref[j] = performance of a model trained ONLY on task j.
        Required for FWT.
    lower_is_better : bool
        True  → EER-like metric (lower = better, e.g. EER %)
        False → accuracy-like metric (higher = better)

    Returns
    -------
    dict with keys:

    Scalar metrics (in original units, e.g. EER or accuracy)
        'AA'   : float  – Average performance at final task
        'AIA'  : float  – Average Incremental performance at final task

    Difference metrics (in score-space units; see module docstring for sign)
        'FM'   : float  – Forgetting Measure  (> 0 = forgot)
        'BWT'  : float  – Backward Transfer   (< 0 = forgot)
        'IM'   : float or None – Intransience  (> 0 = worse than joint)
        'FWT'  : float or None – Forward Transfer (> 0 = positive transfer)

    Curves (one value per step k; NaN where undefined):
        'AA_curve', 'AIA_curve'        – in original units
        'FM_curve', 'BWT_curve'        – in score-space units
    """
    perf_matrix = np.array(perf_matrix, dtype=float)
    K = perf_matrix.shape[0]
    assert perf_matrix.shape == (K, K), "perf_matrix must be square (K x K)"

    # Convert to score: higher is always better
    sign = -1.0 if lower_is_better else 1.0
    s = sign * perf_matrix                                   # s[k, j]: higher = better
    j_ref_s = sign * np.array(joint_ref) if joint_ref is not None else None
    sr_ref_s = sign * np.array(single_ref) if single_ref is not None else None

    results: dict = {}

    # ------------------------------------------------------------------
    # AA_k = (1/(k+1)) * sum_{j=0}^{k} s[k, j]   (Wang eq. after Table I)
    # AIA_k = (1/(k+1)) * sum_{i=0}^{k} AA_i
    # Returned in ORIGINAL units (multiply back by sign).
    # ------------------------------------------------------------------
    AA_curve_s = np.array([np.mean(s[k, :k + 1]) for k in range(K)])
    AIA_curve_s = np.array([np.mean(AA_curve_s[:k + 1]) for k in range(K)])

    results['AA_curve'] = sign * AA_curve_s     # original units
    results['AIA_curve'] = sign * AIA_curve_s
    results['AA'] = float(sign * AA_curve_s[-1])
    results['AIA'] = float(sign * AIA_curve_s[-1])

    # ------------------------------------------------------------------
    # FM_k = (1/k) * sum_{j=0}^{k-1} f_{j,k}
    #   f_{j,k} = max_{i=j..k-1} s[i, j]  -  s[k, j]
    #           = best score on task j before step k  -  score at step k
    # Positive FM → score dropped from best previous → forgetting.
    # (Equivalent to: for EER, current EER > best previous EER.)
    #
    # BWT_k = (1/k) * sum_{j=0}^{k-1} ( s[k, j] - s[j, j] )
    # Negative BWT → score at final step < score when task was first learned
    #              → forgetting.  Positive → backward improvement.
    # ------------------------------------------------------------------
    FM_curve = np.full(K, np.nan)
    BWT_curve = np.full(K, np.nan)

    for k in range(1, K):
        f_vals = []
        bwt_vals = []
        for j in range(k):
            best_prev = np.max(s[j:k, j])        # best score on task j before step k
            f_vals.append(best_prev - s[k, j])   # positive = score fell
            bwt_vals.append(s[k, j] - s[j, j])  # positive = improved vs. step j
        FM_curve[k] = np.mean(f_vals)
        BWT_curve[k] = np.mean(bwt_vals)

    # Returned in SCORE-SPACE units (no sign conversion → consistent semantics).
    results['FM_curve'] = FM_curve
    results['BWT_curve'] = BWT_curve
    results['FM'] = float(FM_curve[-1]) if not np.isnan(FM_curve[-1]) else float('nan')
    results['BWT'] = float(BWT_curve[-1]) if not np.isnan(BWT_curve[-1]) else float('nan')

    # ------------------------------------------------------------------
    # IM_k = joint_ref_score[k] - s[k, k]
    # Positive → joint model outperforms continual on new task → intransient.
    # (For EER: IM > 0 ↔ joint EER < continual EER ↔ continual is worse.)
    # ------------------------------------------------------------------
    if j_ref_s is not None:
        IM = j_ref_s[K - 1] - s[K - 1, K - 1]
        results['IM'] = float(IM)
    else:
        results['IM'] = None

    # ------------------------------------------------------------------
    # FWT_k = (1/(k)) * sum_{j=1}^{k} ( s[j, j] - single_ref_score[j] )
    # Positive → continual outperforms the single-task baseline on new task
    #          → prior tasks provided positive forward transfer.
    # (For EER: FWT > 0 ↔ continual EER < single-task EER.)
    # ------------------------------------------------------------------
    if sr_ref_s is not None and K > 1:
        fwt_vals = [s[j, j] - sr_ref_s[j] for j in range(1, K)]
        results['FWT'] = float(np.mean(fwt_vals))
    else:
        results['FWT'] = None

    return results


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_metrics(
    metrics: dict,
    metric_name: str = "EER",
    lower_is_better: bool = True,
) -> None:
    """Print a formatted summary of CL metrics."""
    dir_note = "lower=better" if lower_is_better else "higher=better"
    scl_note = "lower AA = better" if lower_is_better else "higher AA = better"

    print(f"\n{'='*62}")
    print(f"  Continual Learning Metrics  [{metric_name} | {dir_note}]")
    print(f"{'='*62}")
    print(f"  AA    Avg performance at final task    : {metrics['AA']:.4f}  [{scl_note}]")
    print(f"  AIA   Avg incremental performance      : {metrics['AIA']:.4f}  [{scl_note}]")

    fm = metrics['FM']
    if not np.isnan(fm):
        flag = "FORGOT" if fm > 0 else "stable/improved"
        print(f"  FM    Forgetting Measure               : {fm:+.4f}  [> 0 = {flag}]")
    else:
        print(f"  FM    Forgetting Measure               : N/A (K=1)")

    bwt = metrics['BWT']
    if not np.isnan(bwt):
        flag = "FORGOT" if bwt < 0 else "improved/stable"
        print(f"  BWT   Backward Transfer                : {bwt:+.4f}  [< 0 = {flag}]")
    else:
        print(f"  BWT   Backward Transfer                : N/A (K=1)")

    im = metrics.get('IM')
    if im is not None:
        flag = "intransient" if im > 0 else "matches/exceeds joint"
        print(f"  IM    Intransience Measure             : {im:+.4f}  [> 0 = {flag}]")
    else:
        print(f"  IM    Intransience Measure             : N/A (no joint_ref provided)")

    fwt = metrics.get('FWT')
    if fwt is not None:
        flag = "positive transfer" if fwt > 0 else "negative/no transfer"
        print(f"  FWT   Forward Transfer                 : {fwt:+.4f}  [> 0 = {flag}]")
    else:
        print(f"  FWT   Forward Transfer                 : N/A (no single_ref provided)")

    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# Sanity check with toy EER data (Device 3, 4 environments)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Toy EER matrix
    # Rows = training step k (trained on envs 0..k)
    # Cols = environment j being evaluated
    # Expected: EER rises on old envs as model trains more (catastrophic forgetting)
    eer = np.array([
        [0.10, np.nan, np.nan, np.nan],   # trained on env 0 only
        [0.15, 0.08,  np.nan, np.nan],   # trained on envs 0-1; env 0 got worse
        [0.18, 0.12,  0.09,  np.nan],   # trained on envs 0-2
        [0.20, 0.14,  0.11,  0.07],     # trained on envs 0-3 (final)
    ])

    # joint_ref[k] = EER of model jointly trained on envs 0..k, tested on env k
    joint_ref = np.array([0.09, 0.07, 0.08, 0.06])

    # single_ref[j] = EER of model trained ONLY on env j
    single_ref = np.array([0.11, 0.09, 0.10, 0.08])

    metrics = compute_cl_metrics(
        eer,
        joint_ref=joint_ref,
        single_ref=single_ref,
        lower_is_better=True,
    )
    print_metrics(metrics, metric_name="EER", lower_is_better=True)

    # --- Manual verification ---
    print("Manual checks:")
    print(f"  AA  = mean(0.20, 0.14, 0.11, 0.07) = {np.mean([0.20, 0.14, 0.11, 0.07]):.4f}  (expected 0.1300)")

    # FM: for each j, best_prev_EER = min(eer[j:k, j]), f = current - best
    f0 = 0.20 - min(0.10, 0.15, 0.18)  # env 0: best=0.10, current=0.20
    f1 = 0.14 - min(0.08, 0.12)        # env 1: best=0.08, current=0.14
    f2 = 0.11 - 0.09                   # env 2: best=0.09, current=0.11
    print(f"  FM  = mean({f0:.2f}, {f1:.2f}, {f2:.2f}) = {np.mean([f0, f1, f2]):.4f}  (expected +0.0600)")

    # BWT: for EER in score space, BWT = mean(s[k,j] - s[j,j]) = mean(-eer[k,j] + eer[j,j])
    b0 = -(0.20 - 0.10)   # s[3,0]-s[0,0] = -0.20 - (-0.10) = -0.10
    b1 = -(0.14 - 0.08)
    b2 = -(0.11 - 0.09)
    print(f"  BWT = mean({b0:.2f}, {b1:.2f}, {b2:.2f}) = {np.mean([b0, b1, b2]):.4f}  (< 0 = forgetting)")

    # IM: joint_ref_score[3] - s[3,3] = -0.06 - (-0.07) = +0.01
    im = -0.06 - (-0.07)
    print(f"  IM  = {im:.4f}  (> 0 = intransient, joint EER 0.06 < continual EER 0.07)")

    # FWT: for j=1,2,3: s[j,j] - sr_ref_s[j] = -eer[j,j] - (-single_ref[j]) = single_ref-eer[j,j]
    fwt_vals = [0.09 - 0.08, 0.10 - 0.09, 0.08 - 0.07]
    print(f"  FWT = mean({fwt_vals}) = {np.mean(fwt_vals):.4f}  (> 0 = positive transfer)")

    print("\nCurves:")
    print(f"  AA_curve  : {metrics['AA_curve']}")
    print(f"  FM_curve  : {metrics['FM_curve']}")
    print(f"  BWT_curve : {metrics['BWT_curve']}")
