"""F_0.5 scoring exactly as defined in the problem statement: macro-averaged over Source-1 entities,
singletons scored 1.0 when correctly predicted empty and 0.0 on any false match.

Optimized for high-resolution threshold search and vectorized computation.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def score_entity(pred: Iterable[str], truth: Iterable[str]) -> float:
    """Spec-exact per-entity score. Verified against the worked example in the problem statement:
    pred={S2-00047,S2-00193,S3-00812}, truth={S2-00047,S3-00812} -> P=2/3, R=1.0, F0.5=0.714."""
    pred, truth = set(pred), set(truth)
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    return f_beta(tp / len(pred), tp / len(truth))


def macro_f05(pred_lists: Sequence[Iterable[str]], truth_lists: Sequence[Iterable[str]]) -> float:
    scores = [score_entity(p, t) for p, t in zip(pred_lists, truth_lists)]
    return float(np.mean(scores)) if scores else 0.0


# ----------------------------------------------------------------------------- fast grid evaluation
def evaluate_thresholds(s1_pos: np.ndarray, score: np.ndarray, label: np.ndarray,
                         n_true_by_s1: np.ndarray, n_s1_total: int, thr_grid: Sequence[float]
                         ) -> Dict[float, float]:
    """Macro F0.5 (over all `n_s1_total` Source-1 rows, matching the real scoring denominator - an S1 row
    with no candidate pairs at all still counts, and scores 1.0 iff it is a true singleton) for every
    threshold in `thr_grid`, without materialising per-entity id sets: `s1_pos` is the 0..n_s1_total-1
    position of every (s1, candidate) pair, `score`/`label` its predicted probability and ground-truth
    (0/1) label, and `n_true_by_s1[s1_pos]` the number of true matches that Source-1 row has."""
    covered = np.unique(s1_pos)
    uncovered_true0 = int(np.sum(n_true_by_s1[np.setdiff1d(np.arange(n_s1_total), covered)] == 0))
    base = float(uncovered_true0)
    ntrue_g = n_true_by_s1[covered]
    out = {}
    for tau in thr_grid:
        pred = score >= tau
        tp_flag = pred & (label == 1)
        tmp = pd.DataFrame({"s1": s1_pos, "npred": pred, "ntp": tp_flag})
        agg = tmp.groupby("s1", sort=True).sum()
        npred = agg["npred"].to_numpy(dtype=np.float64)
        ntp = agg["ntp"].to_numpy(dtype=np.float64)
        p = np.divide(ntp, npred, out=np.zeros_like(ntp), where=npred > 0)
        r = np.divide(ntp, ntrue_g, out=np.zeros_like(ntp), where=ntrue_g > 0)
        f = np.where(ntp > 0, (1.25 * p * r) / np.maximum(0.25 * p + r, 1e-12),
                     np.where((npred == 0) & (ntrue_g == 0), 1.0, 0.0))
        out[float(tau)] = float((base + f.sum()) / n_s1_total)
    return out


def best_threshold(s1_pos: np.ndarray, score: np.ndarray, label: np.ndarray, n_true_by_s1: np.ndarray,
                    n_s1_total: int, thr_grid: Sequence[float]):
    """Find optimal threshold with coarse-to-fine grid search for maximum F0.5.

    Key design choice for F0.5 (precision-heavy): when multiple thresholds yield the same F0.5 score,
    we pick the HIGHEST threshold (most conservative / most precise). This is critical because:
    - A low threshold that happens to score perfectly on training data will be overconfident on test data
    - False merges on singletons are penalised harshly (0.0 per entity vs 1.0)
    - The model's calibration shifts between train and test (different TF-IDF vocabularies)
    Also enforces a minimum threshold floor of 0.3 to prevent trivially low thresholds.
    """
    MIN_THRESHOLD = 0.3  # floor: never use a threshold below this for F0.5

    # Phase 1: coarse grid
    scores = evaluate_thresholds(s1_pos, score, label, n_true_by_s1, n_s1_total, thr_grid)
    best_f05 = max(scores.values())
    # Among all thresholds achieving best_f05, pick the HIGHEST (most conservative)
    tau_coarse = max(t for t, f in scores.items() if f == best_f05)

    # Phase 2: fine-grained search around the coarse optimum (±0.05 in 0.005 steps)
    fine_lo = max(0.05, tau_coarse - 0.05)
    fine_hi = min(0.99, tau_coarse + 0.05)
    fine_grid = tuple(round(fine_lo + 0.005 * i, 4) for i in range(int((fine_hi - fine_lo) / 0.005) + 1))
    fine_scores = evaluate_thresholds(s1_pos, score, label, n_true_by_s1, n_s1_total, fine_grid)
    scores.update(fine_scores)

    # Phase 3: ultra-fine search around the fine optimum (±0.005 in 0.001 steps)
    best_f05_so_far = max(scores.values())
    tau_fine = max(t for t, f in scores.items() if f == best_f05_so_far)
    ultra_lo = max(0.01, tau_fine - 0.005)
    ultra_hi = min(0.999, tau_fine + 0.005)
    ultra_grid = tuple(round(ultra_lo + 0.001 * i, 4) for i in range(int((ultra_hi - ultra_lo) / 0.001) + 1))
    ultra_scores = evaluate_thresholds(s1_pos, score, label, n_true_by_s1, n_s1_total, ultra_grid)
    scores.update(ultra_scores)

    # Final selection: highest F0.5, tie-broken by highest threshold, floored at MIN_THRESHOLD
    best_f05_final = max(scores.values())
    candidates = [(t, f) for t, f in scores.items() if f == best_f05_final and t >= MIN_THRESHOLD]
    if not candidates:
        # If no threshold >= MIN_THRESHOLD achieves the best, find the best among >= MIN_THRESHOLD
        above_floor = {t: f for t, f in scores.items() if t >= MIN_THRESHOLD}
        if above_floor:
            best_above = max(above_floor.values())
            candidates = [(t, f) for t, f in above_floor.items() if f == best_above]
        else:
            # Fallback to the overall best
            candidates = [(t, f) for t, f in scores.items() if f == best_f05_final]

    tau = max(t for t, _ in candidates)
    return tau, scores[tau], scores
