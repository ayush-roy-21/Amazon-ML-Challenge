"""Candidate generation ("blocking"). Narrows the O(n_s1 * (n_s2+n_s3)) matching problem down to a short,
high-recall candidate list per Source-1 record *before* the expensive pairwise-feature + model-scoring
stage runs - the classifier only ever sees these candidates, so recall lost here is recall the model can
never recover. Retrieval is done by chunked batch sparse dot-product with country penalty.

Optimized from the original per-row multiprocessing approach to use:
1. Batch sparse matrix multiplication (S1_chunk @ TGT.T) — 10-50x faster than per-row dot products.
2. ThreadPoolExecutor (scipy releases GIL) instead of multiprocessing (no serialization overhead).
3. np.argpartition for O(N) top-k selection instead of O(N log N) full sort.
4. Country penalty applied vectorized on the score matrix.
5. Aggressive memory management with float32 and gc.collect().
"""
from __future__ import annotations

import gc
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp

from .utils import log

BLOCK_VIEWS = ("nchar", "nword", "ncomb", "aword", "nphon")
Ranked = Dict[str, Tuple[np.ndarray, np.ndarray]]


def _safe_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x).strip().lower()
    if s in ("nan", "none", "null", "unknown"):
        return ""
    return s


def _get_n_workers(cfg):
    n_jobs = getattr(cfg, "n_jobs", -1)
    if n_jobs <= 0:
        try:
            n_jobs = len(os.sched_getaffinity(0))
        except AttributeError:
            n_jobs = os.cpu_count() or 4
    return n_jobs


def fetch_and_rank(feat, s1_rows: np.ndarray, tgt_rows: np.ndarray, ctry: Sequence[str], cfg,
                    depth_scale: float = None) -> Ranked:
    """Compute top-k candidates for each S1 row from target rows, for each TF-IDF view.

    Uses chunked batch sparse matrix multiplication with country penalty, threaded for parallelism
    (scipy sparse ops release the GIL).
    """
    scale = cfg.kmax_factor if depth_scale is None else depth_scale
    n_s1 = len(s1_rows)
    n_tgt = len(tgt_rows)
    chunk_size = 500  # SAFE MODE: reduced from 5000 to prevent OOM on 12.5M  # Tuned for memory: ~5k rows × n_tgt scores fits comfortably in RAM

    # Pre-process country arrays for vectorized penalty application
    ctry_arr = np.asarray(ctry, dtype=object)
    ctry_s1 = np.array([_safe_str(c) for c in ctry_arr[s1_rows]], dtype=object)
    ctry_tgt = np.array([_safe_str(c) for c in ctry_arr[tgt_rows]], dtype=object)
    tgt_has_ctry = np.array([c != "" for c in ctry_tgt], dtype=bool)
    penalty = getattr(cfg, "country_penalty", 1.0)
    use_penalty = penalty != 1.0

    n_workers = _get_n_workers(cfg)
    out: Ranked = {}

    for v in BLOCK_VIEWS:
        if v not in feat.views:
            continue
        mat = feat.views[v]
        depth = max(1, int(round(cfg.base_k[v] * scale)))
        k = min(depth, n_tgt)

        # Pre-compute transposed target matrix (vocab × n_tgt) → CSC for fast column slicing
        tgt_mat_T = mat[tgt_rows].T.tocsc()

        all_idx = np.full((n_s1, k), -1, dtype=np.int64)
        all_sim = np.zeros((n_s1, k), dtype=np.float32)

        def process_chunk(start):
            end = min(start + chunk_size, n_s1)
            s1_chunk = s1_rows[start:end]
            s1_mat = mat[s1_chunk]

            # (chunk_size × vocab) @ (vocab × n_tgt) → (chunk_size × n_tgt) dense scores
            scores = s1_mat.dot(tgt_mat_T).toarray().astype(np.float32)

            # Apply country penalty: when both S1 and TGT have known country and they differ
            if use_penalty:
                for i, c1 in enumerate(ctry_s1[start:end]):
                    if c1:
                        mismatch = tgt_has_ctry & (ctry_tgt != c1)
                        scores[i, mismatch] *= penalty

            # Top-k selection via argpartition (O(N) per row) then sort the k items
            local_k = min(k, scores.shape[1])
            if local_k < scores.shape[1]:
                top_idx = np.argpartition(scores, -local_k, axis=1)[:, -local_k:]
                rows = np.arange(scores.shape[0])[:, None]
                top_scores = scores[rows, top_idx]
                sort_order = np.argsort(-top_scores, axis=1)
                sorted_idx = top_idx[rows, sort_order]
                sorted_scores = top_scores[rows, sort_order]
            else:
                sort_order = np.argsort(-scores, axis=1)
                sorted_idx = sort_order
                sorted_scores = np.take_along_axis(scores, sort_order, axis=1)

            # Convert relative target indices to absolute row-in-R indices
            actual_k = min(local_k, sorted_idx.shape[1])
            abs_idx = tgt_rows[sorted_idx[:, :actual_k]]
            return start, end, abs_idx, sorted_scores[:, :actual_k]

        # ThreadPoolExecutor is ideal here: scipy sparse dot releases the GIL
        chunks = list(range(0, n_s1, chunk_size))
        # EMERGENCY FIX: Restrict to 3 workers to prevent 8-Terabyte RAM explosion
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(process_chunk, chunks))

        for start, end, abs_idx, scores in results:
            n_chunk = end - start
            actual_k = abs_idx.shape[1]
            all_idx[start:end, :actual_k] = abs_idx
            all_sim[start:end, :actual_k] = scores

        out[v] = (all_idx, all_sim)
        log(f"  blocking[{v}]: fetched depth={k} for {n_s1} S1 rows, country_penalty={'active' if use_penalty else 'off'}")

        del tgt_mat_T, results
        gc.collect()

    return out


def union_candidates(ranked: Ranked, cfg, k_scale: float) -> List[Set[int]]:
    """Union the top-k candidates across all views for each S1 row."""
    n = next(iter(ranked.values()))[0].shape[0]
    out: List[Set[int]] = [set() for _ in range(n)]
    for v, (idx, _) in ranked.items():
        k = min(max(1, int(round(cfg.base_k[v] * k_scale))), idx.shape[1])
        if k <= 0:
            continue
        sl = idx[:, :k]
        for r in range(n):
            out[r].update(int(x) for x in sl[r] if x >= 0)
    return out


def blocking_recall(cand_sets: List[Set[int]], s1_rows: np.ndarray, n_records: int,
                     true_keys: np.ndarray) -> Tuple[float, int, int]:
    """Compute recall of the blocking step: what fraction of true pairs appear in the candidate sets."""
    pos = {int(r): i for i, r in enumerate(s1_rows)}
    total = hit = 0
    for key in true_keys.tolist():
        a, b = divmod(int(key), n_records)
        i = pos.get(a)
        if i is None:
            continue
        total += 1
        if b in cand_sets[i]:
            hit += 1
    return (hit / total if total else 1.0), hit, total


def tune_k_scale(feat, s1_rows: np.ndarray, tgt_rows: np.ndarray, ctry: Sequence[str], cfg,
                  true_keys: np.ndarray, n_records: int):
    """Find the smallest k_scale that achieves target recall, reusing a single fetch_and_rank call."""
    ranked = fetch_and_rank(feat, s1_rows, tgt_rows, ctry, cfg)
    chosen, chosen_sets, chosen_rec = None, None, None
    for ks in cfg.k_scales:
        sets = union_candidates(ranked, cfg, ks)
        rec, hit, total = blocking_recall(sets, s1_rows, n_records, true_keys)
        avg = float(np.mean([len(s) for s in sets])) if sets else 0.0
        log(f"k_scale={ks}: blocking recall={rec:.4f} ({hit}/{total} true pairs), avg candidates/S1={avg:.1f}")
        if rec >= cfg.target_recall:
            chosen, chosen_sets, chosen_rec = ks, sets, rec
            break
    if chosen is None:
        chosen = cfg.k_scales[-1]
        chosen_sets = union_candidates(ranked, cfg, chosen)
        chosen_rec, _, _ = blocking_recall(chosen_sets, s1_rows, n_records, true_keys)
        log(f"target recall {cfg.target_recall} not reached at max k_scale={chosen}; using it anyway "
            f"(recall={chosen_rec:.4f})")
    return chosen, chosen_sets, chosen_rec, ranked


def candidates_for(ranked: Ranked, cfg, k_scale: float) -> List[np.ndarray]:
    """Union candidates and return as sorted numpy arrays."""
    sets = union_candidates(ranked, cfg, k_scale)
    return [np.asarray(sorted(s), dtype=np.int64) for s in sets]
