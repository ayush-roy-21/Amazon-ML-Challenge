"""Candidate generation ("blocking"). Narrows the O(n_s1 * (n_s2+n_s3)) matching problem down to a short,
high-recall candidate list per Source-1 record *before* the expensive pairwise-feature + model-scoring
stage runs - the classifier only ever sees these candidates, so recall lost here is recall the model can
never recover. Retrieval is plain cosine top-k on the TF-IDF views already built by `Featurizer`; a
same-vs-different country signal only *penalises* (never hard-filters) a candidate's score, since the
country field can be missing/noisy and a chain can legitimately span countries.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .utils import log

# Subset of Featurizer.views used for retrieval. "achar" (address char n-grams) is left out: "aword" and
# the combined "ncomb" view already give address-driven recall, and dropping a sixth expensive sparse
# nearest-neighbour search keeps blocking cheap without a measurable recall cost.
BLOCK_VIEWS = ("nchar", "nword", "ncomb", "aword", "nphon")

Ranked = Dict[str, Tuple[np.ndarray, np.ndarray]]  # view -> (sorted_row_in_R, sorted_penalised_sim)


def _fetch(src_mat, tgt_mat, k: int, n_jobs: int):
    n_src, n_tgt = src_mat.shape[0], tgt_mat.shape[0]
    k = min(k, n_tgt)
    if n_src == 0 or k <= 0:
        return np.full((n_src, 0), -1, dtype=np.int64), np.zeros((n_src, 0), dtype=np.float32)
    # algorithm="brute" + metric="cosine" on sparse input: sklearn chunks the pairwise-distance computation
    # internally (pairwise_distances_chunked), so this never materialises the full n_src x n_tgt matrix.
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=n_jobs)
    nn.fit(tgt_mat)
    dist, idx = nn.kneighbors(src_mat)
    sim = np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)
    return idx.astype(np.int64), sim


def fetch_and_rank(feat, s1_rows: np.ndarray, tgt_rows: np.ndarray, ctry: Sequence[str], cfg,
                    depth_scale: float = None) -> Ranked:
    """Fetch top ``base_k[v] * depth_scale`` (default: ``cfg.kmax_factor``, the largest scale we will ever
    need) neighbours per view, apply the country penalty, and re-sort by the penalised score. This is the
    only step that touches the TF-IDF matrices; every k_scale considered afterwards just slices these
    already-sorted arrays, so tuning k_scale over `cfg.k_scales` costs no extra nearest-neighbour search."""
    scale = cfg.kmax_factor if depth_scale is None else depth_scale
    ctry = np.asarray(ctry, dtype=object)
    c_s1 = ctry[s1_rows]
    out: Ranked = {}
    for v in BLOCK_VIEWS:
        depth = max(1, int(round(cfg.base_k[v] * scale)))
        idx, sim = _fetch(feat.views[v][s1_rows], feat.views[v][tgt_rows], depth, cfg.n_jobs)
        got = idx.shape[1]
        real = np.where(idx >= 0, tgt_rows[np.clip(idx, 0, None)], -1)
        if got < depth:
            real = np.pad(real, ((0, 0), (0, depth - got)), constant_values=-1)
            sim = np.pad(sim, ((0, 0), (0, depth - got)), constant_values=0.0)
        valid = real >= 0
        c_cand = np.where(valid, ctry[np.clip(real, 0, None)], "")
        mism = valid & (c_s1[:, None] != "") & (c_cand != "") & (c_s1[:, None] != c_cand)
        pen_sim = np.where(mism, sim * cfg.country_penalty, sim)
        pen_sim = np.where(valid, pen_sim, -1.0)  # padding sorts last
        order = np.argsort(-pen_sim, axis=1, kind="stable")
        out[v] = (np.take_along_axis(real, order, axis=1), np.take_along_axis(pen_sim, order, axis=1))
        log(f"  blocking[{v}]: fetched depth={depth} for {len(s1_rows)} rows")
    return out


def union_candidates(ranked: Ranked, cfg, k_scale: float) -> List[Set[int]]:
    """Union, per S1 row, the top ``base_k[v] * k_scale`` candidates (row-in-R ids) across all views."""
    n = next(iter(ranked.values()))[0].shape[0]
    out: List[Set[int]] = [set() for _ in range(n)]
    for v, (idx, _sim) in ranked.items():
        k = min(max(1, int(round(cfg.base_k[v] * k_scale))), idx.shape[1])
        if k <= 0:
            continue
        sl = idx[:, :k]
        for r in range(n):
            out[r].update(int(x) for x in sl[r] if x >= 0)
    return out


def blocking_recall(cand_sets: List[Set[int]], s1_rows: np.ndarray, n_records: int,
                     true_keys: np.ndarray) -> Tuple[float, int, int]:
    """Fraction of true (s1, candidate) pairs whose candidate row is present in that S1's candidate set.
    Only true pairs whose S1 row is one of `s1_rows` are considered (returns 1.0 / 0-0 if there are none)."""
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
    """Fetch once at full depth, then walk `cfg.k_scales` in increasing order and stop at the first that
    reaches `cfg.target_recall` on the (labelled) training data - the smallest, cheapest candidate set that
    still gives the classifier a fair shot at every true match."""
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
    """Same as `union_candidates`, materialised as sorted int64 arrays (for feature building / output)."""
    sets = union_candidates(ranked, cfg, k_scale)
    return [np.asarray(sorted(s), dtype=np.int64) for s in sets]
