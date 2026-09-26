"""Candidate generation ("blocking"). Narrows the O(n_s1 * (n_s2+n_s3)) matching problem down to a short,
high-recall candidate list per Source-1 record *before* the expensive pairwise-feature + model-scoring
stage runs - the classifier only ever sees these candidates, so recall lost here is recall the model can
never recover. Retrieval is done by bucketed pre-filtering followed by localized TF-IDF cosine ranking.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple
from collections import defaultdict
import numpy as np

from .utils import log

BLOCK_VIEWS = ("nchar", "nword", "ncomb", "aword", "nphon")
Ranked = Dict[str, Tuple[np.ndarray, np.ndarray]]

def _prefixes(p, z):
    return (str(p)[:4] if p else ""), (str(z)[:3] if z else "")

def fetch_and_rank(feat, s1_rows: np.ndarray, tgt_rows: np.ndarray, ctry: Sequence[str], cfg,
                    depth_scale: float = None) -> Ranked:
    """Deterministic bucket pre-filtering + localized TF-IDF dot products."""
    scale = cfg.kmax_factor if depth_scale is None else depth_scale
    ctry_arr = np.asarray(ctry, dtype=object)
    
    t_ctry = ctry_arr[tgt_rows]
    
    # Fallback in case feat.arr doesn't have it (though we just patched it)
    if "name_phon" in getattr(feat, "arr", {}):
        t_phon = feat.arr["name_phon"][tgt_rows]
        s_phon = feat.arr["name_phon"][s1_rows]
    else:
        name_phon_arr = feat.R["name_phon"].to_numpy(dtype=object)
        t_phon = name_phon_arr[tgt_rows]
        s_phon = name_phon_arr[s1_rows]

    t_post = np.asarray(feat.postal, dtype=object)[tgt_rows]
    
    buckets_c1, buckets_c2 = defaultdict(list), defaultdict(list)
    buckets_m1, buckets_m2 = defaultdict(list), defaultdict(list)
    buckets_a1, buckets_a2 = defaultdict(list), defaultdict(list)
    
    log(f"  blocking: building inverted indexes for {len(tgt_rows)} targets...")
    for i, (c, p, z) in enumerate(zip(t_ctry, t_phon, t_post)):
        pp, zp = _prefixes(p, z)
        if pp:
            buckets_a1[pp].append(i)
            (buckets_c1[(c, pp)] if c else buckets_m1[pp]).append(i)
        if zp:
            buckets_a2[zp].append(i)
            (buckets_c2[(c, zp)] if c else buckets_m2[zp]).append(i)
            
    s_ctry = ctry_arr[s1_rows]
    s_post = np.asarray(feat.postal, dtype=object)[s1_rows]
    
    cands_per_s1 = []
    log(f"  blocking: fetching candidate pools for {len(s1_rows)} queries...")
    for c, p, z in zip(s_ctry, s_phon, s_post):
        pp, zp = _prefixes(p, z)
        cands = set()
        if pp:
            if c:
                cands.update(buckets_c1.get((c, pp), []))
                cands.update(buckets_m1.get(pp, []))
            else:
                cands.update(buckets_a1.get(pp, []))
        if zp:
            if c:
                cands.update(buckets_c2.get((c, zp), []))
                cands.update(buckets_m2.get(zp, []))
            else:
                cands.update(buckets_a2.get(zp, []))
        cands_per_s1.append(np.array(sorted(cands), dtype=np.int32))
        
    out: Ranked = {}
    max_depth = max(max(1, int(round(cfg.base_k[vv] * scale))) for vv in BLOCK_VIEWS)
    
    for v in BLOCK_VIEWS:
        depth = max(1, int(round(cfg.base_k[v] * scale)))
        all_real = np.full((len(s1_rows), depth), -1, dtype=np.int64)
        all_sim = np.zeros((len(s1_rows), depth), dtype=np.float32)
        out[v] = (all_real, all_sim)
        
    s1_mats = {v: feat.views[v][s1_rows] for v in BLOCK_VIEWS}
    tgt_mats = {v: feat.views[v][tgt_rows] for v in BLOCK_VIEWS}

    for i, cands in enumerate(cands_per_s1):
        if len(cands) == 0:
            continue
            
        if len(cands) <= max_depth:
            for v in BLOCK_VIEWS:
                k = len(cands)
                out[v][0][i, :k] = tgt_rows[cands]
                out[v][1][i, :k] = 1.0
            continue
            
        for v in BLOCK_VIEWS:
            sim = s1_mats[v][i].dot(tgt_mats[v][cands].T).toarray().flatten()
            depth = max(1, int(round(cfg.base_k[v] * scale)))
            k = min(depth, len(sim))
            order = np.argsort(-sim, kind="stable")[:k]
            
            out[v][0][i, :k] = tgt_rows[cands[order]]
            out[v][1][i, :k] = sim[order]
            
    for v in BLOCK_VIEWS:
        depth = max(1, int(round(cfg.base_k[v] * scale)))
        log(f"  blocking[{v}]: localized dot-product fetched depth={depth} for {len(s1_rows)} rows")
        
    return out

def union_candidates(ranked: Ranked, cfg, k_scale: float) -> List[Set[int]]:
    n = next(iter(ranked.values()))[0].shape[0]
    out: List[Set[int]] = [set() for _ in range(n)]
    for v, (idx, _) in ranked.items():
        k = min(max(1, int(round(cfg.base_k[v] * k_scale))), idx.shape[1])
        if k <= 0: continue
        sl = idx[:, :k]
        for r in range(n):
            out[r].update(int(x) for x in sl[r] if x >= 0)
    return out

def blocking_recall(cand_sets: List[Set[int]], s1_rows: np.ndarray, n_records: int,
                     true_keys: np.ndarray) -> Tuple[float, int, int]:
    pos = {int(r): i for i, r in enumerate(s1_rows)}
    total = hit = 0
    for key in true_keys.tolist():
        a, b = divmod(int(key), n_records)
        i = pos.get(a)
        if i is None: continue
        total += 1
        if b in cand_sets[i]: hit += 1
    return (hit / total if total else 1.0), hit, total

def tune_k_scale(feat, s1_rows: np.ndarray, tgt_rows: np.ndarray, ctry: Sequence[str], cfg,
                  true_keys: np.ndarray, n_records: int):
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
    sets = union_candidates(ranked, cfg, k_scale)
    return [np.asarray(sorted(s), dtype=np.int64) for s in sets]
