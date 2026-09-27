"""Candidate generation ("blocking"). Narrows the O(n_s1 * (n_s2+n_s3)) matching problem down to a short,
high-recall candidate list per Source-1 record *before* the expensive pairwise-feature + model-scoring
stage runs - the classifier only ever sees these candidates, so recall lost here is recall the model can
never recover. Retrieval is done by bucketed pre-filtering followed by localized TF-IDF cosine ranking.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import math

from .utils import log

BLOCK_VIEWS = ("nchar", "nword", "ncomb", "aword", "nphon")
Ranked = Dict[str, Tuple[np.ndarray, np.ndarray]]

_GLOBAL_BUCKETS = {}
_GLOBAL_DOT_VARS = {}

def _safe_str(x):
    if x is None: return ""
    if isinstance(x, float) and math.isnan(x): return ""
    s = str(x).strip().lower()
    if s in ("nan", "none", "null"): return ""
    return s

def _prefixes(p, z):
    return _safe_str(p)[:4], _safe_str(z)[:3]

def _cands_for_row(args):
    c, p, z = args
    c_safe = _safe_str(c)
    pp, zp = _prefixes(p, z)
    cands = set()
    b = _GLOBAL_BUCKETS
    if pp:
        if c_safe:
            cands.update(b['c1'].get((c_safe, pp), []))
            cands.update(b['m1'].get(pp, []))
        else:
            cands.update(b['a1'].get(pp, []))
    if zp:
        if c_safe:
            cands.update(b['c2'].get((c_safe, zp), []))
            cands.update(b['m2'].get(zp, []))
        else:
            cands.update(b['a2'].get(zp, []))
    return np.array(sorted(cands), dtype=np.int32)

def _process_dot_chunk(args):
    start, end = args
    chunk_size = end - start
    chunk_out = {}
    cfg_base_k = _GLOBAL_DOT_VARS['base_k']
    scale = _GLOBAL_DOT_VARS['scale']
    cands_arr = _GLOBAL_DOT_VARS['cands_per_s1']
    s1_mats = _GLOBAL_DOT_VARS['s1_mats']
    tgt_mats = _GLOBAL_DOT_VARS['tgt_mats']
    tgt_rows = _GLOBAL_DOT_VARS['tgt_rows']
    
    for v in BLOCK_VIEWS:
        depth = max(1, int(round(cfg_base_k[v] * scale)))
        chunk_out[v] = (np.full((chunk_size, depth), -1, dtype=np.int64),
                        np.zeros((chunk_size, depth), dtype=np.float32))
                        
    for i_local, i_global in enumerate(range(start, end)):
        cands = cands_arr[i_global]
        if len(cands) == 0:
            continue
        for v in BLOCK_VIEWS:
            depth = max(1, int(round(cfg_base_k[v] * scale)))
            if len(cands) <= depth:
                k = len(cands)
                chunk_out[v][0][i_local, :k] = tgt_rows[cands]
                chunk_out[v][1][i_local, :k] = 1.0
            else:
                sim = s1_mats[v][i_global].dot(tgt_mats[v][cands].T).toarray().flatten()
                k = min(depth, len(sim))
                order = np.argsort(-sim, kind="stable")[:k]
                chunk_out[v][0][i_local, :k] = tgt_rows[cands[order]]
                chunk_out[v][1][i_local, :k] = sim[order]
    return chunk_out

def fetch_and_rank(feat, s1_rows: np.ndarray, tgt_rows: np.ndarray, ctry: Sequence[str], cfg,
                    depth_scale: float = None) -> Ranked:
    scale = cfg.kmax_factor if depth_scale is None else depth_scale
    ctry_arr = np.asarray(ctry, dtype=object)
    
    t_ctry = ctry_arr[tgt_rows]
    
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
        c_safe = _safe_str(c)
        pp, zp = _prefixes(p, z)
        if pp:
            buckets_a1[pp].append(i)
            (buckets_c1[(c_safe, pp)] if c_safe else buckets_m1[pp]).append(i)
        if zp:
            buckets_a2[zp].append(i)
            (buckets_c2[(c_safe, zp)] if c_safe else buckets_m2[zp]).append(i)
            
    s_ctry = ctry_arr[s1_rows]
    s_post = np.asarray(feat.postal, dtype=object)[s1_rows]
    
    global _GLOBAL_BUCKETS
    _GLOBAL_BUCKETS = {
        'c1': buckets_c1, 'c2': buckets_c2,
        'm1': buckets_m1, 'm2': buckets_m2,
        'a1': buckets_a1, 'a2': buckets_a2
    }

    n_jobs = getattr(cfg, 'n_jobs', -1)
    if n_jobs <= 0:
        import os
        try:
            n_jobs = len(os.sched_getaffinity(0))
        except AttributeError:
            n_jobs = mp.cpu_count()
        
    log(f"  blocking: fetching candidate pools for {len(s1_rows)} queries using multiprocessing...")
    with mp.get_context("fork").Pool(processes=n_jobs) as pool:
        cands_per_s1 = pool.map(_cands_for_row, zip(s_ctry, s_phon, s_post), chunksize=2000)
        
    out: Ranked = {}
    for v in BLOCK_VIEWS:
        depth = max(1, int(round(cfg.base_k[v] * scale)))
        all_real = np.full((len(s1_rows), depth), -1, dtype=np.int64)
        all_sim = np.zeros((len(s1_rows), depth), dtype=np.float32)
        out[v] = (all_real, all_sim)
        
    s1_mats = {v: feat.views[v][s1_rows] for v in BLOCK_VIEWS}
    tgt_mats = {v: feat.views[v][tgt_rows] for v in BLOCK_VIEWS}

    global _GLOBAL_DOT_VARS
    _GLOBAL_DOT_VARS = {
        'base_k': cfg.base_k,
        'scale': scale,
        'cands_per_s1': cands_per_s1,
        's1_mats': s1_mats,
        'tgt_mats': tgt_mats,
        'tgt_rows': tgt_rows
    }
    
    chunk_size = max(1, len(s1_rows) // (n_jobs * 4))
    
    log(f"  blocking: parallelizing localized dot-products across {n_jobs} processes (multiprocessing)...")
    chunks = [(s, min(s + chunk_size, len(s1_rows))) for s in range(0, len(s1_rows), chunk_size)]
    
    with mp.get_context("fork").Pool(processes=n_jobs) as pool:
        results = pool.map(_process_dot_chunk, chunks)
        
    for chunk_idx, (start, end) in enumerate(chunks):
        chunk_out = results[chunk_idx]
        for v in BLOCK_VIEWS:
            out[v][0][start:end] = chunk_out[v][0]
            out[v][1][start:end] = chunk_out[v][1]
            
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
