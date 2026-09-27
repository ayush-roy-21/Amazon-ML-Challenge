"""TF-IDF views and pairwise feature engineering.

Every feature is a similarity between a Source-1 record (``a`` side) and a Source-2/3 candidate (``b`` side).
Nothing here depends on the identity of a country: country labels are only compared for (dis)agreement.
"""
from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
import concurrent.futures

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

from .textnorm import phon_key
from .utils import log

_HAS_CPDIST = hasattr(process, "cpdist")
NAN = float("nan")


# ----------------------------------------------------------------------------- helpers
def pair_scores(A, B, scorer, workers=-1):
    """Score aligned string pairs (A[k], B[k]) -> float32 array."""
    if len(A) == 0:
        return np.zeros(0, dtype=np.float32)
    if _HAS_CPDIST:
        return np.asarray(process.cpdist(A, B, scorer=scorer, dtype=np.float32, workers=workers), dtype=np.float32)
    return np.fromiter((scorer(a, b) for a, b in zip(A, B)), dtype=np.float32, count=len(A))


def _max_df(n, cap_frac, abs_cap):
    return float(max(min(cap_frac, abs_cap / max(n, 1)), min(1.0, 3.0 / max(n, 1))))


def _tfidf(texts, analyzer, ngram, max_df, binary=False, use_idf=True, norm="l2"):
    if analyzer == "char_wb":
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, lowercase=False, sublinear_tf=(not binary),
                            max_df=max_df, min_df=2, dtype=np.float32, binary=binary, use_idf=use_idf, norm=norm)
    else:
        v = TfidfVectorizer(analyzer="word", tokenizer=str.split, token_pattern=None, lowercase=False,
                            ngram_range=ngram, sublinear_tf=(not binary), max_df=max_df, min_df=2, dtype=np.float32, binary=binary, use_idf=use_idf, norm=norm)
    try:
        return v.fit_transform(texts).tocsr()
    except ValueError:  # empty vocabulary
        return sp.csr_matrix((len(texts), 1), dtype=np.float32)


def _idf_map(token_iter):
    df, n = Counter(), 0
    for toks in token_iter:
        n += 1
        df.update(set(toks))
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}, math.log(n + 1) + 1.0


def _is_subseq(a, b):
    it = iter(b)
    return all(c in it for c in a)


@lru_cache(maxsize=1 << 21)
def _tok_sim(a: str, b: str) -> float:
    """Soft similarity of two name tokens (typos, prefixes/abbreviations, phonetic variants)."""
    if a.isdigit() or b.isdigit():
        return 0.0
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    best = 0.0
    if len(s) >= 3:
        if l.startswith(s):
            best = 0.85
        elif s[0] == l[0] and len(s) / len(l) >= 0.3 and _is_subseq(s, l):
            best = 0.75
    pa, pb = phon_key(a), phon_key(b)
    if pa == pb and len(pa) >= 3:
        best = max(best, 0.85)
    jw = JaroWinkler.similarity(a, b)
    if jw > 0.75:
        best = max(best, (jw - 0.75) / 0.25)
    return best


def tok_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return _tok_sim(a, b) if a < b else _tok_sim(b, a)


NT_FEATS = ["nt_jacc", "nt_ovmin", "nt_idfov_a", "nt_idfov_b", "nt_soft_a", "nt_soft_b", "nt_soft_f1",
            "nt_unm_a", "nt_unm_b", "nt_first_eq", "nt_last_eq", "nt_first_jw", "nt_acro", "nt_num",
            "nt_phon_jacc", "nt_ntok_a", "nt_ntok_b"]
AD_FEATS = ["pc_state", "pc_p3", "pc_p2", "pc_logdiff", "hn_state", "nums_jacc", "nums_common",
            "at_jacc", "at_ovmin", "at_idfov_a", "at_idfov_b", "at_idfov_common"]
_NAN_NT = (NAN,) * len(NT_FEATS)
_NAN_AD = (0.0, NAN, NAN, NAN, 0.0, NAN, NAN, NAN, NAN, NAN, NAN, NAN)

CTX_COLS = ["cos_nchar", "nt_soft_f1", "cos_ncomb", "n_tset"]


# ----------------------------------------------------------------------------- featurizer
class Featurizer:
    def __init__(self, R: pd.DataFrame, cfg):
        self.cfg = cfg
        self.R = R
        n = len(R)
        self.N = n
        L = lambda c: R[c].tolist()  # noqa: E731
        self.name_core, self.name_norm, self.compact = L("name_core"), L("name_norm"), L("name_compact")
        self.addr_all, self.addr_main, self.addr_lm = L("addr_all"), L("addr_main"), L("addr_lm")
        self.addr_head, self.addr_tail = L("addr_head"), L("addr_tail")
        self.postal, self.house, self.ctry = L("postal"), L("house"), L("ctry")
        self.legal, self.acro = L("legal"), L("acro")
        self.name_phon_str = L("name_phon")
        self.toks = L("name_toks")
        self.tsets = [frozenset(t) for t in self.toks]
        self.psets = [frozenset(p.split()) for p in L("name_phon")]
        self.nums_name = [frozenset(x for x in t if any(c.isdigit() for c in x)) for t in self.toks]
        self.atoks = [tuple(a.split()) for a in self.addr_all]
        self.aset = [frozenset(t) for t in self.atoks]
        self.nums_addr = [frozenset(x.split()) for x in L("nums")]
        self.src = R["src"].to_numpy()
        self.idf_name, self.idf_name_d = _idf_map(self.toks)
        self.idf_addr, self.idf_addr_d = _idf_map(self.atoks)

        self.t_wa = [np.array([self.idf_name.get(t, self.idf_name_d) for t in ta], dtype=np.float32) if ta else np.zeros(0, dtype=np.float32) for ta in self.toks]
        self.t_swa = np.array([w.sum() for w in self.t_wa], dtype=np.float32)

        self.a_wa = [np.array([self.idf_addr.get(t, self.idf_addr_d) for t in a], dtype=np.float32) if a else np.zeros(0, dtype=np.float32) for a in self.atoks]
        self.a_swa = np.array([w.sum() for w in self.a_wa], dtype=np.float32)

        fields = {"name_core": self.name_core, "name_norm": self.name_norm, "compact": self.compact,
                  "addr_all": self.addr_all, "addr_main": self.addr_main, "addr_lm": self.addr_lm,
                  "addr_head": self.addr_head, "addr_tail": self.addr_tail, "ctry": self.ctry, "name_phon": self.name_phon_str}
        self.arr = {k: np.array(v, dtype=object) for k, v in fields.items()}
        self.emp = {k: np.array([len(x) == 0 for x in v], dtype=bool) for k, v in fields.items()}

        md_name = _max_df(n, 0.2, cfg.max_df_abs)
        md_addr = _max_df(n, 0.1, cfg.max_df_abs)
        from joblib import Parallel, delayed
        tasks = [
            ("nchar", R["name_core"].tolist(), "char_wb", (3, 4), md_name, False, True, "l2"),
            ("nword", R["name_toks_str"].tolist(), "word", (1, 2), md_name, False, True, "l2"),
            ("nphon", R["name_phon"].tolist(), "word", (1, 1), md_name, False, True, "l2"),
            ("achar", R["addr_all"].tolist(), "char_wb", (3, 4), md_addr, False, True, "l2"),
            ("aword", R["addr_all"].tolist(), "word", (1, 2), md_addr, False, True, "l2"),
            ("nchar_bin", R["name_core"].tolist(), "char_wb", (3, 4), md_name, True, False, None),
        ]
        # require='sharedmem' forces Threads instead of Processes, bypassing the AWS /dev/shm freeze entirely!
        results = Parallel(n_jobs=6, require='sharedmem')(
            delayed(_tfidf)(texts, analyzer, ngram, max_df, binary, use_idf, norm)
            for name, texts, analyzer, ngram, max_df, binary, use_idf, norm in tasks
        )
        V = {tasks[i][0]: results[i] for i in range(len(tasks))}
        w = math.sqrt(0.5)
        V["ncomb"] = sp.hstack([V["nchar"] * w, V["achar"] * w], format="csr", dtype=np.float32)
        self.views = V
        self._nnz = {k: np.diff(X.indptr) for k, X in V.items()}
        log(f"featurizer ready: {n} records, views={ {k: X.shape[1] for k, X in V.items()} }")

    # ------------------------------------------------------------------ small vectorised helpers
    def _cos(self, view, I, J, chunk=100_000):
        X, nnz = self.views[view], self._nnz[view]
        out = np.zeros(len(I), dtype=np.float32)
        for s in range(0, len(I), chunk):
            a, b = X[I[s:s + chunk]], X[J[s:s + chunk]]
            out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
        out[(nnz[I] == 0) | (nnz[J] == 0)] = np.nan
        return out

    def _jaccard_sparse(self, view, I, J, chunk=100_000):
        X, nnz = self.views[view], self._nnz[view]
        out = np.zeros(len(I), dtype=np.float32)
        for s in range(0, len(I), chunk):
            a, b = X[I[s:s + chunk]], X[J[s:s + chunk]]
            inter = np.asarray(a.multiply(b).sum(axis=1)).ravel()
            den = nnz[I[s:s + chunk]] + nnz[J[s:s + chunk]] - inter
            out[s:s + chunk] = np.where(den > 0, inter / den, 0.0)
        out[(nnz[I] == 0) | (nnz[J] == 0)] = np.nan
        return out

    def _sim(self, field, scorer, I, J, scale=100.0):
        a, e = self.arr[field], self.emp[field]
        out = pair_scores(a[I].tolist(), a[J].tolist(), scorer) / np.float32(scale)
        out = out.astype(np.float32)
        out[e[I] | e[J]] = np.nan
        return out

    @staticmethod
    def _soft_match_single(args):
        ta, tb, wa, wb, swa, swb = args
        trip = []
        for x, tx in enumerate(ta):
            for y, ty in enumerate(tb):
                s = tok_sim(tx, ty)
                if s > 0.0:
                    trip.append((-s, x, y))
        trip.sort()
        ua, ub = set(), set()
        ma, mb = [0.0] * len(ta), [0.0] * len(tb)
        for ns, x, y in trip:
            if x in ua or y in ub:
                continue
            ua.add(x)
            ub.add(y)
            ma[x] = mb[y] = -ns
        soft_a = sum(w * s for w, s in zip(wa, ma)) / swa if swa > 0 else 0.0
        soft_b = sum(w * s for w, s in zip(wb, mb)) / swb if swb > 0 else 0.0
        f1 = 2 * soft_a * soft_b / (soft_a + soft_b) if soft_a + soft_b > 0 else 0.0
        unm_a = sum(w * (1.0 - s) for w, s in zip(wa, ma))
        unm_b = sum(w * (1.0 - s) for w, s in zip(wb, mb))
        
        return soft_a, soft_b, f1, unm_a, unm_b

    def _vectorize_nt(self, I, J, workers=-1):
        if workers < 0:
            workers = getattr(self.cfg, "n_jobs", -1)
            if workers < 0:
                import os
                workers = os.cpu_count() or 4

        n = len(I)
        res = {k: np.full(n, NAN, dtype=np.float32) for k in NT_FEATS}

        ta = np.array(self.toks, dtype=object)[I]
        tb = np.array(self.toks, dtype=object)[J]
        valid = np.array([bool(x) and bool(y) for x, y in zip(ta, tb)], dtype=bool)
        idx = np.where(valid)[0]

        if len(idx) == 0:
            return res

        v_ta, v_tb = ta[idx], tb[idx]
        v_I, v_J = I[idx], J[idx]
        
        sa = np.array(self.tsets, dtype=object)[v_I]
        sb = np.array(self.tsets, dtype=object)[v_J]
        
        len_a = np.array([len(x) for x in sa], dtype=np.float32)
        len_b = np.array([len(x) for x in sb], dtype=np.float32)
        ni = np.array([len(a & b) for a, b in zip(sa, sb)], dtype=np.float32)
        
        res["nt_jacc"][idx] = ni / (len_a + len_b - ni)
        res["nt_ovmin"][idx] = ni / np.minimum(len_a, len_b)

        def get_idf_ov_a(k, a_toks, b_set):
            w = self.t_wa[k]
            return sum(w[i] for i, t in enumerate(a_toks) if t in b_set)
        
        ioa = np.array([get_idf_ov_a(v_I[m], v_ta[m], sb[m]) for m in range(len(idx))], dtype=np.float32)
        iob = np.array([get_idf_ov_a(v_J[m], v_tb[m], sa[m]) for m in range(len(idx))], dtype=np.float32)
        
        res["nt_idfov_a"][idx] = ioa / self.t_swa[v_I]
        res["nt_idfov_b"][idx] = iob / self.t_swa[v_J]
        
        res["nt_first_eq"][idx] = np.array([a[0] == b[0] for a, b in zip(v_ta, v_tb)], dtype=np.float32)
        res["nt_last_eq"][idx] = np.array([a[-1] == b[-1] for a, b in zip(v_ta, v_tb)], dtype=np.float32)
        
        ia = np.array(self.acro, dtype=object)[v_I]
        ib = np.array(self.acro, dtype=object)[v_J]
        ca = np.array(self.compact, dtype=object)[v_I]
        cb = np.array(self.compact, dtype=object)[v_J]
        
        acro_arr = [
            1.0 if ((len(ia[m]) >= 2 and ia[m] == cb[m]) or (len(ib[m]) >= 2 and ib[m] == ca[m])) else 0.0
            for m in range(len(idx))
        ]
        res["nt_acro"][idx] = acro_arr
        
        na = np.array(self.nums_name, dtype=object)[v_I]
        nb = np.array(self.nums_name, dtype=object)[v_J]
        num_arr = [
            (1.0 if a == b else 0.0) if (a and b) else NAN
            for a, b in zip(na, nb)
        ]
        res["nt_num"][idx] = num_arr
        
        pa = np.array(self.psets, dtype=object)[v_I]
        pb = np.array(self.psets, dtype=object)[v_J]
        pj_arr = [
            len(a & b) / len(a | b) if (a or b) else NAN
            for a, b in zip(pa, pb)
        ]
        res["nt_phon_jacc"][idx] = pj_arr
        
        res["nt_ntok_a"][idx] = len_a
        res["nt_ntok_b"][idx] = len_b
        
        # Soft matching
        first_a = [x[0] for x in v_ta]
        first_b = [x[0] for x in v_tb]
        res["nt_first_jw"][idx] = pair_scores(first_a, first_b, JaroWinkler.similarity, workers=workers)
        
        args_list = [
            (v_ta[m], v_tb[m], self.t_wa[v_I[m]], self.t_wa[v_J[m]], self.t_swa[v_I[m]], self.t_swa[v_J[m]])
            for m in range(len(idx))
        ]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            soft_res = list(pool.map(self._soft_match_single, args_list))
            
        if soft_res:
            res_arr = np.array(soft_res, dtype=np.float32)
            res["nt_soft_a"][idx] = res_arr[:, 0]
            res["nt_soft_b"][idx] = res_arr[:, 1]
            res["nt_soft_f1"][idx] = res_arr[:, 2]
            res["nt_unm_a"][idx] = res_arr[:, 3]
            res["nt_unm_b"][idx] = res_arr[:, 4]

        return res

    def _vectorize_addr(self, I, J):
        n = len(I)
        res = {k: np.full(n, NAN, dtype=np.float32) for k in AD_FEATS}
        
        pa = np.array(self.postal, dtype=object)[I]
        pb = np.array(self.postal, dtype=object)[J]
        ha = np.array(self.house, dtype=object)[I]
        hb = np.array(self.house, dtype=object)[J]
        na = np.array(self.nums_addr, dtype=object)[I]
        nb = np.array(self.nums_addr, dtype=object)[J]
        sa = np.array(self.aset, dtype=object)[I]
        sb = np.array(self.aset, dtype=object)[J]
        
        res["pc_state"][:] = 0.0
        res["hn_state"][:] = 0.0

        valid_p = (pa != "") & (pb != "")
        idx_p = np.where(valid_p)[0]
        if len(idx_p) > 0:
            v_pa, v_pb = pa[idx_p], pb[idx_p]
            res["pc_state"][idx_p] = np.where(v_pa == v_pb, 1.0, -1.0)
            res["pc_p3"][idx_p] = [a[:3] == b[:3] for a, b in zip(v_pa, v_pb)]
            res["pc_p2"][idx_p] = [a[:2] == b[:2] for a, b in zip(v_pa, v_pb)]
            
            ld_arr = []
            for a, b in zip(v_pa, v_pb):
                try:
                    ld_arr.append(math.log1p(abs(int(a) - int(b))))
                except ValueError:
                    ld_arr.append(NAN)
            res["pc_logdiff"][idx_p] = ld_arr
        
        valid_h = (ha != "") & (hb != "")
        res["hn_state"][valid_h] = np.where(ha[valid_h] == hb[valid_h], 1.0, -1.0)
        
        valid_n = np.array([bool(a) and bool(b) for a, b in zip(na, nb)], dtype=bool)
        idx_n = np.where(valid_n)[0]
        if len(idx_n) > 0:
            v_na, v_nb = na[idx_n], nb[idx_n]
            ni_n = np.array([len(a & b) for a, b in zip(v_na, v_nb)], dtype=np.float32)
            nu_n = np.array([len(a | b) for a, b in zip(v_na, v_nb)], dtype=np.float32)
            res["nums_common"][idx_n] = ni_n
            res["nums_jacc"][idx_n] = ni_n / nu_n
            
        valid_a = np.array([bool(a) and bool(b) for a, b in zip(sa, sb)], dtype=bool)
        idx_a = np.where(valid_a)[0]
        if len(idx_a) > 0:
            v_sa, v_sb = sa[idx_a], sb[idx_a]
            v_I, v_J = I[idx_a], J[idx_a]
            inter = [a & b for a, b in zip(v_sa, v_sb)]
            ni_a = np.array([len(x) for x in inter], dtype=np.float32)
            len_a = np.array([len(x) for x in v_sa], dtype=np.float32)
            len_b = np.array([len(x) for x in v_sb], dtype=np.float32)
            
            res["at_jacc"][idx_a] = ni_a / (len_a + len_b - ni_a)
            res["at_ovmin"][idx_a] = ni_a / np.minimum(len_a, len_b)
            
            def get_wi(k, inter_set):
                return sum(self.idf_addr.get(t, self.idf_addr_d) for t in inter_set)
                
            wi_arr = np.array([get_wi(m, inter[m]) for m in range(len(idx_a))], dtype=np.float32)
            res["at_idfov_common"][idx_a] = wi_arr
            res["at_idfov_a"][idx_a] = wi_arr / self.a_swa[v_I]
            res["at_idfov_b"][idx_a] = wi_arr / self.a_swa[v_J]
            
        return res

    # ------------------------------------------------------------------ main entry
    def pair_features(self, I, J, chunk=200_000) -> pd.DataFrame:
        parts = [self._chunk(I[s:s + chunk], J[s:s + chunk]) for s in range(0, len(I), chunk)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _chunk(self, I, J) -> pd.DataFrame:
        F = {}
        for v in ["nchar", "nword", "nphon", "achar", "aword", "ncomb"]:
            F["cos_" + v] = self._cos(v, I, J)
            
        F["name_char_overlap_34"] = self._jaccard_sparse("nchar_bin", I, J)

        # ---- names (core = legal forms removed)
        F["n_ratio"] = self._sim("name_core", fuzz.ratio, I, J)
        F["n_pratio"] = self._sim("name_core", fuzz.partial_ratio, I, J)
        F["n_tsort"] = self._sim("name_core", fuzz.token_sort_ratio, I, J)
        F["n_tset"] = self._sim("name_core", fuzz.token_set_ratio, I, J)
        F["n_wratio"] = self._sim("name_core", fuzz.WRatio, I, J)
        F["n_jw"] = self._sim("name_core", JaroWinkler.similarity, I, J, 1.0)
        F["n_lev"] = self._sim("name_core", Levenshtein.normalized_similarity, I, J, 1.0)
        F["n_phon_ratio"] = self._sim("name_phon", fuzz.ratio, I, J)
        F["n_full_tset"] = self._sim("name_norm", fuzz.token_set_ratio, I, J)
        F["n_full_ratio"] = self._sim("name_norm", fuzz.ratio, I, J)
        F["n_c_ratio"] = self._sim("compact", fuzz.ratio, I, J)
        F["n_c_jw"] = self._sim("compact", JaroWinkler.similarity, I, J, 1.0)
        F["n_exact_core"] = (self.arr["name_core"][I] == self.arr["name_core"][J]).astype(np.float32)
        F["n_exact_compact"] = (self.arr["compact"][I] == self.arr["compact"][J]).astype(np.float32)
        la = np.array([len(x) for x in self.arr["name_core"][I]], dtype=np.float32)
        lb = np.array([len(x) for x in self.arr["name_core"][J]], dtype=np.float32)
        F["n_len_a"], F["n_len_b"], F["n_len_diff"] = la, lb, np.abs(la - lb)
        
        lga, lgb = np.array(self.legal, dtype=object)[I], np.array(self.legal, dtype=object)[J]
        F["legal_both"] = ((lga != "") & (lgb != "")).astype(np.float32)
        F["legal_eq"] = ((lga == lgb) & (lga != "")).astype(np.float32)
        F["legal_one"] = ((lga != "") ^ (lgb != "")).astype(np.float32)
        F["legal_mismatch_penalty"] = np.where((lga != "") & (lgb != "") & (lga != lgb), -1.0, 0.0).astype(np.float32)

        # ---- addresses
        F["a_tset"] = self._sim("addr_all", fuzz.token_set_ratio, I, J)
        F["a_tsort"] = self._sim("addr_all", fuzz.token_sort_ratio, I, J)
        F["a_ratio"] = self._sim("addr_all", fuzz.ratio, I, J)
        F["a_pratio"] = self._sim("addr_all", fuzz.partial_ratio, I, J)
        F["a_main_tset"] = self._sim("addr_main", fuzz.token_set_ratio, I, J)
        F["a_head_tset"] = self._sim("addr_head", fuzz.token_set_ratio, I, J)
        F["a_tail_tset"] = self._sim("addr_tail", fuzz.token_set_ratio, I, J)
        F["a_lm_tset"] = self._sim("addr_lm", fuzz.token_set_ratio, I, J)
        F["a_missing_a"] = self.emp["addr_all"][I].astype(np.float32)
        F["a_missing_b"] = self.emp["addr_all"][J].astype(np.float32)
        F["a_lm_a"] = (~self.emp["addr_lm"][I]).astype(np.float32)
        F["a_lm_b"] = (~self.emp["addr_lm"][J]).astype(np.float32)
        
        pa, pb = np.array(self.postal, dtype=object)[I], np.array(self.postal, dtype=object)[J]
        F["addr_postal_match"] = np.where((pa != "") & (pb != "") & (pa == pb), 1.0, 0.0).astype(np.float32)

        # ---- country (open-set: only (dis)agreement of the labels is used)
        ca, cb = self.arr["ctry"][I], self.arr["ctry"][J]
        miss = self.emp["ctry"][I] | self.emp["ctry"][J]
        F["ctry_state"] = np.where(miss, 0.0, np.where(ca == cb, 1.0, -1.0)).astype(np.float32)
        F["ctry_fuzzy"] = self._sim("ctry", fuzz.ratio, I, J)
        F["is_s3"] = (self.src[J] == 3).astype(np.float32)

        # ---- python-level token / address features
        nt_dict = self._vectorize_nt(I, J)
        ad_dict = self._vectorize_addr(I, J)
        
        for k, v in nt_dict.items():
            F[k] = v
        for k, v in ad_dict.items():
            F[k] = v
            
        F["n_exact_first_tok"] = F["nt_first_eq"]

        return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in F.items()})


# ----------------------------------------------------------------------------- group context features
def group_rank_features(cols: dict, s1, cand, src) -> pd.DataFrame:
    """For every ``{name: values}`` pair: gap-to-best and rank within the same (s1, src) group (forward -
    "how does this candidate compare with the other candidates competing for the same S1 record") and
    within the same ``cand`` group (reverse - "how does this S1 record compare with the other S1 records
    competing for the same candidate", relevant to unique-assignment).  Pure feature engineering, no
    labels, so this is equally valid on a raw similarity column (blocking time) or on a fitted model's own
    predicted probability (stage-2 stacking - see model.py)."""
    s1, cand, src = np.asarray(s1), np.asarray(cand), np.asarray(src)
    out = {}
    for name, v in cols.items():
        v = np.asarray(v, dtype=np.float32)
        col = pd.Series(v)
        g = col.groupby([s1, src])
        gr = col.groupby(cand)
        mx = g.transform("max").to_numpy(np.float32)
        out[f"{name}_gap"] = mx - v
        out[f"{name}_rk"] = g.rank(method="min", ascending=False).to_numpy(np.float32)
        rmx = gr.transform("max").to_numpy(np.float32)
        out[f"{name}_rgap"] = rmx - v
        out[f"{name}_rrk"] = gr.rank(method="min", ascending=False).to_numpy(np.float32)
    sizer = pd.Series(np.zeros(len(s1), dtype=np.float32))
    out["n_cand_s1"] = sizer.groupby([s1, src]).transform("size").to_numpy(np.float32)
    out["n_s1_cand"] = sizer.groupby(cand).transform("size").to_numpy(np.float32)
    return pd.DataFrame(out)


def add_context(F: pd.DataFrame, s1, cand, src) -> pd.DataFrame:
    """Relative features (see ``group_rank_features``) for the raw-similarity CTX_COLS already in F."""
    extra = group_rank_features({c: F[c].to_numpy() for c in CTX_COLS}, s1, cand, src)
    return pd.concat([F.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
