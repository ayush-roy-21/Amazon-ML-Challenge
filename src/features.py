"""TF-IDF views and pairwise feature engineering.

Every feature is a similarity between a Source-1 record (``a`` side) and a Source-2/3 candidate (``b`` side).
Nothing here depends on the identity of a country: country labels are only compared for (dis)agreement.
"""
from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache

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


def _tfidf(texts, analyzer, ngram, max_df):
    if analyzer == "char_wb":
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, lowercase=False, sublinear_tf=True,
                            max_df=max_df, min_df=2, dtype=np.float32)
    else:
        v = TfidfVectorizer(analyzer="word", tokenizer=str.split, token_pattern=None, lowercase=False,
                            ngram_range=ngram, sublinear_tf=True, max_df=max_df, min_df=2, dtype=np.float32)
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

        fields = {"name_core": self.name_core, "name_norm": self.name_norm, "compact": self.compact,
                  "addr_all": self.addr_all, "addr_main": self.addr_main, "addr_lm": self.addr_lm,
                  "addr_head": self.addr_head, "addr_tail": self.addr_tail, "ctry": self.ctry, "name_phon": self.name_phon_str}
        self.arr = {k: np.array(v, dtype=object) for k, v in fields.items()}
        self.emp = {k: np.array([len(x) == 0 for x in v], dtype=bool) for k, v in fields.items()}

        md_name = _max_df(n, 0.2, cfg.max_df_abs)
        md_addr = _max_df(n, 0.1, cfg.max_df_abs)
        V = {
            "nchar": _tfidf(R["name_core"].tolist(), "char_wb", (3, 4), md_name),
            "nword": _tfidf(R["name_toks_str"].tolist(), "word", (1, 2), md_name),
            "nphon": _tfidf(R["name_phon"].tolist(), "word", (1, 1), md_name),
            "achar": _tfidf(R["addr_all"].tolist(), "char_wb", (3, 4), md_addr),
            "aword": _tfidf(R["addr_all"].tolist(), "word", (1, 2), md_addr),
        }
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

    def _sim(self, field, scorer, I, J, scale=100.0):
        a, e = self.arr[field], self.emp[field]
        out = pair_scores(a[I].tolist(), a[J].tolist(), scorer) / np.float32(scale)
        out = out.astype(np.float32)
        out[e[I] | e[J]] = np.nan
        return out

    # ------------------------------------------------------------------ per-pair python features
    def _name_tok(self, i, j):
        ta, tb = self.toks[i], self.toks[j]
        if not ta or not tb:
            return _NAN_NT
        sa, sb = self.tsets[i], self.tsets[j]
        ni = len(sa & sb)
        jacc = ni / (len(sa) + len(sb) - ni)
        ovmin = ni / min(len(sa), len(sb))
        idf, d = self.idf_name, self.idf_name_d
        wa = [idf.get(t, d) for t in ta]
        wb = [idf.get(t, d) for t in tb]
        swa, swb = sum(wa), sum(wb)
        ioa = sum(w for t, w in zip(ta, wa) if t in sb) / swa
        iob = sum(w for t, w in zip(tb, wb) if t in sa) / swb
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
        soft_a = sum(w * s for w, s in zip(wa, ma)) / swa
        soft_b = sum(w * s for w, s in zip(wb, mb)) / swb
        f1 = 2 * soft_a * soft_b / (soft_a + soft_b) if soft_a + soft_b > 0 else 0.0
        unm_a = sum(w * (1.0 - s) for w, s in zip(wa, ma))
        unm_b = sum(w * (1.0 - s) for w, s in zip(wb, mb))
        ia, ib = self.acro[i], self.acro[j]
        acro = 1.0 if ((len(ia) >= 2 and ia == self.compact[j]) or (len(ib) >= 2 and ib == self.compact[i])) else 0.0
        na, nb = self.nums_name[i], self.nums_name[j]
        num = (1.0 if na == nb else 0.0) if (na and nb) else NAN
        pa, pb = self.psets[i], self.psets[j]
        pj = len(pa & pb) / len(pa | pb) if (pa or pb) else NAN
        return (jacc, ovmin, ioa, iob, soft_a, soft_b, f1, unm_a, unm_b, float(ta[0] == tb[0]),
                float(ta[-1] == tb[-1]), JaroWinkler.similarity(ta[0], tb[0]), acro, num, pj,
                float(len(ta)), float(len(tb)))

    def _addr_row(self, i, j):
        pa, pb = self.postal[i], self.postal[j]
        if pa and pb:
            pc = 1.0 if pa == pb else -1.0
            p3, p2 = float(pa[:3] == pb[:3]), float(pa[:2] == pb[:2])
            try:
                ld = math.log1p(abs(int(pa) - int(pb)))
            except ValueError:
                ld = NAN
        else:
            pc, p3, p2, ld = 0.0, NAN, NAN, NAN
        ha, hb = self.house[i], self.house[j]
        hn = 0.0 if not (ha and hb) else (1.0 if ha == hb else -1.0)
        na, nb = self.nums_addr[i], self.nums_addr[j]
        if na and nb:
            nc = len(na & nb)
            nj = nc / len(na | nb)
            nc = float(nc)
        else:
            nj = nc = NAN
        sa, sb = self.aset[i], self.aset[j]
        if sa and sb:
            inter = sa & sb
            ni = len(inter)
            jacc = ni / (len(sa) + len(sb) - ni)
            ovmin = ni / min(len(sa), len(sb))
            idf, d = self.idf_addr, self.idf_addr_d
            wa = sum(idf.get(t, d) for t in sa)
            wb = sum(idf.get(t, d) for t in sb)
            wi = sum(idf.get(t, d) for t in inter)
            return (pc, p3, p2, ld, hn, nj, nc, jacc, ovmin, wi / wa, wi / wb, wi)
        return (pc, p3, p2, ld, hn, nj, nc, NAN, NAN, NAN, NAN, NAN)

    # ------------------------------------------------------------------ main entry
    def pair_features(self, I, J, chunk=200_000) -> pd.DataFrame:
        parts = [self._chunk(I[s:s + chunk], J[s:s + chunk]) for s in range(0, len(I), chunk)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _chunk(self, I, J) -> pd.DataFrame:
        n = len(I)
        F = {}
        for v in self.views:
            F["cos_" + v] = self._cos(v, I, J)
        # ---- names (core = legal forms removed)
        F["n_ratio"] = self._sim("name_core", fuzz.ratio, I, J)
        F["n_pratio"] = self._sim("name_core", fuzz.partial_ratio, I, J)
        F["n_tsort"] = self._sim("name_core", fuzz.token_sort_ratio, I, J)
        F["n_tset"] = self._sim("name_core", fuzz.token_set_ratio, I, J)
        F["n_wratio"] = self._sim("name_core", fuzz.WRatio, I, J)
        F["n_jw"] = self._sim("name_core", JaroWinkler.similarity, I, J, 1.0)
        F["n_lev"] = self._sim("name_core", Levenshtein.normalized_similarity, I, J, 1.0)
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
        # ---- country (open-set: only (dis)agreement of the labels is used)
        ca, cb = self.arr["ctry"][I], self.arr["ctry"][J]
        miss = self.emp["ctry"][I] | self.emp["ctry"][J]
        F["ctry_state"] = np.where(miss, 0.0, np.where(ca == cb, 1.0, -1.0)).astype(np.float32)
        F["ctry_fuzzy"] = self._sim("ctry", fuzz.ratio, I, J)
        F["is_s3"] = (self.src[J] == 3).astype(np.float32)
        # ---- python-level token / address features
        nt = np.empty((n, len(NT_FEATS)), dtype=np.float32)
        ad = np.empty((n, len(AD_FEATS)), dtype=np.float32)
        for k in range(n):
            i, j = int(I[k]), int(J[k])
            nt[k] = self._name_tok(i, j)
            ad[k] = self._addr_row(i, j)
        for c, name in enumerate(NT_FEATS):
            F[name] = nt[:, c]
        for c, name in enumerate(AD_FEATS):
            F[name] = ad[:, c]
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
