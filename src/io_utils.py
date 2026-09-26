"""Reading / writing of the challenge's tab-separated files.

* every input file is read with an explicit tab separator and *no* quote handling, so a stray quote character
  in a business name can never swallow following rows;
* a malformed row (too few / too many tabs) is repaired instead of dropped, so every record survives;
* ID lists are always written sorted, comma separated, without spaces or quoting.
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .textnorm import normalize_frame

COLS = ["entity_id", "business_name", "business_address", "country"]
_NULLS = {"nan", "none", "null", "n/a", "-", "--"}


# ----------------------------------------------------------------------------- reading
def _clean_field(x: str, nullable: bool) -> str:
    x = x.strip()
    if nullable and x.lower() in _NULLS:
        return ""
    return x


def read_tsv(path: str) -> pd.DataFrame:
    """Read one source file -> DataFrame[entity_id, business_name, business_address, country] (all str)."""
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        text = fh.read()
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    lines = [ln for ln in lines if ln.strip() != ""]
    if not lines:
        return pd.DataFrame({c: [] for c in COLS}, dtype=str)
    head = [h.strip().lower() for h in lines[0].split("\t")]
    has_header = "entity_id" in head
    body = lines[1:] if has_header else lines
    pos = {c: head.index(c) for c in COLS if c in head} if has_header else {}
    rows: List[List[str]] = []
    for ln in body:
        f = ln.split("\t")
        if has_header and len(pos) == 4 and len(f) == len(head):
            r = [f[pos[c]] for c in COLS]
        elif len(f) == 4:
            r = f
        elif len(f) > 4:      # a tab inside the address: keep id, name, country; glue the middle back together
            r = [f[0], f[1], "\t".join(f[2:-1]).replace("\t", " "), f[-1]]
        else:                 # missing trailing fields
            r = f + [""] * (4 - len(f))
        rows.append([_clean_field(r[0], False), _clean_field(r[1], False),
                     _clean_field(r[2], True), _clean_field(r[3], True)])
    return pd.DataFrame(rows, columns=COLS, dtype=str)


def read_truth(path: str) -> Dict[str, List[str]]:
    """train_ground_truth.tsv -> {source1_entity_id: [matched ids]} (empty list for singletons)."""
    out: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        lines = [ln.rstrip("\r") for ln in fh.read().split("\n")]
    for k, ln in enumerate(lines):
        if not ln.strip():
            continue
        f = ln.split("\t")
        if k == 0 and f[0].strip().lower() == "source1_entity_id":
            continue
        sid = f[0].strip()
        rest = f[1] if len(f) > 1 else ""
        ids = [x.strip() for x in rest.split(",") if x.strip()]
        out.setdefault(sid, [])
        for x in ids:
            if x not in out[sid]:
                out[sid].append(x)
    return out


def split_paths(data_dir: str, split: str) -> Dict[str, str]:
    """Locate the files of a split ('train' | 'test'); both  <data_dir>/<split>/  and  <data_dir>/  are accepted."""
    for base in (os.path.join(data_dir, split), data_dir):
        p = {k: os.path.join(base, f"{split}_{k}.tsv") for k in ("source1", "source2", "source3")}
        if all(os.path.exists(v) for v in p.values()):
            p["truth"] = os.path.join(base, f"{split}_ground_truth.tsv")
            return p
    raise FileNotFoundError(f"could not find {split}_source1/2/3.tsv under {data_dir!r} (or {data_dir}/{split})")


def load_split(data_dir: str, split: str):
    p = split_paths(data_dir, split)
    s1, s2, s3 = (read_tsv(p[k]) for k in ("source1", "source2", "source3"))
    truth = read_truth(p["truth"]) if os.path.exists(p["truth"]) else None
    return s1, s2, s3, truth


# ----------------------------------------------------------------------------- record frame
def build_records(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    """One frame with all records (S1 rows first, then S2, then S3), normalised columns and a ``src`` column."""
    parts = []
    for k, d in ((1, s1), (2, s2), (3, s3)):
        d = d[COLS].copy()
        d["src"] = k
        parts.append(d)
    R = pd.concat(parts, ignore_index=True)
    N = normalize_frame(R)
    return pd.concat([R.reset_index(drop=True), N.reset_index(drop=True)], axis=1)


def truth_pairs(R: pd.DataFrame, truth: Dict[str, List[str]]):
    """Ground truth -> (int64 keys  s1_row * N + cand_row  of all true pairs present in R, #true matches per S1 row)."""
    ids = R["entity_id"].tolist()
    row = {e: i for i, e in enumerate(ids)}
    src = R["src"].to_numpy()
    s1_rows = np.where(src == 1)[0]
    pos_of_s1 = {int(r): k for k, r in enumerate(s1_rows)}
    n = len(R)
    keys, n_true = [], np.zeros(len(s1_rows), dtype=np.int64)
    for sid, lst in truth.items():
        a = row.get(sid)
        if a is None or src[a] != 1:
            continue
        for m in lst:
            b = row.get(m)
            if b is None or src[b] == 1:
                continue
            keys.append(a * n + b)
            n_true[pos_of_s1[a]] += 1
    return np.unique(np.asarray(keys, dtype=np.int64)), n_true


# ----------------------------------------------------------------------------- writing
def _sorted_ids(ids: Sequence[str]) -> List[str]:
    return sorted(set(ids))


def write_list_tsv(path: str, header: Sequence[str], s1_ids: Sequence[str], lists: Sequence[Sequence[str]]) -> None:
    """Two-column TSV: one row per Source-1 id, comma-joined sorted ids (empty string for none)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(header) + "\n")
        for sid, lst in zip(s1_ids, lists):
            fh.write(f"{sid}\t{','.join(_sorted_ids(lst))}\n")
