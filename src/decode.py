"""Turn scored candidate pairs into the two required id-lists.

Both `candidate_lists` and `match_lists` take the same flat pair representation: `s1_pos` is the
0..n_s1-1 position of every (Source-1, candidate) pair (aligned with `cand_row`), so that every Source-1
record is represented in the output even when it has zero surviving candidates or zero kept matches.

`candidate_pairs.tsv` is just `candidate_lists` on every blocking-stage pair - it does not depend on the
model's score or the decision threshold at all, per the spec ("the final candidate set fed to the matching
model", not a threshold-filtered subset of it).

`matching_results.tsv` is `match_lists`, which additionally applies the decision threshold and, if
`unique` is set, resolves every case where more than one Source-1 record claims the same Source-2/3 record
by keeping only the single highest-scoring claim (a Source-1 record may still match several candidates; a
candidate may not go to more than one Source-1 record - see model.py's module docstring for why this is a
reasonable constraint even though the validator itself does not require it).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def candidate_lists(s1_pos: np.ndarray, n_s1: int, cand_row: np.ndarray,
                     entity_id: Sequence[str]) -> List[List[str]]:
    ids = np.asarray(entity_id, dtype=object)
    out: List[List[str]] = [[] for _ in range(n_s1)]
    for i in range(len(cand_row)):
        out[int(s1_pos[i])].append(str(ids[cand_row[i]]))
    return [sorted(x) for x in out]


def resolve_unique(cand_row: np.ndarray, score: np.ndarray, tau: float, unique: bool) -> np.ndarray:
    """Boolean mask (over the flat pair arrays) of pairs kept as final matches."""
    keep = score >= tau
    if not unique or not keep.any():
        return keep
    idx = np.where(keep)[0]
    order = idx[np.argsort(-score[idx], kind="stable")]
    claimed = set()
    final = np.zeros_like(keep)
    for i in order:
        c = int(cand_row[i])
        if c in claimed:
            continue
        claimed.add(c)
        final[i] = True
    return final


def match_lists(s1_pos: np.ndarray, n_s1: int, cand_row: np.ndarray, keep: np.ndarray,
                 entity_id: Sequence[str]) -> List[List[str]]:
    ids = np.asarray(entity_id, dtype=object)
    out: List[List[str]] = [[] for _ in range(n_s1)]
    for i in np.where(keep)[0]:
        out[int(s1_pos[i])].append(str(ids[cand_row[i]]))
    return [sorted(x) for x in out]


def decode(s1_pos: np.ndarray, n_s1: int, cand_row: np.ndarray, score: np.ndarray, entity_id: Sequence[str],
           tau: float, unique: bool):
    """Convenience wrapper returning (candidate_id_lists, matched_id_lists)."""
    cands = candidate_lists(s1_pos, n_s1, cand_row, entity_id)
    keep = resolve_unique(cand_row, score, tau, unique)
    matches = match_lists(s1_pos, n_s1, cand_row, keep, entity_id)
    return cands, matches
