"""End-to-end orchestration.

``train(data_dir, model_dir)`` fits blocking (recall-tuned k_scale) + the stage-1/2 classifier on the
labelled train split and persists everything needed to reproduce predictions. ``predict(data_dir,
model_dir, out_dir)`` loads that artifact and scores a fresh split (typically ``test``, which has no
labels), writing ``matching_results.tsv`` and ``candidate_pairs.tsv``.

Test data is *never* mixed into the train split's TF-IDF vectorisers or nearest-neighbour indices - a
completely fresh ``Featurizer`` is built for whatever split is being scored. Only the trained classifier
(a function of derived similarity/rank features, never a raw vocabulary index) and the chosen k_scale /
threshold carry over from train to test - see model.py's docstring for why that is what lets the pipeline
handle a country it never trained on at all.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from . import blocking, decode
from . import model as model_mod
from .config import Config
from .features import Featurizer, add_context
from .io_utils import build_records, load_split, truth_pairs, write_list_tsv
from .metrics import best_threshold, macro_f05
from .utils import log


def _pair_arrays(s1_rows: np.ndarray, cand_sets):
    """cand_sets: list aligned with s1_rows, each an array of row-in-R candidate indices. Returns (I, J,
    s1_pos): I/J are row-in-R for the S1/candidate side of every pair; s1_pos is each pair's 0..n_s1-1
    S1 position (used for grouping and for metrics, which need a dense 0..n_s1-1 index)."""
    lens = np.fromiter((len(c) for c in cand_sets), dtype=np.int64, count=len(cand_sets))
    s1_pos = np.repeat(np.arange(len(s1_rows), dtype=np.int64), lens)
    I = np.repeat(s1_rows, lens)
    J = np.concatenate(cand_sets) if lens.sum() else np.zeros(0, dtype=np.int64)
    return I, J, s1_pos


def _build_features(feat: Featurizer, I: np.ndarray, J: np.ndarray, src: np.ndarray) -> pd.DataFrame:
    F = feat.pair_features(I, J)
    return add_context(F, s1=I, cand=J, src=src[J])


# ----------------------------------------------------------------------------- train
def train(data_dir: str, model_dir: str, cfg: Optional[Config] = None) -> dict:
    cfg = cfg or Config()
    os.makedirs(model_dir, exist_ok=True)
    s1, s2, s3, truth = load_split(data_dir, "train")
    if truth is None:
        raise FileNotFoundError("train_ground_truth.tsv not found - training requires labels")
    log(f"train split: S1={len(s1)} S2={len(s2)} S3={len(s3)}")
    R = build_records(s1, s2, s3)
    n = len(R)
    src = R["src"].to_numpy()
    s1_rows = np.where(src == 1)[0]
    tgt_rows = np.where(src != 1)[0]
    ctry = np.asarray(R["ctry"].tolist(), dtype=object)
    entity_id = R["entity_id"].tolist()

    true_keys, n_true_by_s1 = truth_pairs(R, truth)
    feat = Featurizer(R, cfg)
    k_scale, cand_sets, blk_recall, ranked = blocking.tune_k_scale(feat, s1_rows, tgt_rows, ctry, cfg,
                                                                    true_keys, n)
    cand_sets = [np.asarray(sorted(s), dtype=np.int64) for s in cand_sets]  # tune_k_scale returns raw sets
    I, J, s1_pos = _pair_arrays(s1_rows, cand_sets)
    F = _build_features(feat, I, J, src)
    true_set = set(true_keys.tolist())
    y = np.fromiter(((int(i) * n + int(j)) in true_set for i, j in zip(I, J)), dtype=np.int64, count=len(I))
    log(f"built {len(I)} candidate pairs ({int(y.sum())} positive) from {len(s1_rows)} Source-1 records")

    m1, m2, oof_final = model_mod.train_matcher(F, y, s1=I, cand=J, src=src[J], cfg=cfg)
    tau, oof_f05, grid = best_threshold(s1_pos, oof_final, y, n_true_by_s1, len(s1_rows), cfg.thr_grid)
    log(f"chosen threshold={tau}  out-of-fold macro F0.5={oof_f05:.4f}")

    unique = cfg.unique_assign != "no"
    _, matched = decode.decode(s1_pos, len(s1_rows), J, oof_final, entity_id, tau, unique)
    truth_lists = [truth.get(entity_id[r], []) for r in s1_rows]
    post_unique_f05 = macro_f05(matched, truth_lists)
    log(f"out-of-fold macro F0.5 after unique-assignment decoding={post_unique_f05:.4f}")

    country_report = {}
    if cfg.country_holdout:
        row_to_pos = {int(r): k for k, r in enumerate(s1_rows)}
        countries = sorted(set(c for c in ctry[s1_rows] if c))

        def build_for(held_out):
            is_ho = ctry[I] == held_out
            s1ho_rows = s1_rows[ctry[s1_rows] == held_out]
            if is_ho.sum() == 0 or (~is_ho).sum() == 0 or len(s1ho_rows) == 0:
                return None
            tr, ho = ~is_ho, is_ho
            pos_map = {int(r): k for k, r in enumerate(s1ho_rows)}
            s1_pos_ho = np.fromiter((pos_map[int(i)] for i in I[ho]), dtype=np.int64, count=int(ho.sum()))
            n_true_ho = np.array([n_true_by_s1[row_to_pos[int(r)]] for r in s1ho_rows])
            return (F[tr], y[tr], I[tr], F[ho], y[ho], s1_pos_ho, J[ho], src[J[ho]], n_true_ho,
                    len(s1ho_rows))

        country_report = model_mod.country_holdout_report(build_for, countries, cfg)

    joblib.dump({"m1": m1, "m2": m2, "cfg": cfg, "k_scale": k_scale, "tau": tau, "unique_assign": unique},
                os.path.join(model_dir, "model.joblib"))
    report = {"blocking_recall": blk_recall, "k_scale": k_scale, "n_pairs": int(len(I)),
              "n_positive": int(y.sum()), "oof_macro_f05": oof_f05, "threshold": tau,
              "oof_macro_f05_post_unique_assign": post_unique_f05, "stage2_used": m2 is not None,
              "country_holdout_f05": country_report, "threshold_grid": grid}
    with open(os.path.join(model_dir, "train_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    log(f"model + report saved to {model_dir}")
    return report


# ----------------------------------------------------------------------------- predict
def predict(data_dir: str, model_dir: str, out_dir: str) -> dict:
    art = joblib.load(os.path.join(model_dir, "model.joblib"))
    m1, m2, cfg = art["m1"], art["m2"], art["cfg"]
    k_scale, tau, unique = art["k_scale"], art["tau"], art["unique_assign"]

    s1, s2, s3, _truth = load_split(data_dir, "test")
    log(f"test split: S1={len(s1)} S2={len(s2)} S3={len(s3)}")
    R = build_records(s1, s2, s3)
    src = R["src"].to_numpy()
    s1_rows = np.where(src == 1)[0]
    tgt_rows = np.where(src != 1)[0]
    ctry = R["ctry"].tolist()
    entity_id = R["entity_id"].tolist()

    feat = Featurizer(R, cfg)
    ranked = blocking.fetch_and_rank(feat, s1_rows, tgt_rows, ctry, cfg)
    cand_sets = blocking.candidates_for(ranked, cfg, k_scale)
    I, J, s1_pos = _pair_arrays(s1_rows, cand_sets)
    if len(I):
        F = _build_features(feat, I, J, src)
        score = model_mod.score_pairs(m1, m2, F, s1=I, cand=J, src=src[J])
    else:
        score = np.zeros(0, dtype=np.float32)

    cands, matches = decode.decode(s1_pos, len(s1_rows), J, score, entity_id, tau, unique)
    s1_ids = [entity_id[r] for r in s1_rows]
    os.makedirs(out_dir, exist_ok=True)
    write_list_tsv(os.path.join(out_dir, "matching_results.tsv"),
                    ("source1_entity_id", "matched_entity_ids"), s1_ids, matches)
    write_list_tsv(os.path.join(out_dir, "candidate_pairs.tsv"),
                    ("source1_entity_id", "candidate_entity_ids"), s1_ids, cands)
    log(f"wrote {len(s1_ids)} rows to {out_dir}/matching_results.tsv and candidate_pairs.tsv")
    return {"n_s1": len(s1_ids), "n_pairs": int(len(I)), "threshold": tau, "k_scale": k_scale}
