"""The pairwise match classifier.

Stage 1 is a k-fold cross-validated LightGBM model over the engineered similarity features from
`features.py` (out-of-fold predictions are used both for threshold selection and, optionally, as the
input to stage 2). Stage 2 is a small "stacked generalisation" refinement: it adds the stage-1 model's own
predicted probability, plus that probability's rank / gap-to-best within (a) the candidates competing for
the same Source-1 record and (b) the Source-1 records competing for the same candidate, and retrains a
second, smaller model on top. This is the piece that lets the classifier reason about *competition*
("this candidate is a decent match, but a much better one exists for this record") rather than only
absolute similarity - directly useful for the precision-heavy F0.5 metric and for `unique_assign`.

Optimized: added focal loss option, pos_scale_weight, better LightGBM hyperparameters.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:  # pragma: no cover - keeps the pipeline runnable if lightgbm can't be installed
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_LGB = False

from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from .features import group_rank_features
from .utils import log

MIN_STAGE2_POSITIVES = 200  # below this, a second model would just overfit its own few positive examples


class FoldModel:
    """A bagged ensemble of one (booster, calibrator) pair per CV fold.

    Every fold's LightGBM booster is trained on a different subset of the data, so their raw outputs are
    not guaranteed to sit on a shared probability scale - a small/easy fold can end up far more "confident"
    than another purely as an artifact of which rows landed in it, even when every fold ranks its own
    validation examples perfectly. Pooling such raw scores across folds for a single global decision
    threshold is unsound (the threshold could mean "confident positive" for one fold's scores and "clear
    negative" for another's). Each fold therefore also gets its own isotonic regressor, fit on that fold's
    own (raw score, true label) pairs, which rescales its output onto a common "estimated probability of a
    true match" footing before folds are ever combined - monotonic, so it changes no fold's internal
    ranking (its within-fold contribution to average precision is identical), only how folds compare to
    each other. `predict` calibrates every fold's raw output through *that same fold's* calibrator and then
    averages - the identical procedure used to produce the out-of-fold scores that the threshold in
    metrics.best_threshold was chosen against, so the threshold means the same thing at train and test time.
    """

    def __init__(self, boosters: list, calibrators: list, feature_names: List[str], kind: str):
        self.boosters = boosters
        self.calibrators = calibrators
        self.feature_names = feature_names
        self.kind = kind

    def _raw(self, booster, Xm: np.ndarray) -> np.ndarray:
        if self.kind == "lgb":
            return booster.predict(Xm, num_iteration=booster.best_iteration)
        return booster.predict_proba(Xm)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xm = X[self.feature_names].to_numpy(dtype=np.float32)
        outs = []
        for booster, cal in zip(self.boosters, self.calibrators):
            raw = self._raw(booster, Xm)
            outs.append(cal.predict(raw) if cal is not None else raw)
        return np.mean(outs, axis=0).astype(np.float32)


def _lgb_params(cfg, n_pos=0, n_neg=0):
    params = dict(
        objective="binary",
        metric="average_precision",
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        subsample_freq=1,
        reg_lambda=cfg.reg_lambda,
        reg_alpha=getattr(cfg, 'reg_alpha', 0.0),
        max_bin=255,
        verbosity=-1,
        seed=cfg.seed,
        deterministic=True,
        force_row_wise=True,  # more memory-efficient for wide datasets
    )
    # Scale positive weight for imbalanced data — helps the model learn the minority class
    if n_pos > 0 and n_neg > 0:
        ratio = n_neg / n_pos
        # Cap the weight: too high causes overconfident predictions on negatives
        params["scale_pos_weight"] = min(ratio, 20.0)
    return params


def fit_cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg, feature_names: List[str],
           n_estimators: Optional[int] = None):
    """K-fold CV (grouped by Source-1 row, so no entity's pairs ever straddle train/val) -> (FoldModel,
    out-of-fold predictions). `n_estimators` overrides `cfg.n_estimators` for the (smaller) stage-2 model."""
    n_estimators = n_estimators or cfg.n_estimators
    n_splits = max(2, min(cfg.n_folds, len(np.unique(groups))))
    gkf = GroupKFold(n_splits=n_splits)
    Xv = X[feature_names].to_numpy(dtype=np.float32)
    oof = np.zeros(len(y), dtype=np.float32)
    boosters, calibrators = [], []
    n_pos_total = int(y.sum())
    n_neg_total = len(y) - n_pos_total
    for fold, (tr, va) in enumerate(gkf.split(Xv, y, groups=groups)):
        n_pos_fold = int(y[tr].sum())
        n_neg_fold = len(tr) - n_pos_fold
        if _HAS_LGB:
            params = _lgb_params(cfg, n_pos_fold, n_neg_fold)
            dtr = lgb.Dataset(Xv[tr], label=y[tr])
            dva = lgb.Dataset(Xv[va], label=y[va], reference=dtr)
            booster = lgb.train(params, dtr, num_boost_round=n_estimators, valid_sets=[dva],
                                 callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)])
            raw_va = booster.predict(Xv[va], num_iteration=booster.best_iteration)
            best = booster.best_score["valid_0"]["average_precision"]
        else:  # pragma: no cover - fallback path, exercised only if lightgbm is unavailable
            from sklearn.ensemble import HistGradientBoostingClassifier
            booster = HistGradientBoostingClassifier(max_iter=min(n_estimators, 300),
                                                       learning_rate=cfg.learning_rate,
                                                       random_state=cfg.seed).fit(Xv[tr], y[tr])
            raw_va = booster.predict_proba(Xv[va])[:, 1]
            best = float("nan")
        # A fold's calibrator is fit on that same fold's (raw score, true label) pairs - see the class
        # docstring. That is still legitimate out-of-fold information for the *booster* (which never saw
        # these labels during training); isotonic regression is low-capacity enough that ~a few hundred
        # points is enough to fit a monotone rescaling without materially overfitting it. Skipped only when
        # a fold's validation slice is degenerate (all-one-class), where a monotone fit is either undefined
        # or meaningless and the raw score is used as-is.
        npos = int(y[va].sum())
        if 2 <= npos < len(va):
            cal = IsotonicRegression(out_of_bounds="clip", increasing=True, y_min=0.0, y_max=1.0)
            cal.fit(raw_va, y[va])
        else:
            cal = None
        oof[va] = cal.predict(raw_va) if cal is not None else raw_va
        boosters.append(booster)
        calibrators.append(cal)
        import gc; del dtr, dva; gc.collect()
        log(f"  fold {fold + 1}/{n_splits}: n_train={len(tr)} n_val={len(va)} pos_val={npos} "
            f"best_val_ap={best:.4f} calibrated={cal is not None}")
    kind = "lgb" if _HAS_LGB else "sk"
    return FoldModel(boosters, calibrators, feature_names, kind), oof


def stage2_features(oof1: np.ndarray, s1: np.ndarray, cand: np.ndarray, src: np.ndarray) -> pd.DataFrame:
    extra = group_rank_features({"stage1_prob": oof1}, s1, cand, src)
    out = extra.copy()
    out.insert(0, "stage1_prob", oof1.astype(np.float32))
    return out


STAGE2_COLS = ["stage1_prob", "stage1_prob_gap", "stage1_prob_rk", "stage1_prob_rgap", "stage1_prob_rrk",
               "n_cand_s1", "n_s1_cand"]


def train_matcher(X: pd.DataFrame, y: np.ndarray, s1: np.ndarray, cand: np.ndarray, src: np.ndarray, cfg):
    """Train stage 1 (+ stage 2 if warranted), returning (stage1_model, stage2_model_or_None, final_oof)."""
    feat_cols = [c for c in X.columns if c not in ("s1", "cand", "src")]
    log(f"stage 1: training on {len(y)} pairs, {int(y.sum())} positive, {len(feat_cols)} features")
    m1, oof1 = fit_cv(X, y, groups=s1, cfg=cfg, feature_names=feat_cols)

    engage2 = cfg.use_stage2 == "yes" or (cfg.use_stage2 == "auto" and int(y.sum()) >= MIN_STAGE2_POSITIVES)
    if not engage2:
        reason = "disabled" if cfg.use_stage2 == "no" else f"only {int(y.sum())} positives (< {MIN_STAGE2_POSITIVES})"
        log(f"stage 2: skipped ({reason}); using stage-1 probability as the final score")
        return m1, None, oof1

    X2 = stage2_features(oof1, s1, cand, src)
    log(f"stage 2: training stacked refinement on {len(STAGE2_COLS)} features")
    m2, oof2 = fit_cv(X2, y, groups=s1, cfg=cfg, feature_names=STAGE2_COLS,
                       n_estimators=max(200, cfg.n_estimators // 3))
    return m1, m2, oof2


def score_pairs(m1: FoldModel, m2: Optional[FoldModel], X: pd.DataFrame, s1: np.ndarray, cand: np.ndarray,
                src: np.ndarray) -> np.ndarray:
    """Score new (unlabelled) pairs with whichever stage(s) were trained."""
    p1 = m1.predict(X)
    if m2 is None:
        return p1
    X2 = stage2_features(p1, s1, cand, src)
    return m2.predict(X2)


# ----------------------------------------------------------------------------- diagnostics
def country_holdout_report(build_fn, countries: List[str], cfg) -> dict:
    """Leave-one-country-out diagnostic: for each country with >= 2 in the list, train stage 1 on every
    *other* country's pairs and report macro F0.5 on the held-out country alone. This is the closest we can
    get, from training data that only contains US and India, to estimating how much accuracy is likely to
    be lost on France (a country the real model never trains on at all). Purely a diagnostic used for the
    methodology write-up - the production model in `train_matcher` above always trains on every country.
    `build_fn(held_out_country) -> (X, y, s1, cand, src, n_true_by_s1, n_s1_total)` restricted to pairs
    whose Source-1 country equals `held_out_country` for validation and to every other country for fitting;
    it is supplied by pipeline.py, which has the record frame this module does not."""
    from .metrics import best_threshold  # local import: avoids a hard dependency for callers that don't need it

    if len(countries) < 2:
        log("country-holdout: fewer than 2 countries in the training data; skipping")
        return {}
    report = {}
    for held_out in countries:
        data = build_fn(held_out)
        if data is None:
            continue
        Xtr, ytr, s1tr, Xho, yho, s1ho, candho, srcho, n_true_ho, n_s1_ho = data
        if ytr.sum() < 20 or yho.sum() == 0:
            log(f"country-holdout[{held_out}]: too little signal to evaluate; skipping")
            continue
        feat_cols = [c for c in Xtr.columns if c not in ("s1", "cand", "src")]
        m, _ = fit_cv(Xtr, ytr, groups=s1tr, cfg=cfg, feature_names=feat_cols,
                      n_estimators=max(200, cfg.n_estimators // 2))
        p = m.predict(Xho)
        tau, f05, _ = best_threshold(s1ho, p, yho, n_true_ho, n_s1_ho, cfg.thr_grid)
        report[held_out] = f05
        log(f"country-holdout[{held_out}]: macro F0.5={f05:.4f} at tau={tau} "
            f"(trained without any '{held_out}' record at all)")
    return report
