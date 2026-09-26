from dataclasses import dataclass, field


def _grid():
    return tuple(round(0.2 + 0.05 * i, 2) for i in range(16))  # 0.20 ... 0.95


@dataclass
class Config:
    seed: int = 42
    n_jobs: int = -1

    # ---------------- blocking ----------------
    # top-k retrieved per S1 record, per target source, per TF-IDF view (before k_scale)
    base_k: dict = field(default_factory=lambda: {"nchar": 12, "nword": 10, "ncomb": 10, "aword": 6, "nphon": 6})
    kmax_factor: float = 3.0          # retrieval depth = base_k * kmax_factor (allows k_scale up to 3)
    country_penalty: float = 0.6      # cosine multiplier when both country labels are known and differ
    target_recall: float = 0.985      # k_scale is increased on the training data until this recall is reached
    k_scales: tuple = (1.0, 1.5, 2.0, 3.0)
    max_df_abs: int = 2000            # n-grams / tokens present in more docs than this are dropped from TF-IDF

    # ---------------- model ----------------
    n_folds: int = 5
    n_estimators: int = 1500
    learning_rate: float = 0.05
    num_leaves: int = 48
    min_child_samples: int = 30
    subsample: float = 0.8
    colsample_bytree: float = 0.7
    reg_lambda: float = 2.0
    early_stopping_rounds: int = 50
    stage2_conf_threshold: float = 0.5
    use_stage2: str = "auto"          # auto | yes | no
    unique_assign: str = "auto"       # auto | yes | no  (each S2/S3 record belongs to at most one S1 entity)
    thr_grid: tuple = field(default_factory=_grid)
    country_holdout: bool = True

    def fast(self):
        self.n_folds = 3
        self.n_estimators = 400
        self.learning_rate = 0.1
        return self
