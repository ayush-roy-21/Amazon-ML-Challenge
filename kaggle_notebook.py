"""
Amazon ML Challenge 2026 — Business Entity Resolution
Kaggle Notebook Version (single-file, self-contained)

Instructions:
1. Upload dataset files as a Kaggle Dataset
2. Copy this entire file into a Kaggle Notebook
3. Set Runtime → GPU T4 x2 (or at least 30GB RAM CPU)
4. Run All Cells

Output: matching_results.tsv + candidate_pairs.tsv (download from /kaggle/working/output/)
"""

# ═══════════════════════════════════════════════════════════
# CELL 1: Setup & Configuration
# ═══════════════════════════════════════════════════════════

import os
import re
import gc
import time
import unicodedata
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, fbeta_score, precision_recall_curve
from xgboost import XGBClassifier
from tqdm.auto import tqdm

# ── Paths (adjust if your dataset is named differently) ──
# If you uploaded as a Kaggle Dataset named "amazon-ml-challenge-2026":
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_OUTPUT = Path("/kaggle/working/output")
KAGGLE_OUTPUT.mkdir(parents=True, exist_ok=True)

# Auto-detect dataset path
possible_dirs = list(KAGGLE_INPUT.glob("*"))
print(f"Available datasets: {[d.name for d in possible_dirs]}")

# Try to find train/test files
DATASET_DIR = None
for d in possible_dirs:
    if (d / "dataset" / "train" / "train_source1.tsv").exists():
        DATASET_DIR = d / "dataset"
        break
    elif (d / "train" / "train_source1.tsv").exists():
        DATASET_DIR = d
        break
    elif (d / "train_source1.tsv").exists():
        DATASET_DIR = d
        break

if DATASET_DIR is None:
    # Fallback: search for train_source1.tsv anywhere
    for tsv in KAGGLE_INPUT.rglob("train_source1.tsv"):
        DATASET_DIR = tsv.parent
        break

if DATASET_DIR is None:
    raise FileNotFoundError(
        "Cannot find dataset! Make sure you uploaded your data as a Kaggle Dataset. "
        "Expected to find train_source1.tsv somewhere under /kaggle/input/"
    )

# Determine if files are in train/ and test/ subdirs or flat
if (DATASET_DIR / "train").exists():
    TRAIN_DIR = DATASET_DIR / "train"
    TEST_DIR = DATASET_DIR / "test"
else:
    TRAIN_DIR = DATASET_DIR
    TEST_DIR = DATASET_DIR

print(f"Dataset dir: {DATASET_DIR}")
print(f"Train dir: {TRAIN_DIR}")
print(f"Test dir: {TEST_DIR}")

# ── Config ──
TRAIN_SAMPLE_SIZE = 100_000   # Increase on Kaggle (more RAM available). Set 0 for full.
TOP_K_CANDIDATES = 30
NGRAM_RANGE = (2, 4)
MAX_FEATURES = 100_000
BATCH_SIZE = 1000             # Larger batches OK with 30GB RAM
MATCH_THRESHOLD = 0.65
RANDOM_SEED = 42

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.0,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",     # Fast on CPU; use "gpu_hist" if GPU available
}

print("✓ Configuration loaded")


# ═══════════════════════════════════════════════════════════
# CELL 2: Data Loading
# ═══════════════════════════════════════════════════════════

def load_tsv(path):
    """Load a TSV file."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    print(f"  Loaded {path.name}: {df.shape[0]:,} rows")
    return df

print("Loading data...")
train_s1 = load_tsv(TRAIN_DIR / "train_source1.tsv")
train_s2 = load_tsv(TRAIN_DIR / "train_source2.tsv")
train_s3 = load_tsv(TRAIN_DIR / "train_source3.tsv")
gt_df = load_tsv(TRAIN_DIR / "train_ground_truth.tsv")

test_s1 = load_tsv(TEST_DIR / "test_source1.tsv")
test_s2 = load_tsv(TEST_DIR / "test_source2.tsv")
test_s3 = load_tsv(TEST_DIR / "test_source3.tsv")
print("✓ All data loaded")


# ═══════════════════════════════════════════════════════════
# CELL 3: Parse Ground Truth
# ═══════════════════════════════════════════════════════════

def parse_ground_truth(gt):
    """Parse ground truth into {s1_id: set(matched_ids)}"""
    gt_dict = {}
    for _, row in gt.iterrows():
        s1_id = str(row.iloc[0]).strip()
        matched_str = str(row.iloc[1]).strip()
        if matched_str and matched_str != "nan":
            gt_dict[s1_id] = {m.strip() for m in matched_str.split(",") if m.strip()}
        else:
            gt_dict[s1_id] = set()
    return gt_dict

print("Parsing ground truth...")
gt_dict = parse_ground_truth(gt_df)
del gt_df
gc.collect()

n_matched = sum(1 for v in gt_dict.values() if v)
n_singleton = sum(1 for v in gt_dict.values() if not v)
print(f"  Matched: {n_matched:,}, Singletons: {n_singleton:,}")
print("✓ Ground truth parsed")


# ═══════════════════════════════════════════════════════════
# CELL 4: Preprocessing
# ═══════════════════════════════════════════════════════════

ABBREVIATIONS = {
    "corp": "corporation", "inc": "incorporated", "ltd": "limited",
    "llc": "limited liability company", "co": "company",
    "pvt": "private", "intl": "international",
    "svcs": "services", "svc": "service", "mfg": "manufacturing",
    "assoc": "associates", "dept": "department", "mgmt": "management",
    "grp": "group", "tech": "technology", "engg": "engineering",
    "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "rd": "road", "ln": "lane",
    "ste": "suite", "apt": "apartment", "fl": "floor",
    "ct": "court", "pl": "place", "hwy": "highway",
    "nr": "near", "opp": "opposite", "dist": "district",
}

def clean_text(text):
    """Clean text preserving multilingual characters (Hindi, French)."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r'[<>"\'`]', " ", text)
    text = re.sub(r"[^\w\s\-.]", " ", text, flags=re.UNICODE)
    text = re.sub(r"(?<!\w)[.\-](?!\w)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def expand_abbr(text):
    if not text:
        return text
    return " ".join(ABBREVIATIONS.get(t, t) for t in text.split())

def preprocess(df):
    """Preprocess a source DataFrame."""
    df = df.copy()
    for col in ["business_name", "business_address", "country"]:
        if col in df.columns:
            c = f"{col}_clean"
            df[c] = df[col].fillna("").apply(clean_text)
            if col in ("business_name", "business_address"):
                df[c] = df[c].apply(expand_abbr)
    # Text blob: name (2x weight) + address + country
    nc = df.get("business_name_clean", pd.Series(""))
    ac = df.get("business_address_clean", pd.Series(""))
    cc = df.get("country_clean", pd.Series(""))
    df["text_blob"] = (nc + " " + nc + " " + ac + " " + cc).str.strip()
    return df

print("Preprocessing all sources...")
t0 = time.time()
train_s1 = preprocess(train_s1)
train_s2 = preprocess(train_s2)
train_s3 = preprocess(train_s3)
test_s1 = preprocess(test_s1)
test_s2 = preprocess(test_s2)
test_s3 = preprocess(test_s3)
print(f"✓ Preprocessing done in {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════
# CELL 5: Sampling for Training
# ═══════════════════════════════════════════════════════════

def sample_and_filter(s1, s2, s3, gt, sample_size):
    """Sample S1 records and filter S2/S3 to manageable size."""
    if sample_size <= 0 or sample_size >= len(s1):
        return s1, s2, s3, gt

    matched_ids = [sid for sid, mids in gt.items() if mids]
    singleton_ids = [sid for sid, mids in gt.items() if not mids]

    np.random.seed(RANDOM_SEED)
    n_m = int(sample_size * len(matched_ids) / len(gt))
    n_s = sample_size - n_m

    sampled = set(np.random.choice(matched_ids, min(n_m, len(matched_ids)), replace=False))
    sampled |= set(np.random.choice(singleton_ids, min(n_s, len(singleton_ids)), replace=False))

    s1_out = s1[s1["entity_id"].isin(sampled)].copy()
    gt_out = {k: v for k, v in gt.items() if k in sampled}

    # Filter S2/S3: keep GT-referenced records + random negatives
    for src, prefix in [(s2, "S2-"), (s3, "S3-")]:
        gt_ids = {m for mids in gt_out.values() for m in mids if m.startswith(prefix)}
        gt_mask = src["entity_id"].isin(gt_ids)
        other = src[~gt_mask]
        n_rand = min(len(gt_ids) * 5, len(other))
        rand_sample = other.sample(n=n_rand, random_state=RANDOM_SEED) if n_rand > 0 else other.iloc[:0]
        if prefix == "S2-":
            s2_out = pd.concat([src[gt_mask], rand_sample], ignore_index=True)
        else:
            s3_out = pd.concat([src[gt_mask], rand_sample], ignore_index=True)

    print(f"  Sampled S1: {len(s1_out):,}, S2: {len(s2_out):,}, S3: {len(s3_out):,}")
    return s1_out, s2_out, s3_out, gt_out

if TRAIN_SAMPLE_SIZE > 0:
    print(f"Sampling {TRAIN_SAMPLE_SIZE:,} S1 records for training...")
    tr_s1, tr_s2, tr_s3, tr_gt = sample_and_filter(
        train_s1, train_s2, train_s3, gt_dict, TRAIN_SAMPLE_SIZE
    )
else:
    tr_s1, tr_s2, tr_s3, tr_gt = train_s1, train_s2, train_s3, gt_dict

print("✓ Sampling done")
gc.collect()


# ═══════════════════════════════════════════════════════════
# CELL 6: Blocking (TF-IDF Candidate Generation)
# ═══════════════════════════════════════════════════════════

def block_by_country(s1_df, so_df, top_k=TOP_K_CANDIDATES, batch_size=BATCH_SIZE):
    """Country-aware TF-IDF blocking."""
    s1_countries = set(s1_df["country"].fillna("unknown").unique())
    so_countries = set(so_df["country"].fillna("unknown").unique())
    countries = sorted(s1_countries & so_countries)

    all_pairs = []
    for country in countries:
        s1_c = s1_df[s1_df["country"].fillna("unknown") == country]
        so_c = so_df[so_df["country"].fillna("unknown") == country]
        if len(s1_c) == 0 or len(so_c) == 0:
            continue

        print(f"    {country}: {len(s1_c):,} × {len(so_c):,}")

        # Build TF-IDF
        combined = pd.concat([s1_c["text_blob"], so_c["text_blob"]], ignore_index=True)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE,
                              max_features=MAX_FEATURES, sublinear_tf=True, dtype=np.float32)
        vec.fit(combined.fillna(""))
        s1_mat = vec.transform(s1_c["text_blob"].fillna(""))
        so_mat = vec.transform(so_c["text_blob"].fillna(""))

        s1_ids = s1_c["entity_id"].values
        so_ids = so_c["entity_id"].values

        for start in tqdm(range(0, len(s1_ids), batch_size),
                          desc=f"      {country}", leave=False):
            end = min(start + batch_size, len(s1_ids))
            sims = cosine_similarity(s1_mat[start:end], so_mat)
            for i, row_sims in enumerate(sims):
                k = min(top_k, len(row_sims))
                if k == 0:
                    continue
                top_idx = np.argpartition(row_sims, -k)[-k:]
                for idx in top_idx:
                    if row_sims[idx] > 0:
                        all_pairs.append((s1_ids[start+i], so_ids[idx], float(row_sims[idx])))

        del vec, s1_mat, so_mat
        gc.collect()
        print(f"    {country}: {sum(1 for p in all_pairs if True):,} pairs so far")

    return pd.DataFrame(all_pairs, columns=["source1_entity_id", "candidate_entity_id", "tfidf_score"])

print("\n=== BLOCKING (Training) ===")
t0 = time.time()
print("  S1 ↔ S2:")
train_cands_s2 = block_by_country(tr_s1, tr_s2)
print(f"  S2 candidates: {len(train_cands_s2):,}")

print("  S1 ↔ S3:")
train_cands_s3 = block_by_country(tr_s1, tr_s3)
print(f"  S3 candidates: {len(train_cands_s3):,}")
print(f"✓ Blocking done in {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════
# CELL 7: Feature Engineering
# ═══════════════════════════════════════════════════════════

try:
    from rapidfuzz import fuzz
except ImportError:
    os.system("pip install rapidfuzz -q")
    from rapidfuzz import fuzz

def jaccard(s1, s2):
    if not s1 or not s2: return 0.0
    a, b = set(s1.split()), set(s2.split())
    return len(a & b) / len(a | b) if a | b else 0.0

def containment(s1, s2):
    if not s1 or not s2: return 0.0
    a, b = set(s1.split()), set(s2.split())
    return len(a & b) / len(a) if a else 0.0

def num_overlap(s1, s2):
    n1 = set(re.findall(r"\d+", s1))
    n2 = set(re.findall(r"\d+", s2))
    if not n1 or not n2: return 0.0
    return len(n1 & n2) / max(len(n1), len(n2))

def char_ngram_overlap(s1, s2, n=3):
    if not s1 or not s2 or len(s1) < n or len(s2) < n: return 0.0
    a = set(s1[i:i+n] for i in range(len(s1)-n+1))
    b = set(s2[i:i+n] for i in range(len(s2)-n+1))
    return len(a & b) / len(a | b) if a | b else 0.0

def compute_features(row1, row2):
    """Compute pairwise features for a candidate pair."""
    f = {}
    n1 = str(row1.get("business_name_clean", "")).strip()
    n2 = str(row2.get("business_name_clean", "")).strip()
    a1 = str(row1.get("business_address_clean", "")).strip()
    a2 = str(row2.get("business_address_clean", "")).strip()
    c1 = str(row1.get("country_clean", "")).strip()
    c2 = str(row2.get("country_clean", "")).strip()

    # Name features
    f["name_ratio"] = fuzz.ratio(n1, n2) / 100
    f["name_partial"] = fuzz.partial_ratio(n1, n2) / 100
    f["name_tsort"] = fuzz.token_sort_ratio(n1, n2) / 100
    f["name_tset"] = fuzz.token_set_ratio(n1, n2) / 100
    f["name_jaccard"] = jaccard(n1, n2)
    f["name_contain"] = containment(n1, n2)
    f["name_ngram3"] = char_ngram_overlap(n1, n2, 3)

    # Address features
    f["addr_ratio"] = fuzz.ratio(a1, a2) / 100
    f["addr_partial"] = fuzz.partial_ratio(a1, a2) / 100
    f["addr_tsort"] = fuzz.token_sort_ratio(a1, a2) / 100
    f["addr_tset"] = fuzz.token_set_ratio(a1, a2) / 100
    f["addr_jaccard"] = jaccard(a1, a2)
    f["addr_numovlp"] = num_overlap(a1, a2)
    f["addr_ngram3"] = char_ngram_overlap(a1, a2, 3)
    f["addr_contain"] = containment(a1, a2)

    # Country
    f["country_match"] = 1.0 if c1 and c2 and c1 == c2 else 0.0

    # Combined
    cb1, cb2 = f"{n1} {a1}", f"{n2} {a2}"
    f["comb_jaccard"] = jaccard(cb1, cb2)
    f["comb_tsort"] = fuzz.token_sort_ratio(cb1, cb2) / 100

    # Length
    f["name_len_diff"] = abs(len(n1) - len(n2))
    f["addr_len_diff"] = abs(len(a1) - len(a2))
    f["name_len_ratio"] = min(len(n1), len(n2)) / max(len(n1), len(n2)) if max(len(n1), len(n2)) > 0 else 0
    f["addr_len_ratio"] = min(len(a1), len(a2)) / max(len(a1), len(a2)) if max(len(a1), len(a2)) > 0 else 0
    f["name_tok_diff"] = abs(len(n1.split()) - len(n2.split()))
    f["addr_tok_diff"] = abs(len(a1.split()) - len(a2.split()))

    return f

def build_features(cands_df, s1_df, so_df):
    """Build feature matrix for all candidate pairs."""
    s1_idx = s1_df.set_index("entity_id")
    so_idx = so_df.set_index("entity_id")

    rows = []
    for _, c in tqdm(cands_df.iterrows(), total=len(cands_df), desc="  Features"):
        s1_id, c_id = c["source1_entity_id"], c["candidate_entity_id"]
        if s1_id not in s1_idx.index or c_id not in so_idx.index:
            continue
        feats = compute_features(s1_idx.loc[s1_id], so_idx.loc[c_id])
        feats["source1_entity_id"] = s1_id
        feats["candidate_entity_id"] = c_id
        feats["tfidf_score"] = c.get("tfidf_score", 0)
        rows.append(feats)

    return pd.DataFrame(rows)

print("\n=== FEATURE ENGINEERING ===")
t0 = time.time()

print("S1↔S2 features...")
feat_s2 = build_features(train_cands_s2, tr_s1, tr_s2)
print(f"  S2 features: {feat_s2.shape}")

print("S1↔S3 features...")
feat_s3 = build_features(train_cands_s3, tr_s1, tr_s3)
print(f"  S3 features: {feat_s3.shape}")
print(f"✓ Features done in {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════
# CELL 8: Create Labels & Train
# ═══════════════════════════════════════════════════════════

def create_labels(cands_df, gt, prefix):
    """Create binary labels from ground truth."""
    gt_by_source = {}
    for s1, mids in gt.items():
        gt_by_source[s1] = {m for m in mids if m.startswith(prefix)}

    labels = []
    for _, c in cands_df.iterrows():
        s1_id = str(c["source1_entity_id"]).strip()
        c_id = str(c["candidate_entity_id"]).strip()
        labels.append(1 if c_id in gt_by_source.get(s1_id, set()) else 0)
    return pd.Series(labels, name="label")

print("\n=== TRAINING ===")

# Labels
feat_s2["label"] = create_labels(feat_s2, tr_gt, "S2-").values
feat_s3["label"] = create_labels(feat_s3, tr_gt, "S3-").values

combined = pd.concat([feat_s2, feat_s3], ignore_index=True)
exclude = {"source1_entity_id", "candidate_entity_id", "label"}
feature_cols = [c for c in combined.columns if c not in exclude]

X = combined[feature_cols]
y = combined["label"]

n_pos = y.sum()
n_neg = len(y) - n_pos
print(f"  Training set: {len(y):,} pairs ({n_pos:,} positive, {n_neg:,} negative)")

# Adjust class weight
params = XGBOOST_PARAMS.copy()
if n_pos > 0:
    params["scale_pos_weight"] = n_neg / n_pos

clf = XGBClassifier(**params)
print("  Training XGBoost...")
clf.fit(X, y, verbose=50)

# Find best threshold for F0.5
probs = clf.predict_proba(X)[:, 1]
precision, recall, thresholds = precision_recall_curve(y, probs)
best_f05, best_thresh = 0, 0.5
for p, r, t in zip(precision, recall, thresholds):
    if p + r > 0:
        f05 = 1.25 * p * r / (0.25 * p + r)
        if f05 > best_f05:
            best_f05, best_thresh = f05, t

print(f"  Best threshold: {best_thresh:.4f} (F0.5={best_f05:.4f})")
MATCH_THRESHOLD = float(best_thresh)

# Feature importance
fi = pd.DataFrame({"feature": feature_cols, "importance": clf.feature_importances_})
fi = fi.sort_values("importance", ascending=False)
print("\nTop 10 features:")
print(fi.head(10).to_string(index=False))

# Save model
joblib.dump(clf, KAGGLE_OUTPUT / "model.pkl")
joblib.dump(feature_cols, KAGGLE_OUTPUT / "feature_cols.pkl")
print("✓ Model trained and saved")

# Free memory
del combined, X, y, feat_s2, feat_s3, train_cands_s2, train_cands_s3
del tr_s1, tr_s2, tr_s3
gc.collect()


# ═══════════════════════════════════════════════════════════
# CELL 9: Predict on Test Data
# ═══════════════════════════════════════════════════════════

print("\n=== PREDICTION ON TEST DATA ===")
t0 = time.time()

print("Blocking test S1↔S2...")
test_cands_s2 = block_by_country(test_s1, test_s2)
print(f"  Test S2 candidates: {len(test_cands_s2):,}")

print("Blocking test S1↔S3...")
test_cands_s3 = block_by_country(test_s1, test_s3)
print(f"  Test S3 candidates: {len(test_cands_s3):,}")

print("Computing test S2 features...")
test_feat_s2 = build_features(test_cands_s2, test_s1, test_s2)

print("Computing test S3 features...")
test_feat_s3 = build_features(test_cands_s3, test_s1, test_s3)

# Predict
def get_matches(feat_df, threshold):
    if len(feat_df) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "match_prob"])
    probs = clf.predict_proba(feat_df[feature_cols])[:, 1]
    feat_df = feat_df.copy()
    feat_df["match_prob"] = probs
    matched = feat_df[feat_df["match_prob"] >= threshold]
    print(f"  Matches: {len(matched):,} / {len(feat_df):,} (threshold={threshold:.3f})")
    return matched

matches_s2 = get_matches(test_feat_s2, MATCH_THRESHOLD)
matches_s3 = get_matches(test_feat_s3, MATCH_THRESHOLD)
print(f"✓ Prediction done in {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════
# CELL 10: Build & Save Submission
# ═══════════════════════════════════════════════════════════

print("\n=== BUILDING SUBMISSION ===")

all_s1_ids = test_s1["entity_id"].unique()

def build_output(matches_list, cands_list, col_name):
    """Build {s1_id: set(matched/candidate ids)} from DataFrames."""
    records = {s1: set() for s1 in all_s1_ids}
    for df in matches_list:
        if df is not None and len(df) > 0:
            for _, row in df.iterrows():
                s1_id = row["source1_entity_id"]
                c_id = row["candidate_entity_id"]
                if s1_id in records:
                    records[s1_id].add(c_id)
    rows = []
    for s1_id in sorted(records.keys()):
        rows.append({
            "source1_entity_id": s1_id,
            col_name: ",".join(sorted(records[s1_id])),
        })
    return pd.DataFrame(rows)

# matching_results.tsv
matching = build_output([matches_s2, matches_s3], None, "matched_entity_ids")
matching.to_csv(KAGGLE_OUTPUT / "matching_results.tsv", sep="\t", index=False)
n_matched = (matching["matched_entity_ids"] != "").sum()
print(f"  matching_results.tsv: {len(matching):,} rows, {n_matched:,} with matches")

# candidate_pairs.tsv
candidates = build_output([test_cands_s2, test_cands_s3], None, "candidate_entity_ids")
candidates.to_csv(KAGGLE_OUTPUT / "candidate_pairs.tsv", sep="\t", index=False)
print(f"  candidate_pairs.tsv: {len(candidates):,} rows")

print(f"\n✓ Submission files saved to {KAGGLE_OUTPUT}/")
print("  Download matching_results.tsv and upload to the competition portal!")


# ═══════════════════════════════════════════════════════════
# CELL 11: Quick Validation
# ═══════════════════════════════════════════════════════════

print("\n=== VALIDATION ===")

# Check format
errors = []
if len(matching) != len(all_s1_ids):
    errors.append(f"Row count mismatch: {len(matching)} vs {len(all_s1_ids)} S1 IDs")

dupes = matching["source1_entity_id"].duplicated().sum()
if dupes > 0:
    errors.append(f"{dupes} duplicate source1_entity_ids")

# Check IDs
for _, row in matching.head(1000).iterrows():
    mids = str(row["matched_entity_ids"]).strip()
    if mids and mids != "nan":
        for mid in mids.split(","):
            if not mid.startswith(("S2-", "S3-")):
                errors.append(f"Invalid ID prefix: {mid}")
                break

if errors:
    print("  ✗ VALIDATION FAILED:")
    for e in errors:
        print(f"    - {e}")
else:
    print(f"  ✓ PASS — {len(matching):,} rows, {n_matched:,} matched, "
          f"{len(matching)-n_matched:,} singletons")

print("\n✓ ALL DONE!")
