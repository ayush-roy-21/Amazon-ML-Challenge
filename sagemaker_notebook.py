"""
Amazon ML Challenge 2026 — Business Entity Resolution
AWS SageMaker Notebook Version

Instance recommendation: ml.m5.4xlarge (64 GB RAM, 16 vCPUs) — ~$1/hr
Expected runtime: ~6-8 hrs for full data, ~1 hr for 100K sample

Setup:
1. Upload dataset to S3
2. Create SageMaker Notebook Instance (ml.m5.4xlarge)
3. Upload this file as a notebook
4. Run all cells
5. Download output from S3
"""

# ═══════════════════════════════════════════════════════════
# CELL 1: Install dependencies
# ═══════════════════════════════════════════════════════════

import subprocess
subprocess.check_call(["pip", "install", "-q", "rapidfuzz", "xgboost", "tqdm"])

import os
import re
import gc
import time
import unicodedata
import numpy as np
import pandas as pd
import joblib
import boto3
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, fbeta_score, precision_recall_curve
from xgboost import XGBClassifier
from rapidfuzz import fuzz
from tqdm.auto import tqdm

print("✓ All packages loaded")


# ═══════════════════════════════════════════════════════════
# CELL 2: Configuration
# ═══════════════════════════════════════════════════════════

# ── S3 bucket (change to your bucket) ──
S3_BUCKET = "your-bucket-name"          # ← CHANGE THIS
S3_PREFIX = "amazon-ml-challenge"       # ← folder in bucket

# ── OR local paths if you uploaded directly to the notebook instance ──
# If you uploaded files to SageMaker notebook storage, set USE_S3 = False
USE_S3 = False  # Set True if data is in S3, False if uploaded to notebook

if USE_S3:
    LOCAL_DATA = Path("/tmp/data")
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)
else:
    # Files should be in /home/ec2-user/SageMaker/data/
    LOCAL_DATA = Path("/home/ec2-user/SageMaker/data")

OUTPUT_DIR = Path("/home/ec2-user/SageMaker/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Pipeline config ──
TRAIN_SAMPLE_SIZE = 0         # 0 = FULL DATA (SageMaker has enough RAM!)
TOP_K_CANDIDATES = 30
NGRAM_RANGE = (2, 4)
MAX_FEATURES = 100_000
BATCH_SIZE = 1000
MATCH_THRESHOLD = 0.65
RANDOM_SEED = 42

XGBOOST_PARAMS = {
    "n_estimators": 600,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.0,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,              # Use all 16 vCPUs
    "tree_method": "hist",
}

print("✓ Configuration set")
print(f"  Sample size: {'FULL DATA' if TRAIN_SAMPLE_SIZE == 0 else f'{TRAIN_SAMPLE_SIZE:,}'}")
print(f"  Data path: {LOCAL_DATA}")
print(f"  Output: {OUTPUT_DIR}")


# ═══════════════════════════════════════════════════════════
# CELL 3: Download data from S3 (skip if USE_S3=False)
# ═══════════════════════════════════════════════════════════

if USE_S3:
    s3 = boto3.client("s3")
    files = [
        "dataset/train/train_source1.tsv",
        "dataset/train/train_source2.tsv",
        "dataset/train/train_source3.tsv",
        "dataset/train/train_ground_truth.tsv",
        "dataset/test/test_source1.tsv",
        "dataset/test/test_source2.tsv",
        "dataset/test/test_source3.tsv",
    ]
    for f in files:
        local_path = LOCAL_DATA / f
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            s3_key = f"{S3_PREFIX}/{f}"
            print(f"  Downloading s3://{S3_BUCKET}/{s3_key}...")
            s3.download_file(S3_BUCKET, s3_key, str(local_path))
        else:
            print(f"  Already exists: {local_path.name}")
    TRAIN_DIR = LOCAL_DATA / "dataset" / "train"
    TEST_DIR = LOCAL_DATA / "dataset" / "test"
    print("✓ Data downloaded from S3")
else:
    # Auto-detect directory structure
    if (LOCAL_DATA / "dataset" / "train").exists():
        TRAIN_DIR = LOCAL_DATA / "dataset" / "train"
        TEST_DIR = LOCAL_DATA / "dataset" / "test"
    elif (LOCAL_DATA / "train").exists():
        TRAIN_DIR = LOCAL_DATA / "train"
        TEST_DIR = LOCAL_DATA / "test"
    else:
        TRAIN_DIR = LOCAL_DATA
        TEST_DIR = LOCAL_DATA
    print(f"  Train dir: {TRAIN_DIR}")
    print(f"  Test dir: {TEST_DIR}")

# Verify files exist
for f in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"]:
    assert (TRAIN_DIR / f).exists(), f"Missing: {TRAIN_DIR / f}"
for f in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
    assert (TEST_DIR / f).exists(), f"Missing: {TEST_DIR / f}"
print("✓ All data files verified")


# ═══════════════════════════════════════════════════════════
# CELL 4: Load Data
# ═══════════════════════════════════════════════════════════

def load_tsv(path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    print(f"  {path.name}: {df.shape[0]:,} rows")
    return df

print("Loading data...")
t0 = time.time()
train_s1 = load_tsv(TRAIN_DIR / "train_source1.tsv")
train_s2 = load_tsv(TRAIN_DIR / "train_source2.tsv")
train_s3 = load_tsv(TRAIN_DIR / "train_source3.tsv")
gt_df = load_tsv(TRAIN_DIR / "train_ground_truth.tsv")
test_s1 = load_tsv(TEST_DIR / "test_source1.tsv")
test_s2 = load_tsv(TEST_DIR / "test_source2.tsv")
test_s3 = load_tsv(TEST_DIR / "test_source3.tsv")
print(f"✓ All data loaded in {time.time()-t0:.0f}s")
print(f"  Memory: {sum(df.memory_usage(deep=True).sum() for df in [train_s1,train_s2,train_s3,gt_df,test_s1,test_s2,test_s3]) / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════
# CELL 5: Parse Ground Truth
# ═══════════════════════════════════════════════════════════

def parse_ground_truth(gt):
    gt_dict = {}
    for _, row in tqdm(gt.iterrows(), total=len(gt), desc="  Parsing GT"):
        s1_id = str(row.iloc[0]).strip()
        matched_str = str(row.iloc[1]).strip()
        if matched_str and matched_str != "nan":
            gt_dict[s1_id] = {m.strip() for m in matched_str.split(",") if m.strip()}
        else:
            gt_dict[s1_id] = set()
    return gt_dict

print("Parsing ground truth...")
gt_dict = parse_ground_truth(gt_df)
del gt_df; gc.collect()

n_matched = sum(1 for v in gt_dict.values() if v)
print(f"  Matched: {n_matched:,}, Singletons: {len(gt_dict)-n_matched:,}")


# ═══════════════════════════════════════════════════════════
# CELL 6: Preprocessing
# ═══════════════════════════════════════════════════════════

ABBR = {
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
    if not isinstance(text, str) or not text.strip(): return ""
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r'[<>"\'`]', " ", text)
    text = re.sub(r"[^\w\s\-.]", " ", text, flags=re.UNICODE)
    text = re.sub(r"(?<!\w)[.\-](?!\w)", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def expand_abbr(text):
    if not text: return text
    return " ".join(ABBR.get(t, t) for t in text.split())

def preprocess(df):
    df = df.copy()
    for col in ["business_name", "business_address", "country"]:
        if col in df.columns:
            c = f"{col}_clean"
            df[c] = df[col].fillna("").apply(clean_text)
            if col in ("business_name", "business_address"):
                df[c] = df[c].apply(expand_abbr)
    nc = df.get("business_name_clean", pd.Series("", index=df.index))
    ac = df.get("business_address_clean", pd.Series("", index=df.index))
    cc = df.get("country_clean", pd.Series("", index=df.index))
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
# CELL 7: Sampling (optional — set TRAIN_SAMPLE_SIZE=0 for full)
# ═══════════════════════════════════════════════════════════

def sample_and_filter(s1, s2, s3, gt, sample_size):
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

    s2_out, s3_out = s2, s3
    for src_df, prefix, name in [(s2, "S2-", "S2"), (s3, "S3-", "S3")]:
        gt_ids = {m for mids in gt_out.values() for m in mids if m.startswith(prefix)}
        gt_mask = src_df["entity_id"].isin(gt_ids)
        other = src_df[~gt_mask]
        n_rand = min(len(gt_ids) * 5, len(other))
        rand_sample = other.sample(n=n_rand, random_state=RANDOM_SEED) if n_rand > 0 else other.iloc[:0]
        filtered = pd.concat([src_df[gt_mask], rand_sample], ignore_index=True)
        if prefix == "S2-":
            s2_out = filtered
        else:
            s3_out = filtered
        print(f"  Filtered {name}: {len(filtered):,}")

    print(f"  Sampled S1: {len(s1_out):,}")
    return s1_out, s2_out, s3_out, gt_out

if TRAIN_SAMPLE_SIZE > 0:
    print(f"Sampling {TRAIN_SAMPLE_SIZE:,} records...")
    tr_s1, tr_s2, tr_s3, tr_gt = sample_and_filter(train_s1, train_s2, train_s3, gt_dict, TRAIN_SAMPLE_SIZE)
else:
    print("Using FULL training data (no sampling)")
    tr_s1, tr_s2, tr_s3, tr_gt = train_s1, train_s2, train_s3, gt_dict

gc.collect()
print("✓ Ready for blocking")


# ═══════════════════════════════════════════════════════════
# CELL 8: Blocking (Country-Aware TF-IDF)
# ═══════════════════════════════════════════════════════════

def block_by_country(s1_df, so_df, top_k=TOP_K_CANDIDATES, batch_size=BATCH_SIZE, label=""):
    s1_countries = set(s1_df["country"].fillna("unknown").unique())
    so_countries = set(so_df["country"].fillna("unknown").unique())
    countries = sorted(s1_countries & so_countries)
    all_pairs = []

    for country in countries:
        s1_c = s1_df[s1_df["country"].fillna("unknown") == country]
        so_c = so_df[so_df["country"].fillna("unknown") == country]
        if len(s1_c) == 0 or len(so_c) == 0: continue
        print(f"    {label}/{country}: {len(s1_c):,} × {len(so_c):,}")

        combined = pd.concat([s1_c["text_blob"], so_c["text_blob"]], ignore_index=True)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE,
                              max_features=MAX_FEATURES, sublinear_tf=True, dtype=np.float32)
        vec.fit(combined.fillna(""))
        s1_mat = vec.transform(s1_c["text_blob"].fillna(""))
        so_mat = vec.transform(so_c["text_blob"].fillna(""))
        s1_ids = s1_c["entity_id"].values
        so_ids = so_c["entity_id"].values

        for start in tqdm(range(0, len(s1_ids), batch_size), desc=f"      {country}", leave=False):
            end = min(start + batch_size, len(s1_ids))
            sims = cosine_similarity(s1_mat[start:end], so_mat)
            for i, row_sims in enumerate(sims):
                k = min(top_k, len(row_sims))
                if k == 0: continue
                top_idx = np.argpartition(row_sims, -k)[-k:]
                for idx in top_idx:
                    if row_sims[idx] > 0:
                        all_pairs.append((s1_ids[start+i], so_ids[idx], float(row_sims[idx])))

        del vec, s1_mat, so_mat; gc.collect()
        print(f"    {country}: done ({len(all_pairs):,} pairs total)")

    return pd.DataFrame(all_pairs, columns=["source1_entity_id", "candidate_entity_id", "tfidf_score"])

print("\n=== BLOCKING (Training) ===")
t0 = time.time()
train_cands_s2 = block_by_country(tr_s1, tr_s2, label="S2")
print(f"  S2 candidates: {len(train_cands_s2):,}")
train_cands_s3 = block_by_country(tr_s1, tr_s3, label="S3")
print(f"  S3 candidates: {len(train_cands_s3):,}")
print(f"✓ Blocking done in {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════
# CELL 9: Feature Engineering
# ═══════════════════════════════════════════════════════════

def jaccard(s1, s2):
    if not s1 or not s2: return 0.0
    a, b = set(s1.split()), set(s2.split())
    return len(a & b) / len(a | b) if a | b else 0.0

def containment(s1, s2):
    if not s1 or not s2: return 0.0
    a, b = set(s1.split()), set(s2.split())
    return len(a & b) / len(a) if a else 0.0

def num_overlap(s1, s2):
    n1, n2 = set(re.findall(r"\d+", s1)), set(re.findall(r"\d+", s2))
    if not n1 or not n2: return 0.0
    return len(n1 & n2) / max(len(n1), len(n2))

def char_ngram(s1, s2, n=3):
    if not s1 or not s2 or len(s1)<n or len(s2)<n: return 0.0
    a = set(s1[i:i+n] for i in range(len(s1)-n+1))
    b = set(s2[i:i+n] for i in range(len(s2)-n+1))
    return len(a & b) / len(a | b) if a | b else 0.0

def compute_features(r1, r2):
    f = {}
    n1 = str(r1.get("business_name_clean","")).strip()
    n2 = str(r2.get("business_name_clean","")).strip()
    a1 = str(r1.get("business_address_clean","")).strip()
    a2 = str(r2.get("business_address_clean","")).strip()
    c1 = str(r1.get("country_clean","")).strip()
    c2 = str(r2.get("country_clean","")).strip()

    f["name_ratio"] = fuzz.ratio(n1,n2)/100
    f["name_partial"] = fuzz.partial_ratio(n1,n2)/100
    f["name_tsort"] = fuzz.token_sort_ratio(n1,n2)/100
    f["name_tset"] = fuzz.token_set_ratio(n1,n2)/100
    f["name_jaccard"] = jaccard(n1,n2)
    f["name_contain"] = containment(n1,n2)
    f["name_ngram3"] = char_ngram(n1,n2,3)

    f["addr_ratio"] = fuzz.ratio(a1,a2)/100
    f["addr_partial"] = fuzz.partial_ratio(a1,a2)/100
    f["addr_tsort"] = fuzz.token_sort_ratio(a1,a2)/100
    f["addr_tset"] = fuzz.token_set_ratio(a1,a2)/100
    f["addr_jaccard"] = jaccard(a1,a2)
    f["addr_numovlp"] = num_overlap(a1,a2)
    f["addr_ngram3"] = char_ngram(a1,a2,3)
    f["addr_contain"] = containment(a1,a2)

    f["country_match"] = 1.0 if c1 and c2 and c1==c2 else 0.0
    cb1, cb2 = f"{n1} {a1}", f"{n2} {a2}"
    f["comb_jaccard"] = jaccard(cb1,cb2)
    f["comb_tsort"] = fuzz.token_sort_ratio(cb1,cb2)/100

    f["name_len_diff"] = abs(len(n1)-len(n2))
    f["addr_len_diff"] = abs(len(a1)-len(a2))
    f["name_len_ratio"] = min(len(n1),len(n2))/max(len(n1),len(n2)) if max(len(n1),len(n2))>0 else 0
    f["addr_len_ratio"] = min(len(a1),len(a2))/max(len(a1),len(a2)) if max(len(a1),len(a2))>0 else 0
    f["name_tok_diff"] = abs(len(n1.split())-len(n2.split()))
    f["addr_tok_diff"] = abs(len(a1.split())-len(a2.split()))
    return f

def build_features(cands_df, s1_df, so_df):
    s1_idx = s1_df.set_index("entity_id")
    so_idx = so_df.set_index("entity_id")
    rows = []
    for _, c in tqdm(cands_df.iterrows(), total=len(cands_df), desc="  Features"):
        s1_id, c_id = c["source1_entity_id"], c["candidate_entity_id"]
        if s1_id not in s1_idx.index or c_id not in so_idx.index: continue
        feats = compute_features(s1_idx.loc[s1_id], so_idx.loc[c_id])
        feats["source1_entity_id"] = s1_id
        feats["candidate_entity_id"] = c_id
        feats["tfidf_score"] = c.get("tfidf_score", 0)
        rows.append(feats)
    return pd.DataFrame(rows)

print("\n=== FEATURE ENGINEERING ===")
t0 = time.time()
feat_s2 = build_features(train_cands_s2, tr_s1, tr_s2)
feat_s3 = build_features(train_cands_s3, tr_s1, tr_s3)
print(f"  S2: {feat_s2.shape}, S3: {feat_s3.shape}")
print(f"✓ Features done in {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════
# CELL 10: Create Labels & Train XGBoost
# ═══════════════════════════════════════════════════════════

def create_labels(cands_df, gt, prefix):
    gt_src = {s1: {m for m in mids if m.startswith(prefix)} for s1, mids in gt.items()}
    return pd.Series([
        1 if str(c["candidate_entity_id"]).strip() in gt_src.get(str(c["source1_entity_id"]).strip(), set()) else 0
        for _, c in cands_df.iterrows()
    ], name="label")

print("\n=== TRAINING ===")
feat_s2["label"] = create_labels(feat_s2, tr_gt, "S2-").values
feat_s3["label"] = create_labels(feat_s3, tr_gt, "S3-").values

combined = pd.concat([feat_s2, feat_s3], ignore_index=True)
exclude = {"source1_entity_id", "candidate_entity_id", "label"}
feature_cols = [c for c in combined.columns if c not in exclude]

X, y = combined[feature_cols], combined["label"]
n_pos, n_neg = y.sum(), len(y) - y.sum()
print(f"  {len(y):,} pairs ({n_pos:,} pos, {n_neg:,} neg)")

params = XGBOOST_PARAMS.copy()
if n_pos > 0: params["scale_pos_weight"] = n_neg / n_pos

clf = XGBClassifier(**params)
clf.fit(X, y, verbose=100)

# Best F0.5 threshold
probs = clf.predict_proba(X)[:, 1]
prec, rec, thresh = precision_recall_curve(y, probs)
best_f05, best_thresh = 0, 0.5
for p, r, t in zip(prec, rec, thresh):
    if p+r > 0:
        f05 = 1.25*p*r / (0.25*p + r)
        if f05 > best_f05: best_f05, best_thresh = f05, t

MATCH_THRESHOLD = float(best_thresh)
print(f"  Best threshold: {MATCH_THRESHOLD:.4f} (F0.5={best_f05:.4f})")

fi = pd.DataFrame({"feature": feature_cols, "imp": clf.feature_importances_}).sort_values("imp", ascending=False)
print("\nTop 10 features:")
print(fi.head(10).to_string(index=False))

# Save
joblib.dump(clf, OUTPUT_DIR / "model.pkl")
joblib.dump(feature_cols, OUTPUT_DIR / "feature_cols.pkl")

del combined, X, y, feat_s2, feat_s3, train_cands_s2, train_cands_s3
del tr_s1, tr_s2, tr_s3
gc.collect()
print("✓ Model trained and saved")


# ═══════════════════════════════════════════════════════════
# CELL 11: Predict on Test Data
# ═══════════════════════════════════════════════════════════

print("\n=== PREDICTION ON TEST DATA ===")
t0 = time.time()

print("Blocking test S1↔S2...")
test_cands_s2 = block_by_country(test_s1, test_s2, label="test-S2")
print(f"  Test S2 candidates: {len(test_cands_s2):,}")

print("Blocking test S1↔S3...")
test_cands_s3 = block_by_country(test_s1, test_s3, label="test-S3")
print(f"  Test S3 candidates: {len(test_cands_s3):,}")

print("Features test S2...")
test_feat_s2 = build_features(test_cands_s2, test_s1, test_s2)
print("Features test S3...")
test_feat_s3 = build_features(test_cands_s3, test_s1, test_s3)

def get_matches(feat_df, threshold):
    if len(feat_df) == 0:
        return pd.DataFrame(columns=["source1_entity_id","candidate_entity_id","match_prob"])
    p = clf.predict_proba(feat_df[feature_cols])[:, 1]
    feat_df = feat_df.copy()
    feat_df["match_prob"] = p
    m = feat_df[feat_df["match_prob"] >= threshold]
    print(f"  Matches: {len(m):,}/{len(feat_df):,} (threshold={threshold:.3f})")
    return m

matches_s2 = get_matches(test_feat_s2, MATCH_THRESHOLD)
matches_s3 = get_matches(test_feat_s3, MATCH_THRESHOLD)
print(f"✓ Prediction done in {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════
# CELL 12: Build & Save Submission
# ═══════════════════════════════════════════════════════════

print("\n=== SUBMISSION ===")
all_s1 = test_s1["entity_id"].unique()

def build_output(dfs, col_name):
    records = {s1: set() for s1 in all_s1}
    for df in dfs:
        if df is not None and len(df) > 0:
            for _, r in df.iterrows():
                s1, cid = r["source1_entity_id"], r["candidate_entity_id"]
                if s1 in records: records[s1].add(cid)
    return pd.DataFrame([
        {"source1_entity_id": s1, col_name: ",".join(sorted(records[s1]))}
        for s1 in sorted(records.keys())
    ])

matching = build_output([matches_s2, matches_s3], "matched_entity_ids")
matching.to_csv(OUTPUT_DIR / "matching_results.tsv", sep="\t", index=False)
n_m = (matching["matched_entity_ids"] != "").sum()
print(f"  matching_results.tsv: {len(matching):,} rows, {n_m:,} matched")

candidates = build_output([test_cands_s2, test_cands_s3], "candidate_entity_ids")
candidates.to_csv(OUTPUT_DIR / "candidate_pairs.tsv", sep="\t", index=False)
print(f"  candidate_pairs.tsv: {len(candidates):,} rows")

# Upload to S3 if configured
if USE_S3:
    for f in ["matching_results.tsv", "candidate_pairs.tsv", "model.pkl"]:
        s3.upload_file(str(OUTPUT_DIR / f), S3_BUCKET, f"{S3_PREFIX}/output/{f}")
        print(f"  Uploaded to s3://{S3_BUCKET}/{S3_PREFIX}/output/{f}")

print(f"\n✓ ALL DONE! Files saved to {OUTPUT_DIR}/")
print("  Download matching_results.tsv and submit to the competition!")
