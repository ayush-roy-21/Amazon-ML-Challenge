# Amazon ML Challenge 2026 — Business Entity Resolution (Team IQRA)

This repository contains the end-to-end Machine Learning pipeline for the Amazon ML Challenge 2026. The goal is to identify and match real-world business entities across three different data sources, handling noise, missing data, and multilingual text (English, Hindi, French).

## 🚀 Pipeline Architecture

This solution uses a highly advanced, multi-stage blocking and classification approach to handle the massive scale of the dataset (2.2M Source 1 records vs 10M+ Source 2/3 records):

1. **Multilingual Preprocessing**: Custom text normalization that preserves Devanagari (Hindi) and French characters. Generates phonetic keys for names to catch misspellings.
2. **Multi-View Blocking**: Uses Nearest-Neighbour retrieval over multiple TF-IDF views (character n-grams, word n-grams, phonetic n-grams). Implements a **country penalty** rather than a hard filter to robustly handle missing or noisy country labels.
3. **90+ Pairwise Features**: Computes extensive similarity metrics (Jaro-Winkler, Levenshtein, Token Sort, Set ratios) across different text views (names, addresses, concatenated fields) using `rapidfuzz` and `cpdist` vectorization.
4. **LightGBM Classifier**: A highly optimized gradient-boosted tree model trained with Cross-Validation and Isotonic Calibration. Includes a **Stacked Stage-2 Refinement** model for borderline cases.
5. **Threshold Decoding**: Dynamically scans probability thresholds to maximize the strict **F0.5 macro-averaged score** required by the competition.

## 📂 Project Structure

```
Amazon ML challenge/
├── dataset/                     # Place data files here
│   ├── train/                   # Contains train_source1/2/3.tsv and train_ground_truth.tsv
│   └── test/                    # Contains test_source1/2/3.tsv
├── src/                         # Core pipeline modules
│   ├── textnorm.py              # Text normalisation (names/addresses/countries)
│   ├── features.py              # TF-IDF views + ~90 pairwise similarity/rank features
│   ├── blocking.py              # NN candidate retrieval, country penalty, k_scale search
│   ├── model.py                 # LightGBM CV training & calibration
│   ├── decode.py                # Threshold decoding into the two output id-lists
│   ├── train.py                 # Entry point: train blocking + model on labelled split
│   ├── predict.py               # Entry point: score a split and write TSVs
│   └── validate_submission.py   # Re-implementation of the spec's output-format checks
├── run_pipeline.py              # Master wrapper script for easy execution
└── requirements.txt             # Python dependencies
```

## ⚙️ How to Run

### 1. Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```
Ensure all 7 `.tsv` dataset files are placed in the `dataset/train/` and `dataset/test/` directories.

### 2. Execution via Wrapper (Recommended)
You can use the simple `run_pipeline.py` script to orchestrate the training, prediction, and validation stages automatically:

```bash
# Run full pipeline (Train -> Predict -> Validate)
python run_pipeline.py

# Fast mode (uses fewer folds/trees for a quick smoke test)
python run_pipeline.py --fast

# Predict only (if model is already trained)
python run_pipeline.py --predict-only
```

### 3. Execution via Direct Modules (Advanced)
If you prefer running the stages individually:

```bash
# 1. Train the model (saves to models/ directory)
python3 -m src.train --data-dir dataset --model-dir models

# 2. Predict on test set (saves to output/ directory)
python3 -m src.predict --data-dir dataset --model-dir models --out-dir output

# 3. Validate format
python3 -m src.validate_submission --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## 📊 Output Format
The pipeline generates two files in the `output/` directory, formatted exactly to competition specifications:
1. `matching_results.tsv`: Contains `source1_entity_id` and a comma-separated list of `matched_entity_ids` (or empty for singletons).
2. `candidate_pairs.tsv`: Contains the shortlisted candidate pairs evaluated by the model.
