"""CLI entry point: score a split (typically ``test``) with a trained model and write the two output TSVs.

    python3 -m src.predict --data-dir dataset --model-dir model --out-dir output
"""
from __future__ import annotations

import argparse
import json

from .pipeline import predict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="directory containing test/test_source1.tsv etc.")
    ap.add_argument("--model-dir", required=True, help="directory with a model.joblib from src.train")
    ap.add_argument("--out-dir", required=True, help="where to write matching_results.tsv / candidate_pairs.tsv")
    args = ap.parse_args()
    summary = predict(args.data_dir, args.model_dir, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
