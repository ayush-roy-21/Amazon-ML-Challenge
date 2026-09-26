"""CLI entry point: fit blocking + the match classifier on a labelled train split.

    python3 -m src.train --data-dir dataset --model-dir model
    python3 -m src.train --data-dir dataset --model-dir model --fast   # quick smoke run, fewer trees/folds
"""
from __future__ import annotations

import argparse
import json

from .config import Config
from .pipeline import train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="directory containing train/train_source1.tsv etc.")
    ap.add_argument("--model-dir", required=True, help="where to save the trained model artifact")
    ap.add_argument("--fast", action="store_true", help="use fewer folds/trees for a quick smoke run")
    ap.add_argument("--no-stage2", action="store_true", help="disable the stacked stage-2 refinement model")
    ap.add_argument("--no-country-holdout", action="store_true", help="skip the leave-one-country-out report")
    args = ap.parse_args()

    cfg = Config().fast() if args.fast else Config()
    if args.no_stage2:
        cfg.use_stage2 = "no"
    if args.no_country_holdout:
        cfg.country_holdout = False

    report = train(args.data_dir, args.model_dir, cfg)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
