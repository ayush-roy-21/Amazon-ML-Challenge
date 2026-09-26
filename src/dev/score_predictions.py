"""Local, dev-only sanity check: macro F0.5 of a matching_results.tsv against a ground-truth file.

This is NOT the official leaderboard scorer (we were never given one) - it is a thin, obviously-correct
re-application of the exact spec formula in `metrics.py`, useful for checking a held-out split (e.g. the
synthetic generator's `dev_only_test_ground_truth.tsv`) locally before submitting.

    python3 -m src.dev.score_predictions \
        --predictions output/matching_results.tsv --truth dataset/dev_only_test_ground_truth.tsv
(run from the `business_entity_resolution/` directory, same as src.train / src.predict)
"""
from __future__ import annotations

import argparse

from ..io_utils import read_truth
from ..metrics import macro_f05


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()

    pred = read_truth(args.predictions)
    truth = read_truth(args.truth)
    if set(pred) != set(truth):
        missing = set(truth) - set(pred)
        extra = set(pred) - set(truth)
        print(f"warning: prediction/truth id sets differ ({len(missing)} missing, {len(extra)} extra)")
    ids = list(truth)
    score = macro_f05([pred.get(i, []) for i in ids], [truth[i] for i in ids])
    print(f"macro F0.5 over {len(ids)} Source-1 entities: {score:.4f}")


if __name__ == "__main__":
    main()
