"""
Master orchestrator for the Amazon ML Challenge 2026 — Business Entity Resolution.

This wraps the advanced LightGBM pipeline.
Usage:
    python run_pipeline.py                  # Full pipeline (train + predict + validate)
    python run_pipeline.py --predict-only   # Skip training, load saved model
    python run_pipeline.py --fast           # Run faster (fewer folds/trees) for smoke testing
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, desc):
    print(f"\n{'='*80}")
    print(f"🚀 STAGE: {desc}")
    print(f"{'='*80}")
    t0 = time.time()
    
    # Run the command
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: Stage '{desc}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)
        
    print(f"✅ Completed '{desc}' in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run the full LightGBM entity resolution pipeline.")
    parser.add_argument("--predict-only", action="store_true", help="Skip training, only predict")
    parser.add_argument("--fast", action="store_true", help="Use fewer folds/trees for faster execution")
    parser.add_argument("--data-dir", default="dataset", help="Directory containing train/ and test/")
    parser.add_argument("--model-dir", default="models", help="Directory to save/load the model")
    parser.add_argument("--out-dir", default="output", help="Directory to save the final TSVs")
    args = parser.parse_args()

    # Ensure directories exist
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # 1. TRAIN
    if not args.predict_only:
        cmd_train = [sys.executable, "-m", "src.train", "--data-dir", args.data_dir, "--model-dir", args.model_dir]
        if args.fast:
            cmd_train.append("--fast")
        run_command(cmd_train, "Training LightGBM Model")
    else:
        print(f"\n⏭️ Skipping training (--predict-only). Using existing models in {args.model_dir}/")

    # 2. PREDICT
    cmd_predict = [
        sys.executable, "-m", "src.predict", 
        "--data-dir", args.data_dir, 
        "--model-dir", args.model_dir, 
        "--out-dir", args.out_dir
    ]
    run_command(cmd_predict, "Predicting on Test Set")

    # 3. VALIDATE
    cmd_validate = [
        sys.executable, "-m", "src.validate_submission",
        "--matching", f"{args.out_dir}/matching_results.tsv",
        "--candidate", f"{args.out_dir}/candidate_pairs.tsv",
        "--test-dir", f"{args.data_dir}/test"
    ]
    run_command(cmd_validate, "Validating Submission Format")

    print(f"\n🎉 Pipeline fully completed in {(time.time() - total_start) / 60:.1f} minutes!")
    print(f"📁 Final outputs are ready in: {args.out_dir}/")


if __name__ == "__main__":
    main()
