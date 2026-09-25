#!/usr/bin/env python3
"""Master End-to-End Pipeline Runner for Amazon Business Entity Resolution Challenge.

Automates the complete sequence:
  Phase 0: Dataset presence & format verification
  Phase 1: Ground truth audit, target exclusivity check & 5-fold CV split
  Phase 2: Hour-1 Baseline Rule Matcher submission generation & local validation
  Phase 3: High-recall Multi-Route Blocking (7 routes, target >= 98% recall)
  Phase 4: Calibrated GBDT & Dedicated Singleton Gate Training + Macro F0.5 Threshold Sweep
  Phase 5: Test set inference, conflict resolution, submission file generation & validation
"""

import argparse
import os
import subprocess
import sys


def run_cmd(cmd: str, desc: str):
    print("\n" + "=" * 75)
    print(f" [PIPELINE] {desc}")
    print(f" Command: {cmd}")
    print("=" * 75)
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"[ERROR] Pipeline step failed: {desc}", file=sys.stderr)
        sys.exit(ret.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run complete ML challenge pipeline.")
    parser.add_argument("--step", default="gpu", choices=["all", "gpu", "baseline", "blocking", "train", "audit"],
                        help="Pipeline step to execute (default: gpu).")
    parser.add_argument("--train-dir", default="dataset/train", help="Path to training dataset.")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test dataset.")
    parser.add_argument("--output-dir", default="output", help="Output directory for submissions.")
    parser.add_argument("--n-train", type=int, default=35000, help="Number of training entities for GBDT.")
    parser.add_argument("--n-val", type=int, default=15000, help="Number of validation entities for GBDT.")
    args = parser.parse_args()

    python_bin = sys.executable

    # Phase 0: Verify dataset
    train_s1 = os.path.join(args.train_dir, "train_source1.tsv")
    gt_file = os.path.join(args.train_dir, "train_ground_truth.tsv")

    if not os.path.exists(train_s1) or not os.path.exists(gt_file):
        print(f"[WARN] Training files not yet found in '{args.train_dir}'.")
        print("Please copy the dataset files into 'dataset/train/' and 'dataset/test/':")
        print("  - dataset/train/train_source1.tsv")
        print("  - dataset/train/train_source2.tsv")
        print("  - dataset/train/train_source3.tsv")
        print("  - dataset/train/train_ground_truth.tsv")
        print("  - dataset/test/test_source1.tsv (and S2/S3 test files)")
        sys.exit(1)

    # Fast Full GPU Pipeline (Phase 1 through Phase 5)
    if args.step in ["gpu", "all"]:
        run_cmd(
            f"{python_bin} src/run_gpu_pipeline.py --train-dir {args.train_dir} --test-dir {args.test_dir} "
            f"--output-dir {args.output_dir} --n-train {args.n_train} --n-val {args.n_val}",
            "Full GPU Pipeline: Training on RTX 3060, Calibrated Sweep & Full Test Inference"
        )
    elif args.step == "audit":
        run_cmd(f"{python_bin} src/data_audit_and_split.py {args.train_dir}", "Phase 1: Data Audit & Grouped 5-Fold CV")
    elif args.step == "baseline":
        run_cmd(f"{python_bin} src/baseline_rule_matcher.py", "Phase 2: Baseline Rule Matcher & Hour-1 Submission")
    elif args.step == "blocking":
        run_cmd(f"{python_bin} src/multi_route_blocker.py", "Phase 3: Multi-Route Blocker (>= 98% Recall Target)")
    elif args.step == "train":
        run_cmd(f"{python_bin} src/train_gbdt_and_singleton_gate.py", "Phase 4: Train Calibrated GBDT & Singleton Gate")

    print("\n" + "=" * 75)
    print(" [SUCCESS] Pipeline execution complete!")
    print("=" * 75)


if __name__ == "__main__":
    main()
