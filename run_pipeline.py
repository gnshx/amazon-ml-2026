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
    parser.add_argument("--step", default="all", choices=["all", "baseline", "blocking", "train", "audit"],
                        help="Pipeline step to execute.")
    parser.add_argument("--train-dir", default="dataset/train", help="Path to training dataset.")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test dataset.")
    parser.add_argument("--output-dir", default="output", help="Output directory for submissions.")
    args = parser.parse_args()

    venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
    python_bin = venv_py if os.path.exists(venv_py) else sys.executable

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

    # Phase 1: Audit & Split
    if args.step in ["all", "audit"]:
        run_cmd(f"{python_bin} src/data_audit_and_split.py {args.train_dir}", "Phase 1: Data Audit & Grouped 5-Fold CV")

    # Phase 2: Baseline
    if args.step in ["all", "baseline"]:
        run_cmd(f"{python_bin} src/baseline_rule_matcher.py", "Phase 2: Baseline Rule Matcher & Hour-1 Submission")

    # Phase 3: Multi-route blocking
    if args.step in ["all", "blocking"]:
        run_cmd(f"{python_bin} src/multi_route_blocker.py", "Phase 3: Multi-Route Blocker (>= 98% Recall Target)")

    # Phase 4: Training & Singleton Gate
    if args.step in ["all", "train"]:
        run_cmd(f"{python_bin} src/train_gbdt_and_singleton_gate.py", "Phase 4: Train Calibrated GBDT & Singleton Gate")

    print("\n" + "=" * 75)
    print(" [SUCCESS] Pipeline execution complete!")
    print("=" * 75)


if __name__ == "__main__":
    main()
