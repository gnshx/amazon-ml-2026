#!/usr/bin/env python3
"""High-Performance Streaming Data Audit & Grouped 5-Fold Cross-Validation Splitter.

Performs:
1. Fast line-by-line streaming audit of ground truth linkages, singletons, and match distributions.
2. Target Exclusivity Audit: Verifies whether any Target ID (S2/S3) is shared across multiple S1 entities.
3. Connected-Component Grouped 5-Fold Cross-Validation split generation.
"""

from collections import defaultdict
import csv
import json
import os
import random
import sys
import time
from typing import Dict, List, Set, Tuple


def audit_ground_truth(gt_path: str):
    """Audits ground truth linkages, singletons, and target exclusivity via line-by-line streaming."""
    print("=" * 75)
    print(" GROUND TRUTH & TARGET EXCLUSIVITY AUDIT")
    print(f" File: {gt_path}")
    print("=" * 75)

    start_time = time.time()
    total_s1 = 0
    match_counts = defaultdict(int)
    s2_count = 0
    s3_count = 0

    s1_to_targets = {}
    target_to_s1 = defaultdict(list)

    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        print(f"[INFO] Header columns: {header}")

        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            raw_matches = parts[1].strip() if len(parts) > 1 else ""

            targets = [x.strip() for x in raw_matches.split(",") if x.strip()] if raw_matches else []
            s1_to_targets[s1_id] = targets
            match_counts[len(targets)] += 1
            total_s1 += 1

            for t in targets:
                target_to_s1[t].append(s1_id)
                if "s2" in t.lower():
                    s2_count += 1
                elif "s3" in t.lower():
                    s3_count += 1

            if line_idx % 500000 == 0:
                print(f"       Processed {line_idx:,} rows... ({time.time() - start_time:.1f}s)")

    elapsed = time.time() - start_time
    singletons = match_counts[0]
    singleton_pct = (singletons / total_s1 * 100) if total_s1 else 0

    print(f"\n[AUDIT RESULTS] Completed in {elapsed:.2f} seconds")
    print("-" * 50)
    print(f"Total Source 1 entities:         {total_s1:,}")
    print(f"Singletons (0 matches):          {singletons:,} ({singleton_pct:.2f}%)")
    print(f"Non-Singletons (>=1 match):      {total_s1 - singletons:,} ({100 - singleton_pct:.2f}%)")
    for k in sorted(match_counts.keys())[:10]:
        if k > 0:
            print(f"  - Exactly {k} match{'es' if k > 1 else ''}:           {match_counts[k]:,} ({match_counts[k]/total_s1*100:.2f}%)")
    if max(match_counts.keys()) > 10:
        over_10 = sum(match_counts[k] for k in match_counts if k > 10)
        print(f"  - >10 matches:                 {over_10:,} ({over_10/total_s1*100:.2f}%)")
        print(f"  - Max matches for a single S1: {max(match_counts.keys())}")

    print(f"\nTotal Target Links:              {s2_count + s3_count:,}")
    print(f"  - Source 2 links:              {s2_count:,} ({s2_count/(s2_count+s3_count)*100:.1f}%)")
    print(f"  - Source 3 links:              {s3_count:,} ({s3_count/(s2_count+s3_count)*100:.1f}%)")

    # Target Exclusivity Audit
    print("\n" + "=" * 50)
    print(" TARGET EXCLUSIVITY AUDIT (1-to-1 Reality Check)")
    print("=" * 50)
    total_unique_targets = len(target_to_s1)
    multi_assigned_targets = {t: s1s for t, s1s in target_to_s1.items() if len(s1s) > 1}
    print(f"Total Unique Target IDs:         {total_unique_targets:,}")
    print(f"Target IDs matching multiple S1: {len(multi_assigned_targets):,} ({len(multi_assigned_targets)/total_unique_targets*100:.4f}%)")

    if len(multi_assigned_targets) > 0:
        print("\n[CRITICAL FINDING]: Target records ARE NOT strictly exclusive!")
        print(f"Found {len(multi_assigned_targets):,} target IDs mapped to multiple Source 1 entities.")
        print("Example multi-matched targets:")
        for t, s1s in list(multi_assigned_targets.items())[:5]:
            print(f"  - Target '{t}' matched to {len(s1s)} Source 1 IDs: {s1s[:4]}...")
        print("-> CONCLUSION: Blindly enforcing 1-to-1 mutual-best will DELETE valid true links!")
    else:
        print("\n[CRITICAL FINDING]: All target IDs are 100% strictly exclusive to a single Source 1 entity!")
        print("-> CONCLUSION: 1-to-1 mutual-best constraint is mathematically valid and safe to enforce.")

    return s1_to_targets, target_to_s1


def build_connected_component_folds(s1_to_targets: Dict[str, List[str]], n_splits: int = 5, seed: int = 42) -> Dict[str, int]:
    """Partitions Source 1 entities into n balanced folds grouped by connected components."""
    print("\n" + "=" * 50)
    print(" GENERATING ZERO-LEAKAGE CONNECTED-COMPONENT CV FOLDS")
    print("=" * 50)
    start_time = time.time()

    # Build adjacency list
    adj = defaultdict(set)
    for s1, targets in s1_to_targets.items():
        for t in targets:
            adj[s1].add(t)
            adj[t].add(s1)

    # Find connected components (BFS)
    visited = set()
    components = []

    all_s1_ids = list(s1_to_targets.keys())
    for s1 in all_s1_ids:
        if s1 in visited:
            continue
        comp_s1 = []
        queue = [s1]
        visited.add(s1)

        while queue:
            curr = queue.pop(0)
            if curr in s1_to_targets:
                comp_s1.append(curr)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(comp_s1)

    print(f"[INFO] Discovered {len(components):,} connected components in {time.time() - start_time:.1f}s.")

    # Sort components descending for greedy bin packing
    random.seed(seed)
    random.shuffle(components)
    components.sort(key=lambda c: len(c), reverse=True)

    # Bin pack into n_splits folds
    fold_s1_lists = [[] for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]

    for comp in components:
        min_fold_idx = fold_sizes.index(min(fold_sizes))
        fold_s1_lists[min_fold_idx].extend(comp)
        fold_sizes[min_fold_idx] += len(comp)

    s1_to_fold = {}
    print(f"\nFold Balance Summary ({n_splits} Folds):")
    for fold_idx, s1_list in enumerate(fold_s1_lists):
        for s1 in s1_list:
            s1_to_fold[s1] = fold_idx
        sing_count = sum(1 for s in s1_list if len(s1_to_targets[s]) == 0)
        print(f"  Fold {fold_idx}: {len(s1_list):,} entities | Singletons: {sing_count:,} ({sing_count/len(s1_list)*100:.1f}%)")

    return s1_to_fold


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset/train"
    gt_file = os.path.join(data_dir, "train_ground_truth.tsv")

    if not os.path.exists(gt_file):
        print(f"[ERROR] Ground truth file not found: {gt_file}", file=sys.stderr)
        sys.exit(1)

    s1_to_targets, _ = audit_ground_truth(gt_file)
    s1_to_fold = build_connected_component_folds(s1_to_targets, n_splits=5)

    out_file = os.path.join(data_dir, "cv_splits_5fold.json")
    print(f"\nSaving 5-fold cross-validation split to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(s1_to_fold, f)
    print(f"[SUCCESS] Saved {len(s1_to_fold):,} entity fold assignments to: {out_file}")


if __name__ == "__main__":
    main()
