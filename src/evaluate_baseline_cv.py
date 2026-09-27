#!/usr/bin/env python3
"""Evaluate Baseline Rule Matcher on Local 5-Fold Validation Set (Fold 0).

Measures:
1. Macro F0.5
2. Singleton Accuracy
3. Non-Singleton F0.5
4. Candidate Blocking Recall on Fold 0
"""

from collections import defaultdict
import json
import os
import re
import sys
import time
import unicodedata
from typing import Dict, List, Set, Tuple

from metrics import evaluate_macro_f05, compute_entity_f05


LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited", "co", "company", "dba", "lp", "pllc",
    "pvt", "private", "llp", "opc", "sarl", "sas", "sa", "eurl", "sci", "snc"
}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower().replace("&", " and ")
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_core_name(name: str) -> str:
    norm = normalize_text(name)
    tokens = norm.split()
    filtered = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return " ".join(filtered) if filtered else norm


def main():
    train_dir = "dataset/train"
    splits_file = os.path.join(train_dir, "cv_splits_5fold.json")
    if not os.path.exists(splits_file):
        print(f"[ERROR] Splits file {splits_file} not found.")
        sys.exit(1)

    print("=" * 75)
    print(" [LOCAL CV BENCHMARK] Evaluating Baseline on Fold 0")
    print("=" * 75)

    with open(splits_file, "r", encoding="utf-8") as f:
        s1_to_fold = json.load(f)

    # Filter fold 0 S1 IDs (take 50,000 for lightning-fast reference or full fold 0)
    # Let's take a representative sample of 50,000 entities from Fold 0 for fast validation
    fold0_all = {s1_id for s1_id, fold in s1_to_fold.items() if fold == 0}
    sample_size = min(50000, len(fold0_all))
    val_s1_ids = set(sorted(list(fold0_all))[:sample_size])
    print(f"[INFO] Evaluating on {len(val_s1_ids):,} representative entities from Fold 0...")

    # Load Ground Truth for validation entities
    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    val_gt = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            if s1_id in val_s1_ids:
                targets = [x.strip() for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1].strip() else []
                val_gt[s1_id] = targets

    # Load validation S1 names
    val_s1_names = {}
    s1_file = os.path.join(train_dir, "train_source1.tsv")
    with open(s1_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_idx = 0
        name_idx = 1
        for i, col in enumerate(header):
            if "id" in col.lower():
                id_idx = i
            elif "name" in col.lower():
                name_idx = i

        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[id_idx].strip()
            if s1_id in val_s1_ids:
                name = parts[name_idx].strip() if len(parts) > name_idx else ""
                val_s1_names[s1_id] = extract_core_name(name)

    print(f"[INFO] Loaded {len(val_s1_names):,} validation Source 1 entities.")

    # Build target inverted index from train_source2 & train_source3
    print("\nIndexing training targets (S2 + S3)...")
    start_time = time.time()
    target_index = defaultdict(list)
    total_indexed = 0

    # Needed query names
    needed_names = set(val_s1_names.values())

    for source_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(train_dir, source_file)
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            id_idx = 0
            name_idx = 1
            for i, col in enumerate(header):
                if "id" in col.lower():
                    id_idx = i
                elif "name" in col.lower():
                    name_idx = i

            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) <= max(id_idx, name_idx):
                    continue
                t_id = parts[id_idx].strip()
                name = parts[name_idx].strip()
                core = extract_core_name(name)

                # Only index if relevant to validation set to save time and memory
                if core in needed_names:
                    if len(target_index[core]) < 30:
                        target_index[core].append(t_id)

                total_indexed += 1
                if total_indexed % 3000000 == 0:
                    print(f"       Processed {total_indexed:,} targets... ({time.time() - start_time:.1f}s)")

    print(f"[INFO] Filtered index built in {time.time() - start_time:.1f}s.")

    # Perform matching
    val_preds = {}
    val_candidates = {}
    total_gold_links = sum(len(g) for g in val_gt.values())
    captured_links = 0

    for s1_id in val_s1_ids:
        core = val_s1_names.get(s1_id, "")
        cands = target_index.get(core, [])[:30] if core else []
        val_candidates[s1_id] = cands

        # Baseline rule: if unique match or <=3 matches with distinctive name
        matches = []
        if len(cands) == 1:
            matches = [cands[0]]
        elif len(cands) <= 3 and len(core) >= 8:
            matches = cands

        val_preds[s1_id] = matches

        # Measure blocking recall
        gold = set(val_gt.get(s1_id, []))
        captured = len(gold.intersection(set(cands)))
        captured_links += captured

    # Score
    res = evaluate_macro_f05(val_preds, val_gt)
    cand_recall = captured_links / total_gold_links if total_gold_links else 0.0

    print("\n" + "=" * 75)
    print(" [LOCAL FOLD 0 BENCHMARK RESULTS - BASELINE RULE MATCHER]")
    print("=" * 75)
    print(f"  Macro F0.5 Score:          {res['macro_f05']:.4f}")
    print(f"  Singleton Accuracy:        {res['singleton_accuracy']:.4f} ({res['singleton_count']} singletons)")
    print(f"  Non-Singleton F0.5:        {res['non_singleton_f05']:.4f} ({res['non_singleton_count']} non-singletons)")
    print(f"  Candidate Recall:          {cand_recall:.4f} ({captured_links:,} / {total_gold_links:,} true links captured)")
    print("=" * 75)


if __name__ == "__main__":
    main()
