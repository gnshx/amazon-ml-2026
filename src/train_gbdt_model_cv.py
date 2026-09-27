#!/usr/bin/env python3
"""Train Calibrated LightGBM Pairwise Matcher with Zero-Leakage CV.

Pipeline:
1. Uses cv_splits_5fold.json to guarantee zero connected-component leakage.
2. High-Recall Multi-Route Blocker with measured link-level recall (targets >= 90%).
3. Robust pairwise training set construction: True links (positives) + Hard negative candidates.
4. Upgraded FeatureExtractor (40 features) handling missing data, postal/street number isolation, and French/Indic tokens.
5. LightGBM classifier training with balanced / tuned precision objective.
6. Entity-level threshold optimization directly maximizing the competition Macro F0.5.
7. Detailed benchmarking report: Macro F0.5, Singleton Accuracy, Non-Singleton F0.5, Feature Importances.
"""

from collections import defaultdict
import csv
import json
import os
import random
import sys
import time
from typing import Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz

from feature_extractor import (
    FeatureExtractor,
    normalize_text,
    extract_core_name,
    extract_sorted_key,
    extract_distinctive_name_tokens,
    extract_postal_and_number,
    is_non_ascii_name,
)
from metrics import evaluate_macro_f05


def load_dataset_samples(
    train_dir: str,
    n_train_s1: int = 30000,
    n_val_s1: int = 15000,
    seed: int = 42,
) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, List[str]], Dict[str, List[str]]]:
    """Loads representative train (from Fold 1) and validation (from Fold 0) entities."""
    random.seed(seed)
    splits_file = os.path.join(train_dir, "cv_splits_5fold.json")
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Missing {splits_file}. Run Phase 1 data audit first.")

    with open(splits_file, "r") as f:
        s1_to_fold = json.load(f)

    # Load Ground Truth
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    print("[1/5] Loading ground truth links...")
    all_gt = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            matches = [x.strip() for x in row[1].split(",") if x.strip()] if len(row) > 1 and row[1].strip() else []
            all_gt[s1_id] = matches

    # Partition candidate pools by fold
    fold0_s1 = [s for s, fld in s1_to_fold.items() if fld == 0]
    fold1_s1 = [s for s, fld in s1_to_fold.items() if fld == 1]

    random.shuffle(fold0_s1)
    random.shuffle(fold1_s1)

    val_s1_ids = set(fold0_s1[:n_val_s1])
    train_s1_ids = set(fold1_s1[:n_train_s1])
    all_needed_s1 = train_s1_ids.union(val_s1_ids)

    print(f"Selected {len(train_s1_ids):,} training entities (Fold 1) and {len(val_s1_ids):,} validation entities (Fold 0).")

    # Load Source 1 records
    print("[2/5] Streaming Source 1 metadata...")
    s1_train_data = {}
    s1_val_data = {}
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            if s1_id in all_needed_s1:
                name = row[1] if len(row) > 1 else ""
                addr = row[2] if len(row) > 2 else ""
                country = row[3] if len(row) > 3 else ""
                rec = {"name": name, "address": addr, "country": country}
                if s1_id in train_s1_ids:
                    s1_train_data[s1_id] = rec
                else:
                    s1_val_data[s1_id] = rec

    train_gt = {s: all_gt[s] for s in s1_train_data}
    val_gt = {s: all_gt[s] for s in s1_val_data}

    return s1_train_data, s1_val_data, train_gt, val_gt


def build_blocker_index_and_generate_candidates(
    train_dir: str,
    s1_all_data: Dict[str, dict],
    gold_links: Dict[str, List[str]],
    max_cands_per_entity: int = 40,
) -> Tuple[Dict[str, List[str]], Dict[str, dict]]:
    """Builds multi-route blocking inverted index and generates candidate pools."""
    print("[3/5] Indexing target records (S2 + S3) with expanded recall keys...")
    t0 = time.time()

    # Pre-compute needed query keys for all S1 entities
    needed_cores = set()
    needed_sorted = set()
    needed_brand = set()
    needed_concat = set()
    needed_postal_num = set()
    needed_addr_num = set()

    s1_meta = {}
    for s1_id, data in s1_all_data.items():
        name = data["name"]
        addr = data["address"]
        core = extract_core_name(name)
        sorted_k = extract_sorted_key(name)
        norm = normalize_text(name)
        concat_k = norm.replace(" ", "")
        brand = extract_distinctive_name_tokens(name)
        postal, num, dist_tokens = extract_postal_and_number(addr)

        if core: needed_cores.add(core)
        if sorted_k: needed_sorted.add(sorted_k)
        if len(concat_k) >= 5: needed_concat.add(concat_k)
        for b in brand: needed_brand.add(b)
        if postal and num:
            needed_postal_num.add(f"{postal}_{num}")
        if num and dist_tokens:
            for dt in sorted(dist_tokens)[:2]:
                needed_addr_num.add(f"{num}_{dt}")

        s1_meta[s1_id] = {
            "core": core,
            "sorted": sorted_k,
            "concat": concat_k,
            "brand": brand,
            "postal_num": f"{postal}_{num}" if (postal and num) else "",
            "addr_keys": [f"{num}_{dt}" for dt in sorted(dist_tokens)[:2]] if (num and dist_tokens) else [],
            "country": data["country"].strip().lower()
        }

    # Inverted index tables (expanded cap to prevent premature truncation)
    idx_core = defaultdict(list)
    idx_sorted = defaultdict(list)
    idx_concat = defaultdict(list)
    idx_brand = defaultdict(list)
    idx_postal_num = defaultdict(list)
    idx_addr_num = defaultdict(list)
    target_data = {}

    CAP_CORE = 60
    CAP_SORTED = 50
    CAP_CONCAT = 30
    CAP_BRAND = 40
    CAP_POSTAL_NUM = 30
    CAP_ADDR = 25

    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(train_dir, fname)
        print(f"  Streaming {fname}...")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                t_id = row[0].strip()
                t_name = row[1] if len(row) > 1 else ""
                t_addr = row[2] if len(row) > 2 else ""
                t_country = row[3].strip().lower() if len(row) > 3 else ""

                t_core = extract_core_name(t_name)
                t_sorted = extract_sorted_key(t_name)
                t_norm = normalize_text(t_name)
                t_concat = t_norm.replace(" ", "")
                t_brand = extract_distinctive_name_tokens(t_name)
                t_postal, t_num, t_dist = extract_postal_and_number(t_addr)

                hit = False
                if t_core in needed_cores and len(idx_core[t_core]) < CAP_CORE:
                    idx_core[t_core].append(t_id)
                    hit = True
                if t_sorted in needed_sorted and len(idx_sorted[t_sorted]) < CAP_SORTED:
                    idx_sorted[t_sorted].append(t_id)
                    hit = True
                if len(t_concat) >= 5 and t_concat in needed_concat and len(idx_concat[t_concat]) < CAP_CONCAT:
                    idx_concat[t_concat].append(t_id)
                    hit = True
                for b in t_brand:
                    if b in needed_brand and len(idx_brand[b]) < CAP_BRAND:
                        idx_brand[b].append(t_id)
                        hit = True
                if t_postal and t_num:
                    pn_k = f"{t_postal}_{t_num}"
                    if pn_k in needed_postal_num and len(idx_postal_num[pn_k]) < CAP_POSTAL_NUM:
                        idx_postal_num[pn_k].append(t_id)
                        hit = True
                if t_num and t_dist:
                    for dt in sorted(t_dist)[:2]:
                        ak = f"{t_num}_{dt}"
                        if ak in needed_addr_num and len(idx_addr_num[ak]) < CAP_ADDR:
                            idx_addr_num[ak].append(t_id)
                            hit = True

                if hit:
                    target_data[t_id] = {
                        "name": t_name,
                        "address": t_addr,
                        "country": t_country,
                        "core": t_core,
                    }

    print(f"Indexed targets in {time.time() - t0:.1f}s. Loaded {len(target_data):,} relevant targets.")

    # Generate Candidate Pools per S1
    print("Generating candidate sets and evaluating true-link recall...")
    s1_candidates = {}
    total_true_links = 0
    captured_true_links = 0

    for s1_id, meta in s1_meta.items():
        cands = set()
        cands.update(idx_core.get(meta["core"], []))
        cands.update(idx_sorted.get(meta["sorted"], []))
        if len(meta["concat"]) >= 5:
            cands.update(idx_concat.get(meta["concat"], []))
        for b in list(meta["brand"])[:3]:
            cands.update(idx_brand.get(b, []))
        if meta["postal_num"]:
            cands.update(idx_postal_num.get(meta["postal_num"], []))
        for ak in meta["addr_keys"]:
            cands.update(idx_addr_num.get(ak, []))

        # Filter candidates strictly by country compatibility
        s1_ctry = meta["country"]
        filtered = []
        for c in cands:
            t_rec = target_data.get(c)
            if not t_rec: continue
            if s1_ctry and t_rec["country"] and s1_ctry != t_rec["country"]:
                continue
            filtered.append(c)

        # Candidate ranking: sort by core token overlap / fuzzy similarity
        if len(filtered) > max_cands_per_entity:
            # Rank candidates to keep the most promising ones
            s1_core = meta["core"]
            scored = []
            for c in filtered:
                t_core = target_data[c]["core"]
                sim = fuzz.ratio(s1_core, t_core) if (s1_core and t_core) else 50
                scored.append((c, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            filtered = [x[0] for x in scored[:max_cands_per_entity]]

        s1_candidates[s1_id] = filtered

        # Track Recall for gold links
        golds = gold_links.get(s1_id, [])
        if golds:
            total_true_links += len(golds)
            captured_true_links += len(set(golds).intersection(set(filtered)))

    recall_pct = (captured_true_links / total_true_links * 100) if total_true_links else 0.0
    print(f"[RECALL CHECK] True Gold Links: {total_true_links:,} | Captured in Candidates: {captured_true_links:,} | Blocker Recall: {recall_pct:.2f}%")

    return s1_candidates, target_data


def build_pairwise_training_matrix(
    s1_dict: Dict[str, dict],
    candidates: Dict[str, List[str]],
    gold_links: Dict[str, List[str]],
    target_data: Dict[str, dict],
    fe: FeatureExtractor,
    max_hard_negatives: int = 5,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Constructs pairwise feature matrix X, binary labels y, and feature column names."""
    print("Constructing pairwise training feature matrix...")
    t0 = time.time()
    rows = []
    labels = []
    feature_names = None

    for s1_id, cand_list in candidates.items():
        s1_data = s1_dict[s1_id]
        gold_set = set(gold_links.get(s1_id, []))
        pool_size = len(cand_list)

        neg_count = 0
        for rank, cand_id in enumerate(cand_list, start=1):
            cand_rec = target_data.get(cand_id)
            if not cand_rec: continue

            is_match = 1 if cand_id in gold_set else 0

            # For negatives, only retain top hard negatives to maintain balanced dataset
            if is_match == 0:
                if neg_count >= max_hard_negatives:
                    continue
                neg_count += 1

            feat_dict = fe.extract_pair_features(
                s1_data=s1_data,
                cand_data=cand_rec,
                cand_id=cand_id,
                cand_rank=rank,
                pool_size=pool_size
            )

            if feature_names is None:
                feature_names = sorted(feat_dict.keys())

            row = [feat_dict[k] for k in feature_names]
            rows.append(row)
            labels.append(is_match)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"Matrix built in {time.time() - t0:.1f}s. Shape: {X.shape} | Positives: {np.sum(y==1):,} | Negatives: {np.sum(y==0):,}")
    return X, y, feature_names


def main():
    print("=" * 80)
    print(" [LIGHTGBM PAIRWISE MATCHER] Training & Zero-Leakage Cross-Validation")
    print("=" * 80)

    train_dir = "dataset/train"
    s1_train, s1_val, train_gt, val_gt = load_dataset_samples(
        train_dir=train_dir,
        n_train_s1=35000,
        n_val_s1=20000,
        seed=42
    )

    all_s1 = {**s1_train, **s1_val}
    all_gt = {**train_gt, **val_gt}

    # Candidate generation
    candidates, target_data = build_blocker_index_and_generate_candidates(
        train_dir=train_dir,
        s1_all_data=all_s1,
        gold_links=all_gt,
        max_cands_per_entity=35
    )

    fe = FeatureExtractor()

    # Split candidates into train and val
    train_cands = {s: candidates[s] for s in s1_train}
    val_cands = {s: candidates[s] for s in s1_val}

    # Build Training Feature Matrix
    X_train, y_train, feat_names = build_pairwise_training_matrix(
        s1_dict=s1_train,
        candidates=train_cands,
        gold_links=train_gt,
        target_data=target_data,
        fe=fe,
        max_hard_negatives=5
    )

    # Train LightGBM model
    print("\n[4/5] Training LightGBM Pairwise Classifier...")
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos = max(1.0, float(neg_count) / float(2.0 * pos_count)) if pos_count > 0 else 1.0

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 350,
        "learning_rate": 0.06,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "scale_pos_weight": scale_pos,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # Feature Importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 15 Most Informative Features:")
    for i in range(min(15, len(feat_names))):
        idx = sorted_idx[i]
        print(f"  {i+1:2d}. {feat_names[idx]:25s} (importance: {importances[idx]})")

    # Evaluate on Validation Entities (Fold 0)
    print("\n[5/5] Scoring Held-Out Validation Entities (Fold 0)...")
    val_s1_scores = defaultdict(list)
    val_rows = []
    val_meta = []

    for s1_id, c_list in val_cands.items():
        s1_rec = s1_val[s1_id]
        pool_sz = len(c_list)
        for rank, c_id in enumerate(c_list, start=1):
            c_rec = target_data.get(c_id)
            if not c_rec: continue
            f_dict = fe.extract_pair_features(s1_rec, c_rec, c_id, rank, pool_sz)
            val_rows.append([f_dict[k] for k in feat_names])
            val_meta.append((s1_id, c_id))

    if val_rows:
        X_val = np.array(val_rows, dtype=np.float32)
        val_probs = model.predict_proba(X_val)[:, 1]
        for (s1_id, c_id), prob in zip(val_meta, val_probs):
            val_s1_scores[s1_id].append((c_id, float(prob)))

    # Sweep threshold with Target Exclusivity Conflict Resolution to optimize Macro F0.5
    print("\nOptimizing Threshold directly on Per-Entity Macro F0.5 (with Target Exclusivity)...")
    best_macro_f05 = 0.0
    best_tau = 0.90
    best_preds = {}
    best_res = None

    threshold_grid = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
    for tau in threshold_grid:
        # Step A: Filter by threshold and gather target claims
        target_claims = defaultdict(list)
        for s1_id in s1_val:
            for c_id, prob in val_s1_scores.get(s1_id, []):
                if prob >= tau:
                    target_claims[c_id].append((s1_id, prob))

        # Step B: Target Exclusivity Assignment (Winner-Take-Target)
        winner_for_target = {}
        for c_id, claims in target_claims.items():
            best_s1, best_p = max(claims, key=lambda x: x[1])
            winner_for_target[c_id] = best_s1

        # Step C: Retain only won targets
        current_preds = {}
        for s1_id in s1_val:
            retained = []
            for c_id, prob in val_s1_scores.get(s1_id, []):
                if prob >= tau and winner_for_target.get(c_id) == s1_id:
                    retained.append(c_id)
            current_preds[s1_id] = retained

        res = evaluate_macro_f05(current_preds, val_gt)
        print(f"  Threshold tau = {tau:.2f} -> Macro F0.5 = {res['macro_f05']:.4f} | Singleton Acc = {res['singleton_accuracy']:.4f} | Non-Sing F0.5 = {res['non_singleton_f05']:.4f}")
        if res["macro_f05"] > best_macro_f05:
            best_macro_f05 = res["macro_f05"]
            best_tau = tau
            best_preds = current_preds
            best_res = res

    # Compute False Links per Entity
    total_val_entities = len(val_gt)
    total_false_links = sum(
        len(set(best_preds.get(s, [])).difference(set(val_gt[s])))
        for s in val_gt
    )
    false_links_per_entity = total_false_links / total_val_entities if total_val_entities else 0.0

    print("\n" + "=" * 80)
    print(" [LIGHTGBM VALIDATION BENCHMARK RESULTS - FOLD 0]")
    print("=" * 80)
    print(f"  OPTIMAL THRESHOLD (tau):   {best_tau:.2f}")
    print(f"  MACRO F0.5 SCORE:          {best_res['macro_f05']:.4f}")
    print(f"  SINGLETON ACCURACY:        {best_res['singleton_accuracy']:.4f} ({best_res['singleton_count']} singletons)")
    print(f"  NON-SINGLETON F0.5:        {best_res['non_singleton_f05']:.4f} ({best_res['non_singleton_count']} non-singletons)")
    print(f"  FALSE LINKS PER ENTITY:    {false_links_per_entity:.4f} ({total_false_links:,} total false merges)")
    print(f"  TOTAL VALIDATION ENTITIES: {total_val_entities:,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
