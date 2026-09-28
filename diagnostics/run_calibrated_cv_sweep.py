#!/usr/bin/env python3
"""Run Calibrated GBDT on Zero-Leakage Fold 1 -> Fold 0 Benchmark with Threshold Grid Search.

Strict adherence to evaluation principles:
1. Holdout: Fold 0 (25,000 entities, 0 target leakage)
2. Calibrated: scale_pos_weight = 1.0 (pure logloss probabilities)
3. Grid Search: Dense sweep of tau in [0.10 ... 0.95] to maximize macro F0.5
4. S2 vs S3 Joint Grid Search
5. Ground Truth Sanity Check: Compares predicted singletons & matches/non-sing against real GT targets.
"""

from collections import defaultdict
import csv
import json
import os
import pickle
import time
from typing import Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz

import sys
sys.path.insert(0, os.path.abspath("src"))

from metrics import evaluate_macro_f05
from feature_extractor import FeatureExtractor

CACHE_FILE = "scratch/fixed_benchmark_cache_50k.pkl"
SPLITS_FILE = "dataset/train/cv_splits_5fold.json"
GT_FILE = "dataset/train/train_ground_truth.tsv"
OUTPUT_MODEL = "output/calibrated_lgbm_model.txt"

def load_data():
    print(f"Loading cached tables from {CACHE_FILE}...")
    t0 = time.time()
    with open(CACHE_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded cache in {time.time()-t0:.1f}s.")
    
    with open(SPLITS_FILE) as f:
        s1_to_fold = json.load(f)

    # Load GT
    print(f"Loading ground truth from {GT_FILE}...")
    eval_gt = {}
    with open(GT_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if not row: continue
            s1_id = row[0].strip()
            raw = row[1].strip() if len(row) > 1 else ""
            eval_gt[s1_id] = [m.strip() for m in raw.split(",") if m.strip()] if raw else []

    return data, s1_to_fold, eval_gt

def extract_candidates_and_features(s1_records, target_table, target_ids, indices, eval_gt, is_train=False):
    fe = FeatureExtractor()
    idx_core, idx_sorted, idx_concat, idx_brand, idx_num_p, idx_post_p, idx_addr_k = indices
    
    rows = []
    labels = []
    meta = [] # (s1_id, t_id, is_s2)
    s1_cand_map = defaultdict(list)

    total_s1 = len(s1_records)
    print(f"Gathering candidates and extracting features for {total_s1:,} entities (is_train={is_train})...")
    t0 = time.time()

    for i, s1 in enumerate(s1_records):
        s1_id = s1["id"]
        gold_targets = set(eval_gt.get(s1_id, []))
        
        # Candidate gathering with 500/300/200 caps
        cands = set()
        cands.update(idx_core.get(s1["core"], [])[:500])
        for a in s1["aliases"]:
            cands.update(idx_core.get(a, [])[:500])
        cands.update(idx_sorted.get(s1["sorted"], [])[:500])
        if len(s1["concat"]) >= 5:
            cands.update(idx_concat.get(s1["concat"], [])[:300])
        for bt in s1["distinctive"][:6]:
            cands.update(idx_brand.get(bt, [])[:300])
        if s1["num_prefix"]:
            cands.update(idx_num_p.get(s1["num_prefix"], [])[:300])
        if s1["post_prefix"]:
            cands.update(idx_post_p.get(s1["post_prefix"], [])[:300])
        for ak in s1["addr_keys"]:
            cands.update(idx_addr_k.get(ak, [])[:200])

        s1_country = s1["country"]
        s1_data = {"name": s1["norm_name"], "address": s1["clean_addr"], "country": s1_country}

        # Filter candidates by open-set country concordance
        valid_cands = []
        for t_idx in cands:
            t = target_table[t_idx]
            t_country = t[5]
            if s1_country and t_country and s1_country != t_country:
                continue
            valid_cands.append(t_idx)

        # For training: take all true positives in candidates + up to 10 hard negatives
        if is_train:
            pos_t_idxs = [t_idx for t_idx in valid_cands if target_ids[t_idx] in gold_targets]
            neg_t_idxs = [t_idx for t_idx in valid_cands if target_ids[t_idx] not in gold_targets][:10]
            selected_t_idxs = pos_t_idxs + neg_t_idxs
        else:
            selected_t_idxs = valid_cands

        for rank, t_idx in enumerate(selected_t_idxs, start=1):
            t = target_table[t_idx]
            t_id = target_ids[t_idx]
            t_data = {"name": t[3], "address": t[4], "country": t[5]}
            
            feat_dict = fe.extract_pair_features(s1_data, t_data, cand_id=t_id, cand_rank=rank, pool_size=len(selected_t_idxs))
            feat_vals = list(feat_dict.values())
            
            rows.append(feat_vals)
            y = 1 if t_id in gold_targets else 0
            labels.append(y)
            meta.append((s1_id, t_id, 1 if t_id.startswith("S2-") else 0))
            s1_cand_map[s1_id].append(t_id)

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1:,}/{total_s1:,} entities in {time.time()-t0:.1f}s...")

    feat_names = list(feat_dict.keys()) if rows else []
    print(f"Extraction done in {time.time()-t0:.1f}s. Matrix shape: ({len(rows):,}, {len(feat_names)}) | Pos: {sum(labels):,} | Neg: {len(labels)-sum(labels):,}")
    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int32), meta, feat_names, s1_cand_map

def main():
    print("=" * 80)
    print(" CALIBRATED GBDT TRAINING & HOLDOUT THRESHOLD GRID SEARCH")
    print(" Zero-Leakage Fold 1 -> Fold 0 Evaluation | Standard Scorer")
    print("=" * 80)

    data, s1_to_fold, eval_gt = load_data()
    s1_records = data["s1_records"]
    target_table = data["target_table"]
    target_ids = data["target_ids"]
    indices = (
        defaultdict(list, data["idx_core"]),
        defaultdict(list, data["idx_sorted"]),
        defaultdict(list, data["idx_concat"]),
        defaultdict(list, data["idx_brand"]),
        defaultdict(list, data["idx_num_p"]),
        defaultdict(list, data["idx_post_p"]),
        defaultdict(list, data["idx_addr_k"]),
    )

    # Separate S1 entities into Fold 1 (train) and Fold 0 (val holdout)
    fold0_s1 = [s for s in s1_records if s1_to_fold.get(s["id"]) == 0]
    fold1_s1 = [s for s in s1_records if s1_to_fold.get(s["id"]) == 1]
    print(f"Separated entities: Fold 1 (Train) = {len(fold1_s1):,} | Fold 0 (Val Holdout) = {len(fold0_s1):,}")

    # Build Training Matrix (Fold 1)
    print("\n--- Phase 1: Building Training Matrix (Fold 1) ---")
    X_train, y_train, train_meta, feat_names, _ = extract_candidates_and_features(
        fold1_s1, target_table, target_ids, indices, eval_gt, is_train=True
    )

    # Train Calibrated LightGBM (scale_pos_weight = 1.0)
    print("\n--- Phase 2: Training Calibrated LightGBM (scale_pos_weight=1.0) ---")
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 350,
        "learning_rate": 0.05,
        "num_leaves": 45,
        "max_depth": 7,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 25,
        "scale_pos_weight": 1.0,  # CRUCIAL: Unweighted logloss for calibrated probabilities
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**lgb_params)
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"Model trained in {time.time()-t0:.1f}s.")

    # Save model
    os.makedirs("output", exist_ok=True)
    model.booster_.save_model(OUTPUT_MODEL)
    print(f"Saved trained booster to {OUTPUT_MODEL}.")

    # Build Validation Matrix (Fold 0 Holdout)
    print("\n--- Phase 3: Extracting Validation Features (Fold 0 Holdout) ---")
    X_val, y_val, val_meta, _, _ = extract_candidates_and_features(
        fold0_s1, target_table, target_ids, indices, eval_gt, is_train=False
    )

    # Score validation pairs
    print(f"Predicting calibrated probabilities on {len(X_val):,} validation candidate pairs...")
    t0 = time.time()
    val_probs = model.predict_proba(X_val)[:, 1]
    print(f"Inference completed in {time.time()-t0:.1f}s.")

    # Group scores by S1 entity
    val_s1_scores = defaultdict(list)
    for (s1_id, t_id, is_s2), prob in zip(val_meta, val_probs):
        val_s1_scores[s1_id].append((t_id, float(prob), bool(is_s2)))

    val_s1_ids = [s["id"] for s in fold0_s1]
    gt_f0 = {s: eval_gt.get(s, []) for s in val_s1_ids}

    # Step 4: Fine-grained Grid Search over tau in [0.10 ... 0.95]
    print("\n" + "=" * 95)
    print(" PHASE 4: GLOBAL THRESHOLD (tau) GRID SEARCH ON FOLD 0 HOLDOUT")
    print("=" * 95)
    print(f"{'tau':<6} | {'Macro F0.5':<10} | {'Singleton Acc':<14} | {'Pred Sing %':<12} | {'Matches/NS':<12} | {'Total Matches':<14}")
    print("-" * 95)

    tau_candidates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    best_tau = None
    best_f05 = -1.0
    best_metrics = None

    for tau in tau_candidates:
        # Resolve target claims using mutual best with probability
        t_claims = defaultdict(list)
        for s1_id in val_s1_ids:
            for t_id, prob, is_s2 in val_s1_scores.get(s1_id, []):
                if prob >= tau:
                    t_claims[t_id].append((s1_id, prob))

        winner = {t_id: max(claims, key=lambda x: x[1])[0] for t_id, claims in t_claims.items()}

        current_preds = {}
        total_m = 0
        pred_sing = 0
        for s1_id in val_s1_ids:
            retained = [t_id for t_id, prob, is_s2 in val_s1_scores.get(s1_id, [])
                        if prob >= tau and winner.get(t_id) == s1_id]
            current_preds[s1_id] = retained
            total_m += len(retained)
            if not retained:
                pred_sing += 1

        res = evaluate_macro_f05(current_preds, gt_f0)
        macro_f05 = res["macro_f05"]
        sing_acc = res["singleton_accuracy"]
        sing_pct = (pred_sing / len(val_s1_ids)) * 100
        non_sing_count = len(val_s1_ids) - pred_sing
        m_per_ns = total_m / non_sing_count if non_sing_count else 0.0

        marker = " <--- MAX F0.5" if macro_f05 > best_f05 else ""
        print(f"{tau:<6.2f} | {macro_f05:<10.4f} | {sing_acc:<14.4f} | {sing_pct:>10.2f}% | {m_per_ns:>12.3f} | {total_m:<14,} {marker}")

        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_tau = tau
            best_metrics = (macro_f05, sing_acc, sing_pct, m_per_ns, total_m)

    print("\n" + "=" * 95)
    print(" PHASE 5: DUAL S2 / S3 THRESHOLD JOINT GRID SEARCH")
    print("=" * 95)
    print(f"Sweeping tau_S2 and tau_S3 around optimal range...")
    
    best_dual_f05 = -1.0
    best_tau_s2 = None
    best_tau_s3 = None
    best_dual_metrics = None

    search_range = [t for t in [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
                    if abs(t - best_tau) <= 0.25]
    if best_tau not in search_range:
        search_range.append(best_tau)
    search_range.sort()

    for t_s2 in search_range:
        for t_s3 in search_range:
            t_claims = defaultdict(list)
            for s1_id in val_s1_ids:
                for t_id, prob, is_s2 in val_s1_scores.get(s1_id, []):
                    thresh = t_s2 if is_s2 else t_s3
                    if prob >= thresh:
                        t_claims[t_id].append((s1_id, prob))

            winner = {t_id: max(claims, key=lambda x: x[1])[0] for t_id, claims in t_claims.items()}

            current_preds = {}
            total_m = 0
            pred_sing = 0
            for s1_id in val_s1_ids:
                retained = [t_id for t_id, prob, is_s2 in val_s1_scores.get(s1_id, [])
                            if prob >= (t_s2 if is_s2 else t_s3) and winner.get(t_id) == s1_id]
                current_preds[s1_id] = retained
                total_m += len(retained)
                if not retained:
                    pred_sing += 1

            res = evaluate_macro_f05(current_preds, gt_f0)
            if res["macro_f05"] > best_dual_f05:
                best_dual_f05 = res["macro_f05"]
                best_tau_s2 = t_s2
                best_tau_s3 = t_s3
                sing_pct = (pred_sing / len(val_s1_ids)) * 100
                non_sing_count = len(val_s1_ids) - pred_sing
                m_per_ns = total_m / non_sing_count if non_sing_count else 0.0
                best_dual_metrics = (best_dual_f05, res["singleton_accuracy"], sing_pct, m_per_ns, total_m)

    print(f"Optimal Joint Dual Thresholds: tau_S2 = {best_tau_s2:.2f}, tau_S3 = {best_tau_s3:.2f}")
    print(f"  Macro F0.5:         {best_dual_metrics[0]:.4f}")
    print(f"  Singleton Accuracy: {best_dual_metrics[1]:.4f}")
    print(f"  Predicted Sing %:   {best_dual_metrics[2]:.2f}% (Ground Truth Target: 5.58%)")
    print(f"  Matches / Non-Sing: {best_dual_metrics[3]:.3f} (Ground Truth Target: 3.666)")
    print(f"  Total Matches:      {best_dual_metrics[4]:,}")

    # Output JSON summary
    summary = {
        "best_global_tau": best_tau,
        "best_global_f05": best_metrics[0],
        "global_sing_acc": best_metrics[1],
        "global_sing_pct": best_metrics[2],
        "global_m_per_ns": best_metrics[3],
        "best_tau_s2": best_tau_s2,
        "best_tau_s3": best_tau_s3,
        "best_dual_f05": best_dual_metrics[0],
        "dual_sing_acc": best_dual_metrics[1],
        "dual_sing_pct": best_dual_metrics[2],
        "dual_m_per_ns": best_dual_metrics[3],
        "dual_total_matches": best_dual_metrics[4],
    }
    with open("output/calibrated_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved summary to output/calibrated_sweep_summary.json.")

if __name__ == "__main__":
    main()
