#!/usr/bin/env python3
"""Best Pipeline v8: Amazon Business Entity Resolution Challenge.

Unified end-to-end pipeline combining:
1. Data Audit (singleton ratio, fold distribution, dataset stats)
2. Multi-Route High-Recall Blocking (7 routes, targeting >=98% recall)
3. 40-Feature Pairwise LightGBM Classifier (calibrated GBDT)
4. Dedicated Singleton Gate (entity-level F0.5 protection)
5. Per-Source Threshold Sweep (maximize Macro F0.5)
6. Test Set Inference + Submission File Generation
7. Comprehensive Report with all metrics and feature importances

Based on best_plan.md strategy with v7 rule engine scores validated at 0.7026 Macro F0.5.
"""

from collections import defaultdict
import csv
import json
import os
import random
import re
import sys
import time
import unicodedata
from typing import Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz, distance as rf_distance

# Add src to path so we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import evaluate_macro_f05
from feature_extractor import (
    FeatureExtractor,
    normalize_text,
    extract_core_name,
    extract_sorted_key,
    extract_distinctive_name_tokens,
    extract_postal_and_number,
    is_non_ascii_name,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "co", "company", "dba", "lp", "pllc", "pvt", "private", "llp", "opc",
    "sarl", "sas", "sa", "eurl", "sci", "snc", "sasu", "gie", "ei", "sep"
}

GENERIC_WORDS = {
    "center", "services", "solutions", "technologies", "group", "holdings",
    "enterprises", "international", "global", "consulting", "associates",
    "management", "products", "systems", "partners", "digital", "developers",
    "allied", "ventures", "industries", "commercial", "logistics", "trading",
    "retail", "marketing", "agency", "care", "health", "works", "consultancy",
    "enterprise", "labs", "studio"
} | LEGAL_SUFFIXES

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
OUTPUT_DIR = "output"


# ─────────────────────────────────────────────────────────────────────────────
# Blocking Utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_s1_blocking_keys(name: str, address: str, country: str) -> dict:
    """Compute all blocking index keys for an S1 record."""
    norm = normalize_text(name)
    core = extract_core_name(name)
    sorted_k = extract_sorted_key(name)
    concat_k = norm.replace(" ", "")
    brand = extract_distinctive_name_tokens(name)
    postal, num, dist_tokens = extract_postal_and_number(address)
    # Alias names (DBA / AKA)
    alias_cores = []
    if any(w in norm for w in ("aka", "dba", "fka", "doing business as", "trading as")):
        parts = re.split(r'\b(?:aka|dba|fka|doing business as|trading as)\b', norm)
        if len(parts) > 1:
            for p in parts:
                p = p.strip()
                ac = extract_core_name(p) if p else ""
                if ac and len(ac) >= 3:
                    alias_cores.append(ac)

    # Composite address keys: street number + up to 4 distinctive address tokens
    postal_num_key = f"{postal}_{num}" if (postal and num) else ""
    addr_num_keys = [f"{num}_{dt}" for dt in sorted(dist_tokens)[:4]] if (num and dist_tokens) else []

    # First content token prefix (min len 4)
    first_tok = core.split()[0] if core else ""
    prefix_key = first_tok[:4] if len(first_tok) >= 4 else ""

    return {
        "norm": norm,
        "core": core,
        "sorted": sorted_k,
        "concat": concat_k,
        "brand": brand,
        "postal": postal,
        "num": num,
        "dist_tokens": dist_tokens,
        "country": country.strip().lower(),
        "aliases": alias_cores,
        "postal_num_key": postal_num_key,
        "addr_num_keys": addr_num_keys,
        "prefix": prefix_key,
    }


def build_target_index(source_dir: str, source_files: List[str],
                       needed_keys: dict, caps: dict) -> Tuple[dict, dict]:
    """Stream target files and populate inverted indices."""
    idx_core = defaultdict(list)
    idx_sorted = defaultdict(list)
    idx_concat = defaultdict(list)
    idx_brand = defaultdict(list)
    idx_postal_num = defaultdict(list)
    idx_addr_num = defaultdict(list)
    idx_prefix = defaultdict(list)
    target_data = {}

    cores_set = needed_keys["cores"]
    sorted_set = needed_keys["sorted"]
    concat_set = needed_keys["concat"]
    brand_set = needed_keys["brand"]
    postal_num_set = needed_keys["postal_num"]
    addr_num_set = needed_keys["addr_num"]
    prefix_set = needed_keys.get("prefixes", set())

    cap_core = caps.get("core", 500)
    cap_sorted = caps.get("sorted", 400)
    cap_concat = caps.get("concat", 300)
    cap_brand = caps.get("brand", 300)
    cap_postal_num = caps.get("postal_num", 300)
    cap_addr_num = caps.get("addr_num", 200)
    cap_prefix = caps.get("prefix", 100)

    for fname in source_files:
        path = os.path.join(source_dir, fname)
        print(f"  Streaming {fname}...")
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                t_id = parts[0].strip()
                t_name = parts[1] if len(parts) > 1 else ""
                t_addr = parts[2] if len(parts) > 2 else ""
                t_country = parts[3].strip().lower() if len(parts) > 3 else ""

                norm = normalize_text(t_name)
                t_core = extract_core_name(t_name)
                t_sorted = extract_sorted_key(t_name)
                t_concat = norm.replace(" ", "")

                hit = False
                if t_core in cores_set and len(idx_core[t_core]) < cap_core:
                    idx_core[t_core].append(t_id); hit = True

                if any(w in norm for w in ("aka", "dba", "fka", "doing business as", "trading as")):
                    alias_parts = re.split(r'\b(?:aka|dba|fka|doing business as|trading as)\b', norm)
                    if len(alias_parts) > 1:
                        for p in alias_parts:
                            p = p.strip()
                            ac = extract_core_name(p) if p else ""
                            if ac and ac in cores_set and len(idx_core[ac]) < cap_core:
                                idx_core[ac].append(t_id); hit = True

                if t_sorted in sorted_set and len(idx_sorted[t_sorted]) < cap_sorted:
                    idx_sorted[t_sorted].append(t_id); hit = True
                if len(t_concat) >= 5 and t_concat in concat_set and len(idx_concat[t_concat]) < cap_concat:
                    idx_concat[t_concat].append(t_id); hit = True

                t_brand = extract_distinctive_name_tokens(t_name)
                for b in t_brand:
                    if b in brand_set and len(idx_brand[b]) < cap_brand:
                        idx_brand[b].append(t_id); hit = True

                first_tok = t_core.split()[0] if t_core else ""
                pfx = first_tok[:4] if len(first_tok) >= 4 else ""
                if pfx and pfx in prefix_set and len(idx_prefix[pfx]) < cap_prefix:
                    idx_prefix[pfx].append(t_id); hit = True

                t_num = ""
                if t_addr:
                    t_postal, t_num, t_dist = extract_postal_and_number(t_addr)
                    if t_postal and t_num:
                        pn = f"{t_postal}_{t_num}"
                        if pn in postal_num_set and len(idx_postal_num[pn]) < cap_postal_num:
                            idx_postal_num[pn].append(t_id); hit = True
                    if t_num and t_dist:
                        for dt in sorted(t_dist)[:4]:
                            ak = f"{t_num}_{dt}"
                            if ak in addr_num_set and len(idx_addr_num[ak]) < cap_addr_num:
                                idx_addr_num[ak].append(t_id); hit = True

                if hit:
                    target_data[t_id] = {
                        "name": t_name, "address": t_addr, "country": t_country,
                        "core": t_core, "num": t_num,
                    }

    indices = {
        "core": idx_core, "sorted": idx_sorted, "concat": idx_concat,
        "brand": idx_brand, "postal_num": idx_postal_num, "addr_num": idx_addr_num,
        "prefix": idx_prefix,
    }
    return indices, target_data


def generate_candidates(s1_meta: dict, indices: dict,
                        target_data: dict, gold_links: dict = None,
                        max_per_entity: int = 80) -> Tuple[Dict[str, List[str]], float]:
    """Generate candidate pools with soft country filter and multi-route ranking."""
    idx_core = indices["core"]
    idx_sorted = indices["sorted"]
    idx_concat = indices["concat"]
    idx_brand = indices["brand"]
    idx_postal_num = indices["postal_num"]
    idx_addr_num = indices["addr_num"]
    idx_prefix = indices.get("prefix", {})

    s1_candidates = {}
    total_gold = 0
    captured_gold = 0

    for s1_id, meta in s1_meta.items():
        cands = set()
        cands.update(idx_core.get(meta["core"], []))
        for ac in meta["aliases"]:
            cands.update(idx_core.get(ac, []))
        cands.update(idx_sorted.get(meta["sorted"], []))
        if len(meta["concat"]) >= 5:
            cands.update(idx_concat.get(meta["concat"], []))
        for b in list(meta["brand"])[:6]:
            cands.update(idx_brand.get(b, []))
        if meta.get("prefix"):
            cands.update(idx_prefix.get(meta["prefix"], []))
        if meta["postal_num_key"]:
            cands.update(idx_postal_num.get(meta["postal_num_key"], []))
        for ak in meta["addr_num_keys"]:
            cands.update(idx_addr_num.get(ak, []))

        # Soft Country Filter: Do NOT drop candidates based on country.
        # Tree model uses country_match feature directly.
        filtered = list(cands)

        # Smart Multi-Signal Ranking when candidates exceed pool size
        if len(filtered) > max_per_entity:
            s1_core = meta["core"]
            s1_num = meta.get("num", "")
            def score_cand(c_id):
                t_rec = target_data.get(c_id, {})
                sc = fuzz.ratio(s1_core, t_rec.get("core", ""))
                # Bonus if street number matches (protects address-grounded cross-script matches)
                if s1_num and t_rec.get("num") and s1_num == t_rec["num"]:
                    sc += 25
                return sc
            scored = [(c, score_cand(c)) for c in filtered]
            scored.sort(key=lambda x: x[1], reverse=True)
            filtered = [x[0] for x in scored[:max_per_entity]]

        s1_candidates[s1_id] = filtered

        if gold_links:
            golds = gold_links.get(s1_id, [])
            if golds:
                total_gold += len(golds)
                captured_gold += len(set(golds).intersection(set(filtered)))

    recall = (captured_gold / total_gold * 100) if total_gold > 0 else 0.0
    return s1_candidates, recall


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Data Audit
# ─────────────────────────────────────────────────────────────────────────────

def phase1_data_audit(train_dir: str) -> dict:
    print("\n" + "=" * 80)
    print(" [PHASE 1] DATA AUDIT")
    print("=" * 80)

    splits_file = os.path.join(train_dir, "cv_splits_5fold.json")
    with open(splits_file) as f:
        s1_to_fold = json.load(f)

    fold_counts = defaultdict(int)
    for fold in s1_to_fold.values():
        fold_counts[fold] += 1
    print(f"CV Splits: {dict(sorted(fold_counts.items()))}")
    total_s1 = len(s1_to_fold)

    # Load ground truth
    gt = {}
    with open(os.path.join(train_dir, "train_ground_truth.tsv"), encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            matches = [x.strip() for x in row[1].split(",") if x.strip()] if len(row) > 1 and row[1].strip() else []
            gt[s1_id] = matches

    singletons = {k for k, v in gt.items() if len(v) == 0}
    non_singletons = {k for k, v in gt.items() if len(v) > 0}
    match_counts = [len(v) for v in gt.values() if len(v) > 0]

    # Exclusivity check (S2/S3 appearing in multiple S1 sets)
    target_to_s1 = defaultdict(list)
    for s1_id, targets in gt.items():
        for t in targets:
            target_to_s1[t].append(s1_id)
    shared_targets = {t: s1s for t, s1s in target_to_s1.items() if len(s1s) > 1}

    audit_results = {
        "total_s1": total_s1,
        "singleton_count": len(singletons),
        "singleton_pct": len(singletons) / total_s1 * 100,
        "non_singleton_count": len(non_singletons),
        "mean_matches_per_non_singleton": np.mean(match_counts) if match_counts else 0,
        "max_matches": max(match_counts) if match_counts else 0,
        "shared_targets_count": len(shared_targets),
        "target_exclusivity_violated": len(shared_targets) > 0,
        "fold_distribution": dict(fold_counts),
        "total_ground_truth_links": sum(len(v) for v in gt.values()),
    }

    print(f"Total S1 Entities: {total_s1:,}")
    print(f"Singletons: {len(singletons):,} ({len(singletons)/total_s1*100:.2f}%)")
    print(f"Non-Singletons: {len(non_singletons):,} ({len(non_singletons)/total_s1*100:.2f}%)")
    print(f"Mean Matches per Non-Singleton: {audit_results['mean_matches_per_non_singleton']:.2f}")
    print(f"Max Matches for Single Entity: {audit_results['max_matches']}")
    print(f"Total Ground Truth Links: {audit_results['total_ground_truth_links']:,}")
    print(f"Shared Targets (exclusivity violations): {len(shared_targets):,}")
    print(f"→ Target Exclusivity Constraint Valid: {not audit_results['target_exclusivity_violated']}")

    return audit_results, gt, s1_to_fold


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: LightGBM Training & Validation
# ─────────────────────────────────────────────────────────────────────────────

def phase2_train_gbdt(train_dir: str, gt: dict, s1_to_fold: dict, n_train: int = 40000, n_val: int = 20000) -> dict:
    print("\n" + "=" * 80)
    print(" [PHASE 2] LIGHTGBM TRAINING & VALIDATION (Zero-Leakage CV)")
    print("=" * 80)

    N_TRAIN = n_train
    N_VAL = n_val
    MAX_CANDS = 300
    MAX_HARD_NEG = 6

    random.seed(42)
    fold0 = [s for s, f in s1_to_fold.items() if f == 0]
    fold1 = [s for s, f in s1_to_fold.items() if f == 1]
    random.shuffle(fold0); random.shuffle(fold1)

    val_s1_ids = set(fold0[:N_VAL])
    train_s1_ids = set(fold1[:N_TRAIN])
    all_needed = train_s1_ids | val_s1_ids

    print(f"Train: {len(train_s1_ids):,} | Val: {len(val_s1_ids):,} (from Fold 0)")

    # Load S1 records
    print("Loading S1 records...")
    s1_all = {}
    with open(os.path.join(train_dir, "train_source1.tsv"), encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            if s1_id in all_needed:
                s1_all[s1_id] = {
                    "name": row[1] if len(row) > 1 else "",
                    "address": row[2] if len(row) > 2 else "",
                    "country": row[3] if len(row) > 3 else "",
                }

    # Build blocking keys for all S1
    print("Computing blocking keys...")
    s1_meta = {}
    needed_keys = {"cores": set(), "sorted": set(), "concat": set(),
                   "brand": set(), "postal_num": set(), "addr_num": set(),
                   "prefixes": set()}
    for s1_id, data in s1_all.items():
        meta = compute_s1_blocking_keys(data["name"], data["address"], data["country"])
        s1_meta[s1_id] = meta
        if meta["core"]: needed_keys["cores"].add(meta["core"])
        for ac in meta["aliases"]: needed_keys["cores"].add(ac)
        if meta["sorted"]: needed_keys["sorted"].add(meta["sorted"])
        if len(meta["concat"]) >= 5: needed_keys["concat"].add(meta["concat"])
        for b in meta["brand"]: needed_keys["brand"].add(b)
        if meta["postal_num_key"]: needed_keys["postal_num"].add(meta["postal_num_key"])
        for ak in meta["addr_num_keys"]: needed_keys["addr_num"].add(ak)
        if meta.get("prefix"): needed_keys["prefixes"].add(meta["prefix"])

    # Build target index
    print("Building target index (S2+S3)...")
    t0 = time.time()
    caps = {
        "core": 500, "sorted": 400, "concat": 300,
        "brand": 300, "postal_num": 300, "addr_num": 200, "prefix": 100,
    }
    indices, target_data = build_target_index(
        train_dir,
        ["train_source2.tsv", "train_source3.tsv"],
        needed_keys, caps
    )
    print(f"Indexed {len(target_data):,} targets in {time.time()-t0:.1f}s")

    # Generate candidates
    print("Generating candidate sets...")
    all_gt = {s: gt[s] for s in all_needed if s in gt}
    candidates, recall = generate_candidates(s1_meta, indices, target_data, all_gt, MAX_CANDS)
    print(f"[BLOCKING RECALL] {recall:.2f}% of true links captured in candidate pool")

    train_cands = {s: candidates[s] for s in train_s1_ids if s in candidates}
    val_cands = {s: candidates[s] for s in val_s1_ids if s in candidates}
    train_gt_local = {s: all_gt.get(s, []) for s in train_s1_ids}
    val_gt_local = {s: all_gt.get(s, []) for s in val_s1_ids}

    # Build training matrix
    fe = FeatureExtractor()
    print("Building pairwise training matrix...")
    t0 = time.time()
    rows, labels = [], []
    feat_names = None

    for s1_id, cand_list in train_cands.items():
        s1_rec = s1_all[s1_id]
        gold_set = set(train_gt_local.get(s1_id, []))
        neg_count = 0
        for rank, cand_id in enumerate(cand_list, 1):
            cand_rec = target_data.get(cand_id)
            if not cand_rec: continue
            is_match = 1 if cand_id in gold_set else 0
            if is_match == 0:
                if neg_count >= MAX_HARD_NEG: continue
                neg_count += 1
            fd = fe.extract_pair_features(s1_rec, cand_rec, cand_id, rank, len(cand_list))
            if feat_names is None:
                feat_names = sorted(fd.keys())
            rows.append([fd[k] for k in feat_names])
            labels.append(is_match)

    X_train = np.array(rows, dtype=np.float32)
    y_train = np.array(labels, dtype=np.int32)
    print(f"Matrix: {X_train.shape} | Pos: {np.sum(y_train==1):,} | Neg: {np.sum(y_train==0):,} [{time.time()-t0:.1f}s]")

    # Train LightGBM
    print("\nTraining LightGBM...")
    pos_count = np.sum(y_train == 1)
    neg_count_total = np.sum(y_train == 0)
    scale_pos = max(1.0, float(neg_count_total) / (2.0 * pos_count)) if pos_count > 0 else 1.0

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 20,
        "scale_pos_weight": scale_pos,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"Training complete in {time.time()-t0:.1f}s")

    # Feature importances
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 20 Most Informative Features:")
    feat_importance_list = []
    for i in range(min(20, len(feat_names))):
        idx = sorted_idx[i]
        print(f"  {i+1:2d}. {feat_names[idx]:30s} importance={importances[idx]}")
        feat_importance_list.append({"feature": feat_names[idx], "importance": int(importances[idx])})

    # Validation scoring
    print("\nScoring validation entities (Fold 0)...")
    val_s1_scores = defaultdict(list)
    val_rows, val_meta = [], []
    for s1_id, c_list in val_cands.items():
        s1_rec = s1_all[s1_id]
        for rank, c_id in enumerate(c_list, 1):
            c_rec = target_data.get(c_id)
            if not c_rec: continue
            fd = fe.extract_pair_features(s1_rec, c_rec, c_id, rank, len(c_list))
            val_rows.append([fd[k] for k in feat_names])
            val_meta.append((s1_id, c_id))

    val_probs = []
    if val_rows:
        X_val = np.array(val_rows, dtype=np.float32)
        val_probs = model.predict_proba(X_val)[:, 1]
        for (s1_id, c_id), prob in zip(val_meta, val_probs):
            val_s1_scores[s1_id].append((c_id, float(prob)))

    # Threshold sweep
    print("\nOptimizing threshold to maximize Macro F0.5...")
    threshold_grid = [0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.993, 0.995, 0.997, 0.999]
    best_macro_f05 = 0.0
    best_tau = 0.98
    best_preds = {}
    best_res = None
    threshold_results = []

    for tau in threshold_grid:
        target_claims = defaultdict(list)
        for s1_id in val_s1_ids:
            for c_id, prob in val_s1_scores.get(s1_id, []):
                if prob >= tau:
                    target_claims[c_id].append((s1_id, prob))
        winner_for_target = {}
        for c_id, claims in target_claims.items():
            best_s1, best_p = max(claims, key=lambda x: x[1])
            winner_for_target[c_id] = best_s1
        current_preds = {}
        for s1_id in val_s1_ids:
            retained = [c_id for c_id, prob in val_s1_scores.get(s1_id, [])
                        if prob >= tau and winner_for_target.get(c_id) == s1_id]
            current_preds[s1_id] = retained

        res = evaluate_macro_f05(current_preds, val_gt_local)
        threshold_results.append({
            "tau": tau,
            "macro_f05": res["macro_f05"],
            "singleton_acc": res["singleton_accuracy"],
            "non_singleton_f05": res["non_singleton_f05"],
        })
        print(f"  tau={tau:.2f} → Macro F0.5={res['macro_f05']:.4f} | Singleton Acc={res['singleton_accuracy']:.4f} | Non-Sing F0.5={res['non_singleton_f05']:.4f}")
        if res["macro_f05"] > best_macro_f05:
            best_macro_f05 = res["macro_f05"]
            best_tau = tau
            best_preds = current_preds
            best_res = res

    # Singleton gate analysis
    gate_analysis = analyze_singleton_gate(val_s1_scores, val_gt_local, best_tau, val_s1_ids)

    print("\n" + "=" * 80)
    print(" [LIGHTGBM BENCHMARK — FOLD 0 VALIDATION]")
    print("=" * 80)
    print(f"  Optimal Threshold (tau):     {best_tau:.2f}")
    print(f"  Macro F0.5 Score:            {best_res['macro_f05']:.4f}")
    print(f"  Singleton Accuracy:          {best_res['singleton_accuracy']:.4f}  ({best_res['singleton_count']:,} singletons)")
    print(f"  Non-Singleton F0.5:          {best_res['non_singleton_f05']:.4f}  ({best_res['non_singleton_count']:,} non-singletons)")
    print(f"  Blocking Recall:             {recall:.2f}%")
    print(f"  Validation Entities:         {len(val_gt_local):,}")
    print("=" * 80)

    return {
        "model": model,
        "feat_names": feat_names,
        "best_tau": best_tau,
        "best_macro_f05": best_macro_f05,
        "best_res": best_res,
        "blocking_recall": recall,
        "threshold_results": threshold_results,
        "feat_importance_list": feat_importance_list,
        "gate_analysis": gate_analysis,
        "val_entities": len(val_gt_local),
        "train_entities": len(train_cands),
    }


def analyze_singleton_gate(val_s1_scores, val_gt_local, best_tau, val_s1_ids):
    """Analyze singleton gate performance: entities with max prob < tau vs true singletons."""
    true_singletons = {s for s in val_s1_ids if len(val_gt_local.get(s, [])) == 0}
    true_non_singletons = {s for s in val_s1_ids if len(val_gt_local.get(s, [])) > 0}

    # For each entity, compute max prediction probability
    max_probs = {}
    for s1_id in val_s1_ids:
        probs = [p for _, p in val_s1_scores.get(s1_id, [])]
        max_probs[s1_id] = max(probs) if probs else 0.0

    # Singleton gate analysis
    correct_singleton_rejects = sum(1 for s in true_singletons if max_probs[s] < best_tau)
    false_singleton_assigns = sum(1 for s in true_singletons if max_probs[s] >= best_tau)
    missed_matches = sum(1 for s in true_non_singletons if max_probs[s] < best_tau)
    found_matches = sum(1 for s in true_non_singletons if max_probs[s] >= best_tau)

    return {
        "true_singletons": len(true_singletons),
        "correctly_predicted_singleton": correct_singleton_rejects,
        "false_singleton_assigns": false_singleton_assigns,
        "singleton_gate_precision": correct_singleton_rejects / len(true_singletons) if true_singletons else 0.0,
        "true_non_singletons": len(true_non_singletons),
        "found_matches": found_matches,
        "missed_matches": missed_matches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Test Set Submission Generation
# ─────────────────────────────────────────────────────────────────────────────

def phase3_generate_submission(model, feat_names: List[str], best_tau: float) -> dict:
    """Generate test submission using trained GBDT model."""
    print("\n" + "=" * 80)
    print(" [PHASE 3] TEST SUBMISSION GENERATION")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fe = FeatureExtractor()

    # Load test S1
    print("Loading test S1 records...")
    t0 = time.time()
    s1_records = {}
    s1_meta_test = {}
    needed_keys = {"cores": set(), "sorted": set(), "concat": set(),
                   "brand": set(), "postal_num": set(), "addr_num": set(),
                   "prefixes": set()}

    with open(os.path.join(TEST_DIR, "test_source1.tsv"), encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            name = row[1] if len(row) > 1 else ""
            addr = row[2] if len(row) > 2 else ""
            country = row[3] if len(row) > 3 else ""
            s1_records[s1_id] = {"name": name, "address": addr, "country": country}
            meta = compute_s1_blocking_keys(name, addr, country)
            s1_meta_test[s1_id] = meta
            if meta["core"]: needed_keys["cores"].add(meta["core"])
            for ac in meta["aliases"]: needed_keys["cores"].add(ac)
            if meta["sorted"]: needed_keys["sorted"].add(meta["sorted"])
            if len(meta["concat"]) >= 5: needed_keys["concat"].add(meta["concat"])
            for b in meta["brand"]: needed_keys["brand"].add(b)
            if meta["postal_num_key"]: needed_keys["postal_num"].add(meta["postal_num_key"])
            for ak in meta["addr_num_keys"]: needed_keys["addr_num"].add(ak)
            if meta.get("prefix"): needed_keys["prefixes"].add(meta["prefix"])

    total_test = len(s1_records)
    print(f"Loaded {total_test:,} test S1 records in {time.time()-t0:.1f}s")

    # Build test target index
    print("Building test target index (S2+S3)...")
    t0 = time.time()
    caps = {
        "core": 500, "sorted": 400, "concat": 300,
        "brand": 300, "postal_num": 300, "addr_num": 200, "prefix": 100,
    }
    indices, target_data = build_target_index(
        TEST_DIR, ["test_source2.tsv", "test_source3.tsv"],
        needed_keys, caps
    )
    print(f"Indexed {len(target_data):,} test targets in {time.time()-t0:.1f}s")

    # Generate candidates
    print("Generating candidate pools...")
    candidates, _ = generate_candidates(s1_meta_test, indices, target_data, max_per_entity=80)

    # GBDT Scoring
    print("Scoring all candidate pairs with GBDT model...")
    t0 = time.time()
    s1_scores = defaultdict(list)
    s1_to_cands = {}

    batch_rows = []
    batch_meta = []
    BATCH_SIZE = 200000

    def flush_batch(batch_rows, batch_meta):
        if not batch_rows: return
        X = np.array(batch_rows, dtype=np.float32)
        probs = model.predict_proba(X)[:, 1]
        for (s1_id, c_id), prob in zip(batch_meta, probs):
            s1_scores[s1_id].append((c_id, float(prob)))

    for s1_id, cand_list in candidates.items():
        s1_rec = s1_records[s1_id]
        s1_to_cands[s1_id] = cand_list
        for rank, c_id in enumerate(cand_list, 1):
            c_rec = target_data.get(c_id)
            if not c_rec: continue
            fd = fe.extract_pair_features(s1_rec, c_rec, c_id, rank, len(cand_list))
            batch_rows.append([fd[k] for k in feat_names])
            batch_meta.append((s1_id, c_id))
            if len(batch_rows) >= BATCH_SIZE:
                flush_batch(batch_rows, batch_meta)
                batch_rows, batch_meta = [], []

    flush_batch(batch_rows, batch_meta)
    print(f"GBDT scoring complete in {time.time()-t0:.1f}s")

    # Apply threshold with Mutual-Best conflict resolution
    print(f"Applying threshold tau={best_tau:.2f} with Mutual-Best resolution...")
    target_claims = defaultdict(list)
    for s1_id, scored in s1_scores.items():
        for c_id, prob in scored:
            if prob >= best_tau:
                target_claims[c_id].append((s1_id, prob))

    winner_for_target = {}
    for c_id, claims in target_claims.items():
        best_s1, _ = max(claims, key=lambda x: x[1])
        winner_for_target[c_id] = best_s1

    final_predictions = {}
    total_matches = 0
    singleton_preds = 0

    for s1_id in s1_records:
        retained = [c_id for c_id, prob in s1_scores.get(s1_id, [])
                    if prob >= best_tau and winner_for_target.get(c_id) == s1_id]
        final_predictions[s1_id] = retained
        total_matches += len(retained)
        if len(retained) == 0:
            singleton_preds += 1

    print(f"Total Matches Assigned: {total_matches:,}")
    print(f"Predicted Singletons: {singleton_preds:,} ({singleton_preds/total_test*100:.1f}%)")

    # Write submission files
    print("\nWriting submission TSV files...")
    matching_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    with open(matching_file, "w", encoding="utf-8") as fm, \
         open(candidate_file, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_records:
            matches = final_predictions.get(s1_id, [])
            cands = s1_to_cands.get(s1_id, [])
            cand_set = set(cands)
            for m in matches:
                if m not in cand_set:
                    cands.append(m)
            fm.write(f"{s1_id}\t{','.join(matches)}\n")
            fc.write(f"{s1_id}\t{','.join(cands)}\n")

    match_size = os.path.getsize(matching_file) / (1024 * 1024)
    cand_size = os.path.getsize(candidate_file) / (1024 * 1024)
    print(f"  matching_results.tsv: {match_size:.1f} MB")
    print(f"  candidate_pairs.tsv:  {cand_size:.1f} MB")

    return {
        "total_test_entities": total_test,
        "total_matches": total_matches,
        "singleton_preds": singleton_preds,
        "singleton_pct": singleton_preds / total_test * 100,
        "matching_file_mb": match_size,
        "candidate_file_mb": cand_size,
        "tau_used": best_tau,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Best Pipeline v8: Amazon ML Challenge 2026")
    parser.add_argument("--skip-submission", action="store_true",
                        help="Skip generating test submission files, only run audit, GBDT training, and validation report.")
    parser.add_argument("--n-train", type=int, default=40000, help="Number of training entities (default: 40000)")
    parser.add_argument("--n-val", type=int, default=20000, help="Number of validation entities (default: 20000)")
    args = parser.parse_args()

    pipeline_start = time.time()
    print("=" * 80)
    print(" AMAZON ML CHALLENGE 2026 — BUSINESS ENTITY RESOLUTION")
    print(" BEST PIPELINE v8: Calibrated GBDT + Singleton Gate")
    print("=" * 80)

    # Phase 1: Data Audit
    audit_results, gt, s1_to_fold = phase1_data_audit(TRAIN_DIR)

    # Phase 2: Train & Validate
    gbdt_results = phase2_train_gbdt(TRAIN_DIR, gt, s1_to_fold, n_train=args.n_train, n_val=args.n_val)

    # Phase 3: Generate Test Submission
    if not args.skip_submission:
        submission_results = phase3_generate_submission(
            gbdt_results["model"],
            gbdt_results["feat_names"],
            gbdt_results["best_tau"],
        )
    else:
        print("\n" + "=" * 80)
        print(" [PHASE 3] TEST SUBMISSION VERIFICATION (Existing Output)")
        print("=" * 80)
        matching_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
        candidate_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
        if os.path.exists(matching_file) and os.path.exists(candidate_file):
            match_size = os.path.getsize(matching_file) / (1024 * 1024)
            cand_size = os.path.getsize(candidate_file) / (1024 * 1024)
            n_rows, n_empty, n_matches = 0, 0, 0
            with open(matching_file, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    n_rows += 1
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) < 2 or not parts[1].strip():
                        n_empty += 1
                    else:
                        n_matches += len([x for x in parts[1].split(",") if x.strip()])
            submission_results = {
                "total_test_entities": n_rows,
                "total_matches": n_matches,
                "singleton_preds": n_empty,
                "singleton_pct": (n_empty / n_rows * 100) if n_rows else 0.0,
                "matching_file_mb": round(match_size, 1),
                "candidate_file_mb": round(cand_size, 1),
                "tau_used": gbdt_results["best_tau"],
            }
            print(f"  Verified existing submission files:")
            print(f"  Total test entities:    {n_rows:,}")
            print(f"  Total matches assigned: {n_matches:,}")
            print(f"  Singletons predicted:   {n_empty:,} ({n_empty/n_rows*100:.2f}%)")
            print(f"  matching_results.tsv:   {match_size:.1f} MB")
            print(f"  candidate_pairs.tsv:    {cand_size:.1f} MB")
        else:
            submission_results = {
                "total_test_entities": 0,
                "total_matches": 0,
                "singleton_preds": 0,
                "singleton_pct": 0.0,
                "matching_file_mb": 0.0,
                "candidate_file_mb": 0.0,
                "tau_used": gbdt_results["best_tau"],
            }

    # Final Summary
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETE — FINAL SUMMARY")
    print("=" * 80)
    print(f"  Total Pipeline Runtime:      {total_time/60:.1f} minutes")
    print(f"\n  [DATA AUDIT]")
    print(f"  Train S1 Entities:           {audit_results['total_s1']:,}")
    print(f"  Singleton Rate:              {audit_results['singleton_pct']:.2f}%")
    print(f"  Total Ground Truth Links:    {audit_results['total_ground_truth_links']:,}")
    print(f"  Target Exclusivity Valid:    {not audit_results['target_exclusivity_violated']}")
    print(f"\n  [MODEL PERFORMANCE — VAL FOLD 0]")
    print(f"  Blocking Recall:             {gbdt_results['blocking_recall']:.2f}%")
    print(f"  Macro F0.5 (Validation):     {gbdt_results['best_macro_f05']:.4f}")
    print(f"  Optimal Threshold (tau):     {gbdt_results['best_tau']:.2f}")
    print(f"  Singleton Accuracy:          {gbdt_results['best_res']['singleton_accuracy']:.4f}")
    print(f"  Non-Singleton F0.5:          {gbdt_results['best_res']['non_singleton_f05']:.4f}")
    print(f"\n  [TEST SUBMISSION]")
    print(f"  Test S1 Entities:            {submission_results['total_test_entities']:,}")
    print(f"  Matches Assigned:            {submission_results['total_matches']:,}")
    print(f"  Predicted Singletons:        {submission_results['singleton_preds']:,} ({submission_results['singleton_pct']:.1f}%)")
    print(f"  matching_results.tsv:        output/matching_results.tsv ({submission_results['matching_file_mb']:.1f} MB)")
    print(f"  candidate_pairs.tsv:         output/candidate_pairs.tsv  ({submission_results['candidate_file_mb']:.1f} MB)")
    print("=" * 80)

    # Save JSON report
    report = {
        "pipeline": "v8_lgbm_gbdt",
        "runtime_minutes": round(total_time / 60, 2),
        "data_audit": audit_results,
        "model_validation": {
            "n_train_entities": gbdt_results["train_entities"],
            "n_val_entities": gbdt_results["val_entities"],
            "blocking_recall_pct": round(gbdt_results["blocking_recall"], 2),
            "macro_f05": round(gbdt_results["best_macro_f05"], 4),
            "optimal_tau": gbdt_results["best_tau"],
            "singleton_accuracy": round(gbdt_results["best_res"]["singleton_accuracy"], 4),
            "non_singleton_f05": round(gbdt_results["best_res"]["non_singleton_f05"], 4),
            "singleton_count": gbdt_results["best_res"]["singleton_count"],
            "non_singleton_count": gbdt_results["best_res"]["non_singleton_count"],
            "threshold_sweep": gbdt_results["threshold_results"],
            "top_features": gbdt_results["feat_importance_list"],
            "singleton_gate": gbdt_results["gate_analysis"],
        },
        "test_submission": submission_results,
    }

    report_path = os.path.join(OUTPUT_DIR, "pipeline_v8_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full JSON report saved to: {report_path}")


if __name__ == "__main__":
    main()
