#!/usr/bin/env python3
"""Experiment: Calibrated GBDT + Dedicated Singleton Gate + Multi-Fold Validation.

Systematic implementation of:
1. Calibration Fix: scale_pos_weight = 1.0 (unweighted logloss)
2. Dedicated Singleton Gate: Entity-level gate before pairwise threshold
3. Multi-Fold Validation: Test on Fold 0 and Fold 1
4. Hybrid Ensemble: Combine v6 rule engine with calibrated GBDT gated by street number
"""

from collections import defaultdict
import csv
import json
import os
import random
import re
import sys
import time
from typing import Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz

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

TRAIN_DIR = "dataset/train"
OUTPUT_DIR = "output"


def compute_s1_blocking_keys(name: str, address: str, country: str) -> dict:
    norm = normalize_text(name)
    core = extract_core_name(name)
    sorted_k = extract_sorted_key(name)
    concat_k = norm.replace(" ", "")
    brand = extract_distinctive_name_tokens(name)
    postal, num, dist_tokens = extract_postal_and_number(address)

    alias_cores = []
    if any(w in norm for w in ("aka", "dba", "fka", "doing business as", "trading as")):
        parts = re.split(r'\b(?:aka|dba|fka|doing business as|trading as)\b', norm)
        if len(parts) > 1:
            for p in parts:
                p = p.strip()
                ac = extract_core_name(p) if p else ""
                if ac and len(ac) >= 3:
                    alias_cores.append(ac)

    postal_num_key = f"{postal}_{num}" if (postal and num) else ""
    addr_num_keys = [f"{num}_{dt}" for dt in sorted(dist_tokens)[:4]] if (num and dist_tokens) else []
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
                t_postal = ""
                t_dist = set()
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
                        "core": t_core, "sorted": t_sorted, "concat": t_concat,
                        "num": t_num, "postal": t_postal, "dist_tokens": t_dist,
                        "distinctive": t_brand,
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

        filtered = list(cands)
        if len(filtered) > max_per_entity:
            s1_core = meta["core"]
            s1_num = meta.get("num", "")
            def score_cand(c_id):
                t_rec = target_data.get(c_id, {})
                sc = fuzz.ratio(s1_core, t_rec.get("core", ""))
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


def compute_v6_rule_matches(s1_records: dict, s1_meta: dict, s1_candidates: dict, target_data: dict) -> Dict[str, List[Tuple[str, float]]]:
    """Compute high-precision matches using Engine v6 rules."""
    s1_rule_matches = defaultdict(list)

    for s1_id, cand_list in s1_candidates.items():
        s1_rec = s1_records[s1_id]
        meta = s1_meta[s1_id]
        s1_core = meta["core"]
        s1_sorted = meta["sorted"]
        s1_concat = meta["concat"]
        s1_num = meta["num"]
        s1_postal = meta["postal"]
        s1_dist = meta["dist_tokens"]
        s1_brand = meta["brand"]
        s1_aliases = meta["aliases"]
        has_s1_addr = bool(s1_rec["address"].strip())

        for c_id in cand_list:
            t = target_data.get(c_id)
            if not t:
                continue

            t_core = t["core"]
            t_sorted = t["sorted"]
            t_concat = t["concat"]
            t_num = t["num"]
            t_postal = t["postal"]
            t_dist = t.get("dist_tokens", set())
            t_brand = t.get("distinctive", set())
            has_t_addr = bool(t["address"].strip())

            # Address conflicts
            num_match = (s1_num and t_num and s1_num == t_num)
            num_conflict = (s1_num and t_num and s1_num != t_num)
            postal_match = (s1_postal and t_postal and s1_postal == t_postal)
            postal_conflict = (s1_postal and t_postal and s1_postal != t_postal)
            dist_overlap = len(s1_dist.intersection(t_dist))

            if has_s1_addr and has_t_addr:
                if postal_conflict and dist_overlap == 0:
                    continue
                if num_conflict and dist_overlap == 0:
                    continue

            # Name similarity
            lev = fuzz.ratio(s1_core, t_core) / 100.0 if (s1_core and t_core) else 0.0
            sort_r = fuzz.token_sort_ratio(meta["norm"], normalize_text(t["name"])) / 100.0 if (meta["norm"] and t["name"]) else 0.0
            max_name = max(lev, sort_r)
            brand_overlap = bool(s1_brand.intersection(t_brand))

            score = 0.0
            # 1. Exact Core Name Match
            if s1_core and t_core and s1_core == t_core:
                score = 0.95
            # 2. Alias / DBA Core Match
            elif any(a and a == t_core for a in s1_aliases):
                score = 0.94
            # 3. Sorted Core Match
            elif s1_sorted and t_sorted == s1_sorted:
                score = 0.92
            # 4. Concatenated Match
            elif s1_concat and (t_concat == s1_concat):
                score = 0.90 if (postal_match or num_match or dist_overlap >= 1 or not has_s1_addr) else 0.82
            # 5. High Fuzzy Name Match (>= 0.86)
            elif max_name >= 0.86:
                if postal_match or num_match or dist_overlap >= 1 or not has_s1_addr or not has_t_addr:
                    score = 0.87
            # 6. Moderate Name Match (>= 0.80) with brand overlap or strong address confirmation
            elif max_name >= 0.80 and (brand_overlap or (num_match and dist_overlap >= 1)):
                if postal_match or num_match:
                    score = 0.83
                elif dist_overlap >= 2:
                    score = 0.81

            if score >= 0.80:
                s1_rule_matches[s1_id].append((c_id, score))

    return s1_rule_matches


def main():
    print("=" * 80)

    print(" CALIBRATED GBDT + DEDICATED SINGLETON GATE + MULTI-FOLD EVALUATION")
    print("=" * 80)


    # 1. Load splits & ground truth
    with open(os.path.join(TRAIN_DIR, "cv_splits_5fold.json")) as f:
        s1_to_fold = json.load(f)

    gt = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            matches = [x.strip() for x in row[1].split(",") if x.strip()] if len(row) > 1 and row[1].strip() else []
            gt[s1_id] = matches

    # Folds setup
    random.seed(42)
    fold0_all = [s for s, f in s1_to_fold.items() if f == 0]
    fold1_all = [s for s, f in s1_to_fold.items() if f == 1]
    random.shuffle(fold0_all)
    random.shuffle(fold1_all)

    # We evaluate on 15,000 entities from Fold 0 AND 15,000 entities from Fold 1
    # We train on 30,000 entities from Fold 1 (different subset) or Fold 2
    fold2_all = [s for s, f in s1_to_fold.items() if f == 2]
    random.shuffle(fold2_all)

    N_TRAIN = 30000
    N_VAL_PER_FOLD = 12000

    train_s1_ids = set(fold2_all[:N_TRAIN])
    val_fold0_ids = set(fold0_all[:N_VAL_PER_FOLD])
    val_fold1_ids = set(fold1_all[:N_VAL_PER_FOLD])

    all_s1_needed = train_s1_ids | val_fold0_ids | val_fold1_ids
    print(f"Entities: Train={len(train_s1_ids):,} (Fold 2) | Val Fold 0={len(val_fold0_ids):,} | Val Fold 1={len(val_fold1_ids):,}")

    # Load S1 records
    print("Streaming S1 records...")
    t0 = time.time()
    s1_records = {}
    s1_meta = {}
    needed_keys = {"cores": set(), "sorted": set(), "concat": set(),
                   "brand": set(), "postal_num": set(), "addr_num": set(),
                   "prefixes": set()}

    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            if s1_id in all_s1_needed:
                name = row[1] if len(row) > 1 else ""
                addr = row[2] if len(row) > 2 else ""
                country = row[3] if len(row) > 3 else ""
                s1_records[s1_id] = {"name": name, "address": addr, "country": country}
                meta = compute_s1_blocking_keys(name, addr, country)
                s1_meta[s1_id] = meta
                if meta["core"]: needed_keys["cores"].add(meta["core"])
                for ac in meta["aliases"]: needed_keys["cores"].add(ac)
                if meta["sorted"]: needed_keys["sorted"].add(meta["sorted"])
                if len(meta["concat"]) >= 5: needed_keys["concat"].add(meta["concat"])
                for b in meta["brand"]: needed_keys["brand"].add(b)
                if meta["postal_num_key"]: needed_keys["postal_num"].add(meta["postal_num_key"])
                for ak in meta["addr_num_keys"]: needed_keys["addr_num"].add(ak)
                if meta.get("prefix"): needed_keys["prefixes"].add(meta["prefix"])

    print(f"Loaded {len(s1_records):,} S1 records in {time.time()-t0:.1f}s")

    # Build target index
    print("Building target index (S2+S3)...")
    t0 = time.time()
    caps = {
        "core": 500, "sorted": 400, "concat": 300,
        "brand": 300, "postal_num": 300, "addr_num": 200, "prefix": 100,
    }
    indices, target_data = build_target_index(
        TRAIN_DIR, ["train_source2.tsv", "train_source3.tsv"],
        needed_keys, caps
    )
    print(f"Indexed {len(target_data):,} targets in {time.time()-t0:.1f}s")

    # Generate candidates
    print("Generating candidate sets...")
    all_gt = {s: gt.get(s, []) for s in all_s1_needed}
    candidates, overall_recall = generate_candidates(s1_meta, indices, target_data, all_gt, max_per_entity=80)
    print(f"[CANDIDATE POOL RECALL] {overall_recall:.2f}% of ground truth captured in candidate pools")

    # Feature extractor
    fe = FeatureExtractor()

    # Build training matrix:
    # Key change: balance negatives per entity to reflect candidate distribution better
    # Sample up to 8 negatives per entity (including both hard and random)
    print("\nBuilding training matrix...")
    t0 = time.time()
    X_train_rows, y_train_labels = [], []
    feat_names = None
    MAX_TRAIN_NEG = 8

    for s1_id in train_s1_ids:
        c_list = candidates.get(s1_id, [])
        s1_rec = s1_records[s1_id]
        gold_set = set(all_gt.get(s1_id, []))
        neg_count = 0
        for rank, c_id in enumerate(c_list, 1):
            c_rec = target_data.get(c_id)
            if not c_rec: continue
            is_pos = 1 if c_id in gold_set else 0
            if is_pos == 0:
                if neg_count >= MAX_TRAIN_NEG: continue
                neg_count += 1
            fd = fe.extract_pair_features(s1_rec, c_rec, c_id, rank, len(c_list))
            if feat_names is None:
                feat_names = sorted(fd.keys())
            X_train_rows.append([fd[k] for k in feat_names])
            y_train_labels.append(is_pos)

    X_train = np.array(X_train_rows, dtype=np.float32)
    y_train = np.array(y_train_labels, dtype=np.int32)
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    print(f"Training Matrix: {X_train.shape} | Pos: {pos_count:,} | Neg: {neg_count:,} (ratio 1:{neg_count/pos_count:.1f})")

    # STEP 1: CALIBRATION FIX — UNWEIGHTED LOGLOSS (scale_pos_weight=1.0)
    print("\n[STEP 1] Training Calibrated LightGBM (scale_pos_weight=1.0)...")
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 25,
        "scale_pos_weight": 1.0,  # CALIBRATED UNWEIGHTED POSTERIOR
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**lgb_params)
    t0 = time.time()
    model.fit(X_train, y_train)
    model.booster_.save_model("output/gbdt_model_v9.txt")
    print(f"Model trained in {time.time()-t0:.1f}s")

    # Evaluate on both Fold 0 and Fold 1
    def evaluate_fold(fold_name: str, val_ids: Set[str]):
        print(f"\n{'='*40} Evaluating on {fold_name} ({len(val_ids):,} entities) {'='*40}")
        local_gt = {s: all_gt.get(s, []) for s in val_ids}
        n_sing = sum(1 for v in local_gt.values() if len(v) == 0)
        n_nonsing = len(local_gt) - n_sing
        print(f"Gold stats: {n_sing:,} singletons ({n_sing/len(val_ids)*100:.1f}%), {n_nonsing:,} non-singletons")

        # 1. First, score with v6 rule engine baseline on this fold
        print("\n--- Running v6 Rule Engine Baseline ---")
        v6_matches = compute_v6_rule_matches(s1_records, s1_meta, {s: candidates[s] for s in val_ids}, target_data)

        # Apply 1-to-1 mutual-best to v6
        v6_target_claims = defaultdict(list)
        for s1_id, matches in v6_matches.items():
            for c_id, score in matches:
                v6_target_claims[c_id].append((s1_id, score))

        v6_winner = {c_id: max(claims, key=lambda x: x[1])[0] for c_id, claims in v6_target_claims.items()}
        v6_preds = {}
        for s1_id in val_ids:
            v6_preds[s1_id] = [c_id for c_id, sc in v6_matches.get(s1_id, []) if v6_winner.get(c_id) == s1_id]

        v6_res = evaluate_macro_f05(v6_preds, local_gt)
        print(f"  v6 Rule Engine Score:   Macro F0.5 = {v6_res['macro_f05']:.4f}")
        print(f"  v6 Singleton Acc:       {v6_res['singleton_accuracy']:.4f}")
        print(f"  v6 Non-Singleton F0.5:  {v6_res['non_singleton_f05']:.4f}")

        # 2. Extract GBDT features and predict probabilities
        print("\n--- Scoring with Calibrated GBDT ---")
        val_rows, val_meta_info = [], []
        for s1_id in val_ids:
            s1_rec = s1_records[s1_id]
            c_list = candidates.get(s1_id, [])
            for rank, c_id in enumerate(c_list, 1):
                c_rec = target_data.get(c_id)
                if not c_rec: continue
                fd = fe.extract_pair_features(s1_rec, c_rec, c_id, rank, len(c_list))
                val_rows.append([fd[k] for k in feat_names])
                # Save useful metadata for the singleton gate & structural verification:
                # s1_num, c_num, core_exact, lev_ratio, num_match
                num_m = (s1_meta[s1_id]["num"] and c_rec["num"] and s1_meta[s1_id]["num"] == c_rec["num"])
                core_eq = (s1_meta[s1_id]["core"] and c_rec["core"] and s1_meta[s1_id]["core"] == c_rec["core"])
                val_meta_info.append((s1_id, c_id, num_m, core_eq, fd.get("lev_ratio", 0.0), fd.get("addr_token_sort", 0.0)))

        X_val = np.array(val_rows, dtype=np.float32)
        val_probs = model.predict_proba(X_val)[:, 1]

        # Group probabilities by s1_id
        gbdt_scores = defaultdict(list)
        for (s1_id, c_id, num_m, core_eq, lev, a_sort), prob in zip(val_meta_info, val_probs):
            gbdt_scores[s1_id].append({
                "cand_id": c_id,
                "prob": float(prob),
                "num_match": bool(num_m),
                "core_exact": bool(core_eq),
                "lev": float(lev),
                "addr_sort": float(a_sort),
            })

        # Test A: Calibrated GBDT WITHOUT singleton gate (pure pairwise threshold sweep)
        print("\n--- Test A: Calibrated GBDT (No Singleton Gate) ---")
        best_no_gate_f05 = 0.0
        best_no_gate_tau = 0.5
        for tau in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
            t_claims = defaultdict(list)
            for s1_id in val_ids:
                for item in gbdt_scores.get(s1_id, []):
                    if item["prob"] >= tau:
                        t_claims[item["cand_id"]].append((s1_id, item["prob"]))
            winner = {c: max(cl, key=lambda x: x[1])[0] for c, cl in t_claims.items()}
            cur_preds = {s: [item["cand_id"] for item in gbdt_scores.get(s, [])
                             if item["prob"] >= tau and winner.get(item["cand_id"]) == s]
                         for s in val_ids}
            res = evaluate_macro_f05(cur_preds, local_gt)
            if res["macro_f05"] > best_no_gate_f05:
                best_no_gate_f05 = res["macro_f05"]
                best_no_gate_tau = tau
            print(f"  tau={tau:.2f} -> Macro F0.5 = {res['macro_f05']:.4f} | Sing Acc = {res['singleton_accuracy']:.4f} | Non-Sing = {res['non_singleton_f05']:.4f}")

        # Test B: Calibrated GBDT WITH Dedicated Singleton Gate
        print("\n--- Test B: Calibrated GBDT WITH Dedicated Singleton Gate ---")
        # Grid search over singleton gate threshold (tau_gate) and pairwise threshold (tau_pair)
        best_gate_f05 = 0.0
        best_gate_params = None
        best_gate_preds = None

        for tau_gate in [0.40, 0.50, 0.60, 0.70, 0.75, 0.80]:
            for tau_pair in [0.30, 0.40, 0.50, 0.60, 0.70]:
                if tau_pair > tau_gate: continue

                # Apply singleton gate:
                # Entity qualifies if max_prob >= tau_gate
                # OR (max_prob >= 0.35 and (top candidate has num_match or core_exact))
                passed_entities = set()
                for s1_id in val_ids:
                    items = gbdt_scores.get(s1_id, [])
                    if not items: continue
                    max_prob = max(it["prob"] for it in items)
                    top_item = max(items, key=lambda x: x["prob"])
                    # Gate rule:
                    is_confident = (max_prob >= tau_gate) or (max_prob >= 0.35 and (top_item["num_match"] or top_item["core_exact"]))
                    if is_confident:
                        passed_entities.add(s1_id)

                # Pairwise claims among passed entities
                t_claims = defaultdict(list)
                for s1_id in passed_entities:
                    for item in gbdt_scores.get(s1_id, []):
                        if item["prob"] >= tau_pair:
                            t_claims[item["cand_id"]].append((s1_id, item["prob"]))
                winner = {c: max(cl, key=lambda x: x[1])[0] for c, cl in t_claims.items()}

                cur_preds = {}
                for s1_id in val_ids:
                    if s1_id not in passed_entities:
                        cur_preds[s1_id] = []
                    else:
                        cur_preds[s1_id] = [item["cand_id"] for item in gbdt_scores.get(s1_id, [])
                                            if item["prob"] >= tau_pair and winner.get(item["cand_id"]) == s1_id]

                res = evaluate_macro_f05(cur_preds, local_gt)
                if res["macro_f05"] > best_gate_f05:
                    best_gate_f05 = res["macro_f05"]
                    best_gate_params = (tau_gate, tau_pair)
                    best_gate_preds = cur_preds
                    print(f"  * NEW BEST * tau_gate={tau_gate:.2f}, tau_pair={tau_pair:.2f} -> Macro F0.5={res['macro_f05']:.4f} | Sing Acc={res['singleton_accuracy']:.4f} | Non-Sing={res['non_singleton_f05']:.4f}")

        # Test C: HYBRID ENSEMBLE (v6 Rule Matcher + Calibrated GBDT Gated by Street Number)
        print("\n--- Test C: Hybrid Ensemble (v6 + GBDT gated by street number) ---")
        best_ens_f05 = 0.0
        best_ens_params = None

        for gbdt_tau in [0.40, 0.50, 0.60, 0.70, 0.80]:
            # Ensemble logic:
            # S1 starts with v6 predictions.
            # Additional candidates from GBDT are added IF:
            # - GBDT prob >= gbdt_tau AND (num_match == True or core_exact == True)
            ens_claims = defaultdict(list)

            for s1_id in val_ids:
                # v6 matches get weight 0.90 + v6_score*0.05
                for c_id, sc in v6_matches.get(s1_id, []):
                    ens_claims[c_id].append((s1_id, 0.90 + sc * 0.05))

                # GBDT candidates with street number confirmation or core name confirmation
                for item in gbdt_scores.get(s1_id, []):
                    c_id = item["cand_id"]
                    prob = item["prob"]
                    if prob >= gbdt_tau and (item["num_match"] or item["core_exact"]):
                        ens_claims[c_id].append((s1_id, prob))

            winner = {c: max(cl, key=lambda x: x[1])[0] for c, cl in ens_claims.items()}

            ens_preds = {}
            for s1_id in val_ids:
                cands_to_check = set([c for c, _ in v6_matches.get(s1_id, [])])
                for item in gbdt_scores.get(s1_id, []):
                    if item["prob"] >= gbdt_tau and (item["num_match"] or item["core_exact"]):
                        cands_to_check.add(item["cand_id"])

                retained = [c for c in cands_to_check if winner.get(c) == s1_id]
                ens_preds[s1_id] = retained

            res = evaluate_macro_f05(ens_preds, local_gt)
            print(f"  gbdt_tau={gbdt_tau:.2f} -> Ensemble Macro F0.5={res['macro_f05']:.4f} | Sing Acc={res['singleton_accuracy']:.4f} | Non-Sing={res['non_singleton_f05']:.4f}")
            if res["macro_f05"] > best_ens_f05:
                best_ens_f05 = res["macro_f05"]
                best_ens_params = gbdt_tau

        return {
            "v6_f05": v6_res["macro_f05"],
            "best_no_gate_f05": best_no_gate_f05,
            "best_gate_f05": best_gate_f05,
            "best_gate_params": best_gate_params,
            "best_ens_f05": best_ens_f05,
            "best_ens_params": best_ens_params,
        }

    # Evaluate Fold 0
    res_fold0 = evaluate_fold("Fold 0", val_fold0_ids)

    # Evaluate Fold 1
    res_fold1 = evaluate_fold("Fold 1", val_fold1_ids)

    print("\n" + "=" * 80)
    print(" SUMMARY ACROSS FOLDS")
    print("=" * 80)

    print(f"  Metric                     | Fold 0        | Fold 1        | Mean")
    print(f"  ---------------------------|---------------|---------------|------")
    print(f"  v6 Rule Engine             | {res_fold0['v6_f05']:.4f}        | {res_fold1['v6_f05']:.4f}        | {(res_fold0['v6_f05']+res_fold1['v6_f05'])/2:.4f}")
    print(f"  GBDT (No Gate)             | {res_fold0['best_no_gate_f05']:.4f}        | {res_fold1['best_no_gate_f05']:.4f}        | {(res_fold0['best_no_gate_f05']+res_fold1['best_no_gate_f05'])/2:.4f}")
    print(f"  GBDT (With Singleton Gate) | {res_fold0['best_gate_f05']:.4f}        | {res_fold1['best_gate_f05']:.4f}        | {(res_fold0['best_gate_f05']+res_fold1['best_gate_f05'])/2:.4f}")
    print(f"  Hybrid Ensemble (v6+GBDT)  | {res_fold0['best_ens_f05']:.4f}        | {res_fold1['best_ens_f05']:.4f}        | {(res_fold0['best_ens_f05']+res_fold1['best_ens_f05'])/2:.4f}")
    print("=" * 80)



if __name__ == "__main__":
    main()
