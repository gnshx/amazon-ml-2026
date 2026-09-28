#!/usr/bin/env python3
"""Rapid precision rule sweeper using the cached 50,000-entity benchmark."""

from collections import defaultdict
import csv
import json
import os
import pickle
import sys
import time
from rapidfuzz import fuzz, distance

sys.path.insert(0, "src")
from metrics import evaluate_macro_f05

TRAIN_DIR = "dataset/train"
CACHE_FILE = "scratch/fixed_benchmark_cache_50k.pkl"

def main():
    print("Loading benchmark cache...")
    t0 = time.time()
    with open(CACHE_FILE, "rb") as f:
        data = pickle.load(f)
    s1_records = data["s1_records"]
    target_table = data["target_table"]
    target_ids = data["target_ids"]
    idx_core = defaultdict(list, data["idx_core"])
    idx_sorted = defaultdict(list, data["idx_sorted"])
    idx_concat = defaultdict(list, data["idx_concat"])
    idx_brand = defaultdict(list, data["idx_brand"])
    idx_num_p = defaultdict(list, data["idx_num_p"])
    idx_post_p = defaultdict(list, data["idx_post_p"])
    idx_addr_k = defaultdict(list, data["idx_addr_k"])
    print(f"Loaded cache in {time.time()-t0:.2f}s ({len(s1_records):,} S1, {len(target_table):,} targets).")

    # Load splits & ground truth
    with open(os.path.join(TRAIN_DIR, "cv_splits_5fold.json")) as f:
        s1_to_fold = json.load(f)
    fold0_entities = sorted([s for s, fold in s1_to_fold.items() if fold == 0])[:25000]
    fold1_entities = sorted([s for s, fold in s1_to_fold.items() if fold == 1])[:25000]
    all_eval_ids = set(fold0_entities + fold1_entities)

    eval_gt = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row[0] in all_eval_ids:
                matches = [x.strip() for x in row[1].split(",") if x.strip()] if len(row) > 1 and row[1].strip() else []
                eval_gt[row[0]] = matches

    gt_f0 = {s: eval_gt.get(s, []) for s in fold0_entities}
    gt_f1 = {s: eval_gt.get(s, []) for s in fold1_entities}

    # Match scoring
    t0 = time.time()
    s1_to_scored = defaultdict(list)
    target_claims = defaultdict(list)

    for s1 in s1_records:
        s1_id = s1["id"]
        core_s1 = s1["core"]
        sorted_s1 = s1["sorted"]
        concat_s1 = s1["concat"]
        brand_tokens = s1["distinctive"]
        s1_p, s1_n = s1["postal"], s1["num"]
        s1_dist = s1["dist_addr"]
        s1_country = s1["country"]
        s1_aliases = s1["aliases"]
        num_p_s1 = s1["num_prefix"]
        post_p_s1 = s1["post_prefix"]
        addr_keys_s1 = s1["addr_keys"]
        clean_s1_addr = s1["clean_addr"]
        norm_s1_name = s1["norm_name"]
        s1_is_non_ascii = s1["is_non_ascii"]
        has_s1_addr = bool(clean_s1_addr)

        cands = set()
        cands.update(idx_core.get(core_s1, [])[:30])
        for a in s1_aliases:
            cands.update(idx_core.get(a, [])[:30])
        cands.update(idx_sorted.get(sorted_s1, [])[:30])
        if len(concat_s1) >= 5:
            cands.update(idx_concat.get(concat_s1, [])[:20])
        for bt in brand_tokens[:3]:
            cands.update(idx_brand.get(bt, [])[:20])
        if num_p_s1:
            cands.update(idx_num_p.get(num_p_s1, [])[:20])
        if post_p_s1:
            cands.update(idx_post_p.get(post_p_s1, [])[:20])
        for ak in addr_keys_s1:
            cands.update(idx_addr_k.get(ak, [])[:10])

        scored_matches = []
        for t_idx in cands:
            t = target_table[t_idx]
            (core_t, sorted_t, concat_t, norm_t_name, clean_t_addr, t_country, t_p, t_n, t_dist, target_is_non_ascii, t_aliases) = t

            if s1_country and t_country and s1_country != t_country:
                continue

            has_t_addr = bool(clean_t_addr)
            postal_match = (s1_p and t_p and s1_p == t_p)
            postal_conflict = (s1_p and t_p and s1_p != t_p)
            num_match = (s1_n and t_n and s1_n == t_n)
            num_conflict = (s1_n and t_n and s1_n != t_n)
            dist_overlap = len(s1_dist.intersection(t_dist)) if (s1_dist and t_dist) else 0

            addr_token_sim = (
                fuzz.token_set_ratio(clean_s1_addr, clean_t_addr) / 100.0
                if (has_s1_addr and has_t_addr)
                else None
            )

            if has_s1_addr and has_t_addr:
                if addr_token_sim is not None and addr_token_sim < 0.35:
                    continue
                if s1_dist and t_dist and not postal_match and dist_overlap == 0:
                    continue
                if postal_conflict and addr_token_sim is not None and addr_token_sim < 0.70:
                    continue
                if num_conflict and dist_overlap == 0:
                    continue
            else:
                core_words = core_s1.split()
                if len(core_words) < 2 and len(core_s1) < 14:
                    continue

            lev = fuzz.ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
            sort_r = fuzz.token_sort_ratio(norm_s1_name, norm_t_name) / 100.0 if (norm_s1_name and norm_t_name) else 0.0
            max_name = max(lev, sort_r)

            score = 0.0
            if core_s1 and core_t and core_s1 == core_t:
                score = 0.95
            elif any(a and a == core_t for a in s1_aliases) or any(a and a == core_s1 for a in t_aliases):
                score = 0.94
            elif sorted_s1 and sorted_t == sorted_s1:
                score = 0.92
            elif (concat_s1 and concat_t == concat_s1) or (concat_s1 and norm_t_name.replace(" ", "") == concat_s1):
                score = 0.90 if (postal_match or num_match or dist_overlap >= 1 or not has_s1_addr) else 0.82
            elif (s1_is_non_ascii or target_is_non_ascii) and num_match and dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.80:
                score = 0.88
            elif max_name >= 0.88:
                if postal_match or num_match or dist_overlap >= 1 or not has_s1_addr or not has_t_addr:
                    score = 0.87
            elif max_name >= 0.72:
                if postal_match and (num_match or dist_overlap >= 1):
                    score = 0.84
                elif num_match and dist_overlap >= 1:
                    score = 0.82
                elif dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.82:
                    score = 0.80

            if score >= 0.80:
                scored_matches.append((t_idx, score, addr_token_sim if addr_token_sim is not None else 0.50))

        s1_to_scored[s1_id] = scored_matches
        for t_idx, sc, a_sim in scored_matches:
            target_claims[t_idx].append((s1_id, sc, a_sim))

    print(f"Scoring completed in {time.time()-t0:.2f}s.")

    # Mutual-best winner
    winner_for_target = {}
    for t_idx, claims in target_claims.items():
        best_s1, best_sc, best_asim = max(claims, key=lambda x: (x[1], x[2]))
        winner_for_target[t_idx] = best_s1

    print("\n" + "=" * 95)
    print(f"{'Cutoff':<8} | {'Overall F0.5':<12} | {'Fold 0 F0.5':<11} | {'Fold 1 F0.5':<11} | {'Sing Acc':<10} | {'Non-Sing F0.5':<13} | {'Matches':<8}")
    print("=" * 95)

    for sc_cutoff in [0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88]:
        preds = {}
        tot_m = 0
        for s1 in s1_records:
            s1_id = s1["id"]
            retained = [target_ids[t_idx] for t_idx, sc, a_sim in s1_to_scored.get(s1_id, [])
                        if winner_for_target.get(t_idx) == s1_id and sc >= sc_cutoff]
            preds[s1_id] = retained
            tot_m += len(retained)

        res_ov = evaluate_macro_f05(preds, eval_gt)
        preds_f0 = {s: preds[s] for s in fold0_entities}
        preds_f1 = {s: preds[s] for s in fold1_entities}
        res_f0 = evaluate_macro_f05(preds_f0, gt_f0)
        res_f1 = evaluate_macro_f05(preds_f1, gt_f1)

        print(f">= {sc_cutoff:<5.2f} | {res_ov['macro_f05']:<12.4f} | {res_f0['macro_f05']:<11.4f} | {res_f1['macro_f05']:<11.4f} | {res_ov['singleton_accuracy']:<10.4f} | {res_ov['non_singleton_f05']:<13.4f} | {tot_m:<8,}")
    print("=" * 95)

if __name__ == "__main__":
    main()
