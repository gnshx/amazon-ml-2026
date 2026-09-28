#!/usr/bin/env python3
"""Test high-recall blocker incorporating address composite keys & distinctive tokens."""

from collections import defaultdict
import csv
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.abspath("src"))
from feature_extractor import normalize_text, extract_core_name, extract_sorted_key, extract_distinctive_name_tokens, extract_postal_and_number
from rapidfuzz import fuzz

TRAIN_DIR = "dataset/train"

with open(os.path.join(TRAIN_DIR, "cv_splits_5fold.json")) as f:
    s1_to_fold = json.load(f)

random.seed(42)
fold0_s1 = [s for s, fld in s1_to_fold.items() if fld == 0]
random.shuffle(fold0_s1)
val_s1_ids = set(fold0_s1[:5000])

gt = {}
needed_gold_targets = set()
with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    for row in reader:
        s1_id = row[0].strip()
        if s1_id in val_s1_ids:
            matches = [x.strip() for x in row[1].split(",") if x.strip()] if len(row) > 1 and row[1].strip() else []
            gt[s1_id] = matches
            needed_gold_targets.update(matches)

total_gold_links = sum(len(v) for v in gt.values())
print(f"Validation True Links: {total_gold_links:,}")

# S1 Blocking keys
s1_meta = {}
needed_cores = set()
needed_sorted = set()
needed_concat = set()
needed_brand = set()
needed_addr_city = set()
needed_addr_num = set()

with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    for row in reader:
        s1_id = row[0].strip()
        if s1_id in val_s1_ids:
            name, addr, ctry = row[1], row[2], row[3]
            norm = normalize_text(name)
            core = extract_core_name(name)
            sorted_k = extract_sorted_key(name)
            concat_k = norm.replace(" ", "")
            brand = extract_distinctive_name_tokens(name)
            postal, num, dist_tokens = extract_postal_and_number(addr)

            # Composite address keys: num + each distinctive address token (up to 4 tokens)
            addr_keys = [f"{num}_{dt}" for dt in sorted(dist_tokens)[:4]] if (num and dist_tokens) else []

            s1_meta[s1_id] = {
                "core": core, "sorted": sorted_k, "concat": concat_k,
                "brand": brand, "addr_keys": addr_keys, "country": ctry.strip().lower(),
                "num": num,
            }

            if core: needed_cores.add(core)
            if sorted_k: needed_sorted.add(sorted_k)
            if len(concat_k) >= 5: needed_concat.add(concat_k)
            for b in brand: needed_brand.add(b)
            for ak in addr_keys: needed_addr_num.add(ak)

print(f"Needed keys: cores={len(needed_cores):,}, brand={len(needed_brand):,}, addr_keys={len(needed_addr_num):,}")

# Index targets with high caps
idx_core = defaultdict(list)
idx_sorted = defaultdict(list)
idx_concat = defaultdict(list)
idx_brand = defaultdict(list)
idx_addr_num = defaultdict(list)
target_data = {}

CAP_CORE = 500
CAP_SORTED = 400
CAP_CONCAT = 300
CAP_BRAND = 300
CAP_ADDR = 150

t0 = time.time()
for fname in ["train_source2.tsv", "train_source3.tsv"]:
    path = os.path.join(TRAIN_DIR, fname)
    print(f"Streaming {fname}...")
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
            if t_core in needed_cores and len(idx_core[t_core]) < CAP_CORE:
                idx_core[t_core].append(t_id); hit = True
            if t_sorted in needed_sorted and len(idx_sorted[t_sorted]) < CAP_SORTED:
                idx_sorted[t_sorted].append(t_id); hit = True
            if len(t_concat) >= 5 and t_concat in needed_concat and len(idx_concat[t_concat]) < CAP_CONCAT:
                idx_concat[t_concat].append(t_id); hit = True

            t_brand = extract_distinctive_name_tokens(t_name)
            for b in t_brand:
                if b in needed_brand and len(idx_brand[b]) < CAP_BRAND:
                    idx_brand[b].append(t_id); hit = True

            if t_addr:
                _, t_num, t_dist = extract_postal_and_number(t_addr)
                if t_num and t_dist:
                    for dt in sorted(t_dist)[:4]:
                        ak = f"{t_num}_{dt}"
                        if ak in needed_addr_num and len(idx_addr_num[ak]) < CAP_ADDR:
                            idx_addr_num[ak].append(t_id); hit = True

            if hit or t_id in needed_gold_targets:
                target_data[t_id] = {
                    "name": t_name, "address": t_addr, "country": t_country,
                    "core": t_core,
                }

print(f"Indexed in {time.time()-t0:.1f}s. Loaded {len(target_data):,} targets.")

# Evaluate recall at different candidate pool sizes
for max_pool in [40, 60, 80, 100, 120]:
    cap_gold = 0
    total_cands = 0
    for s1_id, meta in s1_meta.items():
        cands = set()
        cands.update(idx_core.get(meta["core"], []))
        cands.update(idx_sorted.get(meta["sorted"], []))
        if len(meta["concat"]) >= 5:
            cands.update(idx_concat.get(meta["concat"], []))
        for b in list(meta["brand"])[:6]:
            cands.update(idx_brand.get(b, []))
        for ak in meta["addr_keys"]:
            cands.update(idx_addr_num.get(ak, []))

        filtered = list(cands)
        if len(filtered) > max_pool:
            s1_core = meta["core"]
            # Fast score: fuzzy ratio on core
            scored = [(c, fuzz.ratio(s1_core, target_data.get(c, {}).get("core", ""))) for c in filtered]
            scored.sort(key=lambda x: x[1], reverse=True)
            filtered = [x[0] for x in scored[:max_pool]]

        total_cands += len(filtered)
        golds = gt.get(s1_id, [])
        if golds:
            cap_gold += len(set(golds).intersection(set(filtered)))

    recall = cap_gold / total_gold_links * 100
    avg_pool = total_cands / len(s1_meta)
    print(f"Max Pool {max_pool:3d}: Blocker Recall = {recall:.2f}% | Avg Pool = {avg_pool:.1f} | Captured = {cap_gold:,}/{total_gold_links:,}")
