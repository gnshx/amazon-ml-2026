#!/usr/bin/env python3
"""Diagnose candidates missed by Engine V5."""

import json
import os
import re
from rapidfuzz import fuzz

from experiment_precision_engine_v5 import (
    normalize_raw_text, extract_core_name, extract_sorted_key,
    extract_distinctive_tokens, normalize_address, make_prefix_key
)

def main():
    train_dir = "dataset/train"
    with open("dataset/train/cv_splits_5fold.json") as f:
        s1_to_fold = json.load(f)
    val_s1 = set([s for s, f in s1_to_fold.items() if f == 0][:5000])

    gt = {}
    needed_t = set()
    with open("dataset/train/train_ground_truth.tsv") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in val_s1 and len(p) > 1 and p[1].strip():
                targets = [x.strip() for x in p[1].split(",") if x.strip()]
                gt[p[0]] = targets
                needed_t.update(targets)

    s1_data = {}
    with open("dataset/train/train_source1.tsv") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in val_s1:
                s1_data[p[0]] = {"name": p[1], "addr": p[2], "country": p[3]}

    t_data = {}
    for sf in ["train_source2.tsv", "train_source3.tsv"]:
        with open(f"dataset/train/{sf}") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[0] in needed_t:
                    t_data[p[0]] = {"name": p[1], "addr": p[2], "country": p[3]}
                    if len(t_data) >= len(needed_t):
                        break

    print("Searching for missed true pairs where core, sorted, and brand didn't match...")
    count = 0
    for s_id, tgts in gt.items():
        s = s1_data.get(s_id)
        if not s:
            continue
        c_s1 = extract_core_name(normalize_raw_text(s["name"]))
        sort_s1 = extract_sorted_key(normalize_raw_text(s["name"]))
        brand_s1 = set(extract_distinctive_tokens(normalize_raw_text(s["name"])))
        clean_s1, p1, n1, dist1 = normalize_address(s["addr"])

        for t_id in tgts:
            t = t_data.get(t_id)
            if not t:
                continue
            c_t = extract_core_name(normalize_raw_text(t["name"]))
            sort_t = extract_sorted_key(normalize_raw_text(t["name"]))
            brand_t = set(extract_distinctive_tokens(normalize_raw_text(t["name"])))
            clean_t, p2, n2, dist2 = normalize_address(t["addr"])

            # Check if V5 routes matched
            exact_core = (c_s1 == c_t)
            sort_match = (sort_s1 == sort_t)
            brand_overlap = bool(brand_s1 & brand_t)
            num_match = (n1 and n1 == n2 and c_s1[:3] == c_t[:3])
            post_match = (p1 and p1 == p2 and c_s1[:3] == c_t[:3])

            if not (exact_core or sort_match or brand_overlap or num_match or post_match):
                count += 1
                print(f"\n[MISSED #{count}]")
                print(f"  S1:     {s['name']} | {s['addr']}")
                print(f"  Target: {t['name']} | {t['addr']}")
                print(f"  Core S1: '{c_s1}' vs Core T: '{c_t}'")
                print(f"  Brand S1: {brand_s1} vs Brand T: {brand_t}")
                print(f"  Num S1: '{n1}' vs Num T: '{n2}'")
                print(f"  Post S1: '{p1}' vs Post T: '{p2}'")
                print(f"  Dist Addr S1: {dist1} vs Dist Addr T: {dist2}")
                if count >= 10:
                    return

if __name__ == "__main__":
    main()
