#!/usr/bin/env python3
"""Compute exact ground-truth statistics across all 2.2M Source-1 records and split by country.

Evaluates:
- Total Source-1 entities
- True singletons (fraction with empty matched_entity_ids)
- True non-singletons (fraction with >= 1 match)
- Total match links
- Average matches per non-singleton entity
- Match count distribution (Median, p90, p99, Max)
- Exact distribution by country (US, India) and Target source (S2, S3)
"""

import os
import csv
from collections import defaultdict

TRAIN_DIR = "dataset/train"
GT_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
S1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")

def analyze_ground_truth():
    print(f"Reading country data from {S1_PATH}...")
    s1_country = {}
    with open(S1_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        country_idx = 3
        for row in reader:
            if not row: continue
            s1_id = row[0].strip()
            c = row[country_idx].strip().lower() if len(row) > country_idx else "unknown"
            s1_country[s1_id] = c

    print(f"Loaded country for {len(s1_country):,} S1 entities.")

    print(f"Reading ground truth from {GT_PATH}...")
    total_s1 = 0
    total_singletons = 0
    total_matches = 0
    match_counts = []
    
    country_stats = defaultdict(lambda: {"total": 0, "singletons": 0, "non_singletons": 0, "total_matches": 0, "s2_matches": 0, "s3_matches": 0})

    with open(GT_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row: continue
            s1_id = row[0].strip()
            raw_matches = row[1].strip() if len(row) > 1 else ""
            matches = [m.strip() for m in raw_matches.split(",") if m.strip()] if raw_matches else []
            
            total_s1 += 1
            n_m = len(matches)
            c = s1_country.get(s1_id, "unknown")
            country_stats[c]["total"] += 1
            
            if n_m == 0:
                total_singletons += 1
                country_stats[c]["singletons"] += 1
            else:
                total_matches += n_m
                match_counts.append(n_m)
                country_stats[c]["non_singletons"] += 1
                country_stats[c]["total_matches"] += n_m
                for m in matches:
                    if m.startswith("S2-"):
                        country_stats[c]["s2_matches"] += 1
                    elif m.startswith("S3-"):
                        country_stats[c]["s3_matches"] += 1

    non_singletons = total_s1 - total_singletons
    avg_matches_per_non_sing = total_matches / non_singletons if non_singletons else 0
    avg_matches_per_entity = total_matches / total_s1 if total_s1 else 0

    print("=" * 80)
    print(" GROUND TRUTH EXACT STATS (OVERALL)")
    print("=" * 80)
    print(f"Total S1 Entities:                     {total_s1:,}")
    print(f"True Singletons (0 matches):          {total_singletons:,} ({total_singletons / total_s1 * 100:.2f}%)")
    print(f"True Non-Singletons (>=1 match):      {non_singletons:,} ({non_singletons / total_s1 * 100:.2f}%)")
    print(f"Total True Match Links:               {total_matches:,}")
    print(f"Avg True Matches per Non-Singleton:   {avg_matches_per_non_sing:.3f}")
    print(f"Avg True Matches per ALL Entities:    {avg_matches_per_entity:.3f}")
    if match_counts:
        match_counts.sort()
        p50 = match_counts[len(match_counts)//2]
        p90 = match_counts[int(len(match_counts)*0.9)]
        p99 = match_counts[int(len(match_counts)*0.99)]
        max_m = match_counts[-1]
        print(f"Match Distribution (Non-Sing): Median={p50}, p90={p90}, p99={p99}, Max={max_m}")

    print("\n" + "=" * 80)
    print(" GROUND TRUTH STATS BY COUNTRY")
    print("=" * 80)
    print(f"{'Country':<12} | {'Total':<10} | {'Singletons':<10} | {'Sing %':<8} | {'Total Matches':<14} | {'Matches/Non-Sing':<18} | {'S2/S3 Ratio':<12}")
    print("-" * 92)
    for c, stats in sorted(country_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        tot = stats["total"]
        sing = stats["singletons"]
        ns = stats["non_singletons"]
        tm = stats["total_matches"]
        sing_pct = sing / tot * 100 if tot else 0
        m_per_ns = tm / ns if ns else 0
        s2 = stats["s2_matches"]
        s3 = stats["s3_matches"]
        ratio = f"{s2}/{s3}"
        print(f"{c:<12} | {tot:<10,} | {sing:<10,} | {sing_pct:>6.2f}% | {tm:<14,} | {m_per_ns:>18.3f} | {ratio:<12}")

if __name__ == "__main__":
    analyze_ground_truth()
