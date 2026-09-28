#!/usr/bin/env python3
import csv
import os
from collections import defaultdict

TEST_DIR = "dataset/test"
MATCHING_FILE = "output/matching_results.tsv"
CANDIDATE_FILE = "output/candidate_pairs.tsv"

def main():
    print("Loading country mapping from test_source1.tsv...")
    s1_country = {}
    with open(os.path.join(TEST_DIR, "test_source1.tsv"), "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            # country is column 3 (index 3)
            country = row[3].strip().lower() if len(row) > 3 else "unknown"
            s1_country[s1_id] = country
            
    print(f"Loaded {len(s1_country)} test entities.")
    
    country_stats = defaultdict(lambda: {"total": 0, "has_cands": 0, "total_cands": 0, "has_match": 0})
    
    print("Reading candidate_pairs.tsv...")
    with open(CANDIDATE_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            cands_str = row[1].strip() if len(row) > 1 else ""
            cands = cands_str.split(",") if cands_str else []
            
            ctry = s1_country.get(s1_id, "unknown")
            country_stats[ctry]["total"] += 1
            if cands:
                country_stats[ctry]["has_cands"] += 1
                country_stats[ctry]["total_cands"] += len(cands)
                
    print("Reading matching_results.tsv...")
    with open(MATCHING_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            matches_str = row[1].strip() if len(row) > 1 else ""
            
            ctry = s1_country.get(s1_id, "unknown")
            if matches_str:
                country_stats[ctry]["has_match"] += 1
                
    print("\n--- RESULTS BY COUNTRY ---")
    print(f"{'Country':<15} | {'Count':<10} | {'% with Cands':<15} | {'Avg Cands':<10} | {'% with Match':<15}")
    print("-" * 75)
    for ctry, stats in sorted(country_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        total = stats["total"]
        if total == 0: continue
        pct_cands = (stats["has_cands"] / total) * 100
        avg_cands = stats["total_cands"] / total
        pct_match = (stats["has_match"] / total) * 100
        print(f"{ctry:<15} | {total:<10,} | {pct_cands:>13.2f}% | {avg_cands:>10.2f} | {pct_match:>13.2f}%")

if __name__ == "__main__":
    main()
