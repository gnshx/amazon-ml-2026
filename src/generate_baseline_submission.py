#!/usr/bin/env python3
"""High-Performance Streaming Baseline Submission Generator (Hour 1 Milestone).

Streams 1.73M test Source 1 entities and 9.9M test targets:
1. Normalizes text with accent folding, legal suffix stripping, and punctuation collapse.
2. Inverted index on core business names.
3. Streams output directly to disk to maintain < 1GB RAM usage.
4. Produces:
     output/matching_results.tsv
     output/candidate_pairs.tsv
5. Automatically verifies with official utils/validate_submission.py.
"""

from collections import defaultdict
import csv
import gc
import os
import re
import subprocess
import sys
import time
import unicodedata
from typing import Dict, List, Set, Tuple


LEGAL_SUFFIXES = {
    # US
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited", "co", "company", "dba", "lp", "pllc",
    # India
    "pvt", "private", "llp", "opc",
    # France
    "sarl", "sas", "sa", "eurl", "sci", "snc"
}


def normalize_text(text: str) -> str:
    """Accent-folds, lowercases, and strips punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower().replace("&", " and ")
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_core_name(name: str) -> str:
    """Returns normalized core name without legal suffixes."""
    norm = normalize_text(name)
    tokens = norm.split()
    filtered = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return " ".join(filtered) if filtered else norm


def extract_postal_and_number(address: str) -> Tuple[str, str]:
    """Extracts postal code (5-6 digits) and street number."""
    if not address:
        return "", ""
    postal_match = re.search(r'\b(\d{5,6})\b', address)
    postal = postal_match.group(1) if postal_match else ""
    number_match = re.search(r'\b(\d{1,5})\b', address)
    num = number_match.group(1) if number_match else ""
    return postal, num


def build_test_target_index(test_dir: str) -> Dict[str, List[str]]:
    """Builds an inverted index of core_name -> list of target entity IDs from test_source2 & test_source3."""
    print("=" * 75)
    print(" [INDEXING] Building Core-Name Inverted Index for Test Targets (S2 & S3)")
    print("=" * 75)
    start_time = time.time()

    index = defaultdict(list)
    total_indexed = 0

    for source_file in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(test_dir, source_file)
        if not os.path.exists(path):
            print(f"[WARN] Target file {path} not found!")
            continue

        print(f"Indexing {source_file}...")
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            id_idx = 0
            name_idx = 1
            for i, col in enumerate(header):
                if "id" in col.lower():
                    id_idx = i
                elif "name" in col.lower():
                    name_idx = i

            for line_idx, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) <= max(id_idx, name_idx):
                    continue
                t_id = parts[id_idx].strip()
                name = parts[name_idx].strip()
                core = extract_core_name(name)

                # Skip trivial single-character core names to keep index compact and precise
                if len(core) >= 4:
                    # Cap list length for generic names to prevent memory bloat
                    if len(index[core]) < 30:
                        index[core].append(t_id)

                total_indexed += 1
                if total_indexed % 2000000 == 0:
                    print(f"       Processed {total_indexed:,} targets... ({time.time() - start_time:.1f}s)")

    print(f"[SUCCESS] Indexed {total_indexed:,} targets into {len(index):,} unique core-name keys in {time.time() - start_time:.1f}s.")
    return index


def generate_baseline(test_dir: str = "dataset/test", output_dir: str = "output"):
    """Generates baseline matching_results.tsv and candidate_pairs.tsv in streaming mode."""
    os.makedirs(output_dir, exist_ok=True)
    target_index = build_test_target_index(test_dir)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    match_out_path = os.path.join(output_dir, "matching_results.tsv")
    cand_out_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print("\n" + "=" * 75)
    print(" [STREAMING MATCH] Generating Test Submissions via Fast Core-Name Matcher")
    print(f" Input:  {s1_path}")
    print(f" Output: {match_out_path} & {cand_out_path}")
    print("=" * 75)
    start_time = time.time()

    matched_count = 0
    singleton_count = 0
    multi_match_count = 0
    total_candidates_produced = 0
    total_s1 = 0

    with open(s1_path, "r", encoding="utf-8") as f_in, \
         open(match_out_path, "w", encoding="utf-8", newline="") as f_match, \
         open(cand_out_path, "w", encoding="utf-8", newline="") as f_cand:

        # Write exact required headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        header = f_in.readline().strip().split("\t")
        id_idx = 0
        name_idx = 1
        for i, col in enumerate(header):
            if "id" in col.lower():
                id_idx = i
            elif "name" in col.lower():
                name_idx = i

        for line_idx, line in enumerate(f_in, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) <= max(id_idx, name_idx):
                continue
            s1_id = parts[id_idx].strip()
            name = parts[name_idx].strip()
            core = extract_core_name(name)

            cands = []
            matches = []

            if len(core) >= 4 and core in target_index:
                hit_ids = target_index[core]
                # Candidate pool gets all hits (up to 30)
                cands = hit_ids[:30]

                # High-precision baseline rule:
                # If unique exact match or 2-3 matches with distinctive name, retain as matches!
                if len(hit_ids) == 1:
                    matches = [hit_ids[0]]
                elif len(hit_ids) <= 3 and len(core) >= 8:
                    matches = hit_ids

            cand_str = ",".join(cands) if cands else ""
            match_str = ",".join(matches) if matches else ""

            # Write directly to TSVs
            f_match.write(f"{s1_id}\t{match_str}\n")
            f_cand.write(f"{s1_id}\t{cand_str}\n")

            total_s1 += 1
            total_candidates_produced += len(cands)
            if matches:
                matched_count += 1
                if len(matches) > 1:
                    multi_match_count += 1
            else:
                singleton_count += 1

            if line_idx % 500000 == 0:
                print(f"       Processed {line_idx:,} Source 1 entities... ({time.time() - start_time:.1f}s)")

    elapsed = time.time() - start_time
    print("\n" + "-" * 50)
    print(f"[SUMMARY] Generated 100% test predictions in {elapsed:.2f}s:")
    print(f"  Total Source 1 processed:     {total_s1:,}")
    print(f"  Entities with matches:        {matched_count:,} ({matched_count/total_s1*100:.2f}%)")
    print(f"  Entities as singletons (''):  {singleton_count:,} ({singleton_count/total_s1*100:.2f}%)")
    print(f"  Entities with multi-matches:  {multi_match_count:,}")
    print(f"  Total candidate pairs:        {total_candidates_produced:,}")
    print(f"  Mean candidates per S1:       {total_candidates_produced/total_s1:.2f}")

    # Validate output
    print("\n" + "=" * 75)
    print(" [VALIDATION] Running Official Submission Validator")
    print("=" * 75)
    val_cmd = [
        sys.executable, "utils/validate_submission.py",
        "--matching", match_out_path,
        "--candidate", cand_out_path,
        "--test-dir", test_dir
    ]
    res = subprocess.run(val_cmd)
    if res.returncode == 0:
        print("\n" + "=" * 75)
        print(" [READY TO SUBMIT] output/matching_results.tsv PASSED ALL CHECKS!")
        print(" Upload output/matching_results.tsv to the portal now to lock in your score and timestamp!")
        print("=" * 75)
    else:
        print("\n[VALIDATION FAILED] Please check errors above!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    generate_baseline()
