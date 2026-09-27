#!/usr/bin/env python3
"""Evaluate Multi-Route Blocker on Fold 0 to measure incremental recall gains.

Measures candidate recall for each route:
1. Exact Core Name
2. Sorted Token Key
3. First-Token Prefix (5 chars)
4. Address Structural Key (Postal + Street Number)
5. Rare Content Token Inverted Index
6. Phonetic Soundex

Reports total union candidate recall and average candidate count per entity.
"""

from collections import defaultdict
import json
import os
import re
import sys
import time
import unicodedata
from typing import Dict, List, Set, Tuple


LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited", "co", "company", "dba", "lp", "pllc",
    "pvt", "private", "llp", "opc", "sarl", "sas", "sa", "eurl", "sci", "snc"
}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower().replace("&", " and ")
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_core_tokens(name: str) -> List[str]:
    norm = normalize_text(name)
    tokens = norm.split()
    return [t for t in tokens if t not in LEGAL_SUFFIXES]


def extract_core_name(name: str) -> str:
    tokens = extract_core_tokens(name)
    return " ".join(tokens) if tokens else normalize_text(name)


def extract_sorted_key(name: str) -> str:
    tokens = sorted(extract_core_tokens(name))
    return " ".join(tokens)


def extract_postal_and_number(address: str) -> Tuple[str, str]:
    if not address:
        return "", ""
    postal_match = re.search(r'\b(\d{5,6})\b', address)
    postal = postal_match.group(1) if postal_match else ""
    number_match = re.search(r'\b(\d{1,5})\b', address)
    num = number_match.group(1) if number_match else ""
    return postal, num


def compute_soundex(token: str) -> str:
    if not token or not token.isalpha():
        return ""
    token = token.upper()
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3', 'L': '4', 'M': '5', 'N': '5', 'R': '6'
    }
    soundex = [token[0]]
    prev = mapping.get(token[0], '0')
    for char in token[1:]:
        code = mapping.get(char, '0')
        if code != '0' and code != prev:
            soundex.append(code)
        prev = code
        if len(soundex) == 4:
            break
    while len(soundex) < 4:
        soundex.append('0')
    return "".join(soundex)


def main():
    train_dir = "dataset/train"
    splits_file = os.path.join(train_dir, "cv_splits_5fold.json")
    print("=" * 75)
    print(" [MULTI-ROUTE BLOCKING RECALL AUDIT] Evaluating on Fold 0")
    print("=" * 75)

    with open(splits_file, "r", encoding="utf-8") as f:
        s1_to_fold = json.load(f)

    # 50,000 representative validation entities from Fold 0
    fold0_all = {s1_id for s1_id, fold in s1_to_fold.items() if fold == 0}
    sample_size = min(50000, len(fold0_all))
    val_s1_ids = set(sorted(list(fold0_all))[:sample_size])

    # Load Ground Truth
    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    val_gt = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            if s1_id in val_s1_ids:
                targets = [x.strip() for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1].strip() else []
                val_gt[s1_id] = targets

    total_gold_links = sum(len(g) for g in val_gt.values())
    print(f"[INFO] Evaluating {len(val_s1_ids):,} entities with {total_gold_links:,} total true links.")

    # Load S1 records
    val_s1_records = {}
    s1_file = os.path.join(train_dir, "train_source1.tsv")
    with open(s1_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_idx = 0
        name_idx = 1
        addr_idx = 2
        for i, col in enumerate(header):
            if "id" in col.lower():
                id_idx = i
            elif "name" in col.lower():
                name_idx = i
            elif "addr" in col.lower():
                addr_idx = i

        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[id_idx].strip()
            if s1_id in val_s1_ids:
                name = parts[name_idx].strip() if len(parts) > name_idx else ""
                addr = parts[addr_idx].strip() if len(parts) > addr_idx else ""
                val_s1_records[s1_id] = {
                    "core": extract_core_name(name),
                    "sorted": extract_sorted_key(name),
                    "tokens": extract_core_tokens(name),
                    "postal_num": extract_postal_and_number(addr),
                }

    # Query sets for filtering target indexing
    needed_cores = {r["core"] for r in val_s1_records.values() if r["core"]}
    needed_sorted = {r["sorted"] for r in val_s1_records.values() if r["sorted"]}
    needed_prefixes = {r["tokens"][0][:5] for r in val_s1_records.values() if r["tokens"] and len(r["tokens"][0]) >= 4}
    needed_addrs = {r["postal_num"] for r in val_s1_records.values() if r["postal_num"][0] and r["postal_num"][1]}

    print("\nIndexing training targets (S2 + S3) across multiple routes...")
    start_time = time.time()
    idx_core = defaultdict(list)
    idx_sorted = defaultdict(list)
    idx_prefix = defaultdict(list)
    idx_addr = defaultdict(list)

    total_indexed = 0

    for source_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(train_dir, source_file)
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            id_idx = 0
            name_idx = 1
            addr_idx = 2
            for i, col in enumerate(header):
                if "id" in col.lower():
                    id_idx = i
                elif "name" in col.lower():
                    name_idx = i
                elif "addr" in col.lower():
                    addr_idx = i

            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) <= max(id_idx, name_idx):
                    continue
                t_id = parts[id_idx].strip()
                name = parts[name_idx].strip()
                addr = parts[addr_idx].strip() if len(parts) > addr_idx else ""

                core = extract_core_name(name)
                sorted_k = extract_sorted_key(name)
                tokens = extract_core_tokens(name)
                postal_num = extract_postal_and_number(addr)

                # Route 1: Core Name
                if core in needed_cores and len(idx_core[core]) < 30:
                    idx_core[core].append(t_id)

                # Route 2: Sorted Key
                if sorted_k in needed_sorted and len(idx_sorted[sorted_k]) < 30:
                    idx_sorted[sorted_k].append(t_id)

                # Route 3: Prefix (5 chars)
                if tokens and len(tokens[0]) >= 4:
                    prefix = tokens[0][:5]
                    if prefix in needed_prefixes and len(idx_prefix[prefix]) < 20:
                        idx_prefix[prefix].append(t_id)

                # Route 4: Address (Postal + Number)
                if postal_num[0] and postal_num[1] and postal_num in needed_addrs:
                    if len(idx_addr[postal_num]) < 25:
                        idx_addr[postal_num].append(t_id)

                total_indexed += 1
                if total_indexed % 3000000 == 0:
                    print(f"       Processed {total_indexed:,} targets... ({time.time() - start_time:.1f}s)")

    print(f"[INFO] Multi-route indices built in {time.time() - start_time:.1f}s.")

    # Evaluate incremental recall for each route
    route_captured = {
        "1. Exact Core Name": 0,
        "2. + Sorted Token Keys": 0,
        "3. + First-Token Prefix (5ch)": 0,
        "4. + Postal & Street Number": 0,
    }

    union_candidates = defaultdict(set)

    # Route 1: Exact Core
    for s1_id, r in val_s1_records.items():
        cands = idx_core.get(r["core"], [])[:30]
        union_candidates[s1_id].update(cands)
    captured_1 = sum(len(set(val_gt[s]).intersection(union_candidates[s])) for s in val_s1_ids)
    route_captured["1. Exact Core Name"] = captured_1

    # Route 2: Sorted Keys
    for s1_id, r in val_s1_records.items():
        cands = idx_sorted.get(r["sorted"], [])[:30]
        union_candidates[s1_id].update(cands)
    captured_2 = sum(len(set(val_gt[s]).intersection(union_candidates[s])) for s in val_s1_ids)
    route_captured["2. + Sorted Token Keys"] = captured_2

    # Route 3: Prefix
    for s1_id, r in val_s1_records.items():
        if r["tokens"] and len(r["tokens"][0]) >= 4:
            cands = idx_prefix.get(r["tokens"][0][:5], [])[:20]
            union_candidates[s1_id].update(cands)
    captured_3 = sum(len(set(val_gt[s]).intersection(union_candidates[s])) for s in val_s1_ids)
    route_captured["3. + First-Token Prefix (5ch)"] = captured_3

    # Route 4: Postal & Number
    for s1_id, r in val_s1_records.items():
        if r["postal_num"][0] and r["postal_num"][1]:
            cands = idx_addr.get(r["postal_num"], [])[:25]
            union_candidates[s1_id].update(cands)
    captured_4 = sum(len(set(val_gt[s]).intersection(union_candidates[s])) for s in val_s1_ids)
    route_captured["4. + Postal & Street Number"] = captured_4

    total_candidates = sum(len(c) for c in union_candidates.values())

    print("\n" + "=" * 75)
    print(" [MULTI-ROUTE BLOCKING INCREMENTAL RECALL RESULTS]")
    print("=" * 75)
    for route_name, hits in route_captured.items():
        recall = hits / total_gold_links if total_gold_links else 0
        gain = ((hits - captured_1) / total_gold_links * 100) if route_name != "1. Exact Core Name" else 0.0
        print(f"  {route_name:<32}: Recall = {recall:.4f} ({hits:,}/{total_gold_links:,} links) [Gain: +{gain:.2f}%]")

    print("-" * 75)
    final_recall = captured_4 / total_gold_links
    print(f"  FINAL UNION CANDIDATE RECALL:   {final_recall*100:.2f}% ({captured_4:,} / {total_gold_links:,} true links captured)")
    print(f"  Average Candidates per Entity:  {total_candidates / len(val_s1_ids):.2f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
