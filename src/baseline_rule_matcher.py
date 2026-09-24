#!/usr/bin/env python3
"""Trivial Baseline Matcher (Hour 0-2 Milestone).

A zero-ML, fast, deterministic rule matcher:
1. Normalizes core business names (lowercase, punctuation stripped, suffix stripped, accent folded).
2. Extracts postal/ZIP codes and first numbers.
3. Blocks and matches candidates using exact normalized name & postal code agreement.
4. Generates both matching_results.tsv and candidate_pairs.tsv.
5. Runs the submission validator to verify compliance.
6. Computes local validation Macro F0.5 against ground truth.
"""

from collections import defaultdict
import csv
import os
import re
import sys
import unicodedata
from typing import Dict, List, Set, Tuple

# Try importing local metric scorer
try:
    from metrics import evaluate_macro_f05
except ImportError:
    from src.metrics import evaluate_macro_f05


# Multi-country legal suffix dictionary (US, India, France)
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
    # Unicode NFKD & accent fold (café -> cafe)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    text = text.replace("&", " and ")
    # Replace punctuation with spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_core_name(name: str) -> str:
    """Removes common legal suffixes to produce core entity name."""
    norm = normalize_text(name)
    tokens = norm.split()
    filtered = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return " ".join(filtered) if filtered else norm


def extract_postal_code(address: str) -> str:
    """Extracts 5 or 6 digit postal codes."""
    if not address:
        return ""
    match = re.search(r'\b(\d{5,6})\b', address)
    return match.group(1) if match else ""


def load_entity_table(file_path: str) -> Dict[str, Dict[str, str]]:
    """Loads entity table into {id: {name, address, country, core_name, postal}}."""
    entities = {}
    if not os.path.exists(file_path):
        return entities

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return entities
        header = [h.strip().lower() for h in header]

        id_col = 0
        name_col = -1
        addr_col = -1
        country_col = -1

        for idx, col in enumerate(header):
            if "id" in col:
                id_col = idx
            elif "name" in col:
                name_col = idx
            elif "address" in col or "addr" in col:
                addr_col = idx
            elif "country" in col:
                country_col = idx

        for row in reader:
            if not row:
                continue
            e_id = row[id_col].strip() if id_col < len(row) else ""
            name = row[name_col].strip() if name_col != -1 and name_col < len(row) else ""
            addr = row[addr_col].strip() if addr_col != -1 and addr_col < len(row) else ""
            country = row[country_col].strip().lower() if country_col != -1 and country_col < len(row) else ""

            core = extract_core_name(name)
            postal = extract_postal_code(addr)

            entities[e_id] = {
                "name": name,
                "address": addr,
                "country": country,
                "core_name": core,
                "postal": postal,
            }
    return entities


def run_baseline_matching(
    s1_dict: Dict[str, Dict[str, str]],
    target_dict: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Performs fast deterministic rule-based blocking and matching.

    Returns:
        (matches_dict, candidates_dict)
    """
    # Build target inverted indices
    target_by_core_name = defaultdict(list)
    target_by_name_and_postal = defaultdict(list)

    for t_id, data in target_dict.items():
        core = data["core_name"]
        postal = data["postal"]
        if core:
            target_by_core_name[core].append(t_id)
            if postal:
                target_by_name_and_postal[(core, postal)].append(t_id)

    matches_dict = {}
    candidates_dict = {}

    for s1_id, s1_data in s1_dict.items():
        core = s1_data["core_name"]
        postal = s1_data["postal"]

        cands = set()
        matches = set()

        if core:
            # 1. Exact core-name match -> Candidate
            name_cands = target_by_core_name.get(core, [])
            cands.update(name_cands[:50])

            # 2. Exact core-name + Postal code match -> Match (High Precision)
            if postal:
                exact_both = target_by_name_and_postal.get((core, postal), [])
                matches.update(exact_both)

            # If no postal match, but exact core name and single candidate -> Match
            if len(matches) == 0 and len(name_cands) == 1 and len(core) >= 5:
                # Check country agreement if available
                cand_data = target_dict[name_cands[0]]
                if not s1_data["country"] or not cand_data["country"] or s1_data["country"] == cand_data["country"]:
                    matches.add(name_cands[0])

        # Candidate pool must contain all matches
        cands.update(matches)

        candidates_dict[s1_id] = sorted(list(cands))
        matches_dict[s1_id] = sorted(list(matches))

    return matches_dict, candidates_dict


def write_submission_tsv(records: Dict[str, List[str]], file_path: str, id_col: str, val_col: str):
    """Writes standard submission TSV."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([id_col, val_col])
        for s1_id in sorted(records.keys()):
            val_str = ",".join(records[s1_id]) if records[s1_id] else ""
            writer.writerow([s1_id, val_str])
    print(f"[OK] Written {len(records)} rows to: {file_path}")


def main():
    print("=" * 70)
    print(" Amazon Business Entity Resolution - Baseline Rule Matcher")
    print("=" * 70)

    # Determine paths
    train_dir = "dataset/train"
    test_dir = "dataset/test"
    output_dir = "output"

    # Check if train dataset exists
    s1_train_path = os.path.join(train_dir, "train_source1.tsv")
    if not os.path.exists(s1_train_path):
        # Look for any source1 file
        for root, _, files in os.walk("dataset"):
            for f in files:
                if "source1" in f.lower() and "train" in root.lower() and f.endswith(".tsv"):
                    s1_train_path = os.path.join(root, f)
                    train_dir = root
                    break

    if not os.path.exists(s1_train_path):
        print(f"[INFO] Dataset files not yet detected in 'dataset/train'.")
        print("Once you place train_source1.tsv, train_source2.tsv, train_source3.tsv,")
        print("and train_ground_truth.tsv in dataset/train, run this script to instantly")
        print("generate your baseline submission!")
        return

    print(f"[INFO] Using training directory: {train_dir}")
    s1_dict = load_entity_table(s1_train_path)
    s2_dict = load_entity_table(os.path.join(train_dir, "train_source2.tsv"))
    s3_dict = load_entity_table(os.path.join(train_dir, "train_source3.tsv"))

    targets_dict = {**s2_dict, **s3_dict}
    print(f"[INFO] Loaded Source 1: {len(s1_dict)}, Targets (S2+S3): {len(targets_dict)}")

    # Run matching
    matches, candidates = run_baseline_matching(s1_dict, targets_dict)

    # Evaluate against ground truth if present
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    if os.path.exists(gt_path):
        from data_audit_and_split import load_tsv_rows
        _, gt_rows = load_tsv_rows(gt_path)
        gt_dict = {}
        for r in gt_rows:
            s1_id = r.get("source1_entity_id", list(r.values())[0])
            m_str = r.get("matched_entity_ids", list(r.values())[1] if len(r) > 1 else "")
            gt_dict[s1_id] = [x.strip() for x in m_str.split(",") if x.strip()] if m_str else []

        eval_res = evaluate_macro_f05(matches, gt_dict)
        print("\n--- Baseline Local Validation Performance ---")
        print(f"Macro F0.5 Score:     {eval_res['macro_f05']:.4f}")
        print(f"Singleton Accuracy:   {eval_res['singleton_accuracy']:.4f} ({eval_res['singleton_count']} singletons)")
        print(f"Non-Singleton F0.5:   {eval_res['non_singleton_f05']:.4f} ({eval_res['non_singleton_count']} non-singletons)")

    # Output baseline submission files
    match_out = os.path.join(output_dir, "matching_results_baseline.tsv")
    cand_out = os.path.join(output_dir, "candidate_pairs_baseline.tsv")
    write_submission_tsv(matches, match_out, "source1_entity_id", "matched_entity_ids")
    write_submission_tsv(candidates, cand_out, "source1_entity_id", "candidate_entity_ids")

    # Run validator
    from utils.validate_submission import validate_tsv_file
    print("\nValidating generated baseline output...")
    validate_tsv_file(match_out, "source1_entity_id", "matched_entity_ids")
    validate_tsv_file(cand_out, "source1_entity_id", "candidate_entity_ids")
    print("[SUCCESS] Baseline outputs are 100% compliant with submission specifications!")


if __name__ == "__main__":
    main()
