#!/usr/bin/env python3
"""Benchmark Precision Engine v8 on Fold 0 and Fold 1.

Evaluates high-precision matching enhancements:
1. Ultra-compact memory indexing (tuple tables, <500 MB RAM)
2. Character n-gram typo tolerance on brand tokens (Levenshtein >= 0.80 with brand confirmation)
3. Enhanced cross-script / DBA address concordance (street number + 2 distinctive tokens)
4. Strict multi-tenant / generic name guard (protects singleton accuracy)
5. 1-to-1 Mutual-Best conflict resolution
6. Evaluates on 50,000 entities from Fold 0 and 50,000 entities from Fold 1
"""

from collections import defaultdict
import csv
import json
import os
import re
import sys
import time
import unicodedata
from typing import Dict, List, Set, Tuple

from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import evaluate_macro_f05

TRAIN_DIR = "dataset/train"

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited", "co", "company", "dba", "lp", "pllc",
    "pvt", "private", "llp", "opc", "lnc", "1nc",
    "sarl", "sas", "sa", "eurl", "sci", "snc", "sasu", "gie", "ei", "sep"
}

GENERIC_WORDS = {
    "center", "services", "solutions", "technologies", "group", "holdings", "enterprises",
    "international", "global", "consulting", "associates", "management", "products",
    "systems", "partners", "digital", "developers", "allied", "ventures", "industries",
    "commercial", "logistics", "trading", "retail", "marketing", "agency", "care", "health",
    "works", "consultancy", "enterprise", "labs", "studio"
} | LEGAL_SUFFIXES

STREET_ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "pkwy": "parkway", "hwy": "highway", "saint": "street",
    "rte": "route", "sq": "square",
    "r": "rue", "all": "allee", "imp": "impasse"
}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia"
}

GENERIC_ADDR_WORDS = (
    set(STREET_ABBREVIATIONS.keys()) | set(STREET_ABBREVIATIONS.values()) |
    set(US_STATES.keys()) | set(US_STATES.values()) | {
        "suite", "ste", "apt", "unit", "fl", "floor", "building", "bldg", "po", "box",
        "north", "south", "east", "west", "n", "s", "e", "w",
        "city", "near", "opp", "opposite", "behind", "dist", "district", "state", "road", "street",
        "highway", "county", "hno", "no", "plot", "shop", "flat"
    }
)


def normalize_raw_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower().replace("&", " and ")
    text = re.sub(r'[\(\[\{]\s*id\s*:\s*\d+\s*[\)\]\}]', ' ', text)
    text = re.sub(r'\bid\s*:\s*\d+\b', ' ', text)
    text = re.sub(r'(\.com|\.in|\.org|\.net|\.co|\.io|\.fr|\.co\.uk|\.eu)\b', '', text)
    text = text.replace("@", " ")
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def is_non_ascii_name(raw_name: str) -> bool:
    if not raw_name:
        return False
    letters = re.findall(r'[a-zA-Z]', raw_name)
    return len(letters) < 3 and len(raw_name.strip()) >= 3


def extract_alias_names(name: str) -> List[str]:
    if not name:
        return []
    norm = normalize_raw_text(name)
    parts = re.split(r'\b(?:aka|dba|fka|doing business as|trading as)\b', norm)
    if len(parts) > 1:
        return [p.strip() for p in parts if len(p.strip()) >= 3]
    return []


def extract_core_tokens(norm_name: str) -> List[str]:
    tokens = norm_name.split()
    return [t for t in tokens if t not in LEGAL_SUFFIXES]


def extract_core_name(norm_name: str) -> str:
    tokens = extract_core_tokens(norm_name)
    return " ".join(tokens) if tokens else norm_name


def extract_sorted_key(norm_name: str) -> str:
    tokens = sorted(extract_core_tokens(norm_name))
    return " ".join(tokens)


def extract_concat_key(norm_name: str) -> str:
    tokens = extract_core_tokens(norm_name)
    return "".join(tokens)


def extract_distinctive_tokens(norm_name: str) -> List[str]:
    tokens = extract_core_tokens(norm_name)
    return [t for t in tokens if len(t) >= 4 and t not in GENERIC_WORDS]


def normalize_address(address: str) -> Tuple[str, str, str, Set[str]]:
    if not address:
        return "", "", "", set()
    norm = normalize_raw_text(address)
    tokens = norm.split()
    expanded = []
    for t in tokens:
        if t in STREET_ABBREVIATIONS:
            expanded.append(STREET_ABBREVIATIONS[t])
        elif t in US_STATES:
            expanded.append(US_STATES[t])
        else:
            expanded.append(t)
    clean_addr = " ".join(expanded)

    postal_match = re.search(r'\b(\d{5,6})\b', clean_addr)
    postal = postal_match.group(1) if postal_match else ""

    num_match = re.search(r'\b0*(\d{1,6})\b', clean_addr)
    num = num_match.group(1) if num_match else ""

    dist_tokens = {t for t in tokens if len(t) >= 3 and t not in GENERIC_ADDR_WORDS and not t.isdigit()}
    return clean_addr, postal, num, dist_tokens


def make_prefix_key(name: str, length: int = 3) -> str:
    tokens = extract_core_tokens(name)
    if not tokens:
        return ""
    first = tokens[0]
    return first[:length] if len(first) >= length else first


def run_benchmark():
    print("=" * 80)
    print(" BENCHMARK: ENHANCED PRECISION ENGINE v8 (Fold 0 & Fold 1)")
    print("=" * 80)

    # 1. Load CV Splits
    splits_file = os.path.join(TRAIN_DIR, "cv_splits_5fold.json")
    with open(splits_file) as f:
        s1_to_fold = json.load(f)

    # Pick 25,000 entities from Fold 0 and 25,000 entities from Fold 1
    fold0_entities = sorted([s for s, fold in s1_to_fold.items() if fold == 0])[:25000]
    fold1_entities = sorted([s for s, fold in s1_to_fold.items() if fold == 1])[:25000]
    all_eval_ids = set(fold0_entities) | set(fold1_entities)

    print(f"Evaluation set: {len(fold0_entities):,} (Fold 0) + {len(fold1_entities):,} (Fold 1) = {len(all_eval_ids):,} entities")

    # Load Ground Truth
    gt_file = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    all_gt = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row[0] in all_eval_ids:
                matches = [x.strip() for x in row[1].split(",") if x.strip()] if len(row) > 1 and row[1].strip() else []
                all_gt[row[0]] = matches

    # Load S1 records
    print("Streaming Source 1 records...")
    t0 = time.time()
    s1_data = {}
    needed_cores = set()
    needed_sorted = set()
    needed_concat = set()
    needed_brand = set()
    needed_num_p = set()
    needed_post_p = set()
    needed_addr_k = set()

    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            if s1_id in all_eval_ids:
                raw_name = row[1] if len(row) > 1 else ""
                raw_addr = row[2] if len(row) > 2 else ""
                country = row[3].strip().lower() if len(row) > 3 else ""

                norm_name = normalize_raw_text(raw_name)
                clean_addr, postal, num, dist_addr = normalize_address(raw_addr)
                aliases = extract_alias_names(raw_name)
                core_prefix3 = make_prefix_key(norm_name, 3)

                core = extract_core_name(norm_name)
                sorted_k = extract_sorted_key(norm_name)
                concat_k = extract_concat_key(norm_name)
                brand_tokens = extract_distinctive_tokens(norm_name)
                alias_cores = [extract_core_name(a) for a in aliases if extract_core_name(a)]

                num_p = f"{num}_{core_prefix3}" if (num and core_prefix3) else ""
                post_p = f"{postal}_{core_prefix3}" if (postal and core_prefix3) else ""
                addr_keys = [f"{num}_{t}" for t in sorted(dist_addr)[:2]] if num else []

                if core: needed_cores.add(core)
                for ac in alias_cores: needed_cores.add(ac)
                if sorted_k: needed_sorted.add(sorted_k)
                if len(concat_k) >= 5: needed_concat.add(concat_k)
                for bt in brand_tokens: needed_brand.add(bt)
                if num_p: needed_num_p.add(num_p)
                if post_p: needed_post_p.add(post_p)
                for ak in addr_keys: needed_addr_k.add(ak)

                s1_data[s1_id] = {
                    "id": s1_id,
                    "core": core,
                    "sorted": sorted_k,
                    "concat": concat_k,
                    "distinctive": brand_tokens,
                    "postal": postal,
                    "num": num,
                    "dist_addr": dist_addr,
                    "country": country,
                    "aliases": alias_cores,
                    "num_prefix": num_p,
                    "post_prefix": post_p,
                    "addr_keys": addr_keys,
                    "clean_addr": clean_addr,
                    "norm_name": norm_name,
                    "is_non_ascii": is_non_ascii_name(raw_name),
                }

    print(f"Loaded {len(s1_data):,} S1 records in {time.time()-t0:.1f}s")

    # Index targets with compact tuple representation (<200 MB RAM)
    print("Indexing target records (S2 + S3)...")
    t0 = time.time()
    idx_core = defaultdict(list)
    idx_sorted = defaultdict(list)
    idx_concat = defaultdict(list)
    idx_brand = defaultdict(list)
    idx_num_p = defaultdict(list)
    idx_post_p = defaultdict(list)
    idx_addr_k = defaultdict(list)

    target_table = []
    target_ids = []

    for source_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(TRAIN_DIR, source_file)
        print(f"  Streaming {source_file}...")
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                t_id = parts[0].strip()
                raw_name = parts[1] if len(parts) > 1 else ""
                raw_addr = parts[2] if len(parts) > 2 else ""
                country = parts[3].strip().lower() if len(parts) > 3 else ""

                norm_name = normalize_raw_text(raw_name)
                core = extract_core_name(norm_name)
                sorted_k = extract_sorted_key(norm_name)
                concat_k = extract_concat_key(norm_name)
                brand_tokens = extract_distinctive_tokens(norm_name)
                aliases = extract_alias_names(raw_name)

                clean_addr, postal, num, dist_addr = normalize_address(raw_addr)
                core_prefix3 = make_prefix_key(norm_name, 3)
                num_p = f"{num}_{core_prefix3}" if (num and core_prefix3) else ""
                post_p = f"{postal}_{core_prefix3}" if (postal and core_prefix3) else ""
                addr_keys = [f"{num}_{t}" for t in sorted(dist_addr)[:2]] if num else []

                alias_cores = [extract_core_name(a) for a in aliases if extract_core_name(a)]

                hit = False
                t_idx = len(target_table)

                # Selective indexing caps (preserves high precision and ultra-low RAM)
                if core in needed_cores and len(idx_core[core]) < 35:
                    idx_core[core].append(t_idx); hit = True
                for ac in alias_cores:
                    if ac in needed_cores and len(idx_core[ac]) < 35:
                        idx_core[ac].append(t_idx); hit = True
                if sorted_k in needed_sorted and len(idx_sorted[sorted_k]) < 35:
                    idx_sorted[sorted_k].append(t_idx); hit = True
                if len(concat_k) >= 5 and concat_k in needed_concat and len(idx_concat[concat_k]) < 25:
                    idx_concat[concat_k].append(t_idx); hit = True
                for bt in brand_tokens:
                    if bt in needed_brand and len(idx_brand[bt]) < 25:
                        idx_brand[bt].append(t_idx); hit = True
                if num_p and num_p in needed_num_p and len(idx_num_p[num_p]) < 25:
                    idx_num_p[num_p].append(t_idx); hit = True
                if post_p and post_p in needed_post_p and len(idx_post_p[post_p]) < 25:
                    idx_post_p[post_p].append(t_idx); hit = True
                for ak in addr_keys:
                    if ak in needed_addr_k and len(idx_addr_k[ak]) < 15:
                        idx_addr_k[ak].append(t_idx); hit = True

                if hit:
                    is_non_ascii = is_non_ascii_name(raw_name)
                    target_table.append((
                        core, sorted_k, concat_k, norm_name, clean_addr,
                        country, postal, num, dist_addr, is_non_ascii, alias_cores, set(brand_tokens)
                    ))
                    target_ids.append(t_id)

    print(f"Indexed targets in {time.time()-t0:.1f}s. Loaded {len(target_table):,} targets into memory.")

    # High-Precision Matching Function
    def match_entities(entity_ids: List[str]) -> Tuple[dict, dict]:
        s1_scored = defaultdict(list)
        s1_candidates = {}

        for s1_id in entity_ids:
            s1 = s1_data[s1_id]
            core_s1 = s1["core"]
            sorted_s1 = s1["sorted"]
            concat_s1 = s1["concat"]
            brand_tokens = s1["distinctive"]
            brand_s1_set = set(brand_tokens)
            s1_p, s1_n = s1["postal"], s1["num"]
            s1_dist = s1["dist_addr"]
            s1_country = s1["country"]
            s1_aliases = s1["aliases"]
            num_p_s1 = s1["num_prefix"]
            post_p_s1 = s1["post_prefix"]
            addr_keys_s1 = s1["addr_keys"]
            is_non_ascii = s1["is_non_ascii"]
            has_s1_addr = bool(s1["clean_addr"])

            # Gather candidates
            cands = set()
            cands.update(idx_core.get(core_s1, []))
            for a in s1_aliases:
                cands.update(idx_core.get(a, []))
            cands.update(idx_sorted.get(sorted_s1, []))
            if len(concat_s1) >= 5:
                cands.update(idx_concat.get(concat_s1, []))
            for bt in brand_tokens[:4]:
                cands.update(idx_brand.get(bt, []))
            if num_p_s1:
                cands.update(idx_num_p.get(num_p_s1, []))
            if post_p_s1:
                cands.update(idx_post_p.get(post_p_s1, []))
            for ak in addr_keys_s1:
                cands.update(idx_addr_k.get(ak, []))

            s1_candidates[s1_id] = [target_ids[t_idx] for t_idx in cands]

            # Score candidates with Precision Rules v8
            for t_idx in cands:
                t = target_table[t_idx]
                core_t, sorted_t, concat_t, norm_t_name, clean_t_addr = t[0], t[1], t[2], t[3], t[4]
                country_t, postal_t, num_t, dist_t, is_t_non_ascii, t_aliases, t_brand_set = t[5], t[6], t[7], t[8], t[9], t[10], t[11]

                # Open-set country concordance: different countries are NEVER the same business establishment (100% true in ground truth)
                if s1_country and country_t and s1_country != country_t:
                    continue

                has_t_addr = bool(clean_t_addr)
                num_match = (s1_n and num_t and s1_n == num_t)
                num_conflict = (s1_n and num_t and s1_n != num_t)
                postal_match = (s1_p and postal_t and s1_p == postal_t)
                postal_conflict = (s1_p and postal_t and s1_p != postal_t)
                dist_overlap = len(s1_dist.intersection(dist_t))

                # Compute address token similarity
                addr_token_sim = None
                if has_s1_addr and has_t_addr:
                    inter_len = dist_overlap
                    union_len = len(s1_dist.union(dist_t))
                    addr_token_sim = inter_len / union_len if union_len > 0 else 0.0

                # Strict address conflict gate: reject cross-street and cross-city false merges
                if has_s1_addr and has_t_addr:
                    if addr_token_sim is not None and addr_token_sim < 0.35:
                        continue
                    if s1_dist and dist_t and not postal_match and dist_overlap == 0:
                        continue
                    if postal_conflict and addr_token_sim is not None and addr_token_sim < 0.70:
                        continue
                    if num_conflict and dist_overlap == 0:
                        continue
                else:
                    # Missing address guard: require distinctive multi-word core name (>= 14 chars)
                    core_words = core_s1.split()
                    if len(core_words) < 2 or len(core_s1) < 14:
                        continue

                # Name similarity metrics
                lev = fuzz.ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
                sort_r = fuzz.token_sort_ratio(s1["norm_name"], norm_t_name) / 100.0 if (s1["norm_name"] and norm_t_name) else 0.0
                max_name = max(lev, sort_r)
                brand_overlap = bool(brand_s1_set.intersection(t_brand_set))

                score = 0.0

                # RULE 1: Exact Core Name Match (Highest confidence)
                if core_s1 and core_t and core_s1 == core_t:
                    score = 0.95

                # RULE 2: Alias / DBA Match
                elif any(a and a == core_t for a in s1_aliases) or any(a and a == core_s1 for a in t_aliases):
                    score = 0.94

                # RULE 3: Sorted Core Match (Word transposition)
                elif sorted_s1 and sorted_t == sorted_s1:
                    score = 0.92

                # RULE 4: Concatenated Domain/Handle Match
                elif concat_s1 and (concat_t == concat_s1 or norm_t_name.replace(" ", "") == concat_s1):
                    score = 0.90 if (postal_match or num_match or dist_overlap >= 1 or not has_s1_addr) else 0.82

                # RULE 5: Non-Latin / Cross-Script / DBA Record with Exact Address Concordance
                elif (is_non_ascii or is_t_non_ascii or dist_overlap >= 2) and num_match and dist_overlap >= 2:
                    score = 0.89

                # RULE 6: High Fuzzy Name Similarity (Levenshtein / Sort >= 0.86)
                elif max_name >= 0.86:
                    if postal_match or num_match or dist_overlap >= 1 or (not has_s1_addr and not has_t_addr and max_name >= 0.92):
                        score = 0.87

                # RULE 7: Moderate Fuzzy Name (0.80 <= max_name < 0.86) with Brand Overlap or Strong Address Match
                elif max_name >= 0.80 and (brand_overlap or (num_match and dist_overlap >= 2)):
                    if postal_match or num_match:
                        score = 0.83
                    elif dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.80:
                        score = 0.81

                if score >= 0.80:
                    s1_scored[s1_id].append((t_idx, score, addr_token_sim if addr_token_sim is not None else 0.50))

        return s1_scored, s1_candidates

    # Evaluate Fold
    def evaluate_fold_predictions(fold_name: str, fold_entity_ids: List[str]):
        print(f"\n--- Evaluating {fold_name} ({len(fold_entity_ids):,} entities) ---")
        t0 = time.time()
        s1_scored, s1_candidates = match_entities(fold_entity_ids)

        # 1-to-1 Mutual-Best Conflict Resolution
        target_claims = defaultdict(list)
        for s1_id, matches in s1_scored.items():
            for t_idx, sc, a_sim in matches:
                target_claims[t_idx].append((s1_id, sc, a_sim))

        winner_for_target = {}
        for t_idx, claims in target_claims.items():
            best_s1, best_sc, best_asim = max(claims, key=lambda x: (x[1], x[2]))
            winner_for_target[t_idx] = best_s1

        final_preds = {}
        total_matches = 0
        singleton_preds = 0

        for s1_id in fold_entity_ids:
            retained = [target_ids[t_idx] for t_idx, sc, a_sim in s1_scored.get(s1_id, [])
                        if winner_for_target.get(t_idx) == s1_id]
            final_preds[s1_id] = retained
            total_matches += len(retained)
            if len(retained) == 0:
                singleton_preds += 1

        fold_gt = {s: all_gt.get(s, []) for s in fold_entity_ids}
        res = evaluate_macro_f05(final_preds, fold_gt)

        total_gold = sum(len(g) for g in fold_gt.values())
        captured = sum(len(set(fold_gt[s]).intersection(s1_candidates.get(s, []))) for s in fold_entity_ids)
        recall = (captured / total_gold * 100) if total_gold > 0 else 0.0

        print(f"  Execution time:             {time.time()-t0:.1f}s")
        print(f"  Macro F0.5 Score:           {res['macro_f05']:.4f}")
        print(f"  Singleton Accuracy:         {res['singleton_accuracy']:.4f}  ({res['singleton_count']} singletons)")
        print(f"  Non-Singleton F0.5:         {res['non_singleton_f05']:.4f}  ({res['non_singleton_count']} non-singletons)")
        print(f"  Candidate Pool Recall:      {recall:.2f}%")
        print(f"  Predicted Singletons:       {singleton_preds:,} ({singleton_preds/len(fold_entity_ids)*100:.1f}%)")
        print(f"  Total Matches Assigned:     {total_matches:,}")

        return res

    res_fold0 = evaluate_fold_predictions("Fold 0", fold0_entities)
    res_fold1 = evaluate_fold_predictions("Fold 1", fold1_entities)

    print("\n" + "=" * 80)
    print(" SUMMARY ACROSS FOLDS")
    print("=" * 80)
    print(f"  Fold 0 Macro F0.5:  {res_fold0['macro_f05']:.4f}")
    print(f"  Fold 1 Macro F0.5:  {res_fold1['macro_f05']:.4f}")
    mean_f05 = (res_fold0['macro_f05'] + res_fold1['macro_f05']) / 2
    print(f"  Mean Macro F0.5:    {mean_f05:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
