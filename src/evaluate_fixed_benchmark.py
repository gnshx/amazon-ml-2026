#!/usr/bin/env python3
"""Fixed Benchmark Harness: Fold 0 (25k) + Fold 1 (25k) = 50,000 Entities.

Direct mirror of generate_v6_submission.py for evaluating rule changes:
- Baseline v6
- Change 1: Indic/cross-script DBA (dist_overlap >= 1, addr_token_sim >= 0.90, num_match == True)
- Change 2: Short-name length-aware typo tolerance (edit distance <= 1 for len <= 7)
"""

from collections import defaultdict
import csv
import json
import os
import pickle
import re
import sys
import time
import unicodedata
from typing import Dict, List, Set, Tuple

from rapidfuzz import fuzz, distance

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
    "r": "rue", "av": "avenue", "bd": "boulevard", "all": "allee", "imp": "impasse"
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


def main():
    print("=" * 80)
    print(" FIXED BENCHMARK: FOLD 0 (25k) + FOLD 1 (25k) = 50,000 ENTITIES")
    print("=" * 80)

    # 1. Load CV Splits
    splits_file = os.path.join(TRAIN_DIR, "cv_splits_5fold.json")
    with open(splits_file) as f:
        s1_to_fold = json.load(f)

    fold0_entities = sorted([s for s, fold in s1_to_fold.items() if fold == 0])[:25000]
    fold1_entities = sorted([s for s, fold in s1_to_fold.items() if fold == 1])[:25000]
    eval_entities = fold0_entities + fold1_entities
    all_eval_ids = set(eval_entities)

    print(f"Loaded {len(fold0_entities):,} (Fold 0) + {len(fold1_entities):,} (Fold 1) = {len(eval_entities):,} total evaluation entities.")

    # Load Ground Truth
    gt_file = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    eval_gt = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row[0] in all_eval_ids:
                matches = [x.strip() for x in row[1].split(",") if x.strip()] if len(row) > 1 and row[1].strip() else []
                eval_gt[row[0]] = matches

    # Load or build cached benchmark tables
    cache_file = "scratch/fixed_benchmark_cache_50k.pkl"
    if os.path.exists(cache_file):
        print(f"Loading cached benchmark tables from {cache_file}...")
        t0 = time.time()
        with open(cache_file, "rb") as f:
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
        print(f"Loaded cache in {time.time()-t0:.1f}s. Loaded {len(s1_records):,} S1 and {len(target_table):,} targets.")
    else:
        # Load Source 1
        print("Streaming S1 evaluation records...")
        t0 = time.time()
        s1_records = []
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

                    s1_records.append({
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
                    })

        print(f"Loaded {len(s1_records):,} S1 records in {time.time()-t0:.1f}s")

        # Index targets with EXACT caps from generate_v6_submission.py
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
            line_cnt = 0
            with open(path, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    line_cnt += 1
                    if line_cnt % 500000 == 0:
                        print(f"    {source_file}: processed {line_cnt:,} lines...")
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

                    # Caps identical to generate_v6_submission.py
                    if core in needed_cores and len(idx_core[core]) < 30:
                        idx_core[core].append(t_idx); hit = True
                    for ac in alias_cores:
                        if ac in needed_cores and len(idx_core[ac]) < 30:
                            idx_core[ac].append(t_idx); hit = True
                    if sorted_k in needed_sorted and len(idx_sorted[sorted_k]) < 30:
                        idx_sorted[sorted_k].append(t_idx); hit = True
                    if len(concat_k) >= 5 and concat_k in needed_concat and len(idx_concat[concat_k]) < 20:
                        idx_concat[concat_k].append(t_idx); hit = True
                    for bt in brand_tokens:
                        if bt in needed_brand and len(idx_brand[bt]) < 20:
                            idx_brand[bt].append(t_idx); hit = True
                    if num_p and num_p in needed_num_p and len(idx_num_p[num_p]) < 20:
                        idx_num_p[num_p].append(t_idx); hit = True
                    if post_p and post_p in needed_post_p and len(idx_post_p[post_p]) < 20:
                        idx_post_p[post_p].append(t_idx); hit = True
                    for ak in addr_keys:
                        if ak in needed_addr_k and len(idx_addr_k[ak]) < 10:
                            idx_addr_k[ak].append(t_idx); hit = True

                    if hit:
                        is_non_ascii = is_non_ascii_name(raw_name)
                        target_table.append((
                            core, sorted_k, concat_k, norm_name, clean_addr,
                            country, postal, num, dist_addr, is_non_ascii, alias_cores
                        ))
                        target_ids.append(t_id)
            print(f"  Finished {source_file}: cumulative targets loaded = {len(target_table):,}")

        print(f"Indexed all targets in {time.time()-t0:.1f}s. Total {len(target_table):,} targets into memory.")

        # Save cache
        print("Saving benchmark cache to disk...")
        t_save = time.time()
        with open(cache_file, "wb") as f_c:
            pickle.dump({
                "s1_records": s1_records,
                "target_table": target_table,
                "target_ids": target_ids,
                "idx_core": dict(idx_core),
                "idx_sorted": dict(idx_sorted),
                "idx_concat": dict(idx_concat),
                "idx_brand": dict(idx_brand),
                "idx_num_p": dict(idx_num_p),
                "idx_post_p": dict(idx_post_p),
                "idx_addr_k": dict(idx_addr_k),
            }, f_c, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Benchmark cache saved in {time.time()-t_save:.1f}s.")

    # Benchmark runner
    def evaluate_configuration(config_name: str, indic_dba_mode: bool, short_name_typo_mode: bool):
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

            # Candidate gathering
            cands = set()
            cands.update(idx_core.get(core_s1, [])[:500])
            for a in s1_aliases:
                cands.update(idx_core.get(a, [])[:500])
            cands.update(idx_sorted.get(sorted_s1, [])[:500])
            if len(concat_s1) >= 5:
                cands.update(idx_concat.get(concat_s1, [])[:300])
            for bt in brand_tokens[:6]:
                cands.update(idx_brand.get(bt, [])[:300])
            if num_p_s1:
                cands.update(idx_num_p.get(num_p_s1, [])[:300])
            if post_p_s1:
                cands.update(idx_post_p.get(post_p_s1, [])[:300])
            for ak in addr_keys_s1:
                cands.update(idx_addr_k.get(ak, [])[:200])

            scored_matches = []

            for t_idx in cands:
                t = target_table[t_idx]
                (core_t, sorted_t, concat_t, norm_t_name, clean_t_addr, t_country, t_p, t_n, t_dist, target_is_non_ascii, t_aliases) = t

                # Country constraint (100% true in ground truth)
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

                # Distinctive Address Gate
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

                # Name similarity
                lev = fuzz.ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
                sort_r = fuzz.token_sort_ratio(norm_s1_name, norm_t_name) / 100.0 if (norm_s1_name and norm_t_name) else 0.0
                max_name = max(lev, sort_r)

                score = 0.0

                # 1. Exact Core Name Match
                if core_s1 and core_t and core_s1 == core_t:
                    score = 0.95

                # 2. Alias / DBA Core Match
                elif any(a and a == core_t for a in s1_aliases) or any(a and a == core_s1 for a in t_aliases):
                    score = 0.94

                # 3. Sorted Core Match
                elif sorted_s1 and sorted_t == sorted_s1:
                    score = 0.92

                # 4. Domain / Handle Concatenated Match
                elif (concat_s1 and concat_t == concat_s1) or (concat_s1 and norm_t_name.replace(" ", "") == concat_s1):
                    score = 0.90 if (postal_match or num_match or dist_overlap >= 1 or not has_s1_addr) else 0.82

                # 5. Non-Latin Indic Script / DBA Record Match
                if indic_dba_mode:
                    # Enhanced: Check BOTH S1 and target for non-ascii, allow dist_overlap >= 1 if addr_token_sim >= 0.90 and num_match
                    is_indic = s1_is_non_ascii or target_is_non_ascii
                    if score < 0.80 and is_indic and num_match:
                        if dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.80:
                            score = 0.88
                        elif dist_overlap >= 1 and addr_token_sim is not None and addr_token_sim >= 0.90:
                            score = 0.86
                else:
                    # Baseline v6 rule
                    if score < 0.80 and target_is_non_ascii and num_match and dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.80:
                        score = 0.88

                # 6. Fuzzy Name Similarity (with multi-tenant safety)
                if score < 0.80:
                    if max_name >= 0.88:
                        if postal_match or num_match or dist_overlap >= 1 or not has_s1_addr or not has_t_addr:
                            score = 0.87
                    elif max_name >= 0.72:
                        if postal_match and (num_match or dist_overlap >= 1):
                            score = 0.84
                        elif num_match and dist_overlap >= 1:
                            score = 0.82
                        elif dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.82:
                            score = 0.80

                # 7. Short-name typo tolerance
                if short_name_typo_mode and score < 0.80:
                    # If core name is short (<= 7 chars) and edit distance == 1 character
                    min_len = min(len(core_s1), len(core_t))
                    max_len = max(len(core_s1), len(core_t))
                    if 3 <= min_len <= 7 and max_len <= 8:
                        edit_dist = distance.Levenshtein.distance(core_s1, core_t)
                        if edit_dist <= 1:
                            # Require street number match or postal match or brand overlap to ensure precision
                            if num_match or postal_match or dist_overlap >= 1:
                                score = 0.83

                if score >= 0.80:
                    scored_matches.append((t_idx, score, addr_token_sim if addr_token_sim is not None else 0.50))

            s1_to_scored[s1_id] = scored_matches
            for t_idx, sc, a_sim in scored_matches:
                target_claims[t_idx].append((s1_id, sc, a_sim))

        # 1-to-1 Mutual-Best Conflict Resolution
        winner_for_target = {}
        for t_idx, claims in target_claims.items():
            best_s1, best_sc, best_asim = max(claims, key=lambda x: (x[1], x[2]))
            winner_for_target[t_idx] = best_s1

        final_preds = {}
        total_matches = 0
        singleton_preds = 0

        for s1 in s1_records:
            s1_id = s1["id"]
            retained = [target_ids[t_idx] for t_idx, sc, a_sim in s1_to_scored.get(s1_id, [])
                        if winner_for_target.get(t_idx) == s1_id]
            final_preds[s1_id] = retained
            total_matches += len(retained)
            if len(retained) == 0:
                singleton_preds += 1

        # Evaluate overall and per-fold
        res_overall = evaluate_macro_f05(final_preds, eval_gt)
        preds_f0 = {s: final_preds[s] for s in fold0_entities}
        gt_f0 = {s: eval_gt.get(s, []) for s in fold0_entities}
        res_f0 = evaluate_macro_f05(preds_f0, gt_f0)

        preds_f1 = {s: final_preds[s] for s in fold1_entities}
        gt_f1 = {s: eval_gt.get(s, []) for s in fold1_entities}
        res_f1 = evaluate_macro_f05(preds_f1, gt_f1)

        print(f"\n[{config_name}] ({time.time()-t0:.1f}s)")
        print(f"  Overall Macro F0.5:   {res_overall['macro_f05']:.4f}")
        print(f"  Fold 0 Macro F0.5:    {res_f0['macro_f05']:.4f}")
        print(f"  Fold 1 Macro F0.5:    {res_f1['macro_f05']:.4f}")
        print(f"  Singleton Accuracy:   {res_overall['singleton_accuracy']:.4f} ({res_overall['singleton_count']:,} singletons)")
        print(f"  Non-Singleton F0.5:   {res_overall['non_singleton_f05']:.4f} ({res_overall['non_singleton_count']:,} non-singletons)")
        print(f"  Total Matches:        {total_matches:,}")
        print(f"  Predicted Singletons: {singleton_preds:,} ({singleton_preds/len(s1_records)*100:.2f}%)")

        # Test threshold filtering on retained matches
        best_filtered_f05 = res_overall['macro_f05']
        best_sc_thresh = 0.80
        for sc_thresh in [0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87]:
            filtered_preds = {}
            for s1 in s1_records:
                s1_id = s1["id"]
                retained = [target_ids[t_idx] for t_idx, sc, a_sim in s1_to_scored.get(s1_id, [])
                            if winner_for_target.get(t_idx) == s1_id and sc >= sc_thresh]
                filtered_preds[s1_id] = retained
            res_sc = evaluate_macro_f05(filtered_preds, eval_gt)
            if res_sc['macro_f05'] > best_filtered_f05:
                best_filtered_f05 = res_sc['macro_f05']
                best_sc_thresh = sc_thresh
                print(f"  * HIGHER SCORE with sc >= {sc_thresh:.2f} -> Macro F0.5: {res_sc['macro_f05']:.4f} | Sing Acc: {res_sc['singleton_accuracy']:.4f} | Non-Sing: {res_sc['non_singleton_f05']:.4f}")

        # Test Singleton Protection Gate on borderline matches
        for gate_mode in ["reject_weak_singletons", "require_address_on_fuzzy"]:
            gated_preds = {}
            for s1 in s1_records:
                s1_id = s1["id"]
                cand_matches = [(target_ids[t_idx], sc, a_sim) for t_idx, sc, a_sim in s1_to_scored.get(s1_id, [])
                                if winner_for_target.get(t_idx) == s1_id]
                if not cand_matches:
                    gated_preds[s1_id] = []
                    continue

                if gate_mode == "reject_weak_singletons":
                    # If entity only has 1 match and its score is < 0.84 with low address similarity, reject as singleton
                    if len(cand_matches) == 1 and cand_matches[0][1] < 0.84 and cand_matches[0][2] < 0.70:
                        gated_preds[s1_id] = []
                    else:
                        gated_preds[s1_id] = [m[0] for m in cand_matches]

                elif gate_mode == "require_address_on_fuzzy":
                    # If entity only has fuzzy name matches (score <= 0.84) and address similarity is low, reject
                    max_sc = max(m[1] for m in cand_matches)
                    if max_sc <= 0.84 and all(m[2] < 0.80 for m in cand_matches):
                        gated_preds[s1_id] = []
                    else:
                        gated_preds[s1_id] = [m[0] for m in cand_matches]

            res_gate = evaluate_macro_f05(gated_preds, eval_gt)
            print(f"  * Gate [{gate_mode}] -> Macro F0.5: {res_gate['macro_f05']:.4f} | Sing Acc: {res_gate['singleton_accuracy']:.4f} | Non-Sing: {res_gate['non_singleton_f05']:.4f}")

        return res_overall["macro_f05"]

    print("\n" + "=" * 80)
    print(" STEP 1: RUNNING BASELINE V6 ON FIXED BENCHMARK")
    print("=" * 80)
    score_baseline = evaluate_configuration("Baseline v6", indic_dba_mode=False, short_name_typo_mode=False)

    print("\n" + "=" * 80)
    print(" STEP 2: RUNNING CHANGE 1 (INDIC / CROSS-SCRIPT DBA ENHANCEMENT)")
    print("=" * 80)
    score_change1 = evaluate_configuration("Change 1: Indic DBA", indic_dba_mode=True, short_name_typo_mode=False)

    print("\n" + "=" * 80)
    print(" STEP 3: RUNNING CHANGE 2 (SHORT-NAME TYPO TOLERANCE)")
    print("=" * 80)
    score_change2 = evaluate_configuration("Change 2: Short-Name Typo", indic_dba_mode=False, short_name_typo_mode=True)

    print("\n" + "=" * 80)
    print(" STEP 4: RUNNING CHANGE 1 + 2 COMBINED")
    print("=" * 80)
    score_combined = evaluate_configuration("Change 1 + 2 Combined", indic_dba_mode=True, short_name_typo_mode=True)

    print("\n" + "=" * 80)
    print(" BENCHMARK COMPARISON SUMMARY")
    print("=" * 80)
    print(f"  Baseline v6:                        {score_baseline:.4f}")
    print(f"  Change 1 (Indic DBA):               {score_change1:.4f}  (delta: {score_change1 - score_baseline:+.4f})")
    print(f"  Change 2 (Short-Name Typo):         {score_change2:.4f}  (delta: {score_change2 - score_baseline:+.4f})")
    print(f"  Change 1 + 2 Combined:              {score_combined:.4f}  (delta: {score_combined - score_baseline:+.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
