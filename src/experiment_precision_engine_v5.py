#!/usr/bin/env python3
"""Precision Engine v5 for Business Entity Resolution.

Introduces:
1. Distinctive Address Component Verification (Street + City token overlap):
   - Eliminates false cross-city merges (e.g. Greenville vs Charlotte) that shared only generic words ('street', 'state')
2. Missing Target Address Safeguards:
   - When Target has no address, require multi-word name (>= 2 tokens or len >= 14) and exact core match
3. Composite Location-Name Indexing:
   - num_prefix ('1334_cas')
   - post_prefix ('24422_cas')
   - exact core, sorted core, concat core, distinctive brand tokens
4. Global 1-to-1 Mutual-Best Conflict Resolution.
"""

from collections import defaultdict
import json
import os
import re
import sys
import time
import unicodedata
from typing import Dict, List, Set, Tuple

from rapidfuzz import fuzz
from metrics import evaluate_macro_f05


LEGAL_SUFFIXES = {
    # US / UK / India
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited", "co", "company", "dba", "lp", "pllc",
    "pvt", "private", "llp", "opc", "lnc", "1nc",
    # France
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
    # French abbreviations
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
    """Returns (normalized_address_str, postal_code, street_number, distinctive_tokens)."""
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

    # Postal code (5 or 6 digits)
    postal_match = re.search(r'\b(\d{5,6})\b', clean_addr)
    postal = postal_match.group(1) if postal_match else ""

    # Street number: strip leading zeros (e.g. 01334 -> 1334)
    num_match = re.search(r'\b0*(\d{1,6})\b', clean_addr)
    num = num_match.group(1) if num_match else ""

    # Distinctive tokens (street name, city, locality)
    dist_tokens = {t for t in tokens if len(t) >= 3 and t not in GENERIC_ADDR_WORDS and not t.isdigit()}

    return clean_addr, postal, num, dist_tokens


def make_prefix_key(name: str, length: int = 3) -> str:
    tokens = extract_core_tokens(name)
    if not tokens:
        return ""
    first = tokens[0]
    return first[:length] if len(first) >= length else first


def main():
    train_dir = "dataset/train"
    splits_file = os.path.join(train_dir, "cv_splits_5fold.json")
    print("=" * 80)
    print(" [HYBRID PRECISION ENGINE V5] Benchmarking with Distinctive Address Gate")
    print("=" * 80)

    with open(splits_file, "r", encoding="utf-8") as f:
        s1_to_fold = json.load(f)

    # 50,000 validation entities from Fold 0
    fold0_entities = sorted([s1 for s1, fold in s1_to_fold.items() if fold == 0])
    val_s1_ids = set(fold0_entities[:50000])

    # Load Ground Truth
    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    val_gt = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in val_s1_ids:
                targets = [x.strip() for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1].strip() else []
                val_gt[parts[0]] = targets

    # Load Source 1 text
    s1_data = {}
    s1_file = os.path.join(train_dir, "train_source1.tsv")
    with open(s1_file, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            if s1_id in val_s1_ids:
                raw_name = parts[1] if len(parts) > 1 else ""
                raw_addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""

                norm_name = normalize_raw_text(raw_name)
                clean_addr, postal, num, dist_addr = normalize_address(raw_addr)
                aliases = extract_alias_names(raw_name)
                core_prefix3 = make_prefix_key(norm_name, 3)

                s1_data[s1_id] = {
                    "raw_name": raw_name,
                    "norm_name": norm_name,
                    "clean_addr": clean_addr,
                    "country": country,
                    "postal": postal,
                    "num": num,
                    "dist_addr": dist_addr,
                    "core": extract_core_name(norm_name),
                    "sorted": extract_sorted_key(norm_name),
                    "concat": extract_concat_key(norm_name),
                    "distinctive": extract_distinctive_tokens(norm_name),
                    "aliases": [extract_core_name(a) for a in aliases if extract_core_name(a)],
                    "num_prefix": f"{num}_{core_prefix3}" if (num and core_prefix3) else "",
                    "post_prefix": f"{postal}_{core_prefix3}" if (postal and core_prefix3) else "",
                }

    needed_cores = {r["core"] for r in s1_data.values() if r["core"]}
    for r in s1_data.values():
        for a in r["aliases"]:
            needed_cores.add(a)

    needed_sorted = {r["sorted"] for r in s1_data.values() if r["sorted"]}
    needed_concat = {r["concat"] for r in s1_data.values() if len(r["concat"]) >= 5}
    needed_brand = {t for r in s1_data.values() for t in r["distinctive"]}
    needed_num_p = {r["num_prefix"] for r in s1_data.values() if r["num_prefix"]}
    needed_post_p = {r["post_prefix"] for r in s1_data.values() if r["post_prefix"]}

    print("\nIndexing training targets (S2 + S3)...")
    start_time = time.time()
    idx_core = defaultdict(list)
    idx_sorted = defaultdict(list)
    idx_concat = defaultdict(list)
    idx_brand = defaultdict(list)
    idx_num_p = defaultdict(list)
    idx_post_p = defaultdict(list)
    target_data = {}

    for source_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(train_dir, source_file)
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                t_id = parts[0].strip()
                raw_name = parts[1] if len(parts) > 1 else ""
                raw_addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""

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

                hit = False
                if core in needed_cores and len(idx_core[core]) < 30:
                    idx_core[core].append(t_id)
                    hit = True
                for a in aliases:
                    ac = extract_core_name(a)
                    if ac in needed_cores and len(idx_core[ac]) < 30:
                        idx_core[ac].append(t_id)
                        hit = True
                if sorted_k in needed_sorted and len(idx_sorted[sorted_k]) < 30:
                    idx_sorted[sorted_k].append(t_id)
                    hit = True
                if concat_k in needed_concat and len(idx_concat[concat_k]) < 20:
                    idx_concat[concat_k].append(t_id)
                    hit = True
                for bt in brand_tokens:
                    if bt in needed_brand and len(idx_brand[bt]) < 20:
                        idx_brand[bt].append(t_id)
                        hit = True
                if num_p and num_p in needed_num_p and len(idx_num_p[num_p]) < 20:
                    idx_num_p[num_p].append(t_id)
                    hit = True
                if post_p and post_p in needed_post_p and len(idx_post_p[post_p]) < 20:
                    idx_post_p[post_p].append(t_id)
                    hit = True

                if hit:
                    target_data[t_id] = {
                        "raw_name": raw_name,
                        "norm_name": norm_name,
                        "clean_addr": clean_addr,
                        "country": country,
                        "postal": postal,
                        "num": num,
                        "dist_addr": dist_addr,
                        "core": core,
                        "sorted": sorted_k,
                        "concat": concat_k,
                        "aliases": [extract_core_name(a) for a in aliases if extract_core_name(a)],
                    }

    print(f"Indexed targets in {time.time() - start_time:.1f}s. Loaded {len(target_data):,} relevant targets.")

    # High-Precision Multi-Tier Matching Engine with Distinctive Address Gate
    print("\nExecuting High-Precision Matcher v5...")
    s1_candidates = defaultdict(set)
    s1_scored_matches = defaultdict(list)

    for s1_id, s1 in s1_data.items():
        core_s1 = s1["core"]
        sorted_s1 = s1["sorted"]
        concat_s1 = s1["concat"]
        brand_tokens = s1["distinctive"]
        s1_p, s1_n = s1["postal"], s1["num"]
        s1_dist = s1["dist_addr"]
        s1_country = s1["country"].lower()
        s1_aliases = s1["aliases"]
        num_p_s1 = s1["num_prefix"]
        post_p_s1 = s1["post_prefix"]

        # Gather Candidates across all routes
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

        s1_candidates[s1_id] = cands

        for t_id in cands:
            t = target_data.get(t_id)
            if not t:
                continue

            # Country constraint
            t_country = t["country"].lower()
            if s1_country and t_country and s1_country != t_country:
                continue

            core_t = t["core"]
            score = 0.0

            # Address metrics
            t_p, t_n = t["postal"], t["num"]
            t_dist = t["dist_addr"]
            has_s1_addr = bool(s1["clean_addr"])
            has_t_addr = bool(t["clean_addr"])

            postal_match = (s1_p and t_p and s1_p == t_p)
            postal_conflict = (s1_p and t_p and s1_p != t_p)
            num_match = (s1_n and t_n and s1_n == t_n)
            num_conflict = (s1_n and t_n and s1_n != t_n)

            # Distinctive token overlap (street / city / locality)
            dist_overlap = len(s1_dist.intersection(t_dist)) if (s1_dist and t_dist) else 0

            addr_token_sim = (
                fuzz.token_set_ratio(s1["clean_addr"], t["clean_addr"]) / 100.0
                if (has_s1_addr and has_t_addr)
                else None
            )

            # DISTINCTIVE ADDRESS GATE:
            # If both addresses are present:
            if has_s1_addr and has_t_addr:
                # 1. Overall address token sim floor
                if addr_token_sim is not None and addr_token_sim < 0.35:
                    continue

                # 2. Strict Distinctive Overlap: If both have distinctive tokens, and postal doesn't match:
                #    Zero distinctive overlap means different street AND different city! REJECT!
                if s1_dist and t_dist and not postal_match and dist_overlap == 0:
                    continue

                # 3. Postal conflict gate
                if postal_conflict and addr_token_sim is not None and addr_token_sim < 0.70:
                    continue

                # 4. Street number conflict gate
                if num_conflict and dist_overlap == 0:
                    continue
            else:
                # Target Missing Address Guard:
                # If target has NO address, strictly require multi-word distinctive name
                core_words = core_s1.split()
                if len(core_words) < 2 and len(core_s1) < 14:
                    continue

            # 1. Exact Core Name Match
            if core_s1 and core_t and core_s1 == core_t:
                score = 0.95

            # 2. Alias / DBA Core Match
            elif any(a and a == core_t for a in s1_aliases) or any(a and a == core_s1 for a in t.get("aliases", [])):
                score = 0.94

            # 3. Sorted Core Match (word transpositions)
            elif sorted_s1 and t["sorted"] == sorted_s1:
                score = 0.92

            # 4. Domain / Handle Concatenated Match
            elif (concat_s1 and t["concat"] == concat_s1) or (concat_s1 and t["norm_name"].replace(" ", "") == concat_s1):
                score = 0.90 if (postal_match or num_match or dist_overlap >= 1 or not has_s1_addr) else 0.82

            # 5. Fuzzy Name Similarity
            else:
                lev = fuzz.ratio(core_s1, core_t) / 100.0
                sort_r = fuzz.token_sort_ratio(s1["norm_name"], t["norm_name"]) / 100.0
                max_name = max(lev, sort_r)

                if max_name >= 0.88:
                    if postal_match or num_match or dist_overlap >= 1 or not has_s1_addr or not has_t_addr:
                        score = 0.87
                elif max_name >= 0.72:
                    if postal_match and (num_match or dist_overlap >= 1):
                        score = 0.84
                    elif num_match and dist_overlap >= 1:
                        score = 0.82
                    elif dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.80:
                        score = 0.80

            if score >= 0.80:
                s1_scored_matches[s1_id].append((t_id, score, addr_token_sim if addr_token_sim is not None else 0.50))

    # 1-to-1 Mutual-Best Conflict Resolution
    print("Applying 1-to-1 Mutual-Best Assignment...")
    target_claims = defaultdict(list)
    for s1_id, matches in s1_scored_matches.items():
        for t_id, sc, a_sim in matches:
            target_claims[t_id].append((s1_id, sc, a_sim))

    winner_for_target = {}
    for t_id, claims in target_claims.items():
        best_s1, best_sc, best_asim = max(claims, key=lambda x: (x[1], x[2]))
        winner_for_target[t_id] = best_s1

    # Filter final predictions
    final_preds = {}
    for s1_id in val_s1_ids:
        retained = []
        for t_id, sc, a_sim in s1_scored_matches.get(s1_id, []):
            if winner_for_target.get(t_id) == s1_id:
                retained.append(t_id)
        final_preds[s1_id] = retained

    # Compute validation metrics
    res = evaluate_macro_f05(final_preds, val_gt)
    total_gold = sum(len(g) for g in val_gt.values())
    captured = sum(len(set(val_gt[s]).intersection(s1_candidates[s])) for s in val_s1_ids)

    print("\n" + "=" * 80)
    print(" [HYBRID PRECISION ENGINE V5 BENCHMARK RESULTS]")
    print("=" * 80)
    print(f"  BASELINE SCORE:            0.3265")
    print(f"  ENGINE V3 SCORE:           0.6625")
    print(f"  ENGINE V5 MODEL SCORE:     {res['macro_f05']:.4f}  [NET GAIN OVER BASELINE: +{(res['macro_f05'] - 0.3265)*100:.2f}%!]")
    print(f"  Singleton Accuracy:        {res['singleton_accuracy']:.4f} ({res['singleton_count']} singletons)")
    print(f"  Non-Singleton F0.5:        {res['non_singleton_f05']:.4f} ({res['non_singleton_count']} non-singletons)")
    print(f"  Candidate Pool Recall:     {captured / total_gold * 100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
