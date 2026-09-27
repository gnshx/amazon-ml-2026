#!/usr/bin/env python3
"""Ultra-Lean High-Performance Production Submission Generator V4.

Amazon ML Challenge 2026 - Business Entity Resolution.
Target: Maximum Macro F0.5 | Runtime: ~12-14 minutes | Peak RAM: < 10.5 GB (Zero Swap)

Key Enhancements over V2 (74.5% baseline):
1. Integer Bitmask Business Category Conflict Guard (0 MB overhead, eliminates false merges on shared address)
2. RapidFuzz partial_ratio for DBA / brand substring matching
3. High-precision Postal + 4-gram Prefix Route (cap 15, catches spelling/ending variations)
4. Multi-Signal Composite Singleton Protection Gate
5. Tight memory architecture: zero set allocations per target, pure integer bitmasks
"""

import csv
import gc
import os
import pickle
import re
import sys
import time
import unicodedata
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from rapidfuzz import fuzz, distance

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
    "nd": "northtdakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
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

# Integer bitmask mapping for business categories (0 MB extra RAM)
CAT_BITS = {
    "pharmacy": 1, "pharma": 1,
    "clinic": 2, "hospital": 2, "medical": 2,
    "hotel": 4, "motel": 4, "hospitality": 4,
    "restaurant": 8, "cafe": 8, "bakery": 8, "pizza": 8, "burger": 8, "diner": 8,
    "bank": 16, "finance": 16, "insurance": 16,
    "realty": 32, "real": 32,
    "school": 64, "college": 64, "university": 64, "academy": 64,
    "salon": 128, "spa": 128, "barber": 128,
    "gym": 256, "fitness": 256,
    "auto": 512, "motors": 512, "car": 512, "garage": 512,
    "tech": 1024, "software": 1024, "computers": 1024,
    "dental": 2048, "dentist": 2048,
    "law": 4096, "attorney": 4096, "legal": 4096
}


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
    return [t for t in tokens if len(t) >= 3 and t not in GENERIC_WORDS]


def get_category_bitmask(norm_name: str) -> int:
    mask = 0
    for w in norm_name.split():
        mask |= CAT_BITS.get(w, 0)
    return mask


def normalize_address(raw_addr: str) -> Tuple[str, str, str, Set[str]]:
    if not raw_addr:
        return "", "", "", set()
    norm = normalize_raw_text(raw_addr)
    postal_matches = re.findall(r'\b\d{5,6}\b', norm)
    postal = postal_matches[0] if postal_matches else ""

    num_matches = re.findall(r'\b\d{1,5}[a-z]?\b', norm)
    num = ""
    for n in num_matches:
        if n != postal:
            num = n
            break

    words = norm.split()
    standardized = []
    dist_words = set()
    for w in words:
        w_exp = STREET_ABBREVIATIONS.get(w, US_STATES.get(w, w))
        standardized.append(w_exp)
        if len(w) >= 3 and w not in GENERIC_ADDR_WORDS and not w.isdigit():
            dist_words.add(w)

    clean_addr = " ".join(standardized)
    return clean_addr, postal, num, dist_words


def make_prefix_key(name: str, length: int = 3) -> str:
    core = extract_core_name(name).replace(" ", "")
    return core[:length] if len(core) >= length else ""


def main():
    print("=" * 80, flush=True)
    print(" [PRODUCTION PIPELINE V4 LEAN] Amazon ML Challenge 2026", flush=True)
    print(" Zero-Swap Lean Stream | Category Bitmask Guard | Multi-Signal Gate | Partial Ratio", flush=True)
    print("=" * 80, flush=True)

    test_dir = "dataset/test"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    # Pass 1: Stream S1
    print("\n[Step 1/5] Pass 1: Streaming Source 1 to collect query keys...", flush=True)
    t0 = time.time()
    needed_cores = set()
    needed_sorted = set()
    needed_concat = set()
    needed_brand = set()
    needed_num_p = set()
    needed_post_p = set()
    needed_post_num = set()
    needed_post_p4 = set()
    needed_addr_k = set()
    total_s1_count = 0

    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            total_s1_count += 1
            parts = line.rstrip("\r\n").split("\t")
            raw_name = parts[1] if len(parts) > 1 else ""
            raw_addr = parts[2] if len(parts) > 2 else ""

            norm_name = normalize_raw_text(raw_name)
            clean_addr, postal, num, dist_addr = normalize_address(raw_addr)
            aliases = extract_alias_names(raw_name)
            core_prefix3 = make_prefix_key(norm_name, 3)
            core_prefix4 = make_prefix_key(norm_name, 4)

            core = extract_core_name(norm_name)
            sorted_k = extract_sorted_key(norm_name)
            concat_k = extract_concat_key(norm_name)
            brand_tokens = extract_distinctive_tokens(norm_name)
            alias_cores = [extract_core_name(a) for a in aliases if extract_core_name(a)]

            num_p = f"{num}_{core_prefix3}" if (num and core_prefix3) else ""
            post_p = f"{postal}_{core_prefix3}" if (postal and core_prefix3) else ""
            post_num = f"{postal}_{num}" if (postal and num) else ""
            post_p4 = f"{postal}_{core_prefix4}" if (postal and core_prefix4) else ""
            addr_keys = [f"{num}_{t}" for t in sorted(dist_addr)[:2]] if num else []

            if core: needed_cores.add(core)
            for ac in alias_cores: needed_cores.add(ac)
            if sorted_k: needed_sorted.add(sorted_k)
            if len(concat_k) >= 5: needed_concat.add(concat_k)
            for bt in brand_tokens: needed_brand.add(bt)
            if num_p: needed_num_p.add(num_p)
            if post_p: needed_post_p.add(post_p)
            if post_num: needed_post_num.add(post_num)
            if post_p4: needed_post_p4.add(post_p4)
            for ak in addr_keys: needed_addr_k.add(ak)

    print(f"Collected query keys from {total_s1_count:,} Source 1 records in {time.time() - t0:.1f}s.", flush=True)

    # Pass 2: Stream and index targets
    print("\n[Step 2/5] Pass 2: Streaming and indexing test targets (S2 + S3)...", flush=True)
    t0 = time.time()
    idx_core = defaultdict(list)
    idx_sorted = defaultdict(list)
    idx_concat = defaultdict(list)
    idx_brand = defaultdict(list)
    idx_num_p = defaultdict(list)
    idx_post_p = defaultdict(list)
    idx_post_num = defaultdict(list)
    idx_post_p4 = defaultdict(list)
    idx_addr_k = defaultdict(list)

    target_table = []
    target_ids = []

    for source_file in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(test_dir, source_file)
        print(f"  Streaming {source_file}...", flush=True)
        line_cnt = 0
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                line_cnt += 1
                if line_cnt % 1000000 == 0:
                    print(f"    {source_file}: processed {line_cnt:,} lines, loaded {len(target_table):,} targets...", flush=True)
                parts = line.rstrip("\r\n").split("\t")
                t_id = parts[0].strip()
                raw_name = parts[1] if len(parts) > 1 else ""
                raw_addr = parts[2] if len(parts) > 2 else ""
                country = parts[3].strip().lower() if len(parts) > 3 else ""

                norm_name = normalize_raw_text(raw_name)
                clean_addr, postal, num, dist_addr = normalize_address(raw_addr)
                aliases = extract_alias_names(raw_name)
                core_prefix3 = make_prefix_key(norm_name, 3)
                core_prefix4 = make_prefix_key(norm_name, 4)

                core = extract_core_name(norm_name)
                sorted_k = extract_sorted_key(norm_name)
                concat_k = extract_concat_key(norm_name)
                brand_tokens = extract_distinctive_tokens(norm_name)
                alias_cores = [extract_core_name(a) for a in aliases if extract_core_name(a)]

                num_p = f"{num}_{core_prefix3}" if (num and core_prefix3) else ""
                post_p = f"{postal}_{core_prefix3}" if (postal and core_prefix3) else ""
                post_num = f"{postal}_{num}" if (postal and num) else ""
                post_p4 = f"{postal}_{core_prefix4}" if (postal and core_prefix4) else ""
                addr_keys = [f"{num}_{t}" for t in sorted(dist_addr)[:2]] if num else []

                hit = False
                t_idx = len(target_table)

                # Strict safe caps to guarantee zero swap thrashing
                if core in needed_cores and len(idx_core[core]) < 35:
                    idx_core[core].append(t_idx); hit = True
                for ac in alias_cores:
                    if ac in needed_cores and len(idx_core[ac]) < 35:
                        idx_core[ac].append(t_idx); hit = True
                if sorted_k in needed_sorted and len(idx_sorted[sorted_k]) < 35:
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
                if post_num and post_num in needed_post_num and len(idx_post_num[post_num]) < 20:
                    idx_post_num[post_num].append(t_idx); hit = True
                if post_p4 and post_p4 in needed_post_p4 and len(idx_post_p4[post_p4]) < 15:
                    idx_post_p4[post_p4].append(t_idx); hit = True
                for ak in addr_keys:
                    if ak in needed_addr_k and len(idx_addr_k[ak]) < 12:
                        idx_addr_k[ak].append(t_idx); hit = True

                if hit:
                    is_non_ascii = is_non_ascii_name(raw_name)
                    cat_mask = get_category_bitmask(norm_name)
                    # Ultra-compact tuple (pure primitives, no sets)
                    target_table.append((
                        core, sorted_k, concat_k, norm_name, clean_addr,
                        country, postal, num, tuple(sorted(dist_addr)), is_non_ascii, alias_cores, cat_mask
                    ))
                    target_ids.append(t_id)

    print(f"Indexed targets in {time.time() - t0:.1f}s. Loaded {len(target_table):,} relevant targets into memory.", flush=True)

    # Pass 3: In-Process High-Speed Matching
    print("\n[Step 3/5] Pass 3: In-Process High-Speed Matching across all 1.73M entities...", flush=True)
    t0 = time.time()
    CHUNK_SIZE = 100000

    scratch_file = os.path.join(output_dir, "scratch_results_v4_lean.pkl")
    target_claims = {}

    total_processed = 0
    chunk_count = 0

    with open(scratch_file, "wb") as f_scratch:
        with open(s1_path, "r", encoding="utf-8") as f_s1:
            f_s1.readline()
            current_chunk = []
            for line in f_s1:
                parts = line.rstrip("\r\n").split("\t")
                s1_id = parts[0].strip()
                raw_name = parts[1] if len(parts) > 1 else ""
                raw_addr = parts[2] if len(parts) > 2 else ""
                country = parts[3].strip().lower() if len(parts) > 3 else ""

                norm_name = normalize_raw_text(raw_name)
                clean_addr, postal, num, dist_addr = normalize_address(raw_addr)
                aliases = extract_alias_names(raw_name)
                core_prefix3 = make_prefix_key(norm_name, 3)
                core_prefix4 = make_prefix_key(norm_name, 4)

                core = extract_core_name(norm_name)
                sorted_k = extract_sorted_key(norm_name)
                concat_k = extract_concat_key(norm_name)
                brand_tokens = extract_distinctive_tokens(norm_name)
                alias_cores = [extract_core_name(a) for a in aliases if extract_core_name(a)]

                num_p = f"{num}_{core_prefix3}" if (num and core_prefix3) else ""
                post_p = f"{postal}_{core_prefix3}" if (postal and core_prefix3) else ""
                post_num = f"{postal}_{num}" if (postal and num) else ""
                post_p4 = f"{postal}_{core_prefix4}" if (postal and core_prefix4) else ""
                addr_keys = [f"{num}_{t}" for t in sorted(dist_addr)[:2]] if num else []
                s1_cat_mask = get_category_bitmask(norm_name)

                current_chunk.append((
                    s1_id, core, sorted_k, concat_k, brand_tokens, postal, num,
                    dist_addr, country, alias_cores, num_p, post_p, post_num, post_p4, addr_keys,
                    clean_addr, norm_name, is_non_ascii_name(raw_name), s1_cat_mask
                ))

                if len(current_chunk) >= CHUNK_SIZE:
                    chunk_count += 1
                    t_chunk_start = time.time()
                    chunk_results = []

                    for s1_rec in current_chunk:
                        (s1_id, core_s1, sorted_s1, concat_s1, brand_tokens, s1_p, s1_n, s1_dist,
                         s1_country, s1_aliases, num_p_s1, post_p_s1, post_num_s1, post_p4_s1, addr_keys_s1, clean_s1_addr,
                         norm_s1_name, s1_is_non_ascii, s1_cat_mask) = s1_rec

                        cands = set()
                        cands.update(idx_core.get(core_s1, []))
                        for a in s1_aliases:
                            cands.update(idx_core.get(a, []))
                        cands.update(idx_sorted.get(sorted_s1, []))
                        if len(concat_s1) >= 5:
                            cands.update(idx_concat.get(concat_s1, []))
                        for bt in brand_tokens[:5]:
                            cands.update(idx_brand.get(bt, []))
                        if num_p_s1:
                            cands.update(idx_num_p.get(num_p_s1, []))
                        if post_p_s1:
                            cands.update(idx_post_p.get(post_p_s1, []))
                        if post_num_s1:
                            cands.update(idx_post_num.get(post_num_s1, []))
                        if post_p4_s1:
                            cands.update(idx_post_p4.get(post_p4_s1, []))
                        for ak in addr_keys_s1:
                            cands.update(idx_addr_k.get(ak, []))

                        scored_matches = []
                        has_s1_addr = bool(clean_s1_addr)

                        for t_idx in cands:
                            t = target_table[t_idx]
                            (core_t, sorted_t, concat_t, norm_t_name, clean_t_addr, t_country, t_p, t_n, t_dist_tuple, target_is_non_ascii, t_aliases, t_cat_mask) = t

                            if s1_country and t_country and s1_country != t_country:
                                continue

                            has_t_addr = bool(clean_t_addr)
                            postal_match = (s1_p and t_p and s1_p == t_p)
                            postal_conflict = (s1_p and t_p and s1_p != t_p)
                            num_match = (s1_n and t_n and s1_n == t_n)
                            num_conflict = (s1_n and t_n and s1_n != t_n)

                            dist_overlap = 0
                            if s1_dist and t_dist_tuple:
                                for dw in t_dist_tuple:
                                    if dw in s1_dist:
                                        dist_overlap += 1

                            # Category conflict bitmask check (Doc 17 insight)
                            # If both have categories and bits don't intersect, businesses are in conflicting categories!
                            has_cat_conflict = (s1_cat_mask > 0 and t_cat_mask > 0 and (s1_cat_mask & t_cat_mask) == 0)

                            addr_token_sim = (
                                fuzz.token_set_ratio(clean_s1_addr, clean_t_addr) / 100.0
                                if (has_s1_addr and has_t_addr)
                                else None
                            )

                            if has_s1_addr and has_t_addr:
                                if addr_token_sim is not None and addr_token_sim < 0.35:
                                    continue
                                if s1_dist and t_dist_tuple and not postal_match and dist_overlap == 0:
                                    continue
                                if postal_conflict and addr_token_sim is not None and addr_token_sim < 0.70:
                                    continue
                                if num_conflict and dist_overlap == 0:
                                    continue
                            else:
                                core_words = core_s1.split()
                                if len(core_words) < 2 and len(core_s1) < 14:
                                    continue

                            lev = fuzz.ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
                            sort_r = fuzz.token_sort_ratio(norm_s1_name, norm_t_name) / 100.0 if (norm_s1_name and norm_t_name) else 0.0
                            part_r = fuzz.partial_ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t and max(len(core_s1), len(core_t)) > 6) else 0.0
                            max_name = max(lev, sort_r, part_r)

                            # If conflicting categories exist, reject address-only or low-name matches
                            if has_cat_conflict and max_name < 0.85:
                                continue

                            score = 0.0
                            if core_s1 and core_t and core_s1 == core_t:
                                score = 0.95
                            elif any(a and a == core_t for a in s1_aliases) or any(a and a == core_s1 for a in t_aliases):
                                score = 0.94
                            elif sorted_s1 and sorted_t == sorted_s1:
                                score = 0.92
                            elif (concat_s1 and concat_t == concat_s1) or (concat_s1 and norm_t_name.replace(" ", "") == concat_s1):
                                score = 0.90 if (postal_match or num_match or dist_overlap >= 1 or not has_s1_addr) else 0.82
                            elif (s1_is_non_ascii or target_is_non_ascii) and num_match:
                                if dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.80:
                                    score = 0.88
                                elif dist_overlap >= 1 and addr_token_sim is not None and addr_token_sim >= 0.90:
                                    score = 0.86
                            elif max_name >= 0.88:
                                if postal_match or num_match or dist_overlap >= 1 or not has_s1_addr or not has_t_addr:
                                    score = 0.87
                            elif max_name >= 0.72:
                                if postal_match and (num_match or dist_overlap >= 1):
                                    score = 0.84
                                elif num_match and dist_overlap >= 1:
                                    score = 0.82
                                elif dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.82:
                                    score = 0.80
                            elif max_name >= 0.60 and part_r >= 0.82:
                                if postal_match and num_match:
                                    score = 0.83
                                elif postal_match and dist_overlap >= 1:
                                    score = 0.82

                            min_len = min(len(core_s1), len(core_t))
                            max_len = max(len(core_s1), len(core_t))
                            if score < 0.80 and 3 <= min_len <= 7 and max_len <= 8:
                                if distance.Levenshtein.distance(core_s1, core_t) <= 1:
                                    if num_match or postal_match or dist_overlap >= 1:
                                        score = 0.83

                            # Strict precision cutoff
                            if score >= 0.81:
                                a_sim_val = addr_token_sim if addr_token_sim is not None else 0.50
                                scored_matches.append((t_idx, score, a_sim_val, postal_match, num_match, len(core_s1)))

                        total_processed += 1
                        for t_idx, sc, a_sim, pm, nm, _ in scored_matches:
                            prev = target_claims.get(t_idx)
                            rank_tuple = (sc, 1 if nm else 0, 1 if pm else 0, a_sim)
                            if prev is None or rank_tuple > prev[1]:
                                target_claims[t_idx] = (s1_id, rank_tuple)

                        chunk_results.append((s1_id, list(cands), scored_matches))

                    pickle.dump(chunk_results, f_scratch)
                    dt = time.time() - t_chunk_start
                    rate = len(current_chunk) / dt if dt > 0 else 0
                    print(f"  Chunk {chunk_count:2d}: {total_processed:,} / {total_s1_count:,} ({total_processed/total_s1_count*100:.1f}%) | {rate:,.0f} entities/s | Elapsed: {time.time()-t0:.0f}s", flush=True)
                    current_chunk = []
                    gc.collect()

            # Process final remaining chunk
            if current_chunk:
                chunk_count += 1
                chunk_results = []
                for s1_rec in current_chunk:
                    (s1_id, core_s1, sorted_s1, concat_s1, brand_tokens, s1_p, s1_n, s1_dist,
                     s1_country, s1_aliases, num_p_s1, post_p_s1, post_num_s1, post_p4_s1, addr_keys_s1, clean_s1_addr,
                     norm_s1_name, s1_is_non_ascii, s1_cat_mask) = s1_rec

                    cands = set()
                    cands.update(idx_core.get(core_s1, []))
                    for a in s1_aliases: cands.update(idx_core.get(a, []))
                    cands.update(idx_sorted.get(sorted_s1, []))
                    if len(concat_s1) >= 5: cands.update(idx_concat.get(concat_s1, []))
                    for bt in brand_tokens[:5]: cands.update(idx_brand.get(bt, []))
                    if num_p_s1: cands.update(idx_num_p.get(num_p_s1, []))
                    if post_p_s1: cands.update(idx_post_p.get(post_p_s1, []))
                    if post_num_s1: cands.update(idx_post_num.get(post_num_s1, []))
                    if post_p4_s1: cands.update(idx_post_p4.get(post_p4_s1, []))
                    for ak in addr_keys_s1: cands.update(idx_addr_k.get(ak, []))

                    scored_matches = []
                    has_s1_addr = bool(clean_s1_addr)

                    for t_idx in cands:
                        t = target_table[t_idx]
                        (core_t, sorted_t, concat_t, norm_t_name, clean_t_addr, t_country, t_p, t_n, t_dist_tuple, target_is_non_ascii, t_aliases, t_cat_mask) = t
                        if s1_country and t_country and s1_country != t_country: continue

                        has_t_addr = bool(clean_t_addr)
                        postal_match = (s1_p and t_p and s1_p == t_p)
                        postal_conflict = (s1_p and t_p and s1_p != t_p)
                        num_match = (s1_n and t_n and s1_n == t_n)
                        num_conflict = (s1_n and t_n and s1_n != t_n)

                        dist_overlap = 0
                        if s1_dist and t_dist_tuple:
                            for dw in t_dist_tuple:
                                if dw in s1_dist: dist_overlap += 1

                        has_cat_conflict = (s1_cat_mask > 0 and t_cat_mask > 0 and (s1_cat_mask & t_cat_mask) == 0)

                        addr_token_sim = (
                            fuzz.token_set_ratio(clean_s1_addr, clean_t_addr) / 100.0
                            if (has_s1_addr and has_t_addr) else None
                        )

                        if has_s1_addr and has_t_addr:
                            if addr_token_sim is not None and addr_token_sim < 0.35: continue
                            if s1_dist and t_dist_tuple and not postal_match and dist_overlap == 0: continue
                            if postal_conflict and addr_token_sim is not None and addr_token_sim < 0.70: continue
                            if num_conflict and dist_overlap == 0: continue
                        else:
                            core_words = core_s1.split()
                            if len(core_words) < 2 and len(core_s1) < 14: continue

                        lev = fuzz.ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
                        sort_r = fuzz.token_sort_ratio(norm_s1_name, norm_t_name) / 100.0 if (norm_s1_name and norm_t_name) else 0.0
                        part_r = fuzz.partial_ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t and max(len(core_s1), len(core_t)) > 6) else 0.0
                        max_name = max(lev, sort_r, part_r)

                        if has_cat_conflict and max_name < 0.85:
                            continue

                        score = 0.0
                        if core_s1 and core_t and core_s1 == core_t: score = 0.95
                        elif any(a and a == core_t for a in s1_aliases) or any(a and a == core_s1 for a in t_aliases): score = 0.94
                        elif sorted_s1 and sorted_t == sorted_s1: score = 0.92
                        elif (concat_s1 and concat_t == concat_s1) or (concat_s1 and norm_t_name.replace(" ", "") == concat_s1):
                            score = 0.90 if (postal_match or num_match or dist_overlap >= 1 or not has_s1_addr) else 0.82
                        elif (s1_is_non_ascii or target_is_non_ascii) and num_match:
                            if dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.80: score = 0.88
                            elif dist_overlap >= 1 and addr_token_sim is not None and addr_token_sim >= 0.90: score = 0.86
                        elif max_name >= 0.88:
                            if postal_match or num_match or dist_overlap >= 1 or not has_s1_addr or not has_t_addr: score = 0.87
                        elif max_name >= 0.72:
                            if postal_match and (num_match or dist_overlap >= 1): score = 0.84
                            elif num_match and dist_overlap >= 1: score = 0.82
                            elif dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.82: score = 0.80
                        elif max_name >= 0.60 and part_r >= 0.82:
                            if postal_match and num_match: score = 0.83
                            elif postal_match and dist_overlap >= 1: score = 0.82

                        min_len = min(len(core_s1), len(core_t))
                        max_len = max(len(core_s1), len(core_t))
                        if score < 0.80 and 3 <= min_len <= 7 and max_len <= 8:
                            if distance.Levenshtein.distance(core_s1, core_t) <= 1:
                                if num_match or postal_match or dist_overlap >= 1: score = 0.83

                        if score >= 0.81:
                            a_sim_val = addr_token_sim if addr_token_sim is not None else 0.50
                            scored_matches.append((t_idx, score, a_sim_val, postal_match, num_match, len(core_s1)))

                    total_processed += 1
                    for t_idx, sc, a_sim, pm, nm, _ in scored_matches:
                        prev = target_claims.get(t_idx)
                        rank_tuple = (sc, 1 if nm else 0, 1 if pm else 0, a_sim)
                        if prev is None or rank_tuple > prev[1]:
                            target_claims[t_idx] = (s1_id, rank_tuple)

                    chunk_results.append((s1_id, list(cands), scored_matches))

                pickle.dump(chunk_results, f_scratch)
                print(f"  Final Chunk {chunk_count}: {total_processed:,} / {total_s1_count:,} (100.0%) in {time.time()-t0:.1f}s", flush=True)

    print(f"\nStep 3 Matching finished in {time.time() - t0:.1f}s across {total_processed:,} entities.", flush=True)

    # Step 4: 1-to-1 Mutual-Best Conflict Resolution
    print("\n[Step 4/5] Resolving 1-to-1 Mutual-Best Target Owners...", flush=True)
    t0 = time.time()
    winner_for_target = {t_idx: val[0] for t_idx, val in target_claims.items()}
    del target_claims
    gc.collect()
    print(f"Resolved {len(winner_for_target):,} exclusive target winners in {time.time()-t0:.1f}s.", flush=True)

    # Step 5: Stream Final Submission Files with Multi-Signal Singleton Gate
    print("\n[Step 5/5] Streaming final output TSVs with Multi-Signal Singleton Protection Gate...", flush=True)
    t0 = time.time()

    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")

    total_matches = 0
    singleton_preds = 0
    final_count = 0
    pruned_singletons = 0

    with open(matching_file, "w", encoding="utf-8") as fm, open(candidate_file, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        with open(scratch_file, "rb") as f_scratch:
            while True:
                try:
                    sub_res = pickle.load(f_scratch)
                except EOFError:
                    break

                for s1_id, cands, scored_matches in sub_res:
                    final_count += 1
                    retained_items = [
                        (target_ids[t_idx], sc, a_sim, pm, nm, clen)
                        for t_idx, sc, a_sim, pm, nm, clen in scored_matches
                        if winner_for_target.get(t_idx) == s1_id
                    ]

                    # Multi-Signal Singleton Protection Gate
                    if len(retained_items) == 1:
                        cand_id, sc, a_sim, pm, nm, clen = retained_items[0]
                        should_prune = False

                        if sc < 0.90:
                            addr_confirmations = (1 if pm else 0) + (1 if nm else 0) + (1 if a_sim >= 0.65 else 0)
                            if sc < 0.84 and addr_confirmations < 2:
                                should_prune = True
                            elif clen < 5 and (sc < 0.87 and addr_confirmations < 2):
                                should_prune = True

                        if should_prune:
                            retained_items = []
                            pruned_singletons += 1

                    retained = [item[0] for item in retained_items]
                    cand_ids = [target_ids[t_idx] for t_idx in cands]

                    cand_set = set(cand_ids)
                    for m in retained:
                        if m not in cand_set:
                            cand_ids.append(m)

                    fm.write(f"{s1_id}\t{','.join(retained)}\n")
                    fc.write(f"{s1_id}\t{','.join(cand_ids)}\n")

                    total_matches += len(retained)
                    if len(retained) == 0:
                        singleton_preds += 1

    if os.path.exists(scratch_file):
        os.remove(scratch_file)

    match_mb = os.path.getsize(matching_file) / (1024 * 1024)
    cand_mb = os.path.getsize(candidate_file) / (1024 * 1024)

    print(f"Wrote {final_count:,} records in {time.time()-t0:.1f}s.", flush=True)
    print(f"  matching_results.tsv: {match_mb:.1f} MB | {total_matches:,} total links | {singleton_preds:,} singletons ({singleton_preds/final_count*100:.2f}%)", flush=True)
    print(f"  Multi-signal gate pruned: {pruned_singletons:,} low-confidence singletons", flush=True)
    print(f"  candidate_pairs.tsv:  {cand_mb:.1f} MB", flush=True)
    print("\n>>> Pipeline V4 Lean Stream Finished Successfully! <<<", flush=True)


if __name__ == "__main__":
    main()
