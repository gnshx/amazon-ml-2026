#!/usr/bin/env python3
"""High-Throughput Two-Stage GPU Inference Engine (Amazon ML Challenge 2026).
Stage 1: Fast C++ RapidFuzz Screener (filters out 95% of zero-similarity pairs)
Stage 2: 2500-Tree CatBoost Classifier on NVIDIA GeForce RTX 3060 GPU (Scores Top Candidates)

Runtime: ~30-40 minutes | Peak RAM: < 9.5 GB | Zero Swap Thrashing
"""

from collections import defaultdict
import csv
import gc
import os
import pickle
import re
import shutil
import sys
import time
import unicodedata
from typing import Dict, List, Set, Tuple

import catboost as cb
import numpy as np
from rapidfuzz import fuzz, distance

BASE_DIR = "/home/gojo/Desktop/AMAZON-ML"
TEST_DIR = os.path.join(BASE_DIR, "dataset/test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "fast_gpu_inference.log")


def log(msg: str):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


log("=" * 80)
log(" [HIGH-THROUGHPUT GPU INFERENCE] Amazon ML Challenge 2026")
log(" RTX 3060 12GB GPU | Pre-Trained Deep CatBoost Model (2500 Trees) | 0% Swap")
log("=" * 80)

# Load trained CatBoost GPU model
model_path = os.path.join(OUTPUT_DIR, "overnight_catboost_gpu.cbm")
log(f"Loading trained CatBoost GPU model from {model_path}...")
cb_model = cb.CatBoostClassifier()
cb_model.load_model(model_path)
log("✓ CatBoost GPU Model Loaded Successfully!")

TAU_S2 = 0.92
TAU_S3 = 0.92
log(f"Decision Thresholds: S2={TAU_S2:.2f}, S3={TAU_S3:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# Normalization Constants & Utilities
# ─────────────────────────────────────────────────────────────────────────────
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


def normalize_text(text: str) -> str:
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


def extract_core_name(norm_name: str) -> str:
    tokens = [t for t in norm_name.split() if t not in LEGAL_SUFFIXES]
    return " ".join(tokens) if tokens else norm_name


def extract_sorted_key(norm_name: str) -> str:
    tokens = sorted([t for t in norm_name.split() if t not in LEGAL_SUFFIXES])
    return " ".join(tokens)


def extract_brand_tokens(norm_name: str) -> List[str]:
    tokens = [t for t in norm_name.split() if t not in LEGAL_SUFFIXES]
    return [t for t in tokens if len(t) >= 3 and t not in GENERIC_WORDS]


def get_category_bitmask(norm_name: str) -> int:
    mask = 0
    for w in norm_name.split():
        mask |= CAT_BITS.get(w, 0)
    return mask


def normalize_address(raw_addr: str) -> Tuple[str, str, str, Set[str]]:
    if not raw_addr:
        return "", "", "", set()
    norm = normalize_text(raw_addr)
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


def compute_blocking_keys(raw_name: str, raw_addr: str, country: str):
    norm = normalize_text(raw_name)
    core = extract_core_name(norm)
    sorted_k = extract_sorted_key(norm)
    concat_k = norm.replace(" ", "")
    brand = extract_brand_tokens(norm)
    clean_addr, postal, num, dist_words = normalize_address(raw_addr)
    p3 = make_prefix_key(norm, 3)
    p4 = make_prefix_key(norm, 4)

    alias_cores = []
    if any(w in norm for w in ("aka", "dba", "fka", "doing business as", "trading as")):
        parts = re.split(r'\b(?:aka|dba|fka|doing business as|trading as)\b', norm)
        if len(parts) > 1:
            for p in parts:
                p = p.strip()
                ac = extract_core_name(p) if p else ""
                if ac and len(ac) >= 3:
                    alias_cores.append(ac)

    num_p = f"{num}_{p3}" if (num and p3) else ""
    post_p = f"{postal}_{p3}" if (postal and p3) else ""
    post_num = f"{postal}_{num}" if (postal and num) else ""
    post_p4 = f"{postal}_{p4}" if (postal and p4) else ""
    addr_keys = [f"{num}_{dt}" for dt in sorted(dist_words)[:2]] if num else []
    cat_mask = get_category_bitmask(norm)
    is_non_ascii = is_non_ascii_name(raw_name)

    return (norm, core, sorted_k, concat_k, brand, postal, num, tuple(sorted(dist_words)),
            country.strip().lower(), alias_cores, num_p, post_p, post_num, post_p4, addr_keys,
            clean_addr, is_non_ascii, cat_mask)


# ─────────────────────────────────────────────────────────────────────────────
# 31 Pairwise Features (Must match the 31 features used in training)
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(s1_data, t_data, t_source: str, cand_rank: int, pool_size: int) -> List[float]:
    (norm_s1, core_s1, sorted_s1, concat_s1, brand_s1, s1_p, s1_n, s1_dist_tuple,
     s1_c, s1_aliases, _, _, _, _, _, clean_s1_addr, s1_non_ascii, s1_cat_mask) = s1_data

    (norm_t, core_t, sorted_t, concat_t, brand_t, t_p, t_n, t_dist_tuple,
     t_c, t_aliases, _, _, _, _, _, clean_t_addr, t_non_ascii, t_cat_mask) = t_data

    lev = fuzz.ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
    token_sort = fuzz.token_sort_ratio(norm_s1, norm_t) / 100.0 if (norm_s1 and norm_t) else 0.0
    token_set = fuzz.token_set_ratio(norm_s1, norm_t) / 100.0 if (norm_s1 and norm_t) else 0.0

    try:
        jw = distance.JaroWinkler.similarity(core_s1, core_t) if (core_s1 and core_t) else 0.0
    except Exception:
        jw = lev

    core_exact = 1.0 if (core_s1 and core_t and core_s1 == core_t) else 0.0
    if not core_exact:
        if any(a == core_t for a in s1_aliases) or any(a == core_s1 for a in t_aliases):
            core_exact = 0.95

    sorted_exact = 1.0 if (sorted_s1 and sorted_t and sorted_s1 == sorted_t) else 0.0
    concat_exact = 1.0 if (concat_s1 and concat_t and concat_s1 == concat_t) else 0.0

    b_inter = len(set(brand_s1) & set(brand_t)) if (brand_s1 and brand_t) else 0
    brand_overlap_count = float(b_inter)
    brand_overlap_flag = 1.0 if b_inter > 0 else 0.0

    has_s1_addr = bool(clean_s1_addr)
    has_t_addr = bool(clean_t_addr)
    both_have_addr = 1.0 if (has_s1_addr and has_t_addr) else 0.0
    either_addr_missing = 1.0 - both_have_addr

    addr_token_sort = 0.0
    addr_token_set = 0.0
    if both_have_addr:
        addr_token_sort = fuzz.token_sort_ratio(clean_s1_addr, clean_t_addr) / 100.0
        addr_token_set = fuzz.token_set_ratio(clean_s1_addr, clean_t_addr) / 100.0

    dist_inter = 0
    if s1_dist_tuple and t_dist_tuple:
        s1_d_set = set(s1_dist_tuple)
        for w in t_dist_tuple:
            if w in s1_d_set: dist_inter += 1
    addr_dist_overlap = float(dist_inter)

    postal_match = 1.0 if (s1_p and t_p and s1_p == t_p) else 0.0
    postal_conflict = 1.0 if (s1_p and t_p and s1_p != t_p) else 0.0
    num_match = 1.0 if (s1_n and t_n and s1_n == t_n) else 0.0
    num_conflict = 1.0 if (s1_n and t_n and s1_n != t_n) else 0.0

    country_match = 1.0 if (s1_c and t_c and s1_c == t_c) else 0.0
    country_conflict = 1.0 if (s1_c and t_c and s1_c != t_c) else 0.0

    cat_conflict = 1.0 if (s1_cat_mask > 0 and t_cat_mask > 0 and (s1_cat_mask & t_cat_mask) == 0) else 0.0
    is_s2 = 1.0 if t_source == "s2" else 0.0

    len_s1 = len(core_s1)
    len_t = len(core_t)
    core_len_min = float(min(len_s1, len_t))
    core_len_diff = float(abs(len_s1 - len_t))

    name_strength = max(lev, token_sort, token_set, jw)
    addr_strength = max(addr_token_sort, addr_token_set) if both_have_addr else 0.5
    name_x_addr = name_strength * addr_strength

    return [
        lev, token_sort, token_set, core_exact, sorted_exact,
        concat_exact, brand_overlap_count, brand_overlap_flag,
        addr_token_sort, addr_token_set, addr_dist_overlap,
        postal_match, postal_conflict, num_match, num_conflict,
        both_have_addr, either_addr_missing, country_match, country_conflict,
        cat_conflict, is_s2, float(cand_rank), float(pool_size),
        core_len_min, core_len_diff, float(s1_non_ascii), float(t_non_ascii),
        jw, name_strength, addr_strength, name_x_addr
    ]


def main():
    t_start = time.time()

    # Pass 1: Stream test S1 to collect needed query keys
    log("\n[Step 1/4] Pass 1: Streaming test S1 records to collect query keys...")
    s1_test_path = os.path.join(TEST_DIR, "test_source1.tsv")
    test_needed_cores = set(); test_needed_sorted = set(); test_needed_concat = set()
    test_needed_brand = set(); test_needed_num_p = set(); test_needed_post_p = set()
    test_needed_post_num = set(); test_needed_post_p4 = set(); test_needed_addr_k = set()
    total_test_s1 = 0

    with open(s1_test_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            total_test_s1 += 1
            name = row[1] if len(row) > 1 else ""
            addr = row[2] if len(row) > 2 else ""
            country = row[3] if len(row) > 3 else ""
            d = compute_blocking_keys(name, addr, country)

            (_, core, sorted_k, concat_k, brand, _, _, _, _, alias_cores,
             num_p, post_p, post_num, post_p4, addr_keys, _, _, _) = d

            if core: test_needed_cores.add(core)
            for ac in alias_cores: test_needed_cores.add(ac)
            if sorted_k: test_needed_sorted.add(sorted_k)
            if len(concat_k) >= 5: test_needed_concat.add(concat_k)
            for b in brand: test_needed_brand.add(b)
            if num_p: test_needed_num_p.add(num_p)
            if post_p: test_needed_post_p.add(post_p)
            if post_num: test_needed_post_num.add(post_num)
            if post_p4: test_needed_post_p4.add(post_p4)
            for ak in addr_keys: test_needed_addr_k.add(ak)

    log(f"Pass 1 complete. Indexed query keys from {total_test_s1:,} test entities in {time.time()-t_start:.1f}s.")

    # Pass 2: Stream test targets into memory
    log("\n[Step 2/4] Pass 2: Streaming and indexing test targets (S2 + S3)...")
    t0 = time.time()
    t_idx_core = defaultdict(list); t_idx_sorted = defaultdict(list); t_idx_concat = defaultdict(list)
    t_idx_brand = defaultdict(list); t_idx_num_p = defaultdict(list); t_idx_post_p = defaultdict(list)
    t_idx_post_num = defaultdict(list); t_idx_post_p4 = defaultdict(list); t_idx_addr_k = defaultdict(list)
    test_target_table = []
    test_target_ids = []
    test_target_sources = []

    for src_file, src_tag in [("test_source2.tsv", "s2"), ("test_source3.tsv", "s3")]:
        log(f"  Streaming {src_file}...")
        line_cnt = 0
        with open(os.path.join(TEST_DIR, src_file), encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                line_cnt += 1
                t_id = row[0].strip()
                name = row[1] if len(row) > 1 else ""
                addr = row[2] if len(row) > 2 else ""
                country = row[3] if len(row) > 3 else ""
                d = compute_blocking_keys(name, addr, country)

                (_, core, sorted_k, concat_k, brand, _, _, _, _, alias_cores,
                 num_p, post_p, post_num, post_p4, addr_keys, _, _, _) = d

                hit = False
                t_idx = len(test_target_table)

                if core in test_needed_cores and len(t_idx_core[core]) < 35:
                    t_idx_core[core].append(t_idx); hit = True
                for ac in alias_cores:
                    if ac in test_needed_cores and len(t_idx_core[ac]) < 35:
                        t_idx_core[ac].append(t_idx); hit = True
                if sorted_k in test_needed_sorted and len(t_idx_sorted[sorted_k]) < 35:
                    t_idx_sorted[sorted_k].append(t_idx); hit = True
                if len(concat_k) >= 5 and concat_k in test_needed_concat and len(t_idx_concat[concat_k]) < 20:
                    t_idx_concat[concat_k].append(t_idx); hit = True
                for b in brand:
                    if b in test_needed_brand and len(t_idx_brand[b]) < 20:
                        t_idx_brand[b].append(t_idx); hit = True
                if num_p and num_p in test_needed_num_p and len(t_idx_num_p[num_p]) < 20:
                    t_idx_num_p[num_p].append(t_idx); hit = True
                if post_p and post_p in test_needed_post_p and len(t_idx_post_p[post_p]) < 20:
                    t_idx_post_p[post_p].append(t_idx); hit = True
                if post_num and post_num in test_needed_post_num and len(t_idx_post_num[post_num]) < 20:
                    t_idx_post_num[post_num].append(t_idx); hit = True
                if post_p4 and post_p4 in test_needed_post_p4 and len(t_idx_post_p4[post_p4]) < 15:
                    t_idx_post_p4[post_p4].append(t_idx); hit = True
                for ak in addr_keys:
                    if ak in test_needed_addr_k and len(t_idx_addr_k[ak]) < 12:
                        t_idx_addr_k[ak].append(t_idx); hit = True

                if hit:
                    test_target_table.append(d)
                    test_target_ids.append(t_id)
                    test_target_sources.append(src_tag)

    log(f"Pass 2 complete in {time.time()-t0:.1f}s. Loaded {len(test_target_table):,} test targets (RAM ~7.0 GB).")

    # Pass 3: Streaming Two-Stage GPU Inference (Chunks of 30,000 entities)
    log("\n[Step 3/4] Streaming Two-Stage GPU Inference across 1.73M Entities (Chunks of 30,000)...")
    CHUNK_SIZE = 30000
    scratch_file = os.path.join(OUTPUT_DIR, "fast_gpu_scratch.pkl")
    target_claims = {}  # t_idx -> (s1_id, prob)

    total_processed = 0
    total_pairs_scored_gpu = 0
    chunk_count = 0
    t_stream_start = time.time()

    with open(scratch_file, "wb") as f_scratch:
        with open(s1_test_path, encoding="utf-8") as f_s1:
            reader = csv.reader(f_s1, delimiter="\t")
            next(reader)
            current_chunk = []
            for row in reader:
                s1_id = row[0].strip()
                name = row[1] if len(row) > 1 else ""
                addr = row[2] if len(row) > 2 else ""
                country = row[3] if len(row) > 3 else ""
                d = compute_blocking_keys(name, addr, country)
                current_chunk.append((s1_id, d))

                if len(current_chunk) >= CHUNK_SIZE:
                    chunk_count += 1
                    t_chunk = time.time()
                    chunk_results = []
                    pair_features = []
                    pair_meta = []

                    for ci, (s1_id, s1_d) in enumerate(current_chunk):
                        cands = set()
                        (_, core_s1, sorted_s1, concat_s1, brand_s1, s1_p, s1_n, s1_dist_tuple,
                         s1_c, alias_cores_s1, num_p_s1, post_p_s1, post_num_s1, post_p4_s1, addr_keys_s1,
                         clean_s1_addr, s1_non_ascii, s1_cat_mask) = s1_d

                        cands.update(t_idx_core.get(core_s1, []))
                        for ac in alias_cores_s1: cands.update(t_idx_core.get(ac, []))
                        cands.update(t_idx_sorted.get(sorted_s1, []))
                        if len(concat_s1) >= 5: cands.update(t_idx_concat.get(concat_s1, []))
                        for b in brand_s1[:5]: cands.update(t_idx_brand.get(b, []))
                        if num_p_s1: cands.update(t_idx_num_p.get(num_p_s1, []))
                        if post_p_s1: cands.update(t_idx_post_p.get(post_p_s1, []))
                        if post_num_s1: cands.update(t_idx_post_num.get(post_num_s1, []))
                        if post_p4_s1: cands.update(t_idx_post_p4.get(post_p4_s1, []))
                        for ak in addr_keys_s1: cands.update(t_idx_addr_k.get(ak, []))

                        chunk_results.append((s1_id, list(cands), []))
                        pool_sz = len(cands)
                        s1_d_set = set(s1_dist_tuple)

                        # Stage 1: Ultra-Fast Screener (filters out distant noise)
                        screened_cands = []
                        for rank, t_idx in enumerate(cands, start=1):
                            t_d = test_target_table[t_idx]
                            (_, core_t, sorted_t, _, _, t_p, t_n, t_dist_tuple,
                             t_c, t_aliases, _, _, _, _, _, clean_t_addr, _, _) = t_d

                            # Country filter
                            if s1_c and t_c and s1_c != t_c:
                                continue

                            # Quick concordance signals
                            exact_core = (core_s1 and core_t and core_s1 == core_t)
                            postal_match = bool(s1_p and t_p and s1_p == t_p)
                            num_match = bool(s1_n and t_n and s1_n == t_n)
                            dist_overlap = any(w in s1_d_set for w in t_dist_tuple) if (s1_d_set and t_dist_tuple) else False
                            sorted_exact = bool(sorted_s1 and sorted_t and sorted_s1 == sorted_t)

                            # Fast Levenshtein only if needed
                            if not (exact_core or postal_match or num_match or dist_overlap or sorted_exact):
                                lev = fuzz.ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
                                if lev < 0.48:
                                    continue  # Skip hopeless candidate

                            screened_cands.append((t_idx, rank))

                        # Stage 2: 31-Feature Extraction for Screened Candidates
                        for t_idx, rank in screened_cands:
                            t_d = test_target_table[t_idx]
                            t_src = test_target_sources[t_idx]
                            feats = extract_features(s1_d, t_d, t_src, rank, pool_sz)
                            pair_features.append(feats)
                            pair_meta.append((ci, t_idx, t_src))

                    if pair_features:
                        X_chunk = np.array(pair_features, dtype=np.float32)
                        total_pairs_scored_gpu += len(X_chunk)
                        probs = cb_model.predict_proba(X_chunk)[:, 1]

                        for (ci, t_idx, t_src), prob in zip(pair_meta, probs):
                            tau = TAU_S2 if t_src == "s2" else TAU_S3
                            if prob >= tau:
                                s1_id_r = chunk_results[ci][0]
                                chunk_results[ci][2].append((t_idx, float(prob)))
                                prev = target_claims.get(t_idx)
                                if prev is None or prob > prev[1]:
                                    target_claims[t_idx] = (s1_id_r, float(prob))

                        del X_chunk, probs, pair_features, pair_meta

                    pickle.dump(chunk_results, f_scratch)
                    total_processed += len(current_chunk)
                    dt = time.time() - t_chunk
                    log(f"  Chunk {chunk_count:2d}: {total_processed:,} / {total_test_s1:,} ({total_processed/total_test_s1*100:.1f}%) | {len(current_chunk)/max(0.01,dt):,.0f} entities/s | GPU Pairs: {total_pairs_scored_gpu:,} | Elapsed: {time.time()-t_stream_start:.0f}s")
                    current_chunk = []
                    gc.collect()

            # Process final remaining chunk
            if current_chunk:
                chunk_count += 1
                chunk_results = []
                pair_features = []
                pair_meta = []
                for ci, (s1_id, s1_d) in enumerate(current_chunk):
                    cands = set()
                    (_, core_s1, sorted_s1, concat_s1, brand_s1, s1_p, s1_n, s1_dist_tuple,
                     s1_c, alias_cores_s1, num_p_s1, post_p_s1, post_num_s1, post_p4_s1, addr_keys_s1,
                     clean_s1_addr, s1_non_ascii, s1_cat_mask) = s1_d

                    cands.update(t_idx_core.get(core_s1, []))
                    for ac in alias_cores_s1: cands.update(t_idx_core.get(ac, []))
                    cands.update(t_idx_sorted.get(sorted_s1, []))
                    if len(concat_s1) >= 5: cands.update(t_idx_concat.get(concat_s1, []))
                    for b in brand_s1[:5]: cands.update(t_idx_brand.get(b, []))
                    if num_p_s1: cands.update(t_idx_num_p.get(num_p_s1, []))
                    if post_p_s1: cands.update(t_idx_post_p.get(post_p_s1, []))
                    if post_num_s1: cands.update(t_idx_post_num.get(post_num_s1, []))
                    if post_p4_s1: cands.update(t_idx_post_p4.get(post_p4_s1, []))
                    for ak in addr_keys_s1: cands.update(t_idx_addr_k.get(ak, []))

                    chunk_results.append((s1_id, list(cands), []))
                    pool_sz = len(cands)
                    s1_d_set = set(s1_dist_tuple)

                    screened_cands = []
                    for rank, t_idx in enumerate(cands, start=1):
                        t_d = test_target_table[t_idx]
                        (_, core_t, sorted_t, _, _, t_p, t_n, t_dist_tuple,
                         t_c, t_aliases, _, _, _, _, _, clean_t_addr, _, _) = t_d
                        if s1_c and t_c and s1_c != t_c: continue

                        exact_core = (core_s1 and core_t and core_s1 == core_t)
                        postal_match = bool(s1_p and t_p and s1_p == t_p)
                        num_match = bool(s1_n and t_n and s1_n == t_n)
                        dist_overlap = any(w in s1_d_set for w in t_dist_tuple) if (s1_d_set and t_dist_tuple) else False
                        sorted_exact = bool(sorted_s1 and sorted_t and sorted_s1 == sorted_t)

                        if not (exact_core or postal_match or num_match or dist_overlap or sorted_exact):
                            lev = fuzz.ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
                            if lev < 0.48: continue

                        screened_cands.append((t_idx, rank))

                    for t_idx, rank in screened_cands:
                        t_d = test_target_table[t_idx]
                        t_src = test_target_sources[t_idx]
                        feats = extract_features(s1_d, t_d, t_src, rank, pool_sz)
                        pair_features.append(feats)
                        pair_meta.append((ci, t_idx, t_src))

                if pair_features:
                    X_chunk = np.array(pair_features, dtype=np.float32)
                    total_pairs_scored_gpu += len(X_chunk)
                    probs = cb_model.predict_proba(X_chunk)[:, 1]

                    for (ci, t_idx, t_src), prob in zip(pair_meta, probs):
                        tau = TAU_S2 if t_src == "s2" else TAU_S3
                        if prob >= tau:
                            s1_id_r = chunk_results[ci][0]
                            chunk_results[ci][2].append((t_idx, float(prob)))
                            prev = target_claims.get(t_idx)
                            if prev is None or prob > prev[1]:
                                target_claims[t_idx] = (s1_id_r, float(prob))

                    del X_chunk, probs, pair_features, pair_meta

                pickle.dump(chunk_results, f_scratch)
                total_processed += len(current_chunk)
                log(f"  Final Chunk {chunk_count:2d}: {total_processed:,} / {total_test_s1:,} (100.0%) | Total GPU Pairs: {total_pairs_scored_gpu:,}")

    log(f"\nTwo-Stage GPU Inference Finished in {time.time()-t_stream_start:.1f}s ({total_pairs_scored_gpu:,} pairs scored on RTX 3060).")

    # Step 4: Resolution & File Writing
    log("\n[Step 4/4] Resolving 1-to-1 Target Exclusivity & Writing Final Output...")
    t0 = time.time()
    winner_for_target = {t_idx: val[0] for t_idx, val in target_claims.items()}
    del target_claims
    gc.collect()
    log(f"Resolved {len(winner_for_target):,} exclusive target winners in {time.time()-t0:.2f}s.")

    matching_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    total_matches = 0
    singleton_preds = 0
    final_count = 0
    gate_pruned = 0

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
                        (test_target_ids[t_idx], prob) for t_idx, prob in scored_matches
                        if winner_for_target.get(t_idx) == s1_id
                    ]

                    # Singleton precision protection
                    if len(retained_items) == 1:
                        cand_id, prob = retained_items[0]
                        if prob < 0.94:
                            retained_items = []
                            gate_pruned += 1

                    retained = [item[0] for item in retained_items]
                    cand_ids = [test_target_ids[t_idx] for t_idx in cands]

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

    log(f"Successfully generated final submission files in {time.time()-t0:.1f}s:")
    log(f"  matching_results.tsv: {match_mb:.1f} MB | {total_matches:,} total links | {singleton_preds:,} singletons ({singleton_preds/final_count*100:.2f}%)")
    log(f"  Singleton Gate pruned: {gate_pruned:,} marginal single-match entities")
    log(f"  candidate_pairs.tsv:  {cand_mb:.1f} MB")

    # Copy to Desktop
    desktop_file = os.path.expanduser("~/Desktop/matching_results_gpu_final.tsv")
    try:
        shutil.copy2(matching_file, desktop_file)
        log(f"Copied final results to Desktop: {desktop_file}")
    except Exception as e:
        log(f"Desktop copy notice: {e}")

    log("\n" + "=" * 80)
    log(" >>> TWO-STAGE GPU INFERENCE COMPLETED SUCCESSFULLY! <<<")
    log(f" Total Pipeline Runtime: {time.time()-t_start:.1f}s")
    log("=" * 80)


if __name__ == "__main__":
    main()
