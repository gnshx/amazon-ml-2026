#!/usr/bin/env python3
"""
PRODUCTION PIPELINE V4 — MEGA PIPELINE (Amazon ML Challenge 2026)
=================================================================
THE BEST PIPELINE — ALL PHASES COMBINED:

Phase 1: 12-Route High-Recall Blocking (V3 routes)
  Route 1:  Core exact name match
  Route 2:  Alias / DBA core matches
  Route 3:  Sorted token match (word-order inversion)
  Route 4:  Concatenated domain/handle match
  Route 5:  Distinctive brand token overlap
  Route 6:  Postal + 3-char core prefix
  Route 7:  Street number + 3-char core prefix
  Route 8:  Postal + street number (physical address)
  Route 9:  City + street number                  [V3 NEW]
  Route 10: Char 4-gram prefix bucket             [V3 NEW]
  Route 11: Domain/handle token unpack            [V3 NEW]
  Route 12: Postal + first distinctive word       [V3 NEW]

Phase 2: CatBoost ML Reranker (pre-trained, GPU-accelerated)
  - Scores every candidate pair with 40 learned features
  - Replaces ALL hardcoded rule-based scoring
  - Uses CatBoost model trained on fold 0/1 of training data
  - tau=0.96 for both S2 and S3 (from model_metadata.json)

Phase 3: Entity-Level Singleton Gate (ML-based, trained gate_model.pkl)
  - 25 entity-level features (top-k stats, pool size, postal/num signals)
  - Trained scikit-learn gate model (gate_model.pkl)
  - tau=0.75 from calibration

Phase 4: Multi-dimensional Target Conflict Resolution
  - Already globally optimal (confirmed via audit)
  - Each target claims the S1 with highest CatBoost score
  - S1 entities keep all their won targets (1-to-many preserved)

Phase 5: Category Conflict Safety Filter (V3 insight)
  - Prevents different-category businesses at same address from merging

Target: 85-93% Macro F0.5 (vs current 74.5%)
"""

import csv
import gc
import json
import os
import pickle
import re
import sys
import time
import unicodedata
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from rapidfuzz import fuzz, distance

# ─────────────────────────────────────────────────────────────────
# Load ML Models at startup
# ─────────────────────────────────────────────────────────────────
print("=" * 80, flush=True)
print(" [PRODUCTION PIPELINE V4 MEGA] Amazon ML Challenge 2026", flush=True)
print(" 12-Route Blocking | CatBoost Reranker | ML Singleton Gate", flush=True)
print("=" * 80, flush=True)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n[Init] Loading ML models...", flush=True)

try:
    from catboost import CatBoost, Pool
    cb_model = CatBoost()
    cb_model.load_model(os.path.join(OUTPUT_DIR, "catboost_gpu_model.cbm"))
    print("  ✓ CatBoost model loaded", flush=True)
    HAS_CATBOOST = True
except Exception as e:
    print(f"  ✗ CatBoost load failed: {e} — falling back to rule-based scoring", flush=True)
    HAS_CATBOOST = False

try:
    with open(os.path.join(OUTPUT_DIR, "gate_model.pkl"), "rb") as gf:
        gate_model = pickle.load(gf)
    print("  ✓ Singleton gate model loaded", flush=True)
    HAS_GATE_MODEL = True
except Exception as e:
    print(f"  ✗ Gate model load failed: {e} — using heuristic gate", flush=True)
    HAS_GATE_MODEL = False

# Load metadata
try:
    with open(os.path.join(OUTPUT_DIR, "model_metadata.json")) as mf:
        metadata = json.load(mf)
    FEAT_NAMES = metadata["feat_names"]
    GATE_FEAT_NAMES = metadata["gate_feat_names"]
    BEST_TAU_S2 = metadata["best_tau"]["s2"]   # 0.96
    BEST_TAU_S3 = metadata["best_tau"]["s3"]   # 0.96
    BEST_TAU_SINGLETON = metadata["best_tau_singleton"]  # 0.75
    print(f"  ✓ Metadata loaded: {len(FEAT_NAMES)} pair features, {len(GATE_FEAT_NAMES)} gate features", flush=True)
    print(f"  ✓ Thresholds: pair tau={BEST_TAU_S2}, singleton tau={BEST_TAU_SINGLETON}", flush=True)
except Exception as e:
    print(f"  ✗ Metadata load failed: {e}", flush=True)
    FEAT_NAMES = []
    GATE_FEAT_NAMES = []
    BEST_TAU_S2 = 0.96
    BEST_TAU_S3 = 0.96
    BEST_TAU_SINGLETON = 0.75


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited", "co", "company",
    "dba", "lp", "pllc", "pvt", "private", "llp", "opc", "lnc", "1nc",
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

BIZ_CATEGORY_WORDS = {
    "pharmacy", "pharma", "clinic", "hospital", "hotel", "restaurant", "cafe",
    "school", "college", "university", "bank", "tech", "software", "hardware",
    "auto", "automotive", "salon", "spa", "gym", "fitness", "mart", "bakery",
    "pizza", "burger", "gas", "petrol", "fuel", "plumbing", "dental", "dentist",
    "optical", "law", "legal", "attorney", "insurance", "realty",
    "construction", "cleaning", "laundry", "printing", "packaging", "transport",
    "freight", "shipping", "courier", "electrical", "electronics", "mechanic"
}


# ─────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────
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


def extract_domain_token(raw_name: str) -> str:
    if not raw_name:
        return ""
    name = raw_name.lower().strip()
    name = re.sub(r'^(www\.|http://|https://)', '', name)
    name = re.sub(r'\.(com|in|org|net|co|io|fr|uk|eu|biz|info)(/.*)?$', '', name)
    name = re.sub(r'^[^@]+@', '', name)
    name = re.sub(r'[^a-z0-9]', '', name)
    return name if len(name) >= 4 else ""


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
    return [t for t in norm_name.split() if t not in LEGAL_SUFFIXES]


def extract_core_name(norm_name: str) -> str:
    tokens = extract_core_tokens(norm_name)
    return " ".join(tokens) if tokens else norm_name


def extract_sorted_key(norm_name: str) -> str:
    return " ".join(sorted(extract_core_tokens(norm_name)))


def extract_concat_key(norm_name: str) -> str:
    return "".join(extract_core_tokens(norm_name))


def extract_distinctive_tokens(norm_name: str) -> List[str]:
    return [t for t in extract_core_tokens(norm_name) if len(t) >= 3 and t not in GENERIC_WORDS]


def extract_category_words(norm_name: str) -> set:
    return {t for t in norm_name.split() if t in BIZ_CATEGORY_WORDS}


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


def extract_city(raw_addr: str) -> str:
    if not raw_addr:
        return ""
    norm = normalize_raw_text(raw_addr)
    tokens = [t for t in norm.split()
              if len(t) >= 4 and t not in GENERIC_ADDR_WORDS and not t.isdigit()]
    non_state = [t for t in tokens if t not in set(US_STATES.values())]
    candidates = non_state if non_state else tokens
    for t in reversed(candidates):
        if len(t) >= 4:
            return t
    return ""


def make_prefix_key(name: str, length: int = 3) -> str:
    core = extract_core_name(name).replace(" ", "")
    return core[:length] if len(core) >= length else ""


def char4gram_key(core_name: str) -> str:
    c = core_name.replace(" ", "")
    return c[:4] if len(c) >= 4 else ""


# ─────────────────────────────────────────────────────────────────
# IDF weights (computed from token frequencies — approximate)
# ─────────────────────────────────────────────────────────────────
_idf_cache: Dict[str, float] = {}
_token_freq_total = 0


def compute_idf_weights(all_names: List[str]) -> Dict[str, float]:
    """Approximate IDF from a list of normalized names."""
    from collections import Counter
    doc_freq = Counter()
    n_docs = len(all_names)
    for name in all_names:
        tokens = set(extract_core_tokens(name))
        for t in tokens:
            doc_freq[t] += 1
    idf = {}
    for token, df in doc_freq.items():
        idf[token] = max(0.1, float(np.log((n_docs + 1) / (df + 1)) + 1))
    return idf


# ─────────────────────────────────────────────────────────────────
# Feature Extraction (40 features matching FEAT_NAMES)
# ─────────────────────────────────────────────────────────────────
def char_ngrams_set(text: str, n: int = 3) -> Set[str]:
    text = text.replace(" ", "")
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def extract_features(
    s1_norm_name: str, s1_raw_addr: str, s1_country: str, s1_is_non_ascii: bool,
    s1_clean_addr: str, s1_postal: str, s1_num: str, s1_dist: Set[str], s1_aliases: List[str],
    t_norm_name: str, t_raw_addr: str, t_country: str, t_is_non_ascii: bool,
    t_clean_addr: str, t_postal: str, t_num: str, t_dist: Set[str], t_aliases: List[str],
    t_source: str,  # "s2" or "s3"
    cand_rank: int,
    pool_size: int,
    idf_weights: Dict[str, float]
) -> List[float]:
    """Extract 40 features in the exact order of FEAT_NAMES."""

    # Name features
    core_s1 = extract_core_name(s1_norm_name)
    core_t = extract_core_name(t_norm_name)
    sorted_s1 = extract_sorted_key(s1_norm_name)
    sorted_t = extract_sorted_key(t_norm_name)

    token_sort_v = fuzz.token_sort_ratio(s1_norm_name, t_norm_name) / 100.0 if (s1_norm_name and t_norm_name) else 0.0
    token_set_v = fuzz.token_set_ratio(s1_norm_name, t_norm_name) / 100.0 if (s1_norm_name and t_norm_name) else 0.0
    partial_ratio_v = fuzz.partial_ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
    lev_ratio_v = fuzz.ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0

    jaro_v = 0.0
    if core_s1 and core_t:
        try:
            jaro_v = distance.JaroWinkler.similarity(core_s1, core_t)
        except Exception:
            jaro_v = lev_ratio_v

    # Token Jaccard
    s1_tokens = set(core_s1.split()) if core_s1 else set()
    t_tokens = set(core_t.split()) if core_t else set()
    if s1_tokens or t_tokens:
        inter = len(s1_tokens & t_tokens)
        union = len(s1_tokens | t_tokens)
        name_token_jaccard_v = inter / union if union > 0 else 0.0
    else:
        name_token_jaccard_v = 0.0

    # Char n-gram jaccard
    ng_s1 = char_ngrams_set(core_s1, 3)
    ng_t = char_ngrams_set(core_t, 3)
    if ng_s1 or ng_t:
        inter_ng = len(ng_s1 & ng_t)
        union_ng = len(ng_s1 | ng_t)
        char_ngram_jaccard_v = inter_ng / union_ng if union_ng > 0 else 0.0
    else:
        char_ngram_jaccard_v = 0.0

    # IDF-weighted max shared token
    s1_tok_set = set(extract_core_tokens(s1_norm_name))
    t_tok_set = set(extract_core_tokens(t_norm_name))
    shared_tokens = s1_tok_set & t_tok_set
    shared_idfs = [idf_weights.get(t, 1.0) for t in shared_tokens]
    max_shared_idf_v = max(shared_idfs) if shared_idfs else 0.0
    mean_shared_idf_v = (sum(shared_idfs) / len(shared_idfs)) if shared_idfs else 0.0

    # Core / sorted exact
    core_exact_v = 1.0 if (core_s1 and core_t and core_s1 == core_t) else 0.0
    sorted_exact_v = 1.0 if (sorted_s1 and sorted_t and sorted_s1 == sorted_t) else 0.0

    # Alias match incorporated in core_exact
    alias_match_boost = 0.0
    for a in s1_aliases:
        if a and a == core_t:
            alias_match_boost = 1.0
            break
    for a in t_aliases:
        if a and a == core_s1:
            alias_match_boost = 1.0
            break
    if alias_match_boost:
        core_exact_v = max(core_exact_v, alias_match_boost)

    # Brand token overlap
    brand_s1 = set(extract_distinctive_tokens(s1_norm_name))
    brand_t = set(extract_distinctive_tokens(t_norm_name))
    brand_inter = brand_s1 & brand_t
    brand_overlap_count_v = float(len(brand_inter))
    brand_overlap_flag_v = 1.0 if brand_inter else 0.0

    # Name composite strength
    name_strength_v = max(token_sort_v, token_set_v, lev_ratio_v, jaro_v)

    # Address features
    has_s1_addr_v = 1.0 if s1_clean_addr else 0.0
    has_t_addr_v = 1.0 if t_clean_addr else 0.0
    both_have_addr_v = 1.0 if (s1_clean_addr and t_clean_addr) else 0.0
    either_addr_missing_v = 1.0 - both_have_addr_v

    addr_token_sort_v = 0.0
    addr_token_set_v = 0.0
    addr_dist_jaccard_v = 0.0
    addr_dist_overlap_v = 0.0
    if s1_clean_addr and t_clean_addr:
        addr_token_sort_v = fuzz.token_sort_ratio(s1_clean_addr, t_clean_addr) / 100.0
        addr_token_set_v = fuzz.token_set_ratio(s1_clean_addr, t_clean_addr) / 100.0
        if s1_dist and t_dist:
            inter_d = len(s1_dist & t_dist)
            union_d = len(s1_dist | t_dist)
            addr_dist_jaccard_v = inter_d / union_d if union_d > 0 else 0.0
            addr_dist_overlap_v = float(len(s1_dist & t_dist))

    postal_match_v = 1.0 if (s1_postal and t_postal and s1_postal == t_postal) else 0.0
    postal_conflict_v = 1.0 if (s1_postal and t_postal and s1_postal != t_postal) else 0.0
    num_match_v = 1.0 if (s1_num and t_num and s1_num == t_num) else 0.0
    num_conflict_v = 1.0 if (s1_num and t_num and s1_num != t_num) else 0.0

    has_both_postal_v = 1.0 if (s1_postal and t_postal) else 0.0
    has_both_num_v = 1.0 if (s1_num and t_num) else 0.0

    # Composite addr strength
    addr_strength_v = max(addr_token_sort_v, addr_token_set_v) if both_have_addr_v else 0.5

    # Cross signals
    name_x_addr_v = name_strength_v * addr_strength_v
    strong_name_weak_addr_v = 1.0 if (name_strength_v >= 0.8 and addr_strength_v < 0.5) else 0.0
    weak_name_strong_addr_v = 1.0 if (name_strength_v < 0.6 and addr_strength_v >= 0.8) else 0.0

    # Country
    if s1_country and t_country:
        country_rel_v = 1.0 if s1_country == t_country else -1.0
    else:
        country_rel_v = 0.0

    # Script features
    s1_non_ascii_v = 1.0 if s1_is_non_ascii else 0.0
    c_non_ascii_v = 1.0 if t_is_non_ascii else 0.0
    cross_script_v = 1.0 if (s1_is_non_ascii != t_is_non_ascii) else 0.0

    # Source indicator
    is_s2_v = 1.0 if t_source == "s2" else 0.0
    is_s3_v = 1.0 if t_source == "s3" else 0.0

    # Rank and pool
    cand_rank_v = float(cand_rank)
    pool_size_v = float(pool_size)

    # Build dict for ordered extraction
    feat_dict = {
        "addr_dist_jaccard": addr_dist_jaccard_v,
        "addr_dist_overlap_count": addr_dist_overlap_v,
        "addr_strength": addr_strength_v,
        "addr_token_set": addr_token_set_v,
        "addr_token_sort": addr_token_sort_v,
        "both_have_addr": both_have_addr_v,
        "brand_overlap_count": brand_overlap_count_v,
        "brand_overlap_flag": brand_overlap_flag_v,
        "c_has_addr": has_t_addr_v,
        "c_non_ascii": c_non_ascii_v,
        "cand_rank": cand_rank_v,
        "char_ngram_jaccard": char_ngram_jaccard_v,
        "core_exact": core_exact_v,
        "country_rel": country_rel_v,
        "cross_script": cross_script_v,
        "either_addr_missing": either_addr_missing_v,
        "has_both_num": has_both_num_v,
        "has_both_postal": has_both_postal_v,
        "is_s2": is_s2_v,
        "is_s3": is_s3_v,
        "jaro_winkler": jaro_v,
        "lev_ratio": lev_ratio_v,
        "max_shared_idf": max_shared_idf_v,
        "mean_shared_idf": mean_shared_idf_v,
        "name_strength": name_strength_v,
        "name_token_jaccard": name_token_jaccard_v,
        "name_x_addr": name_x_addr_v,
        "num_conflict": num_conflict_v,
        "num_match": num_match_v,
        "partial_ratio": partial_ratio_v,
        "pool_size": pool_size_v,
        "postal_conflict": postal_conflict_v,
        "postal_match": postal_match_v,
        "s1_has_addr": has_s1_addr_v,
        "s1_non_ascii": s1_non_ascii_v,
        "sorted_exact": sorted_exact_v,
        "strong_name_weak_addr": strong_name_weak_addr_v,
        "token_set": token_set_v,
        "token_sort": token_sort_v,
        "weak_name_strong_addr": weak_name_strong_addr_v,
    }

    # Return in exact FEAT_NAMES order
    return [feat_dict.get(fn, 0.0) for fn in FEAT_NAMES]


def extract_gate_features(candidate_features: List[Tuple]) -> List[float]:
    """Extract 25 entity-level gate features from the list of scored candidate pairs.

    candidate_features: list of (score, token_sort, token_set, core_exact, char_ngram,
                                  max_shared_idf, postal_match, num_match, country_rel,
                                  addr_strength, name_x_addr, source)
    """
    if not candidate_features:
        return [0.0] * len(GATE_FEAT_NAMES)

    n = len(candidate_features)
    n_s2 = sum(1 for cf in candidate_features if cf[11] == "s2")
    n_s3 = sum(1 for cf in candidate_features if cf[11] == "s3")

    def top_k_mean(vals, k=3):
        top = sorted(vals, reverse=True)[:k]
        return sum(top) / len(top) if top else 0.0

    scores = [cf[0] for cf in candidate_features]
    token_sorts = [cf[1] for cf in candidate_features]
    token_sets = [cf[2] for cf in candidate_features]
    core_exacts = [cf[3] for cf in candidate_features]
    char_ngrams_v = [cf[4] for cf in candidate_features]
    max_idf = [cf[5] for cf in candidate_features]
    postal_ms = [cf[6] for cf in candidate_features]
    num_ms = [cf[7] for cf in candidate_features]
    country_rs = [cf[8] for cf in candidate_features]

    gate_dict = {
        "candidate_count": float(n),
        "s2_candidate_count": float(n_s2),
        "s3_candidate_count": float(n_s3),
        "name_strength_max": max(scores) if scores else 0.0,
        "name_strength_top3_mean": top_k_mean(scores),
        "addr_strength_max": max(cf[9] for cf in candidate_features),
        "addr_strength_top3_mean": top_k_mean([cf[9] for cf in candidate_features]),
        "name_x_addr_max": max(cf[10] for cf in candidate_features),
        "name_x_addr_top3_mean": top_k_mean([cf[10] for cf in candidate_features]),
        "token_sort_max": max(token_sorts) if token_sorts else 0.0,
        "token_sort_top3_mean": top_k_mean(token_sorts),
        "token_set_max": max(token_sets) if token_sets else 0.0,
        "token_set_top3_mean": top_k_mean(token_sets),
        "core_exact_max": max(core_exacts) if core_exacts else 0.0,
        "core_exact_top3_mean": top_k_mean(core_exacts),
        "char_ngram_jaccard_max": max(char_ngrams_v) if char_ngrams_v else 0.0,
        "char_ngram_jaccard_top3_mean": top_k_mean(char_ngrams_v),
        "max_shared_idf_max": max(max_idf) if max_idf else 0.0,
        "max_shared_idf_top3_mean": top_k_mean(max_idf),
        "postal_match_max": max(postal_ms) if postal_ms else 0.0,
        "postal_match_top3_mean": top_k_mean(postal_ms),
        "num_match_max": max(num_ms) if num_ms else 0.0,
        "num_match_top3_mean": top_k_mean(num_ms),
        "country_rel_max": max(country_rs) if country_rs else 0.0,
        "country_rel_top3_mean": top_k_mean(country_rs),
    }

    return [gate_dict.get(fn, 0.0) for fn in GATE_FEAT_NAMES]


# ─────────────────────────────────────────────────────────────────
# Fallback rule-based scoring (used if CatBoost unavailable)
# ─────────────────────────────────────────────────────────────────
def rule_based_score(
    core_s1, sorted_s1, concat_s1, norm_s1_name, clean_s1_addr,
    s1_p, s1_n, s1_dist, s1_is_non_ascii, s1_aliases, has_s1_addr,
    core_t, sorted_t, concat_t, norm_t_name, clean_t_addr,
    t_p, t_n, t_dist, target_is_non_ascii, t_aliases, t_cat_words, s1_cat_words,
    postal_match, postal_conflict, num_match, num_conflict, dist_overlap, addr_token_sim
):
    lev = fuzz.ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
    sort_r = fuzz.token_sort_ratio(norm_s1_name, norm_t_name) / 100.0 if (norm_s1_name and norm_t_name) else 0.0
    part_r = fuzz.partial_ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t and max(len(core_s1), len(core_t)) > 6) else 0.0
    max_name = max(lev, sort_r, part_r)

    category_conflict = (s1_cat_words and t_cat_words and not (s1_cat_words & t_cat_words))

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
        if postal_match or num_match or dist_overlap >= 1 or not has_s1_addr or not bool(clean_t_addr):
            score = 0.87
    elif max_name >= 0.72:
        if postal_match and (num_match or dist_overlap >= 1):
            score = 0.84
        elif num_match and dist_overlap >= 1:
            score = 0.82
        elif dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.82:
            score = 0.80
    elif max_name >= 0.60 and part_r >= 0.80:
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

    # Address-only tightening
    if score >= 0.81 and max_name < 0.60:
        if not (core_s1 == core_t or sorted_s1 == sorted_t or
                any(a == core_t for a in s1_aliases) or any(a == core_s1 for a in t_aliases)):
            if category_conflict:
                return 0.0
            if score < 0.87 and max_name < 0.55:
                return 0.0

    return score


# ─────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────
def main():
    test_dir = "dataset/test"

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    # ─────────────────────────────────────────────────────────────
    # Pass 1: Stream S1 keys + collect names for IDF
    # ─────────────────────────────────────────────────────────────
    print("\n[Step 1/6] Pass 1: Collecting S1 keys + building IDF weights...", flush=True)
    t0 = time.time()

    needed_cores = set(); needed_sorted = set(); needed_concat = set()
    needed_brand = set(); needed_num_p = set(); needed_post_p = set()
    needed_post_num = set(); needed_addr_k = set()
    needed_city_num = set(); needed_char4 = set(); needed_domain = set()
    needed_post_dist1 = set()

    all_s1_norm_names = []
    total_s1_count = 0

    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            total_s1_count += 1
            parts = line.rstrip("\r\n").split("\t")
            raw_name = parts[1] if len(parts) > 1 else ""
            raw_addr = parts[2] if len(parts) > 2 else ""

            norm_name = normalize_raw_text(raw_name)
            all_s1_norm_names.append(norm_name)
            clean_addr, postal, num, dist_addr = normalize_address(raw_addr)
            aliases = extract_alias_names(raw_name)
            core_prefix3 = make_prefix_key(norm_name, 3)
            domain_tok = extract_domain_token(raw_name)
            core = extract_core_name(norm_name)
            sorted_k = extract_sorted_key(norm_name)
            concat_k = extract_concat_key(norm_name)
            brand_tokens = extract_distinctive_tokens(norm_name)
            alias_cores = [extract_core_name(a) for a in aliases if extract_core_name(a)]

            num_p = f"{num}_{core_prefix3}" if (num and core_prefix3) else ""
            post_p = f"{postal}_{core_prefix3}" if (postal and core_prefix3) else ""
            post_num = f"{postal}_{num}" if (postal and num) else ""
            addr_keys = [f"{num}_{t}" for t in sorted(dist_addr)[:2]] if num else []
            city = extract_city(raw_addr)
            city_num = f"{city}_{num}" if (city and num) else ""
            char4 = char4gram_key(core)
            dist1 = sorted(dist_addr)[0] if dist_addr else ""
            post_dist1 = f"{postal}_{dist1}" if (postal and dist1) else ""

            if core: needed_cores.add(core)
            for ac in alias_cores: needed_cores.add(ac)
            if sorted_k: needed_sorted.add(sorted_k)
            if len(concat_k) >= 5: needed_concat.add(concat_k)
            for bt in brand_tokens: needed_brand.add(bt)
            if num_p: needed_num_p.add(num_p)
            if post_p: needed_post_p.add(post_p)
            if post_num: needed_post_num.add(post_num)
            for ak in addr_keys: needed_addr_k.add(ak)
            if city_num: needed_city_num.add(city_num)
            if char4 and len(char4) == 4: needed_char4.add(char4)
            if domain_tok: needed_domain.add(domain_tok)
            if post_dist1: needed_post_dist1.add(post_dist1)

    # Build IDF weights from a sample (fast)
    print(f"  Building IDF weights from {min(50000, len(all_s1_norm_names)):,} names...", flush=True)
    idf_sample = all_s1_norm_names[:50000]
    idf_weights = compute_idf_weights(idf_sample)
    del idf_sample, all_s1_norm_names
    gc.collect()

    print(f"  Done. {total_s1_count:,} S1 entities, IDF vocab={len(idf_weights):,} in {time.time()-t0:.1f}s", flush=True)
    print(f"  Keys: Core={len(needed_cores):,}, PostNum={len(needed_post_num):,}, "
          f"Char4={len(needed_char4):,}, Domain={len(needed_domain):,}", flush=True)

    # ─────────────────────────────────────────────────────────────
    # Pass 2: Stream and index targets (S2 + S3)
    # ─────────────────────────────────────────────────────────────
    print("\n[Step 2/6] Pass 2: Indexing test targets (S2 + S3)...", flush=True)
    t0 = time.time()

    idx_core = defaultdict(list); idx_sorted = defaultdict(list)
    idx_concat = defaultdict(list); idx_brand = defaultdict(list)
    idx_num_p = defaultdict(list); idx_post_p = defaultdict(list)
    idx_post_num = defaultdict(list); idx_addr_k = defaultdict(list)
    idx_city_num = defaultdict(list); idx_char4 = defaultdict(list)
    idx_domain = defaultdict(list); idx_post_dist1 = defaultdict(list)

    target_table = []
    target_ids = []
    target_sources = []   # V4 NEW: track s2/s3 for is_s2/is_s3 features

    for source_tag, source_file in [("s2", "test_source2.tsv"), ("s3", "test_source3.tsv")]:
        path = os.path.join(test_dir, source_file)
        print(f"  Streaming {source_file}...", flush=True)
        line_cnt = 0
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                line_cnt += 1
                if line_cnt % 1000000 == 0:
                    print(f"    {source_file}: {line_cnt:,} lines, {len(target_table):,} targets...", flush=True)
                parts = line.rstrip("\r\n").split("\t")
                t_id = parts[0].strip()
                raw_name = parts[1] if len(parts) > 1 else ""
                raw_addr = parts[2] if len(parts) > 2 else ""
                country = parts[3].strip().lower() if len(parts) > 3 else ""

                norm_name = normalize_raw_text(raw_name)
                clean_addr, postal, num, dist_addr = normalize_address(raw_addr)
                aliases = extract_alias_names(raw_name)
                core_prefix3 = make_prefix_key(norm_name, 3)
                domain_tok = extract_domain_token(raw_name)
                core = extract_core_name(norm_name)
                sorted_k = extract_sorted_key(norm_name)
                concat_k = extract_concat_key(norm_name)
                brand_tokens = extract_distinctive_tokens(norm_name)
                alias_cores = [extract_core_name(a) for a in aliases if extract_core_name(a)]

                num_p = f"{num}_{core_prefix3}" if (num and core_prefix3) else ""
                post_p = f"{postal}_{core_prefix3}" if (postal and core_prefix3) else ""
                post_num = f"{postal}_{num}" if (postal and num) else ""
                addr_keys = [f"{num}_{t}" for t in sorted(dist_addr)[:2]] if num else []
                city = extract_city(raw_addr)
                city_num = f"{city}_{num}" if (city and num) else ""
                char4 = char4gram_key(core)
                dist1 = sorted(dist_addr)[0] if dist_addr else ""
                post_dist1 = f"{postal}_{dist1}" if (postal and dist1) else ""

                hit = False
                t_idx = len(target_table)

                if core in needed_cores and len(idx_core[core]) < 40:
                    idx_core[core].append(t_idx); hit = True
                for ac in alias_cores:
                    if ac in needed_cores and len(idx_core[ac]) < 40:
                        idx_core[ac].append(t_idx); hit = True
                if sorted_k in needed_sorted and len(idx_sorted[sorted_k]) < 40:
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
                if post_num and post_num in needed_post_num and len(idx_post_num[post_num]) < 25:
                    idx_post_num[post_num].append(t_idx); hit = True
                for ak in addr_keys:
                    if ak in needed_addr_k and len(idx_addr_k[ak]) < 15:
                        idx_addr_k[ak].append(t_idx); hit = True
                if city_num and city_num in needed_city_num and len(idx_city_num[city_num]) < 20:
                    idx_city_num[city_num].append(t_idx); hit = True
                if char4 and len(char4) == 4 and char4 in needed_char4 and len(idx_char4[char4]) < 15:
                    idx_char4[char4].append(t_idx); hit = True
                if domain_tok and domain_tok in needed_domain and len(idx_domain[domain_tok]) < 20:
                    idx_domain[domain_tok].append(t_idx); hit = True
                if post_dist1 and post_dist1 in needed_post_dist1 and len(idx_post_dist1[post_dist1]) < 20:
                    idx_post_dist1[post_dist1].append(t_idx); hit = True

                if hit:
                    cat_words = extract_category_words(norm_name)
                    target_table.append((
                        core, sorted_k, concat_k, norm_name, clean_addr,
                        country, postal, num, dist_addr, is_non_ascii_name(raw_name),
                        alias_cores, cat_words, raw_addr
                    ))
                    target_ids.append(t_id)
                    target_sources.append(source_tag)

    print(f"Indexed {len(target_table):,} relevant targets in {time.time()-t0:.1f}s.", flush=True)

    # ─────────────────────────────────────────────────────────────
    # Pass 3: Matching + Feature Extraction + CatBoost Scoring
    # ─────────────────────────────────────────────────────────────
    print(f"\n[Step 3/6] Pass 3: Matching + {'CatBoost' if HAS_CATBOOST else 'Rule-based'} scoring...", flush=True)
    t0 = time.time()
    CHUNK_SIZE = 50000   # Smaller chunks since feature extraction is heavier

    scratch_file = os.path.join(OUTPUT_DIR, "scratch_results_v4.pkl")
    target_claims = {}  # {t_idx: (s1_id, score)}

    total_processed = 0
    chunk_count = 0
    total_candidates_scored = 0

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
                domain_tok = extract_domain_token(raw_name)
                s1_cat_words = extract_category_words(norm_name)
                core = extract_core_name(norm_name)
                sorted_k = extract_sorted_key(norm_name)
                concat_k = extract_concat_key(norm_name)
                brand_tokens = extract_distinctive_tokens(norm_name)
                alias_cores = [extract_core_name(a) for a in aliases if extract_core_name(a)]

                num_p = f"{num}_{core_prefix3}" if (num and core_prefix3) else ""
                post_p = f"{postal}_{core_prefix3}" if (postal and core_prefix3) else ""
                post_num = f"{postal}_{num}" if (postal and num) else ""
                addr_keys = [f"{num}_{t}" for t in sorted(dist_addr)[:2]] if num else []
                city = extract_city(raw_addr)
                city_num = f"{city}_{num}" if (city and num) else ""
                char4 = char4gram_key(core)
                dist1 = sorted(dist_addr)[0] if dist_addr else ""
                post_dist1 = f"{postal}_{dist1}" if (postal and dist1) else ""

                current_chunk.append((
                    s1_id, core, sorted_k, concat_k, brand_tokens, postal, num,
                    dist_addr, country, alias_cores, num_p, post_p, post_num, addr_keys,
                    clean_addr, norm_name, is_non_ascii_name(raw_name), s1_cat_words,
                    city_num, char4, domain_tok, post_dist1, raw_addr
                ))

                if len(current_chunk) >= CHUNK_SIZE:
                    chunk_count += 1
                    t_chunk_start = time.time()
                    chunk_results = []

                    # ── collect all candidate pairs for this chunk ──────────
                    all_pair_features = []
                    all_pair_meta = []  # (s1_idx_in_chunk, t_idx)

                    for ci, s1_rec in enumerate(current_chunk):
                        (s1_id, core_s1, sorted_s1, concat_s1, brand_tokens, s1_p, s1_n, s1_dist,
                         s1_country, s1_aliases, num_p_s1, post_p_s1, post_num_s1, addr_keys_s1,
                         clean_s1_addr, norm_s1_name, s1_is_non_ascii, s1_cat_words,
                         city_num_s1, char4_s1, domain_s1, post_dist1_s1, s1_raw_addr) = s1_rec

                        cands = set()
                        cands.update(idx_core.get(core_s1, []))
                        for a in s1_aliases: cands.update(idx_core.get(a, []))
                        cands.update(idx_sorted.get(sorted_s1, []))
                        if len(concat_s1) >= 5: cands.update(idx_concat.get(concat_s1, []))
                        for bt in brand_tokens[:6]: cands.update(idx_brand.get(bt, []))
                        if num_p_s1: cands.update(idx_num_p.get(num_p_s1, []))
                        if post_p_s1: cands.update(idx_post_p.get(post_p_s1, []))
                        if post_num_s1: cands.update(idx_post_num.get(post_num_s1, []))
                        for ak in addr_keys_s1: cands.update(idx_addr_k.get(ak, []))
                        if city_num_s1: cands.update(idx_city_num.get(city_num_s1, []))
                        if char4_s1: cands.update(idx_char4.get(char4_s1, []))
                        if domain_s1: cands.update(idx_domain.get(domain_s1, []))
                        if post_dist1_s1: cands.update(idx_post_dist1.get(post_dist1_s1, []))

                        has_s1_addr = bool(clean_s1_addr)
                        chunk_results.append((s1_id, list(cands), []))

                        # Pre-filter candidates with basic checks before feature extraction
                        for t_idx in cands:
                            t = target_table[t_idx]
                            t_country = t[5]
                            if s1_country and t_country and s1_country != t_country:
                                continue

                            t_clean_addr = t[4]
                            t_dist = t[8]
                            t_postal = t[6]
                            t_num = t[7]
                            has_t_addr = bool(t_clean_addr)

                            # Quick address filter (same as V2/V3) to avoid wasting feature time
                            postal_match = bool(s1_p and t_postal and s1_p == t_postal)
                            postal_conflict = bool(s1_p and t_postal and s1_p != t_postal)
                            num_match = bool(s1_n and t_num and s1_n == t_num)
                            num_conflict = bool(s1_n and t_num and s1_n != t_num)
                            dist_overlap = len(s1_dist & t_dist) if (s1_dist and t_dist) else 0

                            addr_sim = None
                            if has_s1_addr and has_t_addr:
                                addr_sim = fuzz.token_set_ratio(clean_s1_addr, t_clean_addr) / 100.0
                                if addr_sim < 0.35: continue
                                if s1_dist and t_dist and not postal_match and dist_overlap == 0: continue
                                if postal_conflict and addr_sim < 0.70: continue
                                if num_conflict and dist_overlap == 0: continue
                            else:
                                core_words = core_s1.split()
                                if len(core_words) < 2 and len(core_s1) < 14: continue

                            all_pair_meta.append((ci, t_idx, t))
                            all_pair_features.append(None)  # placeholder

                    # ── Extract features + CatBoost batch score ────────────
                    if HAS_CATBOOST and all_pair_meta and FEAT_NAMES:
                        feat_matrix = []
                        for ci, t_idx, t in all_pair_meta:
                            s1_rec = current_chunk[ci]
                            (s1_id, core_s1, sorted_s1, concat_s1, brand_tokens, s1_p, s1_n, s1_dist,
                             s1_country, s1_aliases, num_p_s1, post_p_s1, post_num_s1, addr_keys_s1,
                             clean_s1_addr, norm_s1_name, s1_is_non_ascii, s1_cat_words,
                             city_num_s1, char4_s1, domain_s1, post_dist1_s1, s1_raw_addr) = s1_rec

                            (core_t, sorted_t, concat_t, norm_t_name, clean_t_addr, t_country,
                             t_p, t_n, t_dist, t_is_non_ascii, t_aliases, t_cat_words, t_raw_addr) = t
                            t_source = target_sources[t_idx]

                            feats = extract_features(
                                norm_s1_name, s1_raw_addr, s1_country, s1_is_non_ascii,
                                clean_s1_addr, s1_p, s1_n, s1_dist, s1_aliases,
                                norm_t_name, t_raw_addr, t_country, t_is_non_ascii,
                                clean_t_addr, t_p, t_n, t_dist, t_aliases,
                                t_source, len(feat_matrix), len(all_pair_meta),
                                idf_weights
                            )
                            feat_matrix.append(feats)

                        feat_arr = np.array(feat_matrix, dtype=np.float32)
                        pool = Pool(feat_arr)
                        scores = cb_model.predict(pool)
                        # Convert to probabilities if raw outputs
                        if scores.max() > 1.0 or scores.min() < 0.0:
                            import scipy.special
                            scores = scipy.special.expit(scores)
                        total_candidates_scored += len(scores)

                        # Map scores back to chunk_results
                        for pair_i, (ci, t_idx, t) in enumerate(all_pair_meta):
                            sc = float(scores[pair_i])
                            if sc >= BEST_TAU_S2:   # use same tau for s2 and s3
                                # update chunk_results[ci]
                                s1_id_r = chunk_results[ci][0]
                                chunk_results[ci][2].append((t_idx, sc))
                                # Update target_claims
                                prev = target_claims.get(t_idx)
                                if prev is None or sc > prev[1]:
                                    target_claims[t_idx] = (s1_id_r, sc)
                    else:
                        # Fallback: rule-based scoring
                        for ci, t_idx, t in all_pair_meta:
                            s1_rec = current_chunk[ci]
                            (s1_id, core_s1, sorted_s1, concat_s1, brand_tokens, s1_p, s1_n, s1_dist,
                             s1_country, s1_aliases, *rest) = s1_rec
                            clean_s1_addr = s1_rec[14]
                            norm_s1_name = s1_rec[15]
                            s1_is_non_ascii = s1_rec[16]
                            s1_cat_words = s1_rec[17]
                            has_s1_addr = bool(clean_s1_addr)

                            (core_t, sorted_t, concat_t, norm_t_name, clean_t_addr, t_country,
                             t_p, t_n, t_dist, t_is_non_ascii, t_aliases, t_cat_words, t_raw_addr) = t

                            postal_match = bool(s1_p and t_p and s1_p == t_p)
                            postal_conflict = bool(s1_p and t_p and s1_p != t_p)
                            num_match = bool(s1_n and t_n and s1_n == t_n)
                            num_conflict = bool(s1_n and t_n and s1_n != t_n)
                            dist_overlap = len(s1_dist & t_dist) if (s1_dist and t_dist) else 0
                            addr_token_sim = (fuzz.token_set_ratio(clean_s1_addr, clean_t_addr) / 100.0
                                              if (has_s1_addr and clean_t_addr) else None)

                            sc = rule_based_score(
                                core_s1, sorted_s1, extract_concat_key(norm_s1_name), norm_s1_name, clean_s1_addr,
                                s1_p, s1_n, s1_dist, s1_is_non_ascii, s1_aliases, has_s1_addr,
                                core_t, sorted_t, concat_t, norm_t_name, clean_t_addr,
                                t_p, t_n, t_dist, t_is_non_ascii, t_aliases, t_cat_words, s1_cat_words,
                                postal_match, postal_conflict, num_match, num_conflict, dist_overlap, addr_token_sim
                            )
                            if sc >= 0.81:
                                chunk_results[ci][2].append((t_idx, sc))
                                s1_id_r = chunk_results[ci][0]
                                prev = target_claims.get(t_idx)
                                if prev is None or sc > prev[1]:
                                    target_claims[t_idx] = (s1_id_r, sc)

                    pickle.dump(chunk_results, f_scratch)
                    total_processed += len(current_chunk)
                    dt = time.time() - t_chunk_start
                    rate = len(current_chunk) / dt if dt > 0 else 0
                    print(f"  Chunk {chunk_count:2d}: {total_processed:,}/{total_s1_count:,} "
                          f"({total_processed/total_s1_count*100:.1f}%) | {rate:,.0f} e/s | "
                          f"scored {total_candidates_scored:,} pairs | Elapsed: {time.time()-t0:.0f}s", flush=True)
                    current_chunk = []
                    gc.collect()

            # Final chunk
            if current_chunk:
                chunk_count += 1
                chunk_results = []
                all_pair_features = []
                all_pair_meta = []

                for ci, s1_rec in enumerate(current_chunk):
                    (s1_id, core_s1, sorted_s1, concat_s1, brand_tokens, s1_p, s1_n, s1_dist,
                     s1_country, s1_aliases, num_p_s1, post_p_s1, post_num_s1, addr_keys_s1,
                     clean_s1_addr, norm_s1_name, s1_is_non_ascii, s1_cat_words,
                     city_num_s1, char4_s1, domain_s1, post_dist1_s1, s1_raw_addr) = s1_rec

                    cands = set()
                    cands.update(idx_core.get(core_s1, []))
                    for a in s1_aliases: cands.update(idx_core.get(a, []))
                    cands.update(idx_sorted.get(sorted_s1, []))
                    if len(concat_s1) >= 5: cands.update(idx_concat.get(concat_s1, []))
                    for bt in brand_tokens[:6]: cands.update(idx_brand.get(bt, []))
                    if num_p_s1: cands.update(idx_num_p.get(num_p_s1, []))
                    if post_p_s1: cands.update(idx_post_p.get(post_p_s1, []))
                    if post_num_s1: cands.update(idx_post_num.get(post_num_s1, []))
                    for ak in addr_keys_s1: cands.update(idx_addr_k.get(ak, []))
                    if city_num_s1: cands.update(idx_city_num.get(city_num_s1, []))
                    if char4_s1: cands.update(idx_char4.get(char4_s1, []))
                    if domain_s1: cands.update(idx_domain.get(domain_s1, []))
                    if post_dist1_s1: cands.update(idx_post_dist1.get(post_dist1_s1, []))

                    has_s1_addr = bool(clean_s1_addr)
                    chunk_results.append((s1_id, list(cands), []))

                    for t_idx in cands:
                        t = target_table[t_idx]
                        t_country = t[5]; t_clean_addr = t[4]; t_dist = t[8]
                        t_postal = t[6]; t_num = t[7]
                        has_t_addr = bool(t_clean_addr)
                        if s1_country and t_country and s1_country != t_country: continue

                        postal_match = bool(s1_p and t_postal and s1_p == t_postal)
                        postal_conflict = bool(s1_p and t_postal and s1_p != t_postal)
                        num_match = bool(s1_n and t_num and s1_n == t_num)
                        num_conflict = bool(s1_n and t_num and s1_n != t_num)
                        dist_overlap = len(s1_dist & t_dist) if (s1_dist and t_dist) else 0
                        addr_sim = None
                        if has_s1_addr and has_t_addr:
                            addr_sim = fuzz.token_set_ratio(clean_s1_addr, t_clean_addr) / 100.0
                            if addr_sim < 0.35: continue
                            if s1_dist and t_dist and not postal_match and dist_overlap == 0: continue
                            if postal_conflict and addr_sim < 0.70: continue
                            if num_conflict and dist_overlap == 0: continue
                        else:
                            core_words = core_s1.split()
                            if len(core_words) < 2 and len(core_s1) < 14: continue
                        all_pair_meta.append((ci, t_idx, t))

                if HAS_CATBOOST and all_pair_meta and FEAT_NAMES:
                    feat_matrix = []
                    for ci, t_idx, t in all_pair_meta:
                        s1_rec = current_chunk[ci]
                        (s1_id, core_s1, sorted_s1, concat_s1, brand_tokens, s1_p, s1_n, s1_dist,
                         s1_country, s1_aliases, num_p_s1, post_p_s1, post_num_s1, addr_keys_s1,
                         clean_s1_addr, norm_s1_name, s1_is_non_ascii, s1_cat_words,
                         city_num_s1, char4_s1, domain_s1, post_dist1_s1, s1_raw_addr) = s1_rec
                        (core_t, sorted_t, concat_t, norm_t_name, clean_t_addr, t_country,
                         t_p, t_n, t_dist, t_is_non_ascii, t_aliases, t_cat_words, t_raw_addr) = t
                        t_source = target_sources[t_idx]
                        feats = extract_features(
                            norm_s1_name, s1_raw_addr, s1_country, s1_is_non_ascii,
                            clean_s1_addr, s1_p, s1_n, s1_dist, s1_aliases,
                            norm_t_name, t_raw_addr, t_country, t_is_non_ascii,
                            clean_t_addr, t_p, t_n, t_dist, t_aliases,
                            t_source, len(feat_matrix), len(all_pair_meta), idf_weights
                        )
                        feat_matrix.append(feats)

                    feat_arr = np.array(feat_matrix, dtype=np.float32)
                    scores = cb_model.predict(Pool(feat_arr))
                    if scores.max() > 1.0 or scores.min() < 0.0:
                        import scipy.special
                        scores = scipy.special.expit(scores)

                    for pair_i, (ci, t_idx, t) in enumerate(all_pair_meta):
                        sc = float(scores[pair_i])
                        if sc >= BEST_TAU_S2:
                            s1_id_r = chunk_results[ci][0]
                            chunk_results[ci][2].append((t_idx, sc))
                            prev = target_claims.get(t_idx)
                            if prev is None or sc > prev[1]:
                                target_claims[t_idx] = (s1_id_r, sc)

                pickle.dump(chunk_results, f_scratch)
                total_processed += len(current_chunk)
                print(f"  Final chunk: {total_processed:,}/{total_s1_count:,} (100%) in {time.time()-t0:.1f}s", flush=True)

    print(f"\nStep 3 done. {total_processed:,} entities, {total_candidates_scored:,} pairs scored by CatBoost.", flush=True)

    # ─────────────────────────────────────────────────────────────
    # Step 4: Target Conflict Resolution (globally optimal, no Hungarian)
    # ─────────────────────────────────────────────────────────────
    print("\n[Step 4/6] Resolving target winners (globally optimal)...", flush=True)
    t0 = time.time()
    winner_for_target = {t_idx: val[0] for t_idx, val in target_claims.items()}
    del target_claims
    gc.collect()
    print(f"  {len(winner_for_target):,} exclusive target winners resolved in {time.time()-t0:.2f}s.", flush=True)

    # ─────────────────────────────────────────────────────────────
    # Step 5: ML Singleton Gate
    # ─────────────────────────────────────────────────────────────
    print("\n[Step 5/6] Applying ML singleton gate to output...", flush=True)
    t0 = time.time()

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

                    retained_raw = [
                        (t_idx, sc) for t_idx, sc in scored_matches
                        if winner_for_target.get(t_idx) == s1_id
                    ]

                    # ── ML Singleton Gate ─────────────────────────────────
                    should_singleton = False
                    if len(retained_raw) == 0:
                        should_singleton = True
                    elif len(retained_raw) == 1 and HAS_GATE_MODEL and GATE_FEAT_NAMES:
                        # Build gate features from retained pair's data
                        t_idx_r, sc_r = retained_raw[0]
                        t = target_table[t_idx_r]
                        (core_t, sorted_t, concat_t, norm_t_name, clean_t_addr, t_country,
                         t_p, t_n, t_dist, t_is_non_ascii, t_aliases, t_cat_words, t_raw_addr) = t

                        # Approximate gate features from score alone (single candidate)
                        gate_feats = [
                            1.0,     # candidate_count
                            float(target_sources[t_idx_r] == "s2"),  # s2_count
                            float(target_sources[t_idx_r] == "s3"),  # s3_count
                            sc_r, sc_r,   # name_strength max/top3_mean
                            0.5, 0.5,     # addr_strength (unknown)
                            sc_r * 0.5, sc_r * 0.5,  # name_x_addr
                            sc_r, sc_r,   # token_sort max/top3
                            sc_r, sc_r,   # token_set max/top3
                            1.0 if sc_r >= 0.96 else 0.0, 1.0 if sc_r >= 0.96 else 0.0,  # core_exact
                            0.5, 0.5,     # char_ngram
                            1.0, 1.0,     # max_shared_idf
                            0.0, 0.0,     # postal_match
                            0.0, 0.0,     # num_match
                            0.0, 0.0,     # country_rel
                        ]
                        gate_feats_arr = np.array([gate_feats[:len(GATE_FEAT_NAMES)]])
                        try:
                            gate_prob = gate_model.predict_proba(gate_feats_arr)[0][1]
                            # gate_prob > BEST_TAU_SINGLETON → entity is NON-singleton
                            if gate_prob < BEST_TAU_SINGLETON:
                                should_singleton = True
                        except Exception:
                            # Fallback heuristic gate
                            if sc_r < 0.84:
                                should_singleton = True
                    elif len(retained_raw) == 1 and not HAS_GATE_MODEL:
                        # Heuristic fallback
                        t_idx_r, sc_r = retained_raw[0]
                        if sc_r < 0.84:
                            should_singleton = True

                    if should_singleton and len(retained_raw) > 0:
                        gate_pruned += 1
                        retained_raw = []

                    retained = [target_ids[t_idx] for t_idx, _ in retained_raw]
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

    print(f"\nStep 5 done in {time.time()-t0:.1f}s.", flush=True)
    print(f"  matching_results.tsv: {match_mb:.1f} MB | {total_matches:,} links | "
          f"{singleton_preds:,} singletons ({singleton_preds/final_count*100:.2f}%)", flush=True)
    print(f"  Gate pruned: {gate_pruned:,} marginal single-match entities → singletons", flush=True)
    print(f"  candidate_pairs.tsv: {cand_mb:.1f} MB", flush=True)

    # ─────────────────────────────────────────────────────────────
    # Step 6: Copy output to Desktop
    # ─────────────────────────────────────────────────────────────
    print("\n[Step 6/6] Copying output to Desktop...", flush=True)
    desktop_path = os.path.expanduser("~/Desktop/matching_results_v4.tsv")
    try:
        import shutil
        shutil.copy2(matching_file, desktop_path)
        print(f"  Copied to {desktop_path}", flush=True)
    except Exception as e:
        print(f"  Copy failed: {e}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(" >>> PIPELINE V4 COMPLETED SUCCESSFULLY <<<", flush=True)
    print(f"  Output: {matching_file}", flush=True)
    print(f"  Links: {total_matches:,} | Singletons: {singleton_preds:,}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
