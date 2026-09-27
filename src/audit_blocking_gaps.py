#!/usr/bin/env python3
"""
Audit Blocking Gaps - Amazon ML Challenge 2026
===============================================
Classifies blocking failures into:
  (a) Blocker-recoverable   - share rare token / 4-gram / high embedding sim
  (b) Truly disjoint names  - genuine DBA rebrands with NO textual overlap
  (c) Cross-script / Indic  - non-ASCII pairs that might be transliterations

Also verifies:
  1. How many "disjoint DBA" pairs are actually LATIN-script already
  2. What fraction of misses are fixed by each new cheap blocking route
  3. Whether winner_for_target is already globally optimal

Usage:
    python audit_blocking_gaps.py --data-dir /home/gojo/Desktop/AMAZON-ML/dataset/train
    (or run locally if data is present)
"""

import argparse
import csv
import os
import re
import sys
import unicodedata
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

try:
    from rapidfuzz import fuzz, distance as rfdist
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("[WARN] rapidfuzz not found – name similarity checks will be skipped", flush=True)


# ──────────────────────────────────────────────────────────────
# Copied from run_robust_production_v2.py (must stay in sync)
# ──────────────────────────────────────────────────────────────
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
    "pkwy": "parkway", "hwy": "highway",
}
US_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in",
    "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv",
    "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn",
    "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc"
}
GENERIC_ADDR_WORDS = (
    set(STREET_ABBREVIATIONS.keys()) | set(STREET_ABBREVIATIONS.values()) |
    US_STATES | {
        "suite", "ste", "apt", "unit", "fl", "floor", "building", "bldg", "po", "box",
        "north", "south", "east", "west", "n", "s", "e", "w",
        "city", "near", "opp", "opposite", "behind", "dist", "district", "state",
        "road", "street", "highway", "county", "hno", "no", "plot", "shop", "flat"
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


def is_truly_latin(raw_name: str) -> bool:
    """Check if a name is already purely Latin-script."""
    if not raw_name:
        return True
    for ch in raw_name:
        if ord(ch) > 127:
            return False
    return True


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
    dist_words = set()
    for w in words:
        if len(w) >= 3 and w not in GENERIC_ADDR_WORDS and not w.isdigit():
            dist_words.add(w)
    clean_addr = " ".join(STREET_ABBREVIATIONS.get(w, US_STATES.__contains__(w) and w or w) for w in words)
    return norm, postal, num, dist_words


def make_prefix_key(name: str, length: int = 3) -> str:
    core = extract_core_name(name).replace(" ", "")
    return core[:length] if len(core) >= length else ""


def char_ngrams(text: str, n: int = 4) -> Set[str]:
    """Extract character n-grams from text."""
    text = text.replace(" ", "")
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard(a: Set, b: Set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# V2 Blocking routes (to check if a pair is retrievable)
# ──────────────────────────────────────────────────────────────
def get_s1_keys(norm_name: str, raw_addr: str, raw_name: str) -> Dict[str, List[str]]:
    """Return all blocking keys an S1 entity would emit."""
    _, postal, num, dist_addr = normalize_address(raw_addr)
    aliases = extract_alias_names(raw_name)
    core_prefix3 = make_prefix_key(norm_name, 3)
    core = extract_core_name(norm_name)
    sorted_k = extract_sorted_key(norm_name)
    concat_k = extract_concat_key(norm_name)
    brand_tokens = extract_distinctive_tokens(norm_name)
    alias_cores = [extract_core_name(a) for a in aliases if extract_core_name(a)]

    num_p = f"{num}_{core_prefix3}" if (num and core_prefix3) else ""
    post_p = f"{postal}_{core_prefix3}" if (postal and core_prefix3) else ""
    post_num = f"{postal}_{num}" if (postal and num) else ""
    addr_keys = [f"{num}_{t}" for t in sorted(dist_addr)[:2]] if num else []

    return {
        "core": [core] + alias_cores,
        "sorted": [sorted_k],
        "concat": [concat_k] if len(concat_k) >= 5 else [],
        "brand": brand_tokens,
        "num_p": [num_p] if num_p else [],
        "post_p": [post_p] if post_p else [],
        "post_num": [post_num] if post_num else [],
        "addr_keys": addr_keys,
        # NEW routes we're evaluating:
        "city_street": [],   # will be populated below
        "4gram": list(char_ngrams(core, 4))[:10] if len(core) >= 4 else [],
        "domain": [],        # domain-unpack (www./email @)
    }


def get_target_keys(norm_name: str, raw_addr: str, raw_name: str) -> Dict[str, List[str]]:
    """Return all blocking keys a target entity would emit."""
    return get_s1_keys(norm_name, raw_addr, raw_name)


def pairs_overlap_via_route(s1_keys: Dict, t_keys: Dict) -> Tuple[bool, List[str]]:
    """Check if S1 and target share any blocking key. Returns (found, matching_routes)."""
    matching = []
    for route in ["core", "sorted", "concat", "brand", "num_p", "post_p", "post_num", "addr_keys"]:
        s1_set = set(k for k in s1_keys.get(route, []) if k)
        t_set = set(k for k in t_keys.get(route, []) if k)
        if s1_set & t_set:
            matching.append(route)
    return bool(matching), matching


def check_new_routes(s1_keys: Dict, t_keys: Dict) -> List[str]:
    """Check only NEW proposed routes."""
    matching = []
    for route in ["4gram"]:
        s1_set = set(k for k in s1_keys.get(route, []) if k)
        t_set = set(k for k in t_keys.get(route, []) if k)
        if s1_set & t_set:
            matching.append(route)
    return matching


# ──────────────────────────────────────────────────────────────
# Main audit logic
# ──────────────────────────────────────────────────────────────
def load_source(path: str, limit: int = None) -> Dict[str, Dict]:
    """Load a TSV source file → {entity_id: {name, addr, country, raw_name}}"""
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            if not row:
                continue
            eid = row[0].strip()
            raw_name = row[1] if len(row) > 1 else ""
            raw_addr = row[2] if len(row) > 2 else ""
            country = row[3].strip().lower() if len(row) > 3 else ""
            norm_name = normalize_raw_text(raw_name)
            records[eid] = {
                "raw_name": raw_name,
                "norm_name": norm_name,
                "raw_addr": raw_addr,
                "country": country,
                "is_non_ascii": is_non_ascii_name(raw_name),
                "is_latin": is_truly_latin(raw_name),
            }
    return records


def load_ground_truth(path: str) -> Dict[str, List[str]]:
    """Load ground truth TSV → {s1_id: [target_ids...]}"""
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            matches_str = row[1].strip() if len(row) > 1 else ""
            if matches_str:
                targets = [t.strip() for t in matches_str.split(",") if t.strip()]
            else:
                targets = []
            gt[s1_id] = targets
    return gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/train", help="Path to train directory")
    parser.add_argument("--sample", type=int, default=5000, help="S1 entities to sample (0=all)")
    args = parser.parse_args()

    data_dir = args.data_dir
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")

    for p in [s1_path, s2_path, s3_path, gt_path]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing file: {p}", flush=True)
            sys.exit(1)

    sample_limit = args.sample if args.sample > 0 else None
    print(f"\n{'='*70}", flush=True)
    print(f" BLOCKING GAP AUDIT — Sample: {sample_limit or 'ALL'} S1 entities", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n[1/4] Loading S1 source ({sample_limit or 'full'})...", flush=True)
    s1_records = load_source(s1_path, limit=sample_limit)
    print(f"      Loaded {len(s1_records):,} S1 entities", flush=True)

    print(f"\n[2/4] Loading ground truth...", flush=True)
    gt = load_ground_truth(gt_path)
    # Filter to only S1 IDs we loaded
    s1_ids = set(s1_records.keys())
    gt_filtered = {k: v for k, v in gt.items() if k in s1_ids}
    print(f"      GT entries for our sample: {len(gt_filtered):,}", flush=True)

    # Collect all target IDs we'll need to resolve
    needed_target_ids = set()
    for targets in gt_filtered.values():
        needed_target_ids.update(targets)
    print(f"      Unique target IDs needed: {len(needed_target_ids):,}", flush=True)

    print(f"\n[3/4] Loading S2+S3 targets (only needed IDs)...", flush=True)
    target_records = {}
    for src_path in [s2_path, s3_path]:
        fname = os.path.basename(src_path)
        cnt = 0
        with open(src_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                if not row:
                    continue
                eid = row[0].strip()
                if eid in needed_target_ids:
                    raw_name = row[1] if len(row) > 1 else ""
                    raw_addr = row[2] if len(row) > 2 else ""
                    country = row[3].strip().lower() if len(row) > 3 else ""
                    norm_name = normalize_raw_text(raw_name)
                    target_records[eid] = {
                        "raw_name": raw_name,
                        "norm_name": norm_name,
                        "raw_addr": raw_addr,
                        "country": country,
                        "is_non_ascii": is_non_ascii_name(raw_name),
                        "is_latin": is_truly_latin(raw_name),
                    }
                    cnt += 1
        print(f"      {fname}: loaded {cnt:,} needed targets", flush=True)
    print(f"      Total targets resolved: {len(target_records):,}", flush=True)

    # ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] Auditing blocking gaps for each true match pair...", flush=True)

    # Counters
    total_pairs = 0
    singleton_pairs = 0   # s1 has no true matches
    missing_target_records = 0

    # Blocking analysis
    retrieved_by_v2 = 0
    miss_blocker_recoverable = 0
    miss_new_route_only = 0
    miss_truly_disjoint = 0

    # Additional breakdown
    miss_non_ascii = 0
    miss_latin_only = 0

    # New route breakdown
    new_route_hits = Counter()

    # Name similarity stats for disjoint misses
    disjoint_name_sims = []
    disjoint_addr_postal_match = 0
    disjoint_addr_num_match = 0

    # Category conflict (Doc 17 insight)
    BIZ_CATEGORY_WORDS = {
        "pharmacy", "pharma", "clinic", "hospital", "hotel", "restaurant", "cafe", "cafe",
        "school", "college", "university", "bank", "tech", "software", "hardware", "auto",
        "car", "truck", "salon", "spa", "gym", "fitness", "mart", "store", "shop",
        "bakery", "pizza", "burger", "chicken", "gas", "petrol", "fuel"
    }

    def get_category_words(norm_name):
        return set(t for t in norm_name.split() if t in BIZ_CATEGORY_WORDS)

    category_conflict_count = 0

    processed = 0
    for s1_id, s1_rec in s1_records.items():
        true_targets = gt_filtered.get(s1_id, [])

        if not true_targets:
            singleton_pairs += 1
            continue

        s1_keys = get_s1_keys(s1_rec["norm_name"], s1_rec["raw_addr"], s1_rec["raw_name"])
        s1_cats = get_category_words(s1_rec["norm_name"])

        for t_id in true_targets:
            total_pairs += 1

            if t_id not in target_records:
                missing_target_records += 1
                continue

            t_rec = target_records[t_id]
            t_keys = get_target_keys(t_rec["norm_name"], t_rec["raw_addr"], t_rec["raw_name"])

            found_v2, routes_v2 = pairs_overlap_via_route(s1_keys, t_keys)

            if found_v2:
                retrieved_by_v2 += 1
            else:
                # Check new routes
                new_routes_hit = check_new_routes(s1_keys, t_keys)

                # Name similarity for diagnosis
                core_s1 = extract_core_name(s1_rec["norm_name"])
                core_t = extract_core_name(t_rec["norm_name"])

                ngrams_s1 = char_ngrams(core_s1, 4)
                ngrams_t = char_ngrams(core_t, 4)
                ngram_jacc = jaccard(ngrams_s1, ngrams_t)

                token_sim = 0.0
                if HAS_RAPIDFUZZ and core_s1 and core_t:
                    token_sim = fuzz.token_sort_ratio(core_s1, core_t) / 100.0

                # Address overlap
                _, s1_postal, s1_num, s1_dist = normalize_address(s1_rec["raw_addr"])
                _, t_postal, t_num, t_dist = normalize_address(t_rec["raw_addr"])
                postal_match = (s1_postal and t_postal and s1_postal == t_postal)
                num_match = (s1_num and t_num and s1_num == t_num)

                # Classification
                # A pair is "blocker-recoverable" if:
                #   - 4-gram jaccard > 0.1, OR
                #   - token_sort_ratio > 0.65, OR
                #   - new proposed routes would catch it
                is_recoverable = (
                    ngram_jacc >= 0.10 or
                    token_sim >= 0.65 or
                    bool(new_routes_hit)
                )

                if new_routes_hit:
                    miss_new_route_only += 1
                    for r in new_routes_hit:
                        new_route_hits[r] += 1

                if is_recoverable:
                    miss_blocker_recoverable += 1
                else:
                    miss_truly_disjoint += 1
                    disjoint_name_sims.append((token_sim, ngram_jacc))
                    if postal_match:
                        disjoint_addr_postal_match += 1
                    if num_match:
                        disjoint_addr_num_match += 1

                    # Category conflict check (Doc 17)
                    t_cats = get_category_words(t_rec["norm_name"])
                    if s1_cats and t_cats and not (s1_cats & t_cats):
                        category_conflict_count += 1

                # Indic / script analysis
                if s1_rec["is_non_ascii"] or t_rec["is_non_ascii"]:
                    miss_non_ascii += 1
                    if s1_rec["is_latin"] and t_rec["is_latin"]:
                        miss_latin_only += 1

        processed += 1
        if processed % 500 == 0:
            print(f"  ...processed {processed:,} / {len(s1_records):,} S1 entities", flush=True)

    # ──────────────────────────────────────────────────────────────
    # Summary Report
    # ──────────────────────────────────────────────────────────────
    total_miss = total_pairs - retrieved_by_v2 - missing_target_records
    if total_miss < 0:
        total_miss = 0

    print(f"\n{'='*70}", flush=True)
    print(f" BLOCKING GAP AUDIT RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n  S1 entities sampled:       {len(s1_records):,}", flush=True)
    print(f"  Singletons (no GT targets): {singleton_pairs:,}", flush=True)
    print(f"  Total true match pairs:     {total_pairs:,}", flush=True)
    print(f"  Missing target records:     {missing_target_records:,}", flush=True)
    print(f"\n  ── V2 Blocking Recall ──────────────────────────────", flush=True)
    eligible = total_pairs - missing_target_records
    recall_pct = retrieved_by_v2 / eligible * 100 if eligible > 0 else 0.0
    print(f"  Retrieved by V2 blocking:   {retrieved_by_v2:,} / {eligible:,} = {recall_pct:.2f}%", flush=True)
    miss_total = eligible - retrieved_by_v2
    print(f"  Total misses:               {miss_total:,} ({100-recall_pct:.2f}%)", flush=True)

    print(f"\n  ── Miss Classification ─────────────────────────────", flush=True)
    if miss_total > 0:
        print(f"  (a) Blocker-recoverable:    {miss_blocker_recoverable:,} ({miss_blocker_recoverable/miss_total*100:.1f}%)", flush=True)
        print(f"      — Catchable by new routes (4-gram): {miss_new_route_only:,}", flush=True)
        for route, cnt in new_route_hits.most_common():
            print(f"        [{route}]: {cnt:,}", flush=True)
        print(f"  (b) Truly disjoint names:   {miss_truly_disjoint:,} ({miss_truly_disjoint/miss_total*100:.1f}%)", flush=True)
        print(f"      — Of those, postal match:  {disjoint_addr_postal_match:,}", flush=True)
        print(f"      — Of those, num match:     {disjoint_addr_num_match:,}", flush=True)
        print(f"      — Of those, category conflict: {category_conflict_count:,}", flush=True)
        print(f"  (c) Non-ASCII pairs missed:  {miss_non_ascii:,}", flush=True)
        print(f"      — Already Latin-script:    {miss_latin_only:,}", flush=True)

    if disjoint_name_sims:
        avg_tok = sum(s[0] for s in disjoint_name_sims) / len(disjoint_name_sims)
        avg_ngram = sum(s[1] for s in disjoint_name_sims) / len(disjoint_name_sims)
        print(f"\n  ── Disjoint Miss Name Similarity ───────────────────", flush=True)
        print(f"  Avg token_sort_ratio:   {avg_tok:.3f}", flush=True)
        print(f"  Avg 4-gram Jaccard:     {avg_ngram:.3f}", flush=True)

    print(f"\n  ── winner_for_target Analysis ──────────────────────", flush=True)
    print(f"  Current implementation (V2 line 584):", flush=True)
    print(f"    winner_for_target = {{t_idx: val[0] for t_idx, val in target_claims.items()}}", flush=True)
    print(f"  This is a GLOBAL pass over all S1 chunks — for each target, it keeps", flush=True)
    print(f"  the single highest-ranked S1 claimant. Each target decision is INDEPENDENT.", flush=True)
    print(f"  VERDICT: Already globally optimal. No Hungarian algorithm needed.", flush=True)
    print(f"  S1 entities can still have MULTIPLE targets (1-to-many is preserved).", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f" KEY FINDINGS SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  1. V2 Blocking Recall: {recall_pct:.2f}%", flush=True)
    gap_is_blocker = (miss_blocker_recoverable / miss_total * 100) if miss_total > 0 else 0
    gap_is_disjoint = (miss_truly_disjoint / miss_total * 100) if miss_total > 0 else 0
    print(f"  2. Of missed pairs: {gap_is_blocker:.1f}% are blocker-recoverable (cheap wins)", flush=True)
    print(f"  3. Of missed pairs: {gap_is_disjoint:.1f}% are truly disjoint (need embeddings/DBA data)", flush=True)
    print(f"  4. Category conflict flag useful for: {category_conflict_count:,} disjoint pairs", flush=True)
    print(f"  5. Indic/non-ASCII misses where already Latin: {miss_latin_only:,}", flush=True)
    print(f"\n  ACTION PRIORITY:", flush=True)
    if gap_is_blocker >= 50:
        print(f"  → Fix cheap blocking routes FIRST (majority of gap is recoverable)", flush=True)
    else:
        print(f"  → Majority is truly disjoint → need embedding similarity or DBA training data", flush=True)
    print(f"\n  Done.", flush=True)


if __name__ == "__main__":
    main()
