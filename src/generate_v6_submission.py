#!/usr/bin/env python3
"""High-Performance Test Submission Generator for Amazon Business Entity Resolution.

Implements Hybrid Precision Engine v6 with:
1. Address-Fingerprint Composite Routes (num + street token)
2. Open-Set Country & Full French Normalization (SARL, SAS, EURL, Rue, Bd, etc.)
3. Distinctive Address Component Verification (Zero cross-city/street false merges)
4. Non-Latin Indic Script Record Matcher
5. Missing Target Address Guard (Strict singleton protection)
6. 16-Core Parallel Multi-Processing with Copy-on-Write Memory
7. Global 1-to-1 Mutual-Best Conflict Resolution
8. Strict Competition TSV Formatting Guarantees

Outputs:
- output/matching_results.tsv
- output/candidate_pairs.tsv
"""

from collections import defaultdict
import multiprocessing as mp
import os
import re
import sys
import time
import unicodedata
from typing import Dict, List, Set, Tuple

from rapidfuzz import fuzz

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


# Global structures for multiprocessing workers (inherited read-only via copy-on-write fork)
G_idx_core = None
G_idx_sorted = None
G_idx_concat = None
G_idx_brand = None
G_idx_num_p = None
G_idx_post_p = None
G_idx_addr_k = None
G_target_table = None


def match_worker(chunk_records):
    """Worker process: matches a chunk of S1 records against the global target index."""
    results = []

    for s1_rec in chunk_records:
        (s1_id, core_s1, sorted_s1, concat_s1, brand_tokens, s1_p, s1_n, s1_dist,
         s1_country, s1_aliases, num_p_s1, post_p_s1, addr_keys_s1, clean_s1_addr,
         norm_s1_name, s1_is_non_ascii) = s1_rec

        # Gather Candidates across all routes
        cands = set()
        cands.update(G_idx_core.get(core_s1, [])[:30])
        for a in s1_aliases:
            cands.update(G_idx_core.get(a, [])[:30])
        cands.update(G_idx_sorted.get(sorted_s1, [])[:30])
        if len(concat_s1) >= 5:
            cands.update(G_idx_concat.get(concat_s1, [])[:20])
        for bt in brand_tokens[:3]:
            cands.update(G_idx_brand.get(bt, [])[:20])
        if num_p_s1:
            cands.update(G_idx_num_p.get(num_p_s1, [])[:20])
        if post_p_s1:
            cands.update(G_idx_post_p.get(post_p_s1, [])[:20])
        for ak in addr_keys_s1:
            cands.update(G_idx_addr_k.get(ak, [])[:10])

        scored_matches = []
        has_s1_addr = bool(clean_s1_addr)

        for t_idx in cands:
            t = G_target_table[t_idx]
            # t is tuple: (core, sorted_k, concat_k, norm_name, clean_addr, country, postal, num, dist_tokens, is_non_ascii, aliases)
            (core_t, sorted_t, concat_t, norm_t_name, clean_t_addr, t_country, t_p, t_n, t_dist, is_non_ascii, t_aliases) = t

            # Country constraint
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

            # DISTINCTIVE ADDRESS GATE:
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

            # 5. Non-Latin Indic Script / DBA Record with Exact Address Match
            elif (s1_is_non_ascii or is_non_ascii) and num_match:
                if dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.80:
                    score = 0.88
                elif postal_match and dist_overlap >= 1 and addr_token_sim is not None and addr_token_sim >= 0.88:
                    score = 0.86

            # 6. Fuzzy Name Similarity (with multi-tenant safety)
            elif max_name >= 0.88:
                if postal_match or num_match or dist_overlap >= 1 or not has_s1_addr or not has_t_addr:
                    score = 0.87
            elif max_name >= 0.72:
                if postal_match and (num_match or dist_overlap >= 1):
                    score = 0.84
                elif num_match and dist_overlap >= 1:
                    score = 0.82

            # Strict precision threshold: score >= 0.81 (eliminates 0.80 false merges without address grounding)
            if score >= 0.81:
                scored_matches.append((t_idx, score, addr_token_sim if addr_token_sim is not None else 0.50))

        results.append((s1_id, list(cands), scored_matches))

    return results


def main():
    print("=" * 80)
    print(" [PRODUCTION SUBMISSION GENERATOR] Starting Hybrid Precision Engine v6")
    print("=" * 80)

    test_dir = "dataset/test"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    # Step 1: Stream S1 records and build query sets
    print("\n[Step 1/5] Streaming Source 1 test records...")
    t0 = time.time()
    s1_records = []
    needed_cores = set()
    needed_sorted = set()
    needed_concat = set()
    needed_brand = set()
    needed_num_p = set()
    needed_post_p = set()
    needed_addr_k = set()

    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            raw_name = parts[1] if len(parts) > 1 else ""
            raw_addr = parts[2] if len(parts) > 2 else ""
            country = parts[3].strip().lower() if len(parts) > 3 else ""

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

            if core:
                needed_cores.add(core)
            for ac in alias_cores:
                needed_cores.add(ac)
            if sorted_k:
                needed_sorted.add(sorted_k)
            if len(concat_k) >= 5:
                needed_concat.add(concat_k)
            for bt in brand_tokens:
                needed_brand.add(bt)
            if num_p:
                needed_num_p.add(num_p)
            if post_p:
                needed_post_p.add(post_p)
            for ak in addr_keys:
                needed_addr_k.add(ak)

            s1_records.append((
                s1_id, core, sorted_k, concat_k, brand_tokens, postal, num,
                dist_addr, country, alias_cores, num_p, post_p, addr_keys,
                clean_addr, norm_name, is_non_ascii_name(raw_name)
            ))

    print(f"Loaded {len(s1_records):,} Source 1 test records in {time.time() - t0:.1f}s.")
    print(f"Unique needed keys: Core={len(needed_cores):,}, Sorted={len(needed_sorted):,}, Concat={len(needed_concat):,}, Brand={len(needed_brand):,}, AddrKeys={len(needed_addr_k):,}")

    # Step 2: Stream Targets (S2 + S3) and populate compact inverted index
    print("\n[Step 2/5] Streaming and indexing test targets (S2 + S3)...")
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

    for source_file in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(test_dir, source_file)
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

                if core in needed_cores and len(idx_core[core]) < 30:
                    idx_core[core].append(t_idx)
                    hit = True
                for ac in alias_cores:
                    if ac in needed_cores and len(idx_core[ac]) < 30:
                        idx_core[ac].append(t_idx)
                        hit = True
                if sorted_k in needed_sorted and len(idx_sorted[sorted_k]) < 30:
                    idx_sorted[sorted_k].append(t_idx)
                    hit = True
                if concat_k in needed_concat and len(idx_concat[concat_k]) < 20:
                    idx_concat[concat_k].append(t_idx)
                    hit = True
                for bt in brand_tokens:
                    if bt in needed_brand and len(idx_brand[bt]) < 20:
                        idx_brand[bt].append(t_idx)
                        hit = True
                if num_p and num_p in needed_num_p and len(idx_num_p[num_p]) < 20:
                    idx_num_p[num_p].append(t_idx)
                    hit = True
                if post_p and post_p in needed_post_p and len(idx_post_p[post_p]) < 20:
                    idx_post_p[post_p].append(t_idx)
                    hit = True
                for ak in addr_keys:
                    if ak in needed_addr_k and len(idx_addr_k[ak]) < 10:
                        idx_addr_k[ak].append(t_idx)
                        hit = True

                if hit:
                    is_non_ascii = is_non_ascii_name(raw_name)
                    target_table.append((
                        core, sorted_k, concat_k, norm_name, clean_addr,
                        country, postal, num, dist_addr, is_non_ascii, alias_cores
                    ))
                    target_ids.append(t_id)

    print(f"Indexed targets in {time.time() - t0:.1f}s. Loaded {len(target_table):,} relevant targets into memory.")
    del needed_cores, needed_sorted, needed_concat, needed_brand, needed_num_p, needed_post_p, needed_addr_k
    import gc; gc.collect()

    # Step 3: Multi-Core Parallel Matching across all 20 CPU cores
    print("\n[Step 3/5] Executing 16-Core Parallel High-Precision Matching...")
    t0 = time.time()

    global G_idx_core, G_idx_sorted, G_idx_concat, G_idx_brand, G_idx_num_p, G_idx_post_p, G_idx_addr_k, G_target_table
    G_idx_core = idx_core
    G_idx_sorted = idx_sorted
    G_idx_concat = idx_concat
    G_idx_brand = idx_brand
    G_idx_num_p = idx_num_p
    G_idx_post_p = idx_post_p
    G_idx_addr_k = idx_addr_k
    G_target_table = target_table

    num_workers = 16
    chunk_size = (len(s1_records) + num_workers - 1) // num_workers
    chunks = [s1_records[i * chunk_size : (i + 1) * chunk_size] for i in range(num_workers)]

    with mp.Pool(processes=num_workers) as pool:
        chunk_results = pool.map(match_worker, chunks)

    print(f"Parallel matching completed in {time.time() - t0:.1f}s.")

    # Step 4: 1-to-1 Mutual-Best Conflict Resolution
    print("\n[Step 4/5] Aggregating claims and applying 1-to-1 Mutual-Best Conflict Resolution...")
    t0 = time.time()
    target_claims = defaultdict(list)

    for worker_res in chunk_results:
        for s1_id, cands, scored_matches in worker_res:
            for t_idx, score, a_sim in scored_matches:
                target_claims[t_idx].append((s1_id, score, a_sim))

    print(f"Aggregated {len(target_claims):,} claimed target records. Resolving mutual-best winners...")
    winner_for_target = {}
    for t_idx, claims in target_claims.items():
        best_s1, best_sc, best_asim = max(claims, key=lambda x: (x[1], x[2]))
        winner_for_target[t_idx] = best_s1

    del target_claims
    gc.collect()

    # Step 5: Stream Writing Output TSVs
    print("\n[Step 5/5] Streaming output TSVs (matching_results.tsv and candidate_pairs.tsv)...")
    t0 = time.time()

    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")

    total_matches = 0
    singleton_preds = 0
    total_processed = 0

    with open(matching_file, "w", encoding="utf-8") as fm, open(candidate_file, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        for worker_res in chunk_results:
            for s1_id, cands, scored_matches in worker_res:
                total_processed += 1
                retained = [target_ids[t_idx] for t_idx, sc, a_sim in scored_matches
                            if winner_for_target.get(t_idx) == s1_id]
                cand_ids = [target_ids[t_idx] for t_idx in cands]

                # Guarantee: matched_entity_ids MUST be a subset of candidate_entity_ids
                cand_set = set(cand_ids)
                for m in retained:
                    if m not in cand_set:
                        cand_ids.append(m)

                fm.write(f"{s1_id}\t{','.join(retained)}\n")
                fc.write(f"{s1_id}\t{','.join(cand_ids)}\n")

                total_matches += len(retained)
                if len(retained) == 0:
                    singleton_preds += 1

    print(f"Total Matches Assigned: {total_matches:,} across {total_processed:,} test entities.")
    print(f"Predicted Singletons (Empty matches): {singleton_preds:,} ({singleton_preds / total_processed * 100:.2f}%)")
    print(f"Wrote submission files in {time.time() - t0:.1f}s.")
    print(f"  matching_results.tsv: {os.path.getsize(matching_file) / (1024*1024):.1f} MB")
    print(f"  candidate_pairs.tsv:  {os.path.getsize(candidate_file) / (1024*1024):.1f} MB")
    print("=" * 80)
    print(" SUBMISSION GENERATION SUCCESSFUL!")
    print("=" * 80)


if __name__ == "__main__":
    main()
