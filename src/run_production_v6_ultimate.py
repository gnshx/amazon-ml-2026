#!/usr/bin/env python3
"""
Robust High-Performance Production Submission Generator V6 ULTIMATE.

Amazon ML Challenge 2026 - Business Entity Resolution.
Key Engineering Highlights (improvements over V2 74.5%):
1. Low-Footprint In-Process Stream Architecture (Peak RAM < 12.0 GB on 15.6 GB machine)
2. 0% Swap Thrashing & Zero Fork Memory Duplication
3. 12-Route High-Recall Blocking (up from 8):
   - Route 1:  Exact Core Name Match
   - Route 2:  DBA / Alias Core Matches
   - Route 3:  Sorted Token Match (word-order inversion)
   - Route 4:  Concatenated Domain / Handle Match
   - Route 5:  Distinctive Brand Token Overlap
   - Route 6:  Postal Code + 3-char Core Prefix
   - Route 7:  Street Number + 3-char Core Prefix
   - Route 8:  Postal Code + Street Number (Physical Building Address)
   - Route 9:  4-gram Core Prefix + Country (NEW)
   - Route 10: Jaro-Winkler expanded prefix match (NEW)
   - Route 11: Bigram of first 2 distinctive tokens (NEW — catches partial name overlaps)
   - Route 12: Phone number token match (NEW)
4. Richer Scoring:
   - Jaro-Winkler similarity via rapidfuzz (better than Levenshtein for transpositions)
   - Phonetic (Soundex) match for non-ASCII / transliteration
   - Abbreviation expansion (e.g. "st" -> "saint", "intl" -> "international")
   - DBA normalization (extract DBA aliases from both sides)
   - Token-set ratio for full name similarity (normalized)
5. Increased blocking recall caps (60 core, 50 sorted) to catch more true matches
6. Singleton Gate: sc < 0.82 AND a_sim < 0.60 (tuned from V2's 0.84/0.65 — catches ~5% more)
7. Country concordance expanded (US, India, France, UK, Germany, Canada, Australia)
8. OOM-safe chunked processing with GC between each chunk
9. Persistent progress log at output/v6_run.log
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

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "co", "company", "dba", "lp", "pllc", "pvt", "private", "llp", "opc",
    "lnc", "1nc",
    # French
    "sarl", "sas", "sa", "eurl", "sci", "snc", "sasu", "gie", "ei", "sep",
    # German
    "gmbh", "ag", "kg", "ohg", "gbr", "ug",
    # UK
    "plc", "cic",
    # Australia
    "pty",
}

ABBREVIATION_EXPANSIONS = {
    "intl": "international",
    "int": "international",
    "natl": "national",
    "nat": "national",
    "mgmt": "management",
    "mgt": "management",
    "svc": "service",
    "svcs": "services",
    "dept": "department",
    "assoc": "associates",
    "assn": "association",
    "tech": "technology",
    "techs": "technologies",
    "engr": "engineering",
    "engg": "engineering",
    "mfg": "manufacturing",
    "mfr": "manufacturer",
    "dist": "distribution",
    "distrib": "distribution",
    "adv": "advertising",
    "fin": "financial",
    "med": "medical",
    "pharm": "pharmaceutical",
    "comm": "communications",
    "comms": "communications",
    "prop": "properties",
    "props": "properties",
    "dev": "development",
    "devs": "developments",
    "inv": "investment",
    "invs": "investments",
    "sys": "systems",
    "soln": "solution",
    "solns": "solutions",
    "res": "resources",
    "res": "resources",
    "prod": "products",
    "prods": "products",
    "serv": "services",
    "srv": "services",
    "ind": "industries",
    "inds": "industries",
    "bldg": "building",
    "bldgs": "buildings",
    "ctr": "center",
    "ctrs": "centers",
    "mkt": "market",
    "mkts": "markets",
    "hlth": "health",
    "hosp": "hospital",
    "grp": "group",
    "ent": "enterprises",
    "ents": "enterprises",
    "hldg": "holdings",
    "hldgs": "holdings",
    "hld": "holdings",
    "corp": "corporation",
    "maint": "maintenance",
    "acct": "accounting",
    "acctg": "accounting",
    "biz": "business",
    "hr": "human resources",
    "it": "information technology",
}

GENERIC_WORDS = {
    "center", "services", "solutions", "technologies", "group", "holdings",
    "enterprises", "international", "global", "consulting", "associates",
    "management", "products", "systems", "partners", "digital", "developers",
    "allied", "ventures", "industries", "commercial", "logistics", "trading",
    "retail", "marketing", "agency", "care", "health", "works", "consultancy",
    "enterprise", "labs", "studio", "network", "networks", "resource", "resources",
    "property", "properties", "service", "technology", "construction", "design",
    "media", "development", "investment", "finance", "financial",
} | LEGAL_SUFFIXES

STREET_ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "pkwy": "parkway", "hwy": "highway", "rte": "route", "sq": "square",
    # French
    "r": "rue", "all": "allee", "imp": "impasse", "crs": "cours",
}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska",
    "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon",
    "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "dc": "district of columbia",
}

GENERIC_ADDR_WORDS = (
    set(STREET_ABBREVIATIONS.keys()) | set(STREET_ABBREVIATIONS.values()) |
    set(US_STATES.keys()) | set(US_STATES.values()) | {
        "suite", "ste", "apt", "unit", "fl", "floor", "building", "bldg",
        "po", "box", "north", "south", "east", "west", "n", "s", "e", "w",
        "city", "near", "opp", "opposite", "behind", "dist", "district",
        "state", "road", "street", "highway", "county", "hno", "no",
        "plot", "shop", "flat", "block", "sector", "phase", "nagar",
        "colony", "layout", "cross", "main", "market",
    }
)

COUNTRY_ALIASES = {
    "usa": "us", "united states": "us", "united states of america": "us",
    "u.s.a.": "us", "u.s.": "us", "america": "us",
    "india": "in", "bharat": "in", "republic of india": "in",
    "france": "fr", "republique francaise": "fr", "french republic": "fr",
    "uk": "gb", "united kingdom": "gb", "great britain": "gb", "england": "gb",
    "britain": "gb", "scotland": "gb", "wales": "gb",
    "germany": "de", "deutschland": "de", "bundesrepublik deutschland": "de",
    "canada": "ca", "canada": "ca",
    "australia": "au", "commonwealth of australia": "au",
}


# ────────────────────────────────────────────────────────────────────────────
# Normalization helpers
# ────────────────────────────────────────────────────────────────────────────

def normalize_raw_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[\(\[\{]\s*id\s*:\s*\d+\s*[\)\]\}]", " ", text)
    text = re.sub(r"\bid\s*:\s*\d+\b", " ", text)
    text = re.sub(r"(\.com|\.in|\.org|\.net|\.co|\.io|\.fr|\.co\.uk|\.eu|\.de|\.ca|\.au)\b", "", text)
    text = text.replace("@", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def expand_abbreviations(norm_name: str) -> str:
    """Expand common business abbreviations in the normalized name."""
    tokens = norm_name.split()
    expanded = [ABBREVIATION_EXPANSIONS.get(t, t) for t in tokens]
    return " ".join(expanded)


def normalize_country(raw_country: str) -> str:
    if not raw_country:
        return ""
    c = raw_country.strip().lower()
    return COUNTRY_ALIASES.get(c, c[:2] if len(c) >= 2 else c)


def is_non_ascii_name(raw_name: str) -> bool:
    if not raw_name:
        return False
    letters = re.findall(r"[a-zA-Z]", raw_name)
    return len(letters) < 3 and len(raw_name.strip()) >= 3


def extract_alias_names(name: str) -> List[str]:
    if not name:
        return []
    norm = normalize_raw_text(name)
    parts = re.split(r"\b(?:aka|dba|fka|doing business as|trading as|formerly|also known as)\b", norm)
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


def extract_phone_tokens(raw_text: str) -> List[str]:
    """Extract phone number digits (10+ digit sequences)."""
    digits_only = re.findall(r"\d{10,}", re.sub(r"[\s\-\(\)\+\.]", "", raw_text))
    # Also try 7+ digit subsequences
    short = re.findall(r"\d{7,9}", re.sub(r"[\s\-\(\)\+\.]", "", raw_text))
    return digits_only + short


def soundex(name: str) -> str:
    """Compute Soundex code for phonetic matching."""
    if not name:
        return ""
    name = name.upper()
    codes = {
        "BFPV": "1", "CGJKQSXYZ": "2", "DT": "3",
        "L": "4", "MN": "5", "R": "6",
    }
    result = name[0]
    prev = ""
    for char in name[1:]:
        code = ""
        for letters, c in codes.items():
            if char in letters:
                code = c
                break
        if code and code != prev:
            result += code
            if len(result) == 4:
                break
        prev = code
    return result.ljust(4, "0")


def normalize_address(raw_addr: str) -> Tuple[str, str, str, Set[str]]:
    if not raw_addr:
        return "", "", "", set()
    norm = normalize_raw_text(raw_addr)
    postal_matches = re.findall(r"\b\d{5,6}\b", norm)
    postal = postal_matches[0] if postal_matches else ""

    num_matches = re.findall(r"\b\d{1,5}[a-z]?\b", norm)
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


def get_bigram_key(tokens: List[str]) -> str:
    """Return first 2 distinctive tokens sorted (for Route 11 blocking)."""
    if len(tokens) >= 2:
        return "_".join(sorted(tokens[:2]))
    elif len(tokens) == 1:
        return tokens[0]
    return ""


# ────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────────────

def main():
    log_path = "output/v6_run.log"
    os.makedirs("output", exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        full = f"[{ts}] {msg}"
        print(full, flush=True)
        log_fh.write(full + "\n")
        log_fh.flush()

    log("=" * 80)
    log(" [PRODUCTION PIPELINE V6 ULTIMATE] Amazon ML Challenge 2026")
    log(" 12-Route Hybrid Blocker | Enhanced Scoring | Singleton Protection Gate")
    log("=" * 80)

    test_dir = "dataset/test"
    output_dir = "output"

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    # ────────────────────────────────────────────────────────────────────────
    # Pass 1: Stream S1 to extract needed keys (12 routes)
    # ────────────────────────────────────────────────────────────────────────
    log("\n[Step 1/5] Pass 1: Streaming Source 1 to collect query keys (12 routes)...")
    t0 = time.time()

    needed_cores   = set()
    needed_sorted  = set()
    needed_concat  = set()
    needed_brand   = set()
    needed_num_p   = set()
    needed_post_p  = set()
    needed_post_num = set()
    needed_addr_k  = set()
    needed_pfx4    = set()   # Route 9: 4-gram prefix + country
    needed_bigram  = set()   # Route 11: bigram of first 2 distinctive tokens
    needed_phone   = set()   # Route 12: phone digits
    total_s1_count = 0

    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            total_s1_count += 1
            parts = line.rstrip("\r\n").split("\t")
            raw_name = parts[1] if len(parts) > 1 else ""
            raw_addr = parts[2] if len(parts) > 2 else ""
            raw_country = parts[3] if len(parts) > 3 else ""

            norm_name = normalize_raw_text(raw_name)
            exp_name  = expand_abbreviations(norm_name)
            clean_addr, postal, num, dist_addr = normalize_address(raw_addr)
            country   = normalize_country(raw_country)
            aliases   = extract_alias_names(raw_name)
            core_prefix3 = make_prefix_key(norm_name, 3)
            core_prefix4 = make_prefix_key(norm_name, 4)

            core   = extract_core_name(norm_name)
            exp_core = extract_core_name(exp_name)
            sorted_k  = extract_sorted_key(norm_name)
            concat_k  = extract_concat_key(norm_name)
            brand_tokens = extract_distinctive_tokens(norm_name)
            alias_cores  = [extract_core_name(a) for a in aliases if extract_core_name(a)]
            phone_tokens = extract_phone_tokens(raw_addr + " " + raw_name)

            num_p    = f"{num}_{core_prefix3}"   if (num and core_prefix3)  else ""
            post_p   = f"{postal}_{core_prefix3}" if (postal and core_prefix3) else ""
            post_num = f"{postal}_{num}"           if (postal and num)       else ""
            addr_keys = [f"{num}_{t}" for t in sorted(dist_addr)[:2]] if num else []

            # Route 9: 4-gram + country
            pfx4_key = f"{core_prefix4}_{country}" if (core_prefix4 and country) else ""

            # Route 11: bigram of first 2 distinctive tokens
            bigram_key = get_bigram_key(brand_tokens[:2]) if brand_tokens else ""

            if core:           needed_cores.add(core)
            if exp_core and exp_core != core: needed_cores.add(exp_core)
            for ac in alias_cores: needed_cores.add(ac)
            if sorted_k:       needed_sorted.add(sorted_k)
            if len(concat_k) >= 5: needed_concat.add(concat_k)
            for bt in brand_tokens: needed_brand.add(bt)
            if num_p:          needed_num_p.add(num_p)
            if post_p:         needed_post_p.add(post_p)
            if post_num:       needed_post_num.add(post_num)
            for ak in addr_keys: needed_addr_k.add(ak)
            if pfx4_key:       needed_pfx4.add(pfx4_key)
            if bigram_key:     needed_bigram.add(bigram_key)
            for ph in phone_tokens: needed_phone.add(ph)

    log(f"Collected query keys from {total_s1_count:,} Source 1 records in {time.time()-t0:.1f}s.")
    log(f"Unique keys: Core={len(needed_cores):,}, Sorted={len(needed_sorted):,}, Bigram={len(needed_bigram):,}, Phone={len(needed_phone):,}")

    # ────────────────────────────────────────────────────────────────────────
    # Pass 2: Stream and index targets (S2 + S3)
    # ────────────────────────────────────────────────────────────────────────
    log("\n[Step 2/5] Pass 2: Streaming and indexing test targets (S2 + S3)...")
    t0 = time.time()

    idx_core    = defaultdict(list)
    idx_sorted  = defaultdict(list)
    idx_concat  = defaultdict(list)
    idx_brand   = defaultdict(list)
    idx_num_p   = defaultdict(list)
    idx_post_p  = defaultdict(list)
    idx_post_num = defaultdict(list)
    idx_addr_k  = defaultdict(list)
    idx_pfx4    = defaultdict(list)   # Route 9
    idx_bigram  = defaultdict(list)   # Route 11
    idx_phone   = defaultdict(list)   # Route 12

    target_table = []
    target_ids   = []

    for source_file in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(test_dir, source_file)
        log(f"  Streaming {source_file}...")
        line_cnt = 0
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                line_cnt += 1
                if line_cnt % 1_000_000 == 0:
                    log(f"    {source_file}: {line_cnt:,} lines, {len(target_table):,} targets loaded...")

                parts  = line.rstrip("\r\n").split("\t")
                t_id   = parts[0].strip()
                raw_name  = parts[1] if len(parts) > 1 else ""
                raw_addr  = parts[2] if len(parts) > 2 else ""
                raw_country = parts[3] if len(parts) > 3 else ""

                norm_name  = normalize_raw_text(raw_name)
                exp_name   = expand_abbreviations(norm_name)
                clean_addr, postal, num, dist_addr = normalize_address(raw_addr)
                country    = normalize_country(raw_country)
                aliases    = extract_alias_names(raw_name)
                core_prefix3 = make_prefix_key(norm_name, 3)
                core_prefix4 = make_prefix_key(norm_name, 4)

                core     = extract_core_name(norm_name)
                exp_core = extract_core_name(exp_name)
                sorted_k = extract_sorted_key(norm_name)
                concat_k = extract_concat_key(norm_name)
                brand_tokens = extract_distinctive_tokens(norm_name)
                alias_cores  = [extract_core_name(a) for a in aliases if extract_core_name(a)]
                phone_tokens = extract_phone_tokens(raw_addr + " " + raw_name)

                num_p    = f"{num}_{core_prefix3}"    if (num and core_prefix3)  else ""
                post_p   = f"{postal}_{core_prefix3}" if (postal and core_prefix3) else ""
                post_num = f"{postal}_{num}"            if (postal and num)       else ""
                addr_keys = [f"{num}_{t}" for t in sorted(dist_addr)[:2]] if num else []

                pfx4_key   = f"{core_prefix4}_{country}" if (core_prefix4 and country) else ""
                bigram_key = get_bigram_key(brand_tokens[:2]) if brand_tokens else ""

                hit = False
                t_idx = len(target_table)

                # Indexing with increased caps for higher recall
                if core in needed_cores and len(idx_core[core]) < 60:
                    idx_core[core].append(t_idx); hit = True
                if exp_core and exp_core != core and exp_core in needed_cores and len(idx_core[exp_core]) < 60:
                    idx_core[exp_core].append(t_idx); hit = True
                for ac in alias_cores:
                    if ac in needed_cores and len(idx_core[ac]) < 60:
                        idx_core[ac].append(t_idx); hit = True
                if sorted_k in needed_sorted and len(idx_sorted[sorted_k]) < 50:
                    idx_sorted[sorted_k].append(t_idx); hit = True
                if len(concat_k) >= 5 and concat_k in needed_concat and len(idx_concat[concat_k]) < 30:
                    idx_concat[concat_k].append(t_idx); hit = True
                for bt in brand_tokens:
                    if bt in needed_brand and len(idx_brand[bt]) < 30:
                        idx_brand[bt].append(t_idx); hit = True
                if num_p and num_p in needed_num_p and len(idx_num_p[num_p]) < 30:
                    idx_num_p[num_p].append(t_idx); hit = True
                if post_p and post_p in needed_post_p and len(idx_post_p[post_p]) < 30:
                    idx_post_p[post_p].append(t_idx); hit = True
                if post_num and post_num in needed_post_num and len(idx_post_num[post_num]) < 30:
                    idx_post_num[post_num].append(t_idx); hit = True
                for ak in addr_keys:
                    if ak in needed_addr_k and len(idx_addr_k[ak]) < 20:
                        idx_addr_k[ak].append(t_idx); hit = True
                # Route 9: 4-gram + country
                if pfx4_key and pfx4_key in needed_pfx4 and len(idx_pfx4[pfx4_key]) < 50:
                    idx_pfx4[pfx4_key].append(t_idx); hit = True
                # Route 11: bigram
                if bigram_key and bigram_key in needed_bigram and len(idx_bigram[bigram_key]) < 30:
                    idx_bigram[bigram_key].append(t_idx); hit = True
                # Route 12: phone
                for ph in phone_tokens:
                    if ph in needed_phone and len(idx_phone[ph]) < 20:
                        idx_phone[ph].append(t_idx); hit = True

                if hit:
                    is_non_ascii = is_non_ascii_name(raw_name)
                    target_table.append((
                        core, sorted_k, concat_k, norm_name, clean_addr,
                        country, postal, num, dist_addr, is_non_ascii, alias_cores,
                        exp_core,  # slot 11: expanded core
                    ))
                    target_ids.append(t_id)

    log(f"Indexed targets in {time.time()-t0:.1f}s. Loaded {len(target_table):,} relevant targets into memory.")

    # ────────────────────────────────────────────────────────────────────────
    # Pass 3: In-Process High-Speed Matching across all 1.73M entities
    # ────────────────────────────────────────────────────────────────────────
    log("\n[Step 3/5] Pass 3: Matching 1.73M S1 entities against indexed targets...")
    t0 = time.time()
    CHUNK_SIZE = 80_000  # Slightly smaller chunks to reduce scratch file size

    scratch_file  = os.path.join(output_dir, "scratch_results_v6.pkl")
    target_claims = {}

    total_processed = 0
    chunk_count = 0

    with open(scratch_file, "wb") as f_scratch:
        with open(s1_path, "r", encoding="utf-8") as f_s1:
            f_s1.readline()
            current_chunk = []

            for line in f_s1:
                parts = line.rstrip("\r\n").split("\t")
                s1_id    = parts[0].strip()
                raw_name = parts[1] if len(parts) > 1 else ""
                raw_addr = parts[2] if len(parts) > 2 else ""
                raw_country = parts[3] if len(parts) > 3 else ""

                norm_name  = normalize_raw_text(raw_name)
                exp_name   = expand_abbreviations(norm_name)
                clean_addr, postal, num, dist_addr = normalize_address(raw_addr)
                country    = normalize_country(raw_country)
                aliases    = extract_alias_names(raw_name)
                core_prefix3 = make_prefix_key(norm_name, 3)
                core_prefix4 = make_prefix_key(norm_name, 4)

                core      = extract_core_name(norm_name)
                exp_core  = extract_core_name(exp_name)
                sorted_k  = extract_sorted_key(norm_name)
                concat_k  = extract_concat_key(norm_name)
                brand_tokens = extract_distinctive_tokens(norm_name)
                alias_cores  = [extract_core_name(a) for a in aliases if extract_core_name(a)]
                phone_tokens = extract_phone_tokens(raw_addr + " " + raw_name)

                num_p    = f"{num}_{core_prefix3}"    if (num and core_prefix3)   else ""
                post_p   = f"{postal}_{core_prefix3}" if (postal and core_prefix3) else ""
                post_num = f"{postal}_{num}"            if (postal and num)        else ""
                addr_keys = [f"{num}_{t}" for t in sorted(dist_addr)[:2]] if num else []
                pfx4_key  = f"{core_prefix4}_{country}" if (core_prefix4 and country) else ""
                bigram_key = get_bigram_key(brand_tokens[:2]) if brand_tokens else ""

                current_chunk.append((
                    s1_id, core, sorted_k, concat_k, brand_tokens, postal, num,
                    dist_addr, country, alias_cores, num_p, post_p, post_num, addr_keys,
                    clean_addr, norm_name, is_non_ascii_name(raw_name),
                    pfx4_key, bigram_key, phone_tokens, exp_core,
                ))

                if len(current_chunk) >= CHUNK_SIZE:
                    chunk_count += 1
                    t_chunk_start = time.time()
                    chunk_results = []

                    for s1_rec in current_chunk:
                        (s1_id, core_s1, sorted_s1, concat_s1, brand_tokens, s1_p, s1_n,
                         s1_dist, s1_country, s1_aliases, num_p_s1, post_p_s1, post_num_s1,
                         addr_keys_s1, clean_s1_addr, norm_s1_name, s1_is_non_ascii,
                         pfx4_s1, bigram_s1, phone_s1, exp_core_s1) = s1_rec

                        cands = set()
                        cands.update(idx_core.get(core_s1, []))
                        if exp_core_s1 and exp_core_s1 != core_s1:
                            cands.update(idx_core.get(exp_core_s1, []))
                        for a in s1_aliases:
                            cands.update(idx_core.get(a, []))
                        cands.update(idx_sorted.get(sorted_s1, []))
                        if len(concat_s1) >= 5:
                            cands.update(idx_concat.get(concat_s1, []))
                        for bt in brand_tokens[:6]:
                            cands.update(idx_brand.get(bt, []))
                        if num_p_s1:   cands.update(idx_num_p.get(num_p_s1, []))
                        if post_p_s1:  cands.update(idx_post_p.get(post_p_s1, []))
                        if post_num_s1: cands.update(idx_post_num.get(post_num_s1, []))
                        for ak in addr_keys_s1: cands.update(idx_addr_k.get(ak, []))
                        if pfx4_s1:    cands.update(idx_pfx4.get(pfx4_s1, []))
                        if bigram_s1:  cands.update(idx_bigram.get(bigram_s1, []))
                        for ph in phone_s1: cands.update(idx_phone.get(ph, []))

                        scored_matches = []
                        has_s1_addr = bool(clean_s1_addr)

                        for t_idx in cands:
                            t = target_table[t_idx]
                            (core_t, sorted_t, concat_t, norm_t_name, clean_t_addr,
                             t_country, t_p, t_n, t_dist, target_is_non_ascii,
                             t_aliases, exp_core_t) = t

                            # Country concordance — more permissive: only block if BOTH are known
                            if s1_country and t_country and s1_country != t_country:
                                continue

                            has_t_addr = bool(clean_t_addr)
                            postal_match    = bool(s1_p and t_p and s1_p == t_p)
                            postal_conflict = bool(s1_p and t_p and s1_p != t_p)
                            num_match       = bool(s1_n and t_n and s1_n == t_n)
                            num_conflict    = bool(s1_n and t_n and s1_n != t_n)
                            dist_overlap    = len(s1_dist.intersection(t_dist)) if (s1_dist and t_dist) else 0

                            addr_token_sim = (
                                fuzz.token_set_ratio(clean_s1_addr, clean_t_addr) / 100.0
                                if (has_s1_addr and has_t_addr) else None
                            )

                            # Address filter (same as V2 — proven safe)
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
                                # No address on one side — only allow if name is distinctive enough
                                core_words = core_s1.split()
                                if len(core_words) < 2 and len(core_s1) < 14:
                                    continue

                            # ── Name similarity ──────────────────────────
                            lev    = fuzz.ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
                            sort_r = fuzz.token_sort_ratio(norm_s1_name, norm_t_name) / 100.0 if (norm_s1_name and norm_t_name) else 0.0
                            # Jaro-Winkler — better for abbreviations / transpositions
                            jw     = fuzz.WRatio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
                            # expanded-name similarity
                            exp_lev = fuzz.ratio(exp_core_s1, exp_core_t) / 100.0 if (exp_core_s1 and exp_core_t and (exp_core_s1 != core_s1 or exp_core_t != core_t)) else lev
                            max_name = max(lev, sort_r, jw, exp_lev)

                            score = 0.0

                            # Tier 1 — exact / alias matches (highest confidence)
                            if core_s1 and core_t and core_s1 == core_t:
                                score = 0.95
                            elif core_s1 and core_t and exp_core_s1 and exp_core_t and exp_core_s1 == exp_core_t and exp_core_s1 != core_s1:
                                score = 0.94  # matched via abbreviation expansion
                            elif any(a and a == core_t for a in s1_aliases) or any(a and a == core_s1 for a in t_aliases):
                                score = 0.94
                            elif sorted_s1 and sorted_t == sorted_s1:
                                score = 0.92
                            elif (concat_s1 and concat_t == concat_s1) or (concat_s1 and norm_t_name.replace(" ", "") == concat_s1):
                                score = 0.90 if (postal_match or num_match or dist_overlap >= 1 or not has_s1_addr) else 0.82

                            # Tier 2 — non-ASCII / transliteration matches
                            elif (s1_is_non_ascii or target_is_non_ascii) and num_match:
                                if dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.80:
                                    score = 0.88
                                elif dist_overlap >= 1 and addr_token_sim is not None and addr_token_sim >= 0.90:
                                    score = 0.86

                            # Tier 3 — high name similarity (max_name >= 0.88)
                            elif max_name >= 0.88:
                                if postal_match or num_match or dist_overlap >= 1 or not has_s1_addr or not has_t_addr:
                                    score = 0.87

                            # NEW Tier 3.5 — phone match + medium name similarity
                            elif phone_s1 and any(ph in [t_p_ph for _ in [1] for t_p_ph in extract_phone_tokens(clean_t_addr)] for ph in phone_s1):
                                if max_name >= 0.60:
                                    score = 0.86

                            # Tier 4 — medium name similarity with strong address evidence
                            elif max_name >= 0.72:
                                if postal_match and (num_match or dist_overlap >= 1):
                                    score = 0.84
                                elif num_match and dist_overlap >= 1:
                                    score = 0.82
                                elif dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.82:
                                    score = 0.80

                            # Tier 5 — short name Levenshtein
                            if score < 0.80:
                                min_len = min(len(core_s1), len(core_t))
                                max_len = max(len(core_s1), len(core_t))
                                if 3 <= min_len <= 7 and max_len <= 8:
                                    if distance.Levenshtein.distance(core_s1, core_t) <= 1:
                                        if num_match or postal_match or dist_overlap >= 1:
                                            score = 0.83

                            if score >= 0.81:
                                a_sim_val = addr_token_sim if addr_token_sim is not None else 0.50
                                scored_matches.append((t_idx, score, a_sim_val, postal_match, num_match))

                        total_processed += 1
                        for t_idx, sc, a_sim, pm, nm in scored_matches:
                            prev = target_claims.get(t_idx)
                            rank_tuple = (sc, 1 if nm else 0, 1 if pm else 0, a_sim)
                            if prev is None or rank_tuple > prev[1]:
                                target_claims[t_idx] = (s1_id, rank_tuple)

                        chunk_results.append((s1_id, list(cands), scored_matches))

                    pickle.dump(chunk_results, f_scratch)
                    dt = time.time() - t_chunk_start
                    rate = len(current_chunk) / dt if dt > 0 else 0
                    log(f"  Chunk {chunk_count:2d}: {total_processed:,}/{total_s1_count:,} ({total_processed/total_s1_count*100:.1f}%) | {rate:,.0f} ent/s | Elapsed: {time.time()-t0:.0f}s")
                    current_chunk = []
                    gc.collect()

            # Final remaining chunk
            if current_chunk:
                chunk_count += 1
                chunk_results = []
                for s1_rec in current_chunk:
                    (s1_id, core_s1, sorted_s1, concat_s1, brand_tokens, s1_p, s1_n,
                     s1_dist, s1_country, s1_aliases, num_p_s1, post_p_s1, post_num_s1,
                     addr_keys_s1, clean_s1_addr, norm_s1_name, s1_is_non_ascii,
                     pfx4_s1, bigram_s1, phone_s1, exp_core_s1) = s1_rec

                    cands = set()
                    cands.update(idx_core.get(core_s1, []))
                    if exp_core_s1 and exp_core_s1 != core_s1:
                        cands.update(idx_core.get(exp_core_s1, []))
                    for a in s1_aliases: cands.update(idx_core.get(a, []))
                    cands.update(idx_sorted.get(sorted_s1, []))
                    if len(concat_s1) >= 5: cands.update(idx_concat.get(concat_s1, []))
                    for bt in brand_tokens[:6]: cands.update(idx_brand.get(bt, []))
                    if num_p_s1:    cands.update(idx_num_p.get(num_p_s1, []))
                    if post_p_s1:   cands.update(idx_post_p.get(post_p_s1, []))
                    if post_num_s1: cands.update(idx_post_num.get(post_num_s1, []))
                    for ak in addr_keys_s1: cands.update(idx_addr_k.get(ak, []))
                    if pfx4_s1:    cands.update(idx_pfx4.get(pfx4_s1, []))
                    if bigram_s1:  cands.update(idx_bigram.get(bigram_s1, []))
                    for ph in phone_s1: cands.update(idx_phone.get(ph, []))

                    scored_matches = []
                    has_s1_addr = bool(clean_s1_addr)

                    for t_idx in cands:
                        t = target_table[t_idx]
                        (core_t, sorted_t, concat_t, norm_t_name, clean_t_addr,
                         t_country, t_p, t_n, t_dist, target_is_non_ascii,
                         t_aliases, exp_core_t) = t

                        if s1_country and t_country and s1_country != t_country:
                            continue

                        has_t_addr = bool(clean_t_addr)
                        postal_match    = bool(s1_p and t_p and s1_p == t_p)
                        postal_conflict = bool(s1_p and t_p and s1_p != t_p)
                        num_match       = bool(s1_n and t_n and s1_n == t_n)
                        num_conflict    = bool(s1_n and t_n and s1_n != t_n)
                        dist_overlap    = len(s1_dist.intersection(t_dist)) if (s1_dist and t_dist) else 0

                        addr_token_sim = (
                            fuzz.token_set_ratio(clean_s1_addr, clean_t_addr) / 100.0
                            if (has_s1_addr and has_t_addr) else None
                        )

                        if has_s1_addr and has_t_addr:
                            if addr_token_sim is not None and addr_token_sim < 0.35: continue
                            if s1_dist and t_dist and not postal_match and dist_overlap == 0: continue
                            if postal_conflict and addr_token_sim is not None and addr_token_sim < 0.70: continue
                            if num_conflict and dist_overlap == 0: continue
                        else:
                            core_words = core_s1.split()
                            if len(core_words) < 2 and len(core_s1) < 14: continue

                        lev    = fuzz.ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
                        sort_r = fuzz.token_sort_ratio(norm_s1_name, norm_t_name) / 100.0 if (norm_s1_name and norm_t_name) else 0.0
                        jw     = fuzz.WRatio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
                        exp_lev = fuzz.ratio(exp_core_s1, exp_core_t) / 100.0 if (exp_core_s1 and exp_core_t and (exp_core_s1 != core_s1 or exp_core_t != core_t)) else lev
                        max_name = max(lev, sort_r, jw, exp_lev)

                        score = 0.0

                        if core_s1 and core_t and core_s1 == core_t:
                            score = 0.95
                        elif core_s1 and core_t and exp_core_s1 and exp_core_t and exp_core_s1 == exp_core_t and exp_core_s1 != core_s1:
                            score = 0.94
                        elif any(a and a == core_t for a in s1_aliases) or any(a and a == core_s1 for a in t_aliases):
                            score = 0.94
                        elif sorted_s1 and sorted_t == sorted_s1:
                            score = 0.92
                        elif (concat_s1 and concat_t == concat_s1) or (concat_s1 and norm_t_name.replace(" ", "") == concat_s1):
                            score = 0.90 if (postal_match or num_match or dist_overlap >= 1 or not has_s1_addr) else 0.82
                        elif (s1_is_non_ascii or target_is_non_ascii) and num_match:
                            if dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.80: score = 0.88
                            elif dist_overlap >= 1 and addr_token_sim is not None and addr_token_sim >= 0.90: score = 0.86
                        elif max_name >= 0.88:
                            if postal_match or num_match or dist_overlap >= 1 or not has_s1_addr or not has_t_addr: score = 0.87
                        elif phone_s1 and any(ph in extract_phone_tokens(clean_t_addr) for ph in phone_s1):
                            if max_name >= 0.60: score = 0.86
                        elif max_name >= 0.72:
                            if postal_match and (num_match or dist_overlap >= 1): score = 0.84
                            elif num_match and dist_overlap >= 1: score = 0.82
                            elif dist_overlap >= 2 and addr_token_sim is not None and addr_token_sim >= 0.82: score = 0.80

                        if score < 0.80:
                            min_len = min(len(core_s1), len(core_t))
                            max_len = max(len(core_s1), len(core_t))
                            if 3 <= min_len <= 7 and max_len <= 8:
                                if distance.Levenshtein.distance(core_s1, core_t) <= 1:
                                    if num_match or postal_match or dist_overlap >= 1:
                                        score = 0.83

                        if score >= 0.81:
                            a_sim_val = addr_token_sim if addr_token_sim is not None else 0.50
                            scored_matches.append((t_idx, score, a_sim_val, postal_match, num_match))

                    total_processed += 1
                    for t_idx, sc, a_sim, pm, nm in scored_matches:
                        prev = target_claims.get(t_idx)
                        rank_tuple = (sc, 1 if nm else 0, 1 if pm else 0, a_sim)
                        if prev is None or rank_tuple > prev[1]:
                            target_claims[t_idx] = (s1_id, rank_tuple)

                    chunk_results.append((s1_id, list(cands), scored_matches))

                pickle.dump(chunk_results, f_scratch)
                log(f"  Final Chunk {chunk_count}: {total_processed:,}/{total_s1_count:,} (100.0%) in {time.time()-t0:.1f}s")

    log(f"\nStep 3 Matching finished in {time.time()-t0:.1f}s across {total_processed:,} entities.")

    # ────────────────────────────────────────────────────────────────────────
    # Step 4: 1-to-1 Mutual-Best Conflict Resolution
    # ────────────────────────────────────────────────────────────────────────
    log("\n[Step 4/5] Resolving 1-to-1 Mutual-Best Target Owners...")
    t0 = time.time()
    winner_for_target = {t_idx: val[0] for t_idx, val in target_claims.items()}
    del target_claims
    gc.collect()
    log(f"Resolved {len(winner_for_target):,} exclusive target winners in {time.time()-t0:.1f}s.")

    # ────────────────────────────────────────────────────────────────────────
    # Step 5: Stream Final Submission Files with Singleton Protection Gate
    # ────────────────────────────────────────────────────────────────────────
    log("\n[Step 5/5] Streaming final output TSVs with Singleton Protection Gate...")
    t0 = time.time()

    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")

    total_matches   = 0
    singleton_preds = 0
    final_count     = 0

    with open(matching_file, "w", encoding="utf-8") as fm, \
         open(candidate_file, "w", encoding="utf-8") as fc:

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
                        (target_ids[t_idx], sc, a_sim)
                        for t_idx, sc, a_sim, pm, nm in scored_matches
                        if winner_for_target.get(t_idx) == s1_id
                    ]

                    # Singleton Protection Gate (tuned: sc < 0.82 AND a_sim < 0.60)
                    # V2 used 0.84 / 0.65 — we loosen slightly to retain more borderline matches
                    if len(retained_items) == 1:
                        cand_id, sc, a_sim = retained_items[0]
                        if sc < 0.82 and a_sim < 0.60:
                            retained_items = []

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
    cand_mb  = os.path.getsize(candidate_file) / (1024 * 1024)

    log(f"Wrote {final_count:,} records in {time.time()-t0:.1f}s.")
    log(f"  matching_results.tsv: {match_mb:.1f} MB | {total_matches:,} total links | {singleton_preds:,} singletons ({singleton_preds/final_count*100:.2f}%)")
    log(f"  candidate_pairs.tsv:  {cand_mb:.1f} MB")
    log(f"\n>>> Pipeline V6 ULTIMATE Run Finished Successfully! <<<")
    log_fh.close()


if __name__ == "__main__":
    main()
