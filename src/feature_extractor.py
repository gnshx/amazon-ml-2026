#!/usr/bin/env python3
"""Pairwise Feature Extractor for GBDT & Candidate Scoring.

Extracts robust, domain-invariant pairwise features for candidate pairs (Source 1, Candidate):
- Name Similarity: Token sort, Token set, Levenshtein, Jaro-Winkler, Char n-grams, IDF rarity
- Address Similarity: Normalized token overlap, street number match/conflict, postal match/conflict
- Explicit Missingness Handling: Separate flags for missing addresses/names so tree branches isolate missing data
- Strict Number Disambiguation: Isolates postal codes (5-6 digits) from building/street numbers
- Open-Set Multi-Country Normalization: Handles US, India, and France legal suffixes, accents, and road types
- Script & Entity Signals: Non-Latin Indic script flags, candidate rank, source indicators (S2 vs S3)
"""

from collections import defaultdict
import math
import re
from typing import Dict, List, Set, Tuple
import unicodedata

try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# Comprehensive legal suffixes: US, UK, India, France
LEGAL_SUFFIXES = {
    # US / UK / Common
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited", "co", "company", "dba", "lp", "pllc",
    # India
    "pvt", "private", "llp", "opc",
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
    "r": "rue", "all": "allee", "imp": "impasse"
}

GENERIC_ADDR_WORDS = {
    "suite", "ste", "apt", "unit", "fl", "floor", "building", "bldg", "po", "box",
    "north", "south", "east", "west", "n", "s", "e", "w",
    "city", "near", "opp", "opposite", "behind", "dist", "district", "state", "road", "street",
    "highway", "county", "hno", "no", "plot", "shop", "flat"
} | set(STREET_ABBREVIATIONS.keys()) | set(STREET_ABBREVIATIONS.values())


RE_PAREN_ID = re.compile(r'[\(\[\{]\s*id\s*:\s*\d+\s*[\)\]\}]')
RE_WORD_ID = re.compile(r'\bid\s*:\s*\d+\b')
RE_DOMAIN = re.compile(r'(\.com|\.in|\.org|\.net|\.co|\.io|\.fr|\.co\.uk|\.eu)\b')
RE_NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')
RE_WHITESPACE = re.compile(r'\s+')
RE_LETTERS = re.compile(r'[a-zA-Z]')
RE_POSTAL = re.compile(r'\b(\d{5,6})\b')
RE_STREET_NUM = re.compile(r'\b0*(\d{1,6})\b')


def normalize_text(text: str) -> str:
    """Accent-folds, lowercases, and strips punctuation and domain extensions."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower().replace("&", " and ")
    text = RE_PAREN_ID.sub(' ', text)
    text = RE_WORD_ID.sub(' ', text)
    text = RE_DOMAIN.sub('', text)
    text = text.replace("@", " ")
    text = RE_NON_ALPHANUM.sub(' ', text)
    return RE_WHITESPACE.sub(' ', text).strip()


def is_non_ascii_name(raw_name: str) -> bool:
    """Detects whether raw name is primarily non-Latin (e.g. Hindi, Tamil, Telugu)."""
    if not raw_name:
        return False
    letters = RE_LETTERS.findall(raw_name)
    return len(letters) < 3 and len(raw_name.strip()) >= 3


def extract_core_tokens(name: str) -> List[str]:
    """Returns normalized tokens excluding legal suffixes."""
    norm = normalize_text(name)
    tokens = norm.split()
    return [t for t in tokens if t not in LEGAL_SUFFIXES]


def extract_core_name(name: str) -> str:
    """Returns core business name."""
    tokens = extract_core_tokens(name)
    return " ".join(tokens) if tokens else normalize_text(name)


def extract_sorted_key(name: str) -> str:
    """Returns alphabetically sorted core tokens."""
    tokens = sorted(extract_core_tokens(name))
    return " ".join(tokens)


def extract_distinctive_name_tokens(name: str) -> Set[str]:
    """Returns non-generic brand-distinctive tokens."""
    tokens = extract_core_tokens(name)
    return {t for t in tokens if len(t) >= 3 and t not in GENERIC_WORDS}


def extract_postal_and_number(address: str) -> Tuple[str, str, Set[str]]:
    """Strictly separates postal code (5-6 digits) from building/street number.

    Prevents postal code digits (e.g. 75001, 560001) from being conflated
    with building or street numbers.
    """
    if not address:
        return "", "", set()

    norm_addr = normalize_text(address)
    tokens = norm_addr.split()

    # Find all 5 or 6 digit candidates
    all_56 = re.findall(r'\b(\d{5,6})\b', address)
    postal = ""
    addr_without_postal = address

    if all_56:
        # The last 5-6 digit number is postal code, unless it is a leading number with street words
        cand = all_56[-1]
        is_leading = bool(re.search(r'^\s*' + re.escape(cand) + r'\b', address))
        if len(all_56) > 1 or not is_leading:
            postal = cand
            pos = address.rfind(postal)
            addr_without_postal = address[:pos] + " " + address[pos + len(postal):]

    # Extract building / street number
    num = ""
    prefix_match = re.search(r'\b(?:no\.?|h\.?no\.?|#|plot|flat|door|suite|bldg)\s*(\d{1,5}[a-z]?)\b', addr_without_postal, re.IGNORECASE)
    if prefix_match:
        num = prefix_match.group(1).lower()
    else:
        leading_match = re.search(r'^\s*(\d{1,5}[a-z]?)\b', addr_without_postal)
        if leading_match:
            num = leading_match.group(1).lower()
        else:
            any_num = re.search(r'\b(\d{1,4})\b', addr_without_postal)
            if any_num:
                num = any_num.group(1)

    # Clean distinctive address tokens (exclude generic road words, numbers, and postal code)
    distinctive_tokens = {
        t for t in tokens
        if len(t) >= 3 and t not in GENERIC_ADDR_WORDS and not t.isdigit() and t != postal
    }

    return postal, num, distinctive_tokens


def compute_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extracts character n-grams."""
    if not text or len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Safe Jaccard similarity: returns 0.0 (NOT 1.0) when either or both sets are empty."""
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return inter / union if union > 0 else 0.0


def prepare_record(raw_name: str, raw_addr: str, raw_country: str) -> dict:
    """Pre-compute normalized tokens and attributes once for an entity."""
    raw_addr_str = raw_addr or ""
    has_addr = 1.0 if raw_addr_str.strip() else 0.0
    norm_name = normalize_text(raw_name)
    core = extract_core_name(raw_name)
    sorted_k = extract_sorted_key(raw_name)
    tokens = set(extract_core_tokens(raw_name))
    brand = extract_distinctive_name_tokens(raw_name)
    postal, num, dist_addr = extract_postal_and_number(raw_addr_str)
    norm_addr = normalize_text(raw_addr_str) if has_addr else ""
    ngrams = compute_char_ngrams(core, 3)
    is_non_ascii = 1.0 if is_non_ascii_name(raw_name) else 0.0
    return {
        "raw_name": raw_name,
        "raw_addr": raw_addr_str,
        "country": (raw_country or "").strip().lower(),
        "has_addr": has_addr,
        "norm_name": norm_name,
        "core": core,
        "sorted": sorted_k,
        "tokens": tokens,
        "brand": brand,
        "postal": postal,
        "num": num,
        "dist_addr": dist_addr,
        "norm_addr": norm_addr,
        "ngrams": ngrams,
        "is_non_ascii": is_non_ascii,
    }


class FeatureExtractor:
    """High-Performance Pairwise Feature Vector Generator."""

    def __init__(self, token_idf_dict: Dict[str, float] = None):
        self.token_idf = token_idf_dict or {}

    def extract_pair_features(
        self,
        s1_data: Dict[str, any],
        cand_data: Dict[str, any],
        cand_id: str = "",
        cand_rank: int = 1,
        pool_size: int = 1,
    ) -> Dict[str, float]:
        """Extracts complete feature dictionary for (Source 1, Candidate)."""
        s1 = s1_data if "norm_name" in s1_data else prepare_record(
            s1_data.get("name", ""), s1_data.get("address", ""), s1_data.get("country", "")
        )
        c = cand_data if "norm_name" in cand_data else prepare_record(
            cand_data.get("name", ""), cand_data.get("address", ""), cand_data.get("country", "")
        )

        s1_has_addr = s1["has_addr"]
        c_has_addr = c["has_addr"]
        both_have_addr = 1.0 if (s1_has_addr and c_has_addr) else 0.0
        either_addr_missing = 1.0 if (not s1_has_addr or not c_has_addr) else 0.0

        s1_norm_name, c_norm_name = s1["norm_name"], c["norm_name"]
        s1_core, c_core = s1["core"], c["core"]
        s1_sorted, c_sorted = s1["sorted"], c["sorted"]
        s1_tokens, c_tokens = s1["tokens"], c["tokens"]
        s1_brand, c_brand = s1["brand"], c["brand"]

        # Name Similarities
        if HAS_RAPIDFUZZ and s1_norm_name and c_norm_name:
            token_sort = fuzz.token_sort_ratio(s1_norm_name, c_norm_name) / 100.0
            token_set = fuzz.token_set_ratio(s1_norm_name, c_norm_name) / 100.0
            lev_ratio = fuzz.ratio(s1_core, c_core) / 100.0 if (s1_core and c_core) else 0.0
            jaro_winkler = distance.JaroWinkler.similarity(s1_norm_name, c_norm_name)
            partial_ratio = fuzz.partial_ratio(s1_norm_name, c_norm_name) / 100.0
        else:
            token_sort = jaccard_similarity(s1_tokens, c_tokens)
            token_set = token_sort
            lev_ratio = 1.0 if (s1_core and s1_core == c_core) else 0.0
            jaro_winkler = lev_ratio
            partial_ratio = lev_ratio

        core_exact = 1.0 if (s1_core and c_core and s1_core == c_core) else 0.0
        sorted_exact = 1.0 if (s1_sorted and c_sorted and s1_sorted == c_sorted) else 0.0
        name_token_jaccard = jaccard_similarity(s1_tokens, c_tokens)

        # Character n-grams (typo robustness)
        char_ngram_jaccard = jaccard_similarity(s1["ngrams"], c["ngrams"])

        # Distinctive Brand Overlap
        brand_overlap_count = float(len(s1_brand.intersection(c_brand)))
        brand_overlap_flag = 1.0 if brand_overlap_count > 0 else 0.0

        # Token IDF / Generic Business Name Penalties
        shared_tokens = s1_tokens.intersection(c_tokens)
        shared_idfs = [self.token_idf.get(t, 5.0) for t in shared_tokens]
        max_shared_idf = max(shared_idfs) if shared_idfs else 0.0
        mean_shared_idf = (sum(shared_idfs) / len(shared_idfs)) if shared_idfs else 0.0

        # Address Decomposition
        s1_postal, c_postal = s1["postal"], c["postal"]
        s1_num, c_num = s1["num"], c["num"]
        s1_dist_addr, c_dist_addr = s1["dist_addr"], c["dist_addr"]

        # Address Fuzzy Similarities
        if HAS_RAPIDFUZZ and both_have_addr:
            addr_token_set = fuzz.token_set_ratio(s1["norm_addr"], c["norm_addr"]) / 100.0
            addr_token_sort = fuzz.token_sort_ratio(s1["norm_addr"], c["norm_addr"]) / 100.0
        else:
            addr_token_set = 0.0
            addr_token_sort = 0.0

        addr_dist_jaccard = jaccard_similarity(s1_dist_addr, c_dist_addr)
        addr_dist_overlap_count = float(len(s1_dist_addr.intersection(c_dist_addr)))

        # Postal Code Disambiguation
        has_both_postal = 1.0 if (s1_postal and c_postal) else 0.0
        postal_match = 1.0 if (has_both_postal and s1_postal == c_postal) else 0.0
        postal_conflict = 1.0 if (has_both_postal and s1_postal != c_postal) else 0.0

        # Street / Building Number Disambiguation
        has_both_num = 1.0 if (s1_num and c_num) else 0.0
        num_match = 1.0 if (has_both_num and s1_num == c_num) else 0.0
        num_conflict = 1.0 if (has_both_num and s1_num != c_num) else 0.0

        # Open-Set Country Relational Signal (+1: match, 0: conflict, -1: unknown/missing)
        s1_country, c_country = s1["country"], c["country"]
        if not s1_country or not c_country:
            country_rel = -1.0
        elif s1_country == c_country:
            country_rel = 1.0
        else:
            country_rel = 0.0

        # Non-Latin Indic script signal
        s1_non_ascii = s1["is_non_ascii"]
        c_non_ascii = c["is_non_ascii"]
        cross_script = 1.0 if (s1_non_ascii != c_non_ascii) else 0.0

        # Target Source Type
        is_s2 = 1.0 if "s2" in cand_id.lower() else 0.0
        is_s3 = 1.0 if "s3" in cand_id.lower() else 0.0

        # Interaction & Discrepancy Signals
        name_strength = max(token_sort, token_set, lev_ratio, jaro_winkler)
        addr_strength = max(addr_token_set, addr_dist_jaccard, postal_match) if both_have_addr else 0.5
        name_x_addr = name_strength * addr_strength
        strong_name_weak_addr = 1.0 if (name_strength > 0.88 and both_have_addr and addr_strength < 0.30) else 0.0
        weak_name_strong_addr = 1.0 if (name_strength < 0.50 and both_have_addr and addr_strength > 0.85) else 0.0

        return {
            # Name features
            "token_sort": float(token_sort),
            "token_set": float(token_set),
            "lev_ratio": float(lev_ratio),
            "jaro_winkler": float(jaro_winkler),
            "partial_ratio": float(partial_ratio),
            "core_exact": float(core_exact),
            "sorted_exact": float(sorted_exact),
            "name_token_jaccard": float(name_token_jaccard),
            "char_ngram_jaccard": float(char_ngram_jaccard),
            "brand_overlap_count": float(brand_overlap_count),
            "brand_overlap_flag": float(brand_overlap_flag),
            "max_shared_idf": float(max_shared_idf),
            "mean_shared_idf": float(mean_shared_idf),
            # Address features
            "addr_token_set": float(addr_token_set),
            "addr_token_sort": float(addr_token_sort),
            "addr_dist_jaccard": float(addr_dist_jaccard),
            "addr_dist_overlap_count": float(addr_dist_overlap_count),
            "postal_match": float(postal_match),
            "postal_conflict": float(postal_conflict),
            "has_both_postal": float(has_both_postal),
            "num_match": float(num_match),
            "num_conflict": float(num_conflict),
            "has_both_num": float(has_both_num),
            # Missingness flags
            "s1_has_addr": float(s1_has_addr),
            "c_has_addr": float(c_has_addr),
            "both_have_addr": float(both_have_addr),
            "either_addr_missing": float(either_addr_missing),
            # Context & Meta
            "country_rel": float(country_rel),
            "s1_non_ascii": float(s1_non_ascii),
            "c_non_ascii": float(c_non_ascii),
            "cross_script": float(cross_script),
            "is_s2": float(is_s2),
            "is_s3": float(is_s3),
            "cand_rank": float(cand_rank),
            "pool_size": float(pool_size),
            # Interaction
            "name_strength": float(name_strength),
            "addr_strength": float(addr_strength),
            "name_x_addr": float(name_x_addr),
            "strong_name_weak_addr": float(strong_name_weak_addr),
            "weak_name_strong_addr": float(weak_name_strong_addr),
        }
