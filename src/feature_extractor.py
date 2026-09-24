#!/usr/bin/env python3
"""Pairwise Feature Extractor for GBDT & Singleton Gate.

Extracts 30+ high-ROI similarity features for every candidate pair (S1, Cand):
- Name Similarity: Token sort, Token set, Levenshtein, Jaro-Winkler, Jaccard, Char n-grams, IDF rarity
- Address Similarity: Jaccard, Token set, Postal exact/prefix, Street number match/diff
- Joint & Discrepancy: Evidence product, Evidence min, Strong Name / Weak Addr, Weak Name / Strong Addr
- Context & Meta: Country match (+1, 0, -1), Source indicator (S2 vs S3), Candidate rank and pool size
"""

from collections import defaultdict
import math
import re
from typing import Dict, List, Tuple
import unicodedata

try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# Legal suffixes
LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited", "co", "company", "dba", "lp", "pllc",
    "pvt", "private", "llp", "opc", "sarl", "sas", "sa", "eurl", "sci", "snc"
}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower().replace("&", " and ")
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_core_tokens(name: str) -> List[str]:
    norm = normalize_text(name)
    tokens = norm.split()
    return [t for t in tokens if t not in LEGAL_SUFFIXES]


def extract_core_name(name: str) -> str:
    tokens = extract_core_tokens(name)
    return " ".join(tokens) if tokens else normalize_text(name)


def extract_postal_and_number(address: str) -> Tuple[str, str]:
    if not address:
        return "", ""
    postal_match = re.search(r'\b(\d{5,6})\b', address)
    postal = postal_match.group(1) if postal_match else ""
    number_match = re.search(r'\b(\d{1,5})\b', address)
    num = number_match.group(1) if number_match else ""
    return postal, num


def compute_char_ngrams(text: str, n: int = 3) -> set:
    if not text or len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return inter / union if union > 0 else 0.0


def check_initials_match(name1: str, name2: str) -> float:
    t1 = extract_core_tokens(name1)
    t2 = extract_core_tokens(name2)
    if not t1 or not t2:
        return 0.0
    init1 = "".join([t[0] for t in t1 if t])
    init2 = "".join([t[0] for t in t2 if t])
    return 1.0 if (init1 == "".join(t2) or init2 == "".join(t1) or init1 == init2) else 0.0


class FeatureExtractor:
    """Computes comprehensive pairwise similarity feature vectors."""

    def __init__(self, token_idf_dict: Dict[str, float] = None):
        self.token_idf = token_idf_dict or {}

    def extract_pair_features(
        self,
        s1_data: Dict[str, str],
        cand_data: Dict[str, str],
        cand_id: str,
        pool_size: int = 1,
        cand_rank: int = 1,
    ) -> Dict[str, float]:
        """Extracts feature vector for a candidate pair."""
        s1_name = s1_data.get("name", "")
        s1_addr = s1_data.get("address", "")
        s1_country = s1_data.get("country", "").lower()

        c_name = cand_data.get("name", "")
        c_addr = cand_data.get("address", "")
        c_country = cand_data.get("country", "").lower()

        # Core name representations
        s1_core = extract_core_name(s1_name)
        c_core = extract_core_name(c_name)
        s1_tokens = set(extract_core_tokens(s1_name))
        c_tokens = set(extract_core_tokens(c_name))

        # RapidFuzz / String similarities
        if HAS_RAPIDFUZZ:
            token_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
            token_set = fuzz.token_set_ratio(s1_name, c_name) / 100.0
            lev_ratio = fuzz.ratio(s1_name, c_name) / 100.0
            jaro_winkler = distance.JaroWinkler.similarity(s1_name, c_name)
            partial = fuzz.partial_ratio(s1_name, c_name) / 100.0
            addr_token_set = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0
            addr_ratio = fuzz.ratio(s1_addr, c_addr) / 100.0
        else:
            token_sort = jaccard_similarity(s1_tokens, c_tokens)
            token_set = token_sort
            lev_ratio = 1.0 if s1_core == c_core else 0.5 if s1_core in c_core or c_core in s1_core else 0.0
            jaro_winkler = lev_ratio
            partial = lev_ratio
            addr_token_set = 0.0
            addr_ratio = 0.0

        # Sub-word & token Jaccard
        name_token_jaccard = jaccard_similarity(s1_tokens, c_tokens)
        s1_ngrams = compute_char_ngrams(s1_core, 3)
        c_ngrams = compute_char_ngrams(c_core, 3)
        name_char_jaccard = jaccard_similarity(s1_ngrams, c_ngrams)

        # Address & Numbers
        s1_postal, s1_num = extract_postal_and_number(s1_addr)
        c_postal, c_num = extract_postal_and_number(c_addr)

        postal_exact = 1.0 if (s1_postal and c_postal and s1_postal == c_postal) else 0.0
        postal_prefix = 1.0 if (s1_postal and c_postal and (s1_postal[:3] == c_postal[:3])) else 0.0
        street_num_match = 1.0 if (s1_num and c_num and s1_num == c_num) else 0.0
        
        street_num_diff = 0.0
        if s1_num and c_num and s1_num.isdigit() and c_num.isdigit():
            street_num_diff = min(abs(int(s1_num) - int(c_num)), 100) / 100.0

        s1_addr_tokens = set(normalize_text(s1_addr).split())
        c_addr_tokens = set(normalize_text(c_addr).split())
        addr_jaccard = jaccard_similarity(s1_addr_tokens, c_addr_tokens)

        # Token IDF / Generic name penalty
        shared_tokens = s1_tokens.intersection(c_tokens)
        shared_idfs = [self.token_idf.get(t, 5.0) for t in shared_tokens]
        min_shared_idf = min(shared_idfs) if shared_idfs else 0.0
        mean_shared_idf = (sum(shared_idfs) / len(shared_idfs)) if shared_idfs else 0.0

        # Joint / Interaction Features
        name_evidence = max(token_sort, token_set, jaro_winkler)
        addr_evidence = max(addr_token_set, addr_jaccard, postal_exact)
        name_x_addr = name_evidence * addr_evidence
        min_evidence = min(name_evidence, addr_evidence)

        # Discrepancy flags
        strong_name_weak_addr = 1.0 if (name_evidence > 0.85 and addr_evidence < 0.25) else 0.0
        weak_name_strong_addr = 1.0 if (name_evidence < 0.35 and addr_evidence > 0.80) else 0.0

        # Country match (+1 agree, 0 disagree, -1 missing/unseen)
        if not s1_country or not c_country:
            country_flag = -1.0
        elif s1_country == c_country:
            country_flag = 1.0
        else:
            country_flag = 0.0

        # Source indicator
        is_s2 = 1.0 if "s2" in cand_id.lower() else 0.0
        is_s3 = 1.0 if "s3" in cand_id.lower() else 0.0

        return {
            "token_sort": float(token_sort),
            "token_set": float(token_set),
            "lev_ratio": float(lev_ratio),
            "jaro_winkler": float(jaro_winkler),
            "partial": float(partial),
            "name_token_jaccard": float(name_token_jaccard),
            "name_char_jaccard": float(name_char_jaccard),
            "core_exact": 1.0 if (s1_core and c_core and s1_core == c_core) else 0.0,
            "initials_match": float(check_initials_match(s1_name, c_name)),
            "min_shared_idf": float(min_shared_idf),
            "mean_shared_idf": float(mean_shared_idf),
            "addr_token_set": float(addr_token_set),
            "addr_ratio": float(addr_ratio),
            "addr_jaccard": float(addr_jaccard),
            "postal_exact": float(postal_exact),
            "postal_prefix": float(postal_prefix),
            "street_num_match": float(street_num_match),
            "street_num_diff": float(street_num_diff),
            "name_evidence": float(name_evidence),
            "addr_evidence": float(addr_evidence),
            "name_x_addr": float(name_x_addr),
            "min_evidence": float(min_evidence),
            "strong_name_weak_addr": float(strong_name_weak_addr),
            "weak_name_strong_addr": float(weak_name_strong_addr),
            "country_flag": float(country_flag),
            "is_s2": float(is_s2),
            "is_s3": float(is_s3),
            "cand_rank": float(cand_rank),
            "pool_size": float(pool_size),
        }


if __name__ == "__main__":
    fe = FeatureExtractor()
    feat = fe.extract_pair_features(
        {"name": "Apex Healthcare Pvt Ltd", "address": "123 Main Street 560001", "country": "india"},
        {"name": "Apex Healthcare Services", "address": "123 Main St 560001", "country": "india"},
        cand_id="S2-101"
    )
    print("Feature extractor sample output:")
    for k, v in list(feat.items())[:10]:
        print(f"  {k}: {v}")
    print(f"Total features extracted: {len(feat)}")
