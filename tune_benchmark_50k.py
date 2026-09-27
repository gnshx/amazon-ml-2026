import pickle, time, os, sys
from collections import defaultdict
from rapidfuzz import fuzz, distance

sys.path.insert(0, "src")
from metrics import evaluate_macro_f05

print("Loading cached 50,000 entity benchmark...")
cache_file = "scratch/fixed_benchmark_cache_50k.pkl"
with open(cache_file, "rb") as f:
    data = pickle.load(f)

s1_records = data["s1_records"]
target_table = data["target_table"]
target_ids = data["target_ids"]
idx_core = data["idx_core"]
idx_sorted = data["idx_sorted"]
idx_concat = data["idx_concat"]
idx_brand = data["idx_brand"]
idx_num_p = data["idx_num_p"]
idx_post_p = data["idx_post_p"]
idx_addr_k = data["idx_addr_k"]

print(f"Loaded {len(s1_records):,} S1 entities and {len(target_table):,} targets.")

# Load ground truth for Fold 0 and Fold 1
eval_gt = {}
with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
    f.readline()
    s1_eval_ids = set(s["id"] for s in s1_records)
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        if parts[0] in s1_eval_ids:
            eval_gt[parts[0]] = parts[1].split(",") if len(parts) > 1 and parts[1] else []

print(f"Loaded ground truth for {len(eval_gt):,} benchmark entities.")

# Test various score thresholds and singleton gates
def run_benchmark(sc_cutoff=0.81, min_single_word_len=3, singleton_filter=True):
    target_claims = {}
    s1_to_scored = defaultdict(list)
    
    for s1 in s1_records:
        s1_id = s1["id"]
        core_s1 = s1["core"]
        sorted_s1 = s1["sorted"]
        concat_s1 = s1["concat"]
        brand_tokens = s1["distinctive"]
        s1_p = s1["postal"]
        s1_n = s1["num"]
        s1_dist = s1["dist_addr"]
        s1_country = s1["country"]
        s1_aliases = s1["aliases"]
        num_p_s1 = s1["num_prefix"]
        post_p_s1 = s1["post_prefix"]
        addr_keys_s1 = s1["addr_keys"]
        clean_s1_addr = s1["clean_addr"]
        norm_s1_name = s1["norm_name"]
        s1_is_non_ascii = s1["is_non_ascii"]

        cands = set()
        cands.update(idx_core.get(core_s1, []))
        for a in s1_aliases: cands.update(idx_core.get(a, []))
        cands.update(idx_sorted.get(sorted_s1, []))
        if len(concat_s1) >= 5: cands.update(idx_concat.get(concat_s1, []))
        for bt in brand_tokens: cands.update(idx_brand.get(bt, []))
        if num_p_s1: cands.update(idx_num_p.get(num_p_s1, []))
        if post_p_s1: cands.update(idx_post_p.get(post_p_s1, []))
        for ak in addr_keys_s1: cands.update(idx_addr_k.get(ak, []))

        scored_matches = []
        has_s1_addr = bool(clean_s1_addr)

        for t_idx in cands:
            t = target_table[t_idx]
            (core_t, sorted_t, concat_t, norm_t_name, clean_t_addr, t_country, t_p, t_n, t_dist, target_is_non_ascii, t_aliases) = t

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

            if has_s1_addr and has_t_addr:
                if addr_token_sim is not None and addr_token_sim < 0.35: continue
                if s1_dist and t_dist and not postal_match and dist_overlap == 0: continue
                if postal_conflict and addr_token_sim is not None and addr_token_sim < 0.70: continue
                if num_conflict and dist_overlap == 0: continue
            else:
                core_words = core_s1.split()
                if len(core_words) < 2 and len(core_s1) < min_single_word_len:
                    continue

            lev = fuzz.ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
            sort_r = fuzz.token_sort_ratio(norm_s1_name, norm_t_name) / 100.0 if (norm_s1_name and norm_t_name) else 0.0
            max_name = max(lev, sort_r)

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

            min_len = min(len(core_s1), len(core_t))
            max_len = max(len(core_s1), len(core_t))
            if score < 0.80 and 3 <= min_len <= 7 and max_len <= 8:
                if distance.Levenshtein.distance(core_s1, core_t) <= 1:
                    if num_match or postal_match or dist_overlap >= 1: score = 0.83

            if score >= sc_cutoff:
                scored_matches.append((t_idx, score, addr_token_sim if addr_token_sim is not None else 0.50, postal_match, num_match))

        for t_idx, sc, a_sim, pm, nm in scored_matches:
            prev = target_claims.get(t_idx)
            rank_tuple = (sc, 1 if nm else 0, 1 if pm else 0, a_sim)
            if prev is None or rank_tuple > prev[1]:
                target_claims[t_idx] = (s1_id, rank_tuple)

        s1_to_scored[s1_id] = scored_matches

    winner_for_target = {t_idx: val[0] for t_idx, val in target_claims.items()}
    
    predictions = {}
    for s1 in s1_records:
        s1_id = s1["id"]
        cands_won = [(t_idx, sc, a_sim) for t_idx, sc, a_sim, pm, nm in s1_to_scored.get(s1_id, [])
                     if winner_for_target.get(t_idx) == s1_id]
        
        # Singleton filtering
        if singleton_filter and len(cands_won) == 1:
            t_idx, sc, a_sim = cands_won[0]
            if sc < 0.84 and a_sim < 0.65:
                predictions[s1_id] = []
                continue

        predictions[s1_id] = [target_ids[t_idx] for t_idx, sc, a_sim in cands_won]

    res = evaluate_macro_f05(predictions, eval_gt)
    print(f"Cutoff: {sc_cutoff:.2f} | MinWordLen: {min_single_word_len:2d} | SingFilter: {str(singleton_filter):5s} -> "
          f"Macro F0.5: {res['macro_f05']:.4f} | Sing Acc: {res['singleton_accuracy']:.4f} | Non-Sing: {res['non_singleton_f05']:.4f}")
    return res

print("\nRunning Parameter Sweep on 50k Benchmark:")
for sc in [0.81, 0.82, 0.83]:
    for mwl in [3, 14]:
        for sf in [True, False]:
            run_benchmark(sc, mwl, sf)
