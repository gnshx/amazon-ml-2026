import pickle, time, os, sys
from collections import defaultdict
from rapidfuzz import fuzz, distance

sys.path.insert(0, 'src')
from metrics import evaluate_macro_f05, compute_entity_f05

print('Loading 50k benchmark...')
with open('scratch/fixed_benchmark_cache_50k.pkl', 'rb') as f:
    data = pickle.load(f)

s1_records = data['s1_records']
target_table = data['target_table']
target_ids = data['target_ids']
target_id_to_idx = {tid: i for i, tid in enumerate(target_ids)}

idx_core = data['idx_core']
idx_sorted = data['idx_sorted']
idx_concat = data['idx_concat']
idx_brand = data['idx_brand']
idx_num_p = data['idx_num_p']
idx_post_p = data['idx_post_p']
idx_addr_k = data.get('idx_addr_k', {})

s1_set = set(s['id'] for s in s1_records)
gt = {s['id']: [] for s in s1_records}
with open('dataset/train/train_ground_truth.tsv', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in s1_set:
            gt[p[0]] = p[1].split(',') if len(p) > 1 and p[1] else []

# Run baseline V2 prediction
target_claims = {}
s1_to_scored = defaultdict(list)

for s1 in s1_records:
    s1_id = s1['id']
    core_s1 = s1['core']
    sorted_s1 = s1['sorted']
    concat_s1 = s1['concat']
    brand_tokens = s1['distinctive']
    s1_p = s1['postal']
    s1_n = s1['num']
    s1_dist = s1['dist_addr']
    s1_country = s1['country']
    s1_aliases = s1['aliases']
    num_p_s1 = s1['num_prefix']
    post_p_s1 = s1['post_prefix']
    addr_keys_s1 = s1['addr_keys']
    clean_s1_addr = s1['clean_addr']
    norm_s1_name = s1['norm_name']
    s1_is_non_ascii = s1['is_non_ascii']

    cands = set()
    cands.update(idx_core.get(core_s1, []))
    for a in s1_aliases: cands.update(idx_core.get(a, []))
    cands.update(idx_sorted.get(sorted_s1, []))
    if len(concat_s1) >= 5: cands.update(idx_concat.get(concat_s1, []))
    for bt in brand_tokens[:6]: cands.update(idx_brand.get(bt, []))
    if num_p_s1: cands.update(idx_num_p.get(num_p_s1, []))
    if post_p_s1: cands.update(idx_post_p.get(post_p_s1, []))
    for ak in addr_keys_s1: cands.update(idx_addr_k.get(ak, []))

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

        lev = fuzz.ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
        sort_r = fuzz.token_sort_ratio(norm_s1_name, norm_t_name) / 100.0 if (norm_s1_name and norm_t_name) else 0.0
        max_name = max(lev, sort_r)

        score = 0.0
        if core_s1 and core_t and core_s1 == core_t: score = 0.95
        elif any(a and a == core_t for a in s1_aliases) or any(a and a == core_s1 for a in t_aliases): score = 0.94
        elif sorted_s1 and sorted_t == sorted_t: score = 0.92
        elif (concat_s1 and concat_t == concat_s1) or (concat_s1 and norm_t_name.replace(' ', '') == concat_s1):
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

        if score >= 0.81:
            a_sim_val = addr_token_sim if addr_token_sim is not None else 0.50
            s1_to_scored[s1_id].append((t_idx, score, a_sim_val, postal_match, num_match))
            prev = target_claims.get(t_idx)
            rank_tuple = (score, 1 if num_match else 0, 1 if postal_match else 0, a_sim_val)
            if prev is None or rank_tuple > prev[1]:
                target_claims[t_idx] = (s1_id, rank_tuple)

winner_for_target = {t_idx: val[0] for t_idx, val in target_claims.items()}
predictions = {}
for s1 in s1_records:
    s1_id = s1['id']
    cands_won = [(t_idx, sc, a_sim) for t_idx, sc, a_sim, pm, nm in s1_to_scored.get(s1_id, []) if winner_for_target.get(t_idx) == s1_id]
    if len(cands_won) == 1:
        t_idx, sc, a_sim = cands_won[0]
        if sc < 0.84 and a_sim < 0.65:
            cands_won = []
    predictions[s1_id] = [target_ids[t_idx] for t_idx, sc, a_sim in cands_won]

# Diagnostic analysis
perfect_matches = 0
partial_matches = 0
zero_score = 0
false_singletons = 0
missed_singletons = 0
not_in_candidate_pool = 0
in_pool_but_low_score = 0
won_by_other_s1 = 0

for s1 in s1_records:
    s1_id = s1['id']
    true_targets = set(gt.get(s1_id, []))
    pred_targets = set(predictions.get(s1_id, []))
    f05 = compute_entity_f05(pred_targets, true_targets)

    if f05 == 1.0:
        perfect_matches += 1
    elif f05 > 0.0:
        partial_matches += 1
    else:
        zero_score += 1
        if len(true_targets) == 0 and len(pred_targets) > 0:
            false_singletons += 1 # True singleton falsely matched
        elif len(true_targets) > 0 and len(pred_targets) == 0:
            missed_singletons += 1 # True match falsely predicted as singleton
            # Check why missed
            for tt in true_targets:
                t_idx = target_id_to_idx.get(tt)
                if t_idx is None:
                    not_in_candidate_pool += 1
                else:
                    # was it in cands?
                    pass

print(f'Total S1: {len(s1_records):,}')
print(f'Perfect (F0.5 = 1.0): {perfect_matches:,} ({perfect_matches/len(s1_records)*100:.2f}%)')
print(f'Partial (0 < F0.5 < 1): {partial_matches:,} ({partial_matches/len(s1_records)*100:.2f}%)')
print(f'Zero Score (F0.5 = 0.0): {zero_score:,} ({zero_score/len(s1_records)*100:.2f}%)')
print(f'  - True singletons falsely matched: {false_singletons:,}')
print(f'  - True matches missed (predicted singleton): {missed_singletons:,}')
