import pickle, time, os, sys
from collections import defaultdict
import numpy as np
import catboost as cb
from rapidfuzz import fuzz, distance

sys.path.insert(0, 'src')
from metrics import evaluate_macro_f05

print('Loading 50k benchmark...')
with open('scratch/fixed_benchmark_cache_50k.pkl', 'rb') as f:
    data = pickle.load(f)

s1_records = data['s1_records']
target_table = data['target_table']
target_ids = data['target_ids']
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

print('Loading CatBoost GPU model...')
model = cb.CatBoostClassifier()
model.load_model('output/overnight_catboost_gpu.cbm')
print('Model loaded.')

t0 = time.time()
pairs_to_score = []
pair_meta = [] # (s1_id, t_idx, is_s2)

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
    pool_sz = len(cands)

    for rank, t_idx in enumerate(cands, start=1):
        t = target_table[t_idx]
        (core_t, sorted_t, concat_t, norm_t_name, clean_t_addr, t_country, t_p, t_n, t_dist, target_is_non_ascii, t_aliases) = t

        if s1_country and t_country and s1_country != t_country:
            continue

        has_t_addr = bool(clean_t_addr)
        postal_match = 1.0 if (s1_p and t_p and s1_p == t_p) else 0.0
        postal_conflict = 1.0 if (s1_p and t_p and s1_p != t_p) else 0.0
        num_match = 1.0 if (s1_n and t_n and s1_n == t_n) else 0.0
        num_conflict = 1.0 if (s1_n and t_n and s1_n != t_n) else 0.0
        dist_inter = len(s1_dist.intersection(t_dist)) if (s1_dist and t_dist) else 0

        # Fast screen
        lev = fuzz.ratio(core_s1, core_t) / 100.0 if (core_s1 and core_t) else 0.0
        if not (postal_match or num_match or dist_inter >= 1 or lev >= 0.50):
            continue

        token_sort = fuzz.token_sort_ratio(norm_s1_name, norm_t_name) / 100.0 if (norm_s1_name and norm_t_name) else 0.0
        token_set = fuzz.token_set_ratio(norm_s1_name, norm_t_name) / 100.0 if (norm_s1_name and norm_t_name) else 0.0

        core_exact = 1.0 if (core_s1 and core_t and core_s1 == core_t) else 0.0
        if not core_exact and (any(a == core_t for a in s1_aliases) or any(a == core_s1 for a in t_aliases)):
            core_exact = 0.95
        sorted_exact = 1.0 if (sorted_s1 and sorted_t and sorted_s1 == sorted_t) else 0.0
        concat_exact = 1.0 if (concat_s1 and concat_t and concat_s1 == concat_t) else 0.0

        b_inter = len(set(brand_tokens) & set(t[0].split())) if (brand_tokens and t[0]) else 0
        brand_overlap_count = float(b_inter)
        brand_overlap_flag = 1.0 if b_inter > 0 else 0.0

        both_have_addr = 1.0 if (has_s1_addr and has_t_addr) else 0.0
        either_addr_missing = 1.0 - both_have_addr

        addr_token_sort = 0.0
        addr_token_set = 0.0
        if both_have_addr:
            addr_token_sort = fuzz.token_sort_ratio(clean_s1_addr, clean_t_addr) / 100.0
            addr_token_set = fuzz.token_set_ratio(clean_s1_addr, clean_t_addr) / 100.0

        country_match = 1.0 if (s1_country and t_country and s1_country == t_country) else 0.0
        country_conflict = 1.0 if (s1_country and t_country and s1_country != t_country) else 0.0
        cat_conflict = 0.0
        is_s2 = 1.0 if target_ids[t_idx].startswith('S2') else 0.0

        len_s1 = len(core_s1)
        len_t = len(core_t)
        core_len_min = float(min(len_s1, len_t))
        core_len_diff = float(abs(len_s1 - len_t))

        jw = lev
        name_strength = max(lev, token_sort, token_set, jw)
        addr_strength = max(addr_token_sort, addr_token_set) if both_have_addr else 0.5
        name_x_addr = name_strength * addr_strength

        feats = [
            lev, token_sort, token_set, core_exact, sorted_exact,
            concat_exact, brand_overlap_count, brand_overlap_flag,
            addr_token_sort, addr_token_set, float(dist_inter),
            postal_match, postal_conflict, num_match, num_conflict,
            both_have_addr, either_addr_missing, country_match, country_conflict,
            cat_conflict, is_s2, float(rank), float(pool_sz),
            core_len_min, core_len_diff, float(s1_is_non_ascii), float(target_is_non_ascii),
            jw, name_strength, addr_strength, name_x_addr
        ]
        pairs_to_score.append(feats)
        pair_meta.append((s1_id, t_idx, is_s2))

print(f'Feature extraction took {time.time()-t0:.1f}s across {len(pairs_to_score):,} pairs.')

t1 = time.time()
X = np.array(pairs_to_score, dtype=np.float32)
probs = model.predict_proba(X)[:, 1]
print(f'CatBoost inference took {time.time()-t1:.1f}s.')

for tau in [0.85, 0.88, 0.90, 0.92, 0.94, 0.96]:
    target_claims = {}
    s1_to_scored = defaultdict(list)
    for (s1_id, t_idx, is_s2), prob in zip(pair_meta, probs):
        if prob >= tau:
            s1_to_scored[s1_id].append((t_idx, prob))
            prev = target_claims.get(t_idx)
            if prev is None or prob > prev[1]:
                target_claims[t_idx] = (s1_id, prob)

    winner = {t_idx: val[0] for t_idx, val in target_claims.items()}
    preds = {}
    for s1 in s1_records:
        s1_id = s1['id']
        won = [t_idx for t_idx, prob in s1_to_scored.get(s1_id, []) if winner.get(t_idx) == s1_id]
        preds[s1_id] = [target_ids[t_idx] for t_idx in won]

    res = evaluate_macro_f05(preds, gt)
    print(f'Tau: {tau:.2f} -> Macro F0.5: {res["macro_f05"]*100:.2f}% | Sing: {res["singleton_accuracy"]*100:.2f}% | Non-Sing: {res["non_singleton_f05"]*100:.2f}%')
