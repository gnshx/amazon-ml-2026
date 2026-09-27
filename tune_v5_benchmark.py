import pickle, time, os, sys
from collections import defaultdict
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

print(f'Loaded {len(s1_records):,} S1 and {len(target_ids):,} targets.')

# Load GT strictly for benchmark S1 records
s1_set = set(s['id'] for s in s1_records)
gt = {s['id']: [] for s in s1_records}
with open('dataset/train/train_ground_truth.tsv') as f:
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        s_id = p[0].strip()
        if s_id in s1_set:
            if len(p) > 1 and p[1].strip():
                gt[s_id] = [x.strip() for x in p[1].split(',') if x.strip()]
            else:
                gt[s_id] = []

num_true_sing = sum(1 for v in gt.values() if len(v) == 0)
print(f'GT loaded for 50k: {len(gt):,} entities ({num_true_sing:,} singletons = {num_true_sing/len(gt)*100:.2f}%)')

CAT_BITS = {
    'pharmacy': 1, 'pharma': 1, 'clinic': 2, 'hospital': 2, 'medical': 2,
    'hotel': 4, 'restaurant': 8, 'cafe': 8, 'bank': 16, 'realty': 32,
    'school': 64, 'salon': 128, 'gym': 256, 'auto': 512, 'tech': 1024,
    'dental': 2048, 'law': 4096
}

configs = [
    ('1. Baseline V2 (pure Levenshtein/token_sort)', False, 0.80, 0.83, False),
    ('2. V2 + Category Conflict Guard', True, 0.80, 0.83, False),
    ('3. V2 + Category Guard + Strict Gate (0.83)', True, 0.80, 0.83, True),
    ('4. V2 + Category Guard + Strict Gate (0.84)', True, 0.80, 0.84, True),
    ('5. V2 + Category Guard + Strict Gate (0.85)', True, 0.80, 0.85, True),
    ('6. V2 + Cutoff 0.81 + Category Guard + Strict Gate (0.84)', True, 0.81, 0.84, True),
    ('7. V2 + Cutoff 0.82 + Category Guard + Strict Gate (0.84)', True, 0.82, 0.84, True),
]

for config_name, use_cat_guard, sc_cutoff, sing_cutoff, use_strict_gate in configs:
    t0 = time.time()
    target_claims = {}
    s1_to_scored = {}

    for s1 in s1_records:
        s1_id = s1['id']
        core_s1 = s1['core_name']
        sorted_s1 = s1['sorted_name']
        concat_s1 = s1['concat_name']
        brand_s1 = s1['brand_root']
        num_s1 = s1['num']
        post_s1 = s1['post']
        s1_aliases = s1['aliases']
        norm_s1_name = s1['norm_name']
        has_s1_addr = s1['has_addr']
        s1_tokens = s1['addr_tokens']
        s1_is_non_ascii = s1['is_non_ascii']
        s1_cat = s1['cat_mask']

        cands = set()
        if core_s1:
            m = idx_core.get(core_s1)
            if m: cands.update(m[:100])
        if sorted_s1:
            m = idx_sorted.get(sorted_s1)
            if m: cands.update(m[:50])
        if concat_s1:
            m = idx_concat.get(concat_s1)
            if m: cands.update(m[:30])
        if brand_s1 and (num_s1 or post_s1):
            m = idx_brand.get(brand_s1)
            if m: cands.update(m[:50])
        for alias in s1_aliases:
            if alias:
                m = idx_core.get(alias)
                if m: cands.update(m[:30])
        if num_s1 and post_s1:
            m = idx_num_p.get((num_s1, post_s1))
            if m: cands.update(m[:40])
        if post_s1 and len(core_s1) >= 4:
            m = idx_post_p.get((post_s1, core_s1[:4]))
            if m: cands.update(m[:15])

        if not cands:
            continue

        scored_matches = []
        for t_idx in cands:
            t = target_table[t_idx]
            core_t = t[0]
            sorted_t = t[1]
            concat_t = t[2]
            norm_t_name = t[3]
            num_t = t[4]
            post_t = t[5]
            has_t_addr = t[6]
            t_tokens = t[7]
            t_aliases = t[8]
            target_is_non_ascii = t[9]
            t_cat = t[10]

            if use_cat_guard and s1_cat > 0 and t_cat > 0 and (s1_cat & t_cat) == 0:
                continue

            num_match = (num_s1 and num_t and num_s1 == num_t)
            num_conflict = (num_s1 and num_t and num_s1 != num_t)
            postal_match = (post_s1 and post_t and post_s1 == post_t)
            postal_conflict = (post_s1 and post_t and post_s1 != post_t)

            dist_overlap = len(s1_tokens & t_tokens) if (s1_tokens and t_tokens) else 0
            addr_token_sim = (dist_overlap / max(len(s1_tokens), len(t_tokens))) if (s1_tokens and t_tokens) else None

            if has_s1_addr and has_t_addr:
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
            elif sorted_s1 and sorted_t == sorted_s1: score = 0.92
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

            if score >= sc_cutoff:
                scored_matches.append((t_idx, score, addr_token_sim if addr_token_sim is not None else 0.50, postal_match, num_match, len(core_s1)))

        for t_idx, sc, a_sim, pm, nm, clen in scored_matches:
            prev = target_claims.get(t_idx)
            rank_tuple = (sc, 1 if nm else 0, 1 if pm else 0, a_sim)
            if prev is None or rank_tuple > prev[1]:
                target_claims[t_idx] = (s1_id, rank_tuple)
        s1_to_scored[s1_id] = scored_matches

    winner_for_target = {t_idx: val[0] for t_idx, val in target_claims.items()}
    predictions = {}
    for s1 in s1_records:
        s1_id = s1['id']
        cands_won = [(t_idx, sc, a_sim, pm, nm, clen) for t_idx, sc, a_sim, pm, nm, clen in s1_to_scored.get(s1_id, []) if winner_for_target.get(t_idx) == s1_id]
        if len(cands_won) == 1:
            t_idx, sc, a_sim, pm, nm, clen = cands_won[0]
            if use_strict_gate:
                addr_c = (1 if pm else 0) + (1 if nm else 0) + (1 if a_sim >= 0.65 else 0)
                if sc < sing_cutoff and addr_c < 2: cands_won = []
                elif clen < 5 and addr_c < 2: cands_won = []
            else:
                if sc < sing_cutoff and a_sim < 0.65: cands_won = []

        predictions[s1_id] = [target_ids[t_idx] for t_idx, sc, a_sim, pm, nm, clen in cands_won]

    res = evaluate_macro_f05(predictions, gt)
    print(f'{config_name:58s} | F0.5: {res["macro_f05"]*100:.2f}% | Sing: {res["singleton_accuracy"]*100:.2f}% | Non-Sing: {res["non_singleton_f05"]*100:.2f}% ({time.time()-t0:.1f}s)')
