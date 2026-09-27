import pickle, time, os, sys

with open('scratch/fixed_benchmark_cache_50k.pkl', 'rb') as f:
    data = pickle.load(f)

s1_records = data['s1_records']
target_table = data['target_table']
target_ids = data['target_ids']
target_id_to_idx = {tid: i for i, tid in enumerate(target_ids)}

s1_set = set(s['id'] for s in s1_records)
gt = {s['id']: [] for s in s1_records}
with open('dataset/train/train_ground_truth.tsv', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in s1_set:
            gt[p[0]] = p[1].split(',') if len(p) > 1 and p[1] else []

# Check address presence in true matches:
true_match_cases = 0
s1_has_addr_t_empty = 0
both_have_addr = 0
both_empty_addr = 0
s1_empty_t_has = 0

for s1 in s1_records:
    s1_id = s1['id']
    s1_has_a = bool(s1['clean_addr'])
    for tid in gt.get(s1_id, []):
        t_idx = target_id_to_idx.get(tid)
        if t_idx is not None:
            true_match_cases += 1
            t_has_a = bool(target_table[t_idx][4])
            if s1_has_a and t_has_a: both_have_addr += 1
            elif s1_has_a and not t_has_a: s1_has_addr_t_empty += 1
            elif not s1_has_a and t_has_a: s1_empty_t_has += 1
            else: both_empty_addr += 1

print(f"True match address status across {true_match_cases:,} pairs:")
print(f"  Both have address:            {both_have_addr:,} ({both_have_addr/true_match_cases*100:.1f}%)")
print(f"  S1 has address, Target EMPTY: {s1_has_addr_t_empty:,} ({s1_has_addr_t_empty/true_match_cases*100:.1f}%)")
print(f"  S1 EMPTY, Target has address: {s1_empty_t_has:,} ({s1_empty_t_has/true_match_cases*100:.1f}%)")
print(f"  Both EMPTY:                   {both_empty_addr:,} ({both_empty_addr/true_match_cases*100:.1f}%)")
