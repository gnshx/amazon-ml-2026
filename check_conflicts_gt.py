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

both_conflict_true_matches = 0
num_conflict_only = 0
postal_conflict_only = 0
total_checked = 0

for s1 in s1_records:
    s1_id = s1['id']
    s1_p = s1['postal']
    s1_n = s1['num']
    for tid in gt.get(s1_id, []):
        t_idx = target_id_to_idx.get(tid)
        if t_idx is not None:
            t = target_table[t_idx]
            t_p, t_n = t[6], t[7]
            if s1_p and t_p and s1_n and t_n:
                total_checked += 1
                p_conf = (s1_p != t_p)
                n_conf = (s1_n != t_n)
                if p_conf and n_conf: both_conflict_true_matches += 1
                elif n_conf: num_conflict_only += 1
                elif p_conf: postal_conflict_only += 1

print(f"Address conflicts in True Matches (out of {total_checked:,} pairs with both postal & num):")
print(f"  BOTH Postal & Number conflict: {both_conflict_true_matches} ({both_conflict_true_matches/total_checked*100:.3f}%)")
print(f"  Number conflict only:         {num_conflict_only} ({num_conflict_only/total_checked*100:.2f}%)")
print(f"  Postal conflict only:         {postal_conflict_only} ({postal_conflict_only/total_checked*100:.2f}%)")
