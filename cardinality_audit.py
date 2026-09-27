import pickle, time, os, sys
from collections import defaultdict

sys.path.insert(0, 'src')
from metrics import compute_entity_f05

with open('scratch/fixed_benchmark_cache_50k.pkl', 'rb') as f:
    data = pickle.load(f)

s1_records = data['s1_records']
target_ids = data['target_ids']
s1_set = set(s['id'] for s in s1_records)
gt = {s['id']: [] for s in s1_records}
with open('dataset/train/train_ground_truth.tsv', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in s1_set:
            gt[p[0]] = p[1].split(',') if len(p) > 1 and p[1] else []

# Inspect ground truth cardinality:
gold_lens = [len(gt[s['id']]) for s in s1_records]
print(f"Ground truth match counts:")
print(f"  0 matches (singletons): {sum(1 for l in gold_lens if l == 0)} ({sum(1 for l in gold_lens if l == 0)/len(s1_records)*100:.1f}%)")
print(f"  1 match:                {sum(1 for l in gold_lens if l == 1)} ({sum(1 for l in gold_lens if l == 1)/len(s1_records)*100:.1f}%)")
print(f"  2 matches:              {sum(1 for l in gold_lens if l == 2)} ({sum(1 for l in gold_lens if l == 2)/len(s1_records)*100:.1f}%)")
print(f"  3 matches:              {sum(1 for l in gold_lens if l == 3)} ({sum(1 for l in gold_lens if l == 3)/len(s1_records)*100:.1f}%)")
print(f"  4+ matches:             {sum(1 for l in gold_lens if l >= 4)} ({sum(1 for l in gold_lens if l >= 4)/len(s1_records)*100:.1f}%)")
print(f"  Avg matches per non-sing: {sum(gold_lens)/(len(gold_lens)-sum(1 for l in gold_lens if l == 0)):.2f}")
