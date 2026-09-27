from collections import Counter
import csv

s2_counts = Counter()
s3_counts = Counter()
total_by_entity = Counter()

with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for row in r:
        s1 = row[0]
        targets = row[1].split(',') if row[1] else []
        total_by_entity[len(targets)] += 1
        s2_in_row = sum(1 for t in targets if t.startswith('S2-'))
        s3_in_row = sum(1 for t in targets if t.startswith('S3-'))
        s2_counts[s2_in_row] += 1
        s3_counts[s3_in_row] += 1

print('Total targets distribution:', sorted(total_by_entity.items()))
print('S2 targets per S1 distribution:', sorted(s2_counts.items()))
print('S3 targets per S1 distribution:', sorted(s3_counts.items()))
