#!/usr/bin/env python3
import json
import os
import csv
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

TRAIN_DIR = "dataset/train"

def main():
    print("Loading Validation entities...")
    with open(os.path.join(TRAIN_DIR, "cv_splits_5fold.json")) as f:
        s1_to_fold = json.load(f)

    random.seed(42)
    fold0_s1 = [s for s, fld in s1_to_fold.items() if fld == 0]
    random.shuffle(fold0_s1)
    val_s1_ids = set(fold0_s1[:5000])

    gt = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            if s1_id in val_s1_ids:
                matches = [x.strip() for x in row[1].split(",") if x.strip()] if len(row) > 1 and row[1].strip() else []
                gt[s1_id] = matches

    total_gold_links = sum(len(v) for v in gt.values())

    s1_names = []
    s1_ids_list = []
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            if s1_id in val_s1_ids:
                name = row[1] if len(row) > 1 else ""
                s1_names.append(name.lower())
                s1_ids_list.append(s1_id)

    print("Streaming targets...")
    t_ids = []
    t_names = []
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        with open(os.path.join(TRAIN_DIR, fname), encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                t_ids.append(parts[0].strip())
                t_names.append(parts[1].lower() if len(parts) > 1 else "")

    print(f"Fitting TF-IDF on {len(t_names)} targets...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_df=0.3)
    t_tfidf = vectorizer.fit_transform(t_names)
    print(f"Fitted in {time.time()-t0:.1f}s")

    print("Transforming queries...")
    s1_tfidf = vectorizer.transform(s1_names)
    
    print("Computing dot product...")
    t0 = time.time()
    sim_matrix = s1_tfidf.dot(t_tfidf.T)
    print(f"Dot product in {time.time()-t0:.1f}s")

    print("Extracting top K...")
    cands_tfidf = {}
    cap_gold = 0
    k = 40
    for i in range(sim_matrix.shape[0]):
        row = sim_matrix.getrow(i)
        if row.nnz > 0:
            data = row.data
            indices = row.indices
            if len(data) > k:
                top_k = np.argpartition(data, -k)[-k:]
                top_k_indices = indices[top_k]
            else:
                top_k_indices = indices
            
            cand_list = [t_ids[idx] for idx in top_k_indices]
        else:
            cand_list = []
        
        s1_id = s1_ids_list[i]
        cands_tfidf[s1_id] = cand_list
        golds = gt.get(s1_id, [])
        if golds:
            cap_gold += len(set(golds).intersection(set(cand_list)))

    recall = cap_gold / total_gold_links * 100
    print(f"TF-IDF Route Recall: {recall:.2f}% (Average Pool Size: {sum(len(v) for v in cands_tfidf.values())/len(cands_tfidf):.1f})")

if __name__ == "__main__":
    main()
