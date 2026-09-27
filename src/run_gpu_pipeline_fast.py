#!/usr/bin/env python3
"""
OPTIMIZED GPU Pipeline — Phase 2 Only (Skip Training, Load Pre-Trained Models).
Speed optimizations:
  1. Loads pre-trained CatBoost + gate models (skips ~7 min training).
  2. Parallel feature extraction across 16 CPU cores (joblib).
  3. Increased GPU batch size: 60,000 S1 entities per chunk.
  4. GPU ram_part=0.90 for max VRAM utilization.
  5. Aggressive GC + float32 arrays.
"""

from collections import defaultdict
import csv
import gc
import json
import os
import pickle
import re
import sys
import time
from typing import Dict, List, Tuple

import catboost as cb
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import (
    FeatureExtractor,
    normalize_text,
    extract_core_name,
    extract_sorted_key,
    extract_distinctive_name_tokens,
    extract_postal_and_number,
    is_non_ascii_name,
    prepare_record,
    GENERIC_WORDS,
)

TRAIN_DIR = "dataset/train"
TEST_DIR  = "dataset/test"
OUTPUT_DIR = "output"

GATE_BASE_FEATURES = (
    "name_strength", "addr_strength", "name_x_addr", "token_sort",
    "token_set", "core_exact", "char_ngram_jaccard", "max_shared_idf",
    "postal_match", "num_match", "country_rel",
)

# ─── Reuse blocking helpers from original ────────────────────────────────────

def compute_s1_blocking_keys(name, address, country):
    norm = normalize_text(name)
    core = extract_core_name(name)
    sorted_k = extract_sorted_key(name)
    concat_k = norm.replace(" ", "")
    brand = extract_distinctive_name_tokens(name)
    postal, num, dist_tokens = extract_postal_and_number(address)

    alias_cores = []
    if any(w in norm for w in ("aka", "dba", "fka", "doing business as", "trading as")):
        parts = re.split(r'\b(?:aka|dba|fka|doing business as|trading as)\b', norm)
        if len(parts) > 1:
            for p in parts:
                p = p.strip()
                ac = extract_core_name(p) if p else ""
                if ac and len(ac) >= 3:
                    alias_cores.append(ac)

    postal_num_key = f"{postal}_{num}" if (postal and num) else ""
    addr_num_keys = [f"{num}_{dt}" for dt in sorted(dist_tokens)[:3]] if (num and dist_tokens) else []

    return {
        "norm": norm, "core": core, "sorted": sorted_k,
        "concat": concat_k, "brand": brand, "postal": postal,
        "num": num, "dist_tokens": dist_tokens,
        "country": country.strip().lower(), "aliases": alias_cores,
        "postal_num_key": postal_num_key, "addr_num_keys": addr_num_keys,
    }


def build_filtered_target_index(source_dir, source_files, needed_keys, caps):
    idx_core = defaultdict(list); idx_sorted = defaultdict(list)
    idx_concat = defaultdict(list); idx_brand = defaultdict(list)
    idx_postal_num = defaultdict(list); idx_addr_num = defaultdict(list)
    target_data = {}

    cores_set = needed_keys["cores"]; sorted_set = needed_keys["sorted"]
    concat_set = needed_keys["concat"]; brand_set = needed_keys["brand"]
    postal_num_set = needed_keys["postal_num"]; addr_num_set = needed_keys["addr_num"]

    cap_core = caps.get("core", 60); cap_sorted = caps.get("sorted", 50)
    cap_concat = caps.get("concat", 30); cap_brand = caps.get("brand", 40)
    cap_postal_num = caps.get("postal_num", 30); cap_addr_num = caps.get("addr_num", 25)

    for fname in source_files:
        path = os.path.join(source_dir, fname)
        if not os.path.exists(path):
            continue
        print(f"  Indexing {fname}...")
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                t_id = parts[0].strip()
                t_name = parts[1] if len(parts) > 1 else ""
                t_addr = parts[2] if len(parts) > 2 else ""
                t_country = parts[3].strip().lower() if len(parts) > 3 else ""

                norm = normalize_text(t_name)
                t_core = extract_core_name(t_name)
                t_sorted = extract_sorted_key(t_name)
                t_concat = norm.replace(" ", "")

                hit = False
                if t_core in cores_set and len(idx_core[t_core]) < cap_core:
                    idx_core[t_core].append(t_id); hit = True

                if any(w in norm for w in ("aka","dba","fka","doing business as","trading as")):
                    alias_parts = re.split(r'\b(?:aka|dba|fka|doing business as|trading as)\b', norm)
                    if len(alias_parts) > 1:
                        for p in alias_parts:
                            p = p.strip(); ac = extract_core_name(p) if p else ""
                            if ac and ac in cores_set and len(idx_core[ac]) < cap_core:
                                idx_core[ac].append(t_id); hit = True

                if t_sorted in sorted_set and len(idx_sorted[t_sorted]) < cap_sorted:
                    idx_sorted[t_sorted].append(t_id); hit = True
                if len(t_concat) >= 5 and t_concat in concat_set and len(idx_concat[t_concat]) < cap_concat:
                    idx_concat[t_concat].append(t_id); hit = True

                t_brand = extract_distinctive_name_tokens(t_name)
                for b in t_brand:
                    if b in brand_set and b not in GENERIC_WORDS and len(idx_brand[b]) < cap_brand:
                        idx_brand[b].append(t_id); hit = True

                if t_addr:
                    t_postal, t_num, t_dist = extract_postal_and_number(t_addr)
                    if t_postal and t_num:
                        pn = f"{t_postal}_{t_num}"
                        if pn in postal_num_set and len(idx_postal_num[pn]) < cap_postal_num:
                            idx_postal_num[pn].append(t_id); hit = True
                    if t_num and t_dist:
                        for dt in sorted(t_dist)[:3]:
                            ak = f"{t_num}_{dt}"
                            if ak in addr_num_set and len(idx_addr_num[ak]) < cap_addr_num:
                                idx_addr_num[ak].append(t_id); hit = True

                if hit:
                    target_data[t_id] = prepare_record(t_name, t_addr, t_country)

    return {"core": idx_core, "sorted": idx_sorted, "concat": idx_concat,
            "brand": idx_brand, "postal_num": idx_postal_num, "addr_num": idx_addr_num}, target_data


def generate_candidate_pools(s1_meta, indices, max_per_entity=75):
    idx_core = indices["core"]; idx_sorted = indices["sorted"]
    idx_concat = indices["concat"]; idx_brand = indices["brand"]
    idx_postal_num = indices["postal_num"]; idx_addr_num = indices["addr_num"]
    candidates = {}

    for s1_id, meta in s1_meta.items():
        ordered = []; seen = set()

        def add(values, max_r):
            added = 0
            for val in values:
                if added >= max_r or len(ordered) >= max_per_entity: break
                if val not in seen: seen.add(val); ordered.append(val); added += 1

        add(idx_core.get(meta["core"], []), 20)
        for alias in meta["aliases"]: add(idx_core.get(alias, []), 10)
        add(idx_sorted.get(meta["sorted"], []), 15)
        if len(meta["concat"]) >= 5: add(idx_concat.get(meta["concat"], []), 12)
        for brand in list(meta["brand"])[:4]: add(idx_brand.get(brand, []), 10)
        if meta["postal_num_key"]: add(idx_postal_num.get(meta["postal_num_key"], []), 15)
        for addr_key in meta["addr_num_keys"]: add(idx_addr_num.get(addr_key, []), 10)

        candidates[s1_id] = ordered
    return candidates


def aggregate_singleton_features(candidate_features):
    columns = ["candidate_count", "s2_candidate_count", "s3_candidate_count"]
    for name in GATE_BASE_FEATURES:
        columns.extend((f"{name}_max", f"{name}_top3_mean"))
    result = {}
    for s1_id, rows in candidate_features.items():
        s2_n = sum(1 for r in rows if r.get("is_s2", 0.0) > 0.5)
        s3_n = sum(1 for r in rows if r.get("is_s3", 0.0) > 0.5)
        values = [float(len(rows)), float(s2_n), float(s3_n)]
        for name in GATE_BASE_FEATURES:
            scores = sorted((float(r.get(name, 0.0)) for r in rows), reverse=True)
            values.extend((scores[0] if scores else 0.0,
                           float(np.mean(scores[:3])) if scores else 0.0))
        result[s1_id] = values
    return result, columns


# ─── PARALLEL FEATURE EXTRACTION helper ──────────────────────────────────────

def _extract_batch_chunk(chunk_records, candidates_dict, target_data, feat_names, gate_feat_names_list):
    """Worker: extract features for a sub-chunk of s1 records. Returns (pair_rows, pair_meta, gate_cands)."""
    fe = FeatureExtractor()
    pair_rows = []; pair_meta = []
    gate_cands = {}

    for s1_id, name, addr, country in chunk_records:
        s1_rec = prepare_record(name, addr, country)
        cand_list = candidates_dict.get(s1_id, [])
        pool_sz = len(cand_list)
        gate_cands[s1_id] = []

        for rank, c_id in enumerate(cand_list, start=1):
            c_rec = target_data.get(c_id)
            if not c_rec: continue
            s1_c = s1_rec.get("country", "")
            c_c  = c_rec.get("country", "")
            if s1_c and c_c and s1_c != c_c: continue

            fd = fe.extract_pair_features(s1_rec, c_rec, c_id, rank, pool_sz)
            if feat_names is None:
                feat_names = sorted(fd.keys())
            pair_rows.append([fd[k] for k in feat_names])
            pair_meta.append((s1_id, c_id))

            gate_fd = {k: fd[k] for k in GATE_BASE_FEATURES}
            gate_fd["is_s2"] = fd["is_s2"]; gate_fd["is_s3"] = fd["is_s3"]
            gate_cands[s1_id].append(gate_fd)

    return pair_rows, pair_meta, gate_cands


# ─── MAIN PHASE 2: Optimized GPU Inference ───────────────────────────────────

def run_full_test_gpu_inference_fast(
    cb_model, feat_names, best_tau, gate_model, gate_feat_names,
    tau_singleton, test_dir=TEST_DIR, output_dir=OUTPUT_DIR,
    batch_size_s1=60000, n_jobs=16,
):
    print("\n" + "=" * 80)
    print(" [PHASE 2] FAST GPU INFERENCE — 1.73M S1 ENTITIES")
    print(f" batch_size={batch_size_s1:,}  n_jobs={n_jobs}  GPU=RTX3060")
    print("=" * 80)
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    s1_path = os.path.join(test_dir, "test_source1.tsv")

    # ── Pass 1: Collect all S1 blocking keys ─────────────────────────────────
    print("\nPass 1: Streaming test S1 → collecting query keys...")
    t0 = time.time()
    needed_keys = {"cores": set(), "sorted": set(), "concat": set(),
                   "brand": set(), "postal_num": set(), "addr_num": set()}
    total_test_s1 = 0

    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            total_test_s1 += 1
            parts = line.rstrip("\r\n").split("\t")
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else ""
            meta = compute_s1_blocking_keys(name, addr, country)
            if meta["core"]: needed_keys["cores"].add(meta["core"])
            for ac in meta["aliases"]: needed_keys["cores"].add(ac)
            if meta["sorted"]: needed_keys["sorted"].add(meta["sorted"])
            if len(meta["concat"]) >= 5: needed_keys["concat"].add(meta["concat"])
            for b in meta["brand"]: needed_keys["brand"].add(b)
            if meta["postal_num_key"]: needed_keys["postal_num"].add(meta["postal_num_key"])
            for ak in meta["addr_num_keys"]: needed_keys["addr_num"].add(ak)

    print(f"Pass 1 done in {time.time()-t0:.1f}s — {total_test_s1:,} S1 records.")

    # ── Pass 2: Index test targets ────────────────────────────────────────────
    print("\nPass 2: Building filtered target index from test_source2/3.tsv...")
    t0 = time.time()
    caps = {"core": 60, "sorted": 50, "concat": 30, "brand": 40, "postal_num": 30, "addr_num": 25}
    indices, target_data = build_filtered_target_index(
        test_dir, ["test_source2.tsv", "test_source3.tsv"], needed_keys, caps
    )
    print(f"Pass 2 done in {time.time()-t0:.1f}s — indexed {len(target_data):,} targets.")
    gc.collect()

    # ── Pass 3: Parallel feature extraction + GPU scoring ─────────────────────
    temp_cand_path = os.path.join(output_dir, "temp_candidates_stream.tsv")
    print(f"\nPass 3: Parallel scoring ({n_jobs} cores + GPU) in chunks of {batch_size_s1:,}...")
    fe_master = FeatureExtractor()
    target_claims = {}   # t_id -> (best_s1_id, best_prob)
    s1_passed_matches = defaultdict(list)
    gate_blocked_count = 0
    total_candidate_pairs = 0
    total_gpu_time = 0.0
    batch_num = 0
    t0 = time.time()

    def process_batch(batch_records):
        nonlocal total_candidate_pairs, total_gpu_time, gate_blocked_count

        # Generate candidate pools for entire batch
        batch_s1_meta = {}
        for s1_id, name, addr, country in batch_records:
            batch_s1_meta[s1_id] = compute_s1_blocking_keys(name, addr, country)
        cands_dict = generate_candidate_pools(batch_s1_meta, indices, max_per_entity=75)

        # Write candidate pairs to stream file
        with open(temp_cand_path, "a", encoding="utf-8") as f_tmp:
            for s1_id, name, addr, country in batch_records:
                cand_list = cands_dict.get(s1_id, [])
                f_tmp.write(f"{s1_id}\t{','.join(cand_list)}\n")

        # ── PARALLEL feature extraction across n_jobs cores ──────────────────
        chunk_size = max(1, len(batch_records) // n_jobs)
        chunks = [batch_records[i:i+chunk_size] for i in range(0, len(batch_records), chunk_size)]

        results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
            delayed(_extract_batch_chunk)(chunk, cands_dict, target_data, feat_names, list(GATE_BASE_FEATURES))
            for chunk in chunks
        )

        # Merge results from all workers
        all_pair_rows = []; all_pair_meta = []; batch_gate_candidates = {}
        for pair_rows_w, pair_meta_w, gate_cands_w in results:
            all_pair_rows.extend(pair_rows_w)
            all_pair_meta.extend(pair_meta_w)
            for s1_id, gc_list in gate_cands_w.items():
                batch_gate_candidates.setdefault(s1_id, []).extend(gc_list)

        # Singleton gate prediction
        batch_singleton_prob = {}
        if tau_singleton < 1.0 and batch_gate_candidates:
            gate_rows, _ = aggregate_singleton_features(batch_gate_candidates)
            if gate_rows:
                gate_ids = list(gate_rows)
                X_gate = np.asarray([gate_rows[s] for s in gate_ids], dtype=np.float32)
                batch_singleton_prob = dict(zip(gate_ids, map(float, gate_model.predict_proba(X_gate)[:, 1])))

        # GPU scoring
        if all_pair_rows:
            X_batch = np.array(all_pair_rows, dtype=np.float32)
            total_candidate_pairs += len(X_batch)
            t_gpu = time.time()
            probs = cb_model.predict_proba(X_batch)[:, 1]
            total_gpu_time += (time.time() - t_gpu)

            for (s1_id, c_id), prob in zip(all_pair_meta, probs):
                if tau_singleton < 1.0 and batch_singleton_prob.get(s1_id, 1.0) >= tau_singleton:
                    gate_blocked_count += 1
                    continue
                pair_tau = best_tau["s2"] if "s2" in c_id.lower() else best_tau["s3"]
                if prob >= pair_tau:
                    s1_passed_matches[s1_id].append((c_id, float(prob)))
                    prev = target_claims.get(c_id)
                    if prev is None or prob > prev[1]:
                        target_claims[c_id] = (s1_id, float(prob))

    # Clear temp file and stream S1 in batches
    open(temp_cand_path, "w").close()
    current_batch = []
    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else ""
            current_batch.append((s1_id, name, addr, country))

            if len(current_batch) >= batch_size_s1:
                batch_num += 1
                process_batch(current_batch)
                processed = batch_num * batch_size_s1
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_test_s1 - processed) / rate if rate > 0 else 0
                print(f"  Batch {batch_num:2d}: {min(processed, total_test_s1):,}/{total_test_s1:,} "
                      f"({min(processed, total_test_s1)/total_test_s1*100:.1f}%) | "
                      f"Pairs: {total_candidate_pairs:,} | "
                      f"Elapsed: {elapsed:.0f}s | ETA: {eta/60:.1f}min", flush=True)
                current_batch = []
                gc.collect()

        if current_batch:
            batch_num += 1
            process_batch(current_batch)
            print(f"  Final Batch {batch_num}: {total_test_s1:,}/{total_test_s1:,} (100%) | "
                  f"Total Pairs: {total_candidate_pairs:,}", flush=True)

    throughput = total_candidate_pairs / max(0.001, total_gpu_time)
    print(f"\nGPU Summary: {total_candidate_pairs:,} pairs | {throughput:,.0f} pairs/sec | "
          f"Gate blocked: {gate_blocked_count:,}")

    # ── Pass 4: 1-to-1 conflict resolution ───────────────────────────────────
    print("\nPass 4: 1-to-1 conflict resolution...")
    t0 = time.time()
    winner_for_target = {c_id: val[0] for c_id, val in target_claims.items()}
    del target_claims; gc.collect()
    print(f"Resolved {len(winner_for_target):,} winners in {time.time()-t0:.2f}s.")

    # ── Pass 5: Write output TSVs ─────────────────────────────────────────────
    print("\nPass 5: Writing matching_results.tsv + candidate_pairs.tsv...")
    t0 = time.time()
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")

    total_matches = singleton_preds = final_row_count = 0
    with open(matching_file, "w", encoding="utf-8") as fm, \
         open(candidate_file, "w", encoding="utf-8") as fc, \
         open(temp_cand_path, "r", encoding="utf-8") as f_tmp:

        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        for line in f_tmp:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            final_row_count += 1
            cand_str = parts[1] if len(parts) > 1 else ""
            fc.write(f"{s1_id}\t{cand_str}\n")

            raw_matches = s1_passed_matches.get(s1_id, [])
            final_matches = [
                c_id for c_id, prob in sorted(raw_matches, key=lambda x: -x[1])
                if winner_for_target.get(c_id) == s1_id
            ]
            if final_matches:
                total_matches += 1
                fm.write(f"{s1_id}\t{','.join(final_matches)}\n")
            else:
                singleton_preds += 1
                fm.write(f"{s1_id}\t\n")

    print(f"Wrote {final_row_count:,} rows | Matched: {total_matches:,} | Singleton: {singleton_preds:,}")
    print(f"Output files written in {time.time()-t0:.2f}s")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"PHASE 2 COMPLETE in {total_time/60:.1f} minutes!")
    print(f"  matching_results.tsv -> {matching_file}")
    print(f"  candidate_pairs.tsv  -> {candidate_file}")
    print(f"{'='*60}")

    return {"total_matched": total_matches, "total_singleton": singleton_preds,
            "total_s1": final_row_count, "runtime_min": round(total_time/60, 2)}


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fast Phase-2 GPU Pipeline (pre-trained models)")
    parser.add_argument("--test-dir",   default=TEST_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=60000)
    parser.add_argument("--n-jobs",     type=int, default=16)
    args = parser.parse_args()

    model_path = os.path.join(args.output_dir, "catboost_gpu_model.cbm")
    gate_path  = os.path.join(args.output_dir, "gate_model.pkl")
    meta_path  = os.path.join(args.output_dir, "model_metadata.json")

    for p in [model_path, gate_path, meta_path]:
        if not os.path.exists(p):
            print(f"ERROR: Missing {p}. Run full pipeline first!")
            sys.exit(1)

    print("="*60)
    print(" FAST GPU PIPELINE — PHASE 2 ONLY")
    print("="*60)
    print("Loading pre-trained models...")
    cb_model = cb.CatBoostClassifier()
    cb_model.load_model(model_path)

    with open(gate_path, "rb") as f:
        gate_model = pickle.load(f)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feat_names       = meta["feat_names"]
    best_tau         = meta["best_tau"]
    gate_feat_names  = meta["gate_feat_names"]
    tau_singleton    = meta["best_tau_singleton"]
    best_f05         = meta["best_macro_f05"]

    print(f"Models loaded! Val Macro F0.5={best_f05:.4f} | "
          f"S2 tau={best_tau['s2']:.3f} | S3 tau={best_tau['s3']:.3f} | "
          f"Singleton tau={tau_singleton:.3f}")

    run_full_test_gpu_inference_fast(
        cb_model=cb_model,
        feat_names=feat_names,
        best_tau=best_tau,
        gate_model=gate_model,
        gate_feat_names=gate_feat_names,
        tau_singleton=tau_singleton,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        batch_size_s1=args.batch_size,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
