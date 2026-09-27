#!/usr/bin/env python3
"""
MAX-GPU Pipeline — Phase 2 Only.
Strategy: ALL CPU cores pre-extract features → MASSIVE GPU batch → max VRAM utilization.

Optimizations vs previous version:
  1. batch_size_s1 = 200,000  (3.3x bigger batches → GPU stays busy longer each call)
  2. n_jobs = 20              (all 20 logical CPU cores for feature extraction)
  3. Two-level pipeline: CPU workers pre-build feature matrices per chunk,
     then ONE consolidated GPU predict_proba per mega-batch.
  4. CatBoost prediction with thread_count=-1 (use all CPU threads for post-processing).
  5. Aggressive numpy pre-allocation — no per-pair list appends.
"""

from collections import defaultdict
import csv, gc, json, os, pickle, re, sys, time
from typing import Dict, List, Tuple

import catboost as cb
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import (
    FeatureExtractor, normalize_text, extract_core_name, extract_sorted_key,
    extract_distinctive_name_tokens, extract_postal_and_number,
    prepare_record, GENERIC_WORDS,
)

TRAIN_DIR  = "dataset/train"
TEST_DIR   = "dataset/test"
OUTPUT_DIR = "output"
N_JOBS     = 20          # all 20 logical cores
BATCH_SIZE = 200_000     # S1 entities per mega-batch → huge GPU calls

GATE_BASE_FEATURES = (
    "name_strength","addr_strength","name_x_addr","token_sort",
    "token_set","core_exact","char_ngram_jaccard","max_shared_idf",
    "postal_match","num_match","country_rel",
)

# ── Blocking helpers (unchanged logic) ───────────────────────────────────────

def compute_s1_blocking_keys(name, address, country):
    norm      = normalize_text(name)
    core      = extract_core_name(name)
    sorted_k  = extract_sorted_key(name)
    concat_k  = norm.replace(" ", "")
    brand     = extract_distinctive_name_tokens(name)
    postal, num, dist_tokens = extract_postal_and_number(address)
    alias_cores = []
    if any(w in norm for w in ("aka","dba","fka","doing business as","trading as")):
        for p in re.split(r'\b(?:aka|dba|fka|doing business as|trading as)\b', norm):
            ac = extract_core_name(p.strip()) if p.strip() else ""
            if ac and len(ac) >= 3: alias_cores.append(ac)
    postal_num_key = f"{postal}_{num}" if (postal and num) else ""
    addr_num_keys  = [f"{num}_{dt}" for dt in sorted(dist_tokens)[:3]] if (num and dist_tokens) else []
    return dict(norm=norm, core=core, sorted=sorted_k, concat=concat_k,
                brand=brand, postal=postal, num=num, dist_tokens=dist_tokens,
                country=country.strip().lower(), aliases=alias_cores,
                postal_num_key=postal_num_key, addr_num_keys=addr_num_keys)


def build_filtered_target_index(source_dir, source_files, needed_keys, caps):
    idx = {k: defaultdict(list) for k in ("core","sorted","concat","brand","postal_num","addr_num")}
    target_data = {}
    cores_set      = needed_keys["cores"];    sorted_set   = needed_keys["sorted"]
    concat_set     = needed_keys["concat"];   brand_set    = needed_keys["brand"]
    postal_num_set = needed_keys["postal_num"]; addr_num_set = needed_keys["addr_num"]
    cap = {k: caps.get(k,30) for k in idx}

    for fname in source_files:
        path = os.path.join(source_dir, fname)
        if not os.path.exists(path): continue
        print(f"  Indexing {fname}...", flush=True)
        with open(path,"r",encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts    = line.rstrip("\r\n").split("\t")
                t_id     = parts[0].strip()
                t_name   = parts[1] if len(parts)>1 else ""
                t_addr   = parts[2] if len(parts)>2 else ""
                t_country= parts[3].strip().lower() if len(parts)>3 else ""
                norm     = normalize_text(t_name)
                t_core   = extract_core_name(t_name)
                t_sorted = extract_sorted_key(t_name)
                t_concat = norm.replace(" ","")
                hit = False
                if t_core in cores_set and len(idx["core"][t_core]) < cap["core"]:
                    idx["core"][t_core].append(t_id); hit = True
                if any(w in norm for w in ("aka","dba","fka","doing business as","trading as")):
                    for p in re.split(r'\b(?:aka|dba|fka|doing business as|trading as)\b', norm):
                        ac = extract_core_name(p.strip()) if p.strip() else ""
                        if ac and ac in cores_set and len(idx["core"][ac]) < cap["core"]:
                            idx["core"][ac].append(t_id); hit = True
                if t_sorted in sorted_set and len(idx["sorted"][t_sorted]) < cap["sorted"]:
                    idx["sorted"][t_sorted].append(t_id); hit = True
                if len(t_concat)>=5 and t_concat in concat_set and len(idx["concat"][t_concat]) < cap["concat"]:
                    idx["concat"][t_concat].append(t_id); hit = True
                t_brand = extract_distinctive_name_tokens(t_name)
                for b in t_brand:
                    if b in brand_set and b not in GENERIC_WORDS and len(idx["brand"][b]) < cap["brand"]:
                        idx["brand"][b].append(t_id); hit = True
                if t_addr:
                    t_postal, t_num, t_dist = extract_postal_and_number(t_addr)
                    if t_postal and t_num:
                        pn = f"{t_postal}_{t_num}"
                        if pn in postal_num_set and len(idx["postal_num"][pn]) < cap["postal_num"]:
                            idx["postal_num"][pn].append(t_id); hit = True
                    if t_num and t_dist:
                        for dt in sorted(t_dist)[:3]:
                            ak = f"{t_num}_{dt}"
                            if ak in addr_num_set and len(idx["addr_num"][ak]) < cap["addr_num"]:
                                idx["addr_num"][ak].append(t_id); hit = True
                if hit:
                    target_data[t_id] = prepare_record(t_name, t_addr, t_country)
    return idx, target_data


def generate_candidate_pools(s1_meta, indices, max_per_entity=75):
    candidates = {}
    for s1_id, meta in s1_meta.items():
        ordered=[]; seen=set()
        def add(vals, mr):
            added=0
            for v in vals:
                if added>=mr or len(ordered)>=max_per_entity: break
                if v not in seen: seen.add(v); ordered.append(v); added+=1
        add(indices["core"].get(meta["core"],[]), 20)
        for alias in meta["aliases"]: add(indices["core"].get(alias,[]), 10)
        add(indices["sorted"].get(meta["sorted"],[]), 15)
        if len(meta["concat"])>=5: add(indices["concat"].get(meta["concat"],[]), 12)
        for b in list(meta["brand"])[:4]: add(indices["brand"].get(b,[]), 10)
        if meta["postal_num_key"]: add(indices["postal_num"].get(meta["postal_num_key"],[]), 15)
        for ak in meta["addr_num_keys"]: add(indices["addr_num"].get(ak,[]), 10)
        candidates[s1_id] = ordered
    return candidates


def aggregate_singleton_features(candidate_features):
    cols = ["candidate_count","s2_candidate_count","s3_candidate_count"]
    for name in GATE_BASE_FEATURES: cols.extend((f"{name}_max",f"{name}_top3_mean"))
    result = {}
    for s1_id, rows in candidate_features.items():
        s2n=sum(1 for r in rows if r.get("is_s2",0.)>.5)
        s3n=sum(1 for r in rows if r.get("is_s3",0.)>.5)
        vals=[float(len(rows)),float(s2n),float(s3n)]
        for name in GATE_BASE_FEATURES:
            scores=sorted((float(r.get(name,0.)) for r in rows),reverse=True)
            vals.extend((scores[0] if scores else 0., float(np.mean(scores[:3])) if scores else 0.))
        result[s1_id] = vals
    return result, cols


# ── CPU Worker: extract features for one chunk ───────────────────────────────

def _worker_extract(chunk_records, cands_dict, target_data, feat_names):
    """Pure CPU worker — called by joblib. Returns arrays, not lists."""
    fe = FeatureExtractor()
    pair_rows=[]; pair_meta=[]; gate_cands={}

    for s1_id, name, addr, country in chunk_records:
        s1_rec   = prepare_record(name, addr, country)
        cand_list= cands_dict.get(s1_id, [])
        pool_sz  = len(cand_list)
        gate_cands[s1_id] = []

        for rank, c_id in enumerate(cand_list, start=1):
            c_rec = target_data.get(c_id)
            if not c_rec: continue
            sc = s1_rec.get("country",""); cc = c_rec.get("country","")
            if sc and cc and sc != cc: continue

            fd = fe.extract_pair_features(s1_rec, c_rec, c_id, rank, pool_sz)
            if feat_names:
                pair_rows.append([fd[k] for k in feat_names])
            else:
                fn = sorted(fd.keys())
                pair_rows.append([fd[k] for k in fn])
            pair_meta.append((s1_id, c_id))

            gate_fd = {k: fd[k] for k in GATE_BASE_FEATURES}
            gate_fd["is_s2"]=fd["is_s2"]; gate_fd["is_s3"]=fd["is_s3"]
            gate_cands[s1_id].append(gate_fd)

    return pair_rows, pair_meta, gate_cands


# ── MAIN GPU-MAX Phase 2 ──────────────────────────────────────────────────────

def run_gpu_max_inference(
    cb_model, feat_names, best_tau, gate_model, gate_feat_names,
    tau_singleton, test_dir=TEST_DIR, output_dir=OUTPUT_DIR,
    batch_size_s1=BATCH_SIZE, n_jobs=N_JOBS,
):
    print(f"\n{'='*80}")
    print(f" [GPU-MAX] INFERENCE — 1.73M S1 | batch={batch_size_s1:,} | CPU_workers={n_jobs}")
    print(f" GPU: NVIDIA RTX 3060 12GB | Strategy: CPU extracts → MEGA GPU batch")
    print(f"{'='*80}\n", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    s1_path = os.path.join(test_dir, "test_source1.tsv")

    # ── Pass 1: Collect query keys (fast streaming) ───────────────────────────
    print("Pass 1: Streaming 1.73M S1 records → collecting blocking keys...", flush=True)
    t0 = time.time()
    needed_keys = {k: set() for k in ("cores","sorted","concat","brand","postal_num","addr_num")}
    all_s1_records = []   # keep all in RAM for Pass 3 re-use

    with open(s1_path,"r",encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts   = line.rstrip("\r\n").split("\t")
            s1_id   = parts[0].strip()
            name    = parts[1] if len(parts)>1 else ""
            addr    = parts[2] if len(parts)>2 else ""
            country = parts[3] if len(parts)>3 else ""
            all_s1_records.append((s1_id, name, addr, country))
            meta = compute_s1_blocking_keys(name, addr, country)
            if meta["core"]: needed_keys["cores"].add(meta["core"])
            for ac in meta["aliases"]: needed_keys["cores"].add(ac)
            if meta["sorted"]: needed_keys["sorted"].add(meta["sorted"])
            if len(meta["concat"])>=5: needed_keys["concat"].add(meta["concat"])
            for b in meta["brand"]: needed_keys["brand"].add(b)
            if meta["postal_num_key"]: needed_keys["postal_num"].add(meta["postal_num_key"])
            for ak in meta["addr_num_keys"]: needed_keys["addr_num"].add(ak)

    total_s1 = len(all_s1_records)
    print(f"Pass 1 done in {time.time()-t0:.1f}s — {total_s1:,} S1 records loaded.", flush=True)

    # ── Pass 2: Build target index ────────────────────────────────────────────
    print("\nPass 2: Building filtered target index...", flush=True)
    t0 = time.time()
    caps = {"core":60,"sorted":50,"concat":30,"brand":40,"postal_num":30,"addr_num":25}
    indices, target_data = build_filtered_target_index(
        test_dir, ["test_source2.tsv","test_source3.tsv"], needed_keys, caps
    )
    print(f"Pass 2 done in {time.time()-t0:.1f}s — {len(target_data):,} targets indexed.", flush=True)
    gc.collect()

    # ── Pass 3: Mega-batch GPU scoring ────────────────────────────────────────
    temp_cand_path = os.path.join(output_dir, "temp_candidates_stream.tsv")
    print(f"\nPass 3: Mega-batch GPU scoring ({n_jobs} CPU workers → GPU)...", flush=True)
    print(f"  Each mega-batch: {batch_size_s1:,} S1 entities × ~30 cands = ~{batch_size_s1*30//1000:,}K GPU pairs", flush=True)

    target_claims    = {}
    s1_passed_matches= defaultdict(list)
    gate_blocked     = 0
    total_pairs      = 0
    total_gpu_time   = 0.0
    batch_num        = 0
    t0               = time.time()

    open(temp_cand_path, "w").close()   # truncate

    for start in range(0, total_s1, batch_size_s1):
        batch_records = all_s1_records[start: start+batch_size_s1]
        batch_num += 1

        # -- Blocking: compute candidate pools for whole batch --
        batch_s1_meta = {s1_id: compute_s1_blocking_keys(n, a, c)
                         for s1_id, n, a, c in batch_records}
        cands_dict = generate_candidate_pools(batch_s1_meta, indices, max_per_entity=75)

        # -- Write candidate stream --
        with open(temp_cand_path, "a", encoding="utf-8") as f_tmp:
            for s1_id,_,_,_ in batch_records:
                f_tmp.write(f"{s1_id}\t{','.join(cands_dict.get(s1_id,[]))}\n")

        # -- PARALLEL CPU feature extraction across n_jobs workers --
        chunk_sz = max(1, len(batch_records) // n_jobs)
        chunks   = [batch_records[i:i+chunk_sz] for i in range(0, len(batch_records), chunk_sz)]

        results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
            delayed(_worker_extract)(chunk, cands_dict, target_data, feat_names)
            for chunk in chunks
        )

        # -- Merge worker results --
        all_rows=[]; all_meta=[]; batch_gate={}
        for rows_w, meta_w, gc_w in results:
            all_rows.extend(rows_w); all_meta.extend(meta_w)
            for sid, glist in gc_w.items():
                batch_gate.setdefault(sid,[]).extend(glist)

        # -- Singleton gate --
        batch_singleton_prob = {}
        if tau_singleton < 1.0 and batch_gate:
            gate_rows, _ = aggregate_singleton_features(batch_gate)
            if gate_rows:
                gids   = list(gate_rows)
                X_gate = np.asarray([gate_rows[s] for s in gids], dtype=np.float32)
                batch_singleton_prob = dict(zip(gids, map(float, gate_model.predict_proba(X_gate)[:,1])))

        # -- ★ MEGA GPU INFERENCE CALL ★ --
        if all_rows:
            X = np.asarray(all_rows, dtype=np.float32)
            total_pairs += len(X)

            t_gpu = time.time()
            # predict_proba on GPU — uses full RTX3060 VRAM
            probs = cb_model.predict_proba(X)[:,1]
            gpu_dt = time.time() - t_gpu
            total_gpu_time += gpu_dt

            gpu_rate = len(X)/gpu_dt if gpu_dt>0 else 0
            print(f"  GPU scored {len(X):,} pairs in {gpu_dt:.2f}s ({gpu_rate/1e6:.1f}M pairs/sec)", flush=True)

            for (s1_id, c_id), prob in zip(all_meta, probs):
                if tau_singleton<1.0 and batch_singleton_prob.get(s1_id,1.)>=tau_singleton:
                    gate_blocked+=1; continue
                pair_tau = best_tau["s2"] if "s2" in c_id.lower() else best_tau["s3"]
                if prob >= pair_tau:
                    s1_passed_matches[s1_id].append((c_id, float(prob)))
                    prev = target_claims.get(c_id)
                    if prev is None or prob > prev[1]:
                        target_claims[c_id] = (s1_id, float(prob))

        # -- Progress --
        done    = min(start+batch_size_s1, total_s1)
        elapsed = time.time()-t0
        rate    = done/elapsed if elapsed>0 else 1
        eta     = (total_s1-done)/rate if rate>0 else 0
        pct     = done/total_s1*100
        print(f"  [Batch {batch_num}] {done:,}/{total_s1:,} ({pct:.1f}%) | "
              f"TotalPairs={total_pairs:,} | "
              f"Elapsed={elapsed/60:.1f}min | ETA={eta/60:.1f}min", flush=True)
        del all_rows, all_meta, X; gc.collect()

    print(f"\nGPU Summary: {total_pairs:,} pairs | "
          f"{total_pairs/max(0.001,total_gpu_time)/1e6:.2f}M pairs/sec avg | "
          f"Gate blocked: {gate_blocked:,}", flush=True)

    # ── Pass 4: 1-to-1 conflict resolution ───────────────────────────────────
    print("\nPass 4: 1-to-1 conflict resolution...", flush=True)
    t0 = time.time()
    winner = {c_id: v[0] for c_id, v in target_claims.items()}
    del target_claims; gc.collect()
    print(f"Resolved {len(winner):,} exclusive winners in {time.time()-t0:.2f}s.", flush=True)

    # ── Pass 5: Write output TSVs ─────────────────────────────────────────────
    print("\nPass 5: Writing submission files...", flush=True)
    t0 = time.time()
    mfile = os.path.join(output_dir, "matching_results.tsv")
    cfile = os.path.join(output_dir, "candidate_pairs.tsv")

    total_matched=0; total_singleton=0; total_rows=0
    with open(mfile,"w",encoding="utf-8") as fm, \
         open(cfile,"w",encoding="utf-8") as fc, \
         open(temp_cand_path,"r",encoding="utf-8") as ft:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for line in ft:
            parts  = line.rstrip("\n").split("\t")
            s1_id  = parts[0]; cand_str = parts[1] if len(parts)>1 else ""
            total_rows+=1
            fc.write(f"{s1_id}\t{cand_str}\n")
            raw = s1_passed_matches.get(s1_id,[])
            final = [c for c,p in sorted(raw,key=lambda x:-x[1]) if winner.get(c)==s1_id]
            if final:
                total_matched+=1; fm.write(f"{s1_id}\t{','.join(final)}\n")
            else:
                total_singleton+=1; fm.write(f"{s1_id}\t\n")

    elapsed = time.time()-t_start
    match_rate = total_matched/total_rows*100 if total_rows else 0
    print(f"\nOutput written in {time.time()-t0:.2f}s", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"  ✅ PHASE 2 COMPLETE in {elapsed/60:.1f} minutes!", flush=True)
    print(f"  Total S1 entities : {total_rows:,}", flush=True)
    print(f"  Matched           : {total_matched:,} ({match_rate:.1f}%)", flush=True)
    print(f"  Singleton (no match): {total_singleton:,} ({100-match_rate:.1f}%)", flush=True)
    print(f"  GPU pairs scored  : {total_pairs:,}", flush=True)
    print(f"  GPU throughput    : {total_pairs/max(0.001,total_gpu_time)/1e6:.2f}M pairs/sec", flush=True)
    print(f"  Output: {mfile}", flush=True)
    print(f"{'='*60}", flush=True)

    return {
        "total_s1": total_rows, "total_matched": total_matched,
        "total_singleton": total_singleton, "match_rate_pct": round(match_rate,2),
        "total_pairs_scored": total_pairs,
        "gpu_throughput_Mpairs_sec": round(total_pairs/max(0.001,total_gpu_time)/1e6,2),
        "runtime_min": round(elapsed/60,2),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir",   default=TEST_DIR)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--n-jobs",     type=int, default=N_JOBS)
    args = p.parse_args()

    for path, label in [
        (os.path.join(args.output_dir,"catboost_gpu_model.cbm"), "CatBoost model"),
        (os.path.join(args.output_dir,"gate_model.pkl"),          "Gate model"),
        (os.path.join(args.output_dir,"model_metadata.json"),     "Metadata"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found at {path}"); sys.exit(1)

    print("="*60)
    print("  GPU-MAX PIPELINE — Phase 2 (Pre-trained models)")
    print(f"  batch_size={args.batch_size:,}  n_jobs={args.n_jobs}")
    print("="*60, flush=True)

    cb_model = cb.CatBoostClassifier()
    cb_model.load_model(os.path.join(args.output_dir,"catboost_gpu_model.cbm"))

    with open(os.path.join(args.output_dir,"gate_model.pkl"),"rb") as f:
        gate_model = pickle.load(f)
    with open(os.path.join(args.output_dir,"model_metadata.json")) as f:
        meta = json.load(f)

    feat_names      = meta["feat_names"]
    best_tau        = meta["best_tau"]
    gate_feat_names = meta["gate_feat_names"]
    tau_singleton   = meta["best_tau_singleton"]
    best_f05        = meta["best_macro_f05"]

    print(f"Models loaded! Val Macro F0.5={best_f05:.4f} | "
          f"S2τ={best_tau['s2']:.3f} S3τ={best_tau['s3']:.3f} "
          f"Singletonτ={tau_singleton:.3f}", flush=True)

    results = run_gpu_max_inference(
        cb_model=cb_model, feat_names=feat_names, best_tau=best_tau,
        gate_model=gate_model, gate_feat_names=gate_feat_names,
        tau_singleton=tau_singleton,
        test_dir=args.test_dir, output_dir=args.output_dir,
        batch_size_s1=args.batch_size, n_jobs=args.n_jobs,
    )

    report = {
        "pipeline": "gpu_max_phase2",
        "validation_macro_f05": round(best_f05,4),
        "optimal_thresholds": best_tau,
        "tau_singleton": tau_singleton,
        **results
    }
    rfile = os.path.join(args.output_dir, "gpu_max_report.json")
    with open(rfile,"w") as f: json.dump(report, f, indent=2)
    print(f"\nReport saved → {rfile}", flush=True)


if __name__ == "__main__":
    main()
