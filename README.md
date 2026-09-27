# Amazon ML Challenge 2026: Large-Scale Business Entity Resolution

High-throughput, memory-bounded entity resolution pipeline resolving **1,732,544 query business entities** against an open-set corpus of **8,468,089 target reference entities** under the **Macro F0.5** evaluation metric.

## 📊 Benchmark & Leaderboard Evolution

| Version / Model | Leaderboard / Eval F0.5 | Precision | Recall | Status & Outcome |
| :--- | :---: | :---: | :---: | :--- |
| **V1 Baseline** | 31.0% | 24.2% | 45.1% | ❌ Naive token overlap & unindexed pairwise search; false positive explosion |
| **V2 Production Baseline** | **74.5%** | **78.2%** | **63.1%** | 🏆 **Gold Standard** (8-route blocking, mutual-best 1-to-1 bipartite resolution) |
| **V4 Lean Pipeline** | 73.0% | 75.8% | 64.0% | ⚠️ **Regression (-1.5%)** caused by `fuzz.partial_ratio` substring false positives |
| **CatBoost GPU Tabular GBDT**| 57.0% | 61.3% | 44.7% | ❌ **Major Failure (-17.5%)** on 50k benchmark; decision trees split on surface lexical tokens without dense semantic invariance |
| **V5 Calibrated Pipeline** | 67.4% | 69.1% | 61.8% | ⚠️ **Regression (-7.1%)** caused by aggressive singleton pruning on empty target addresses |
| **V6 ULTIMATE (CPU)** | **75.2% (est)** | High | High | 12-Route blocking, corporate expansion, tuned singleton gate (`sc < 0.82`), in-process streaming (21 hrs runtime) |
| **V7 ULTIMATE (GPU)** | **Targeting >76%** | High | High | 🚀 **100% GPU-accelerated TF-IDF SpMM blocking & PyTorch batch scoring** (~30-60m runtime) |

## 🔬 In-Depth Post-Mortems of Failure Modes & Architectural Fixes

### 1. The V4 Lean Regression: The Toxic Substring Trap (`fuzz.partial_ratio`)
* **Hypothesis:** Business names frequently contain abbreviations, DBAs, or appended brand extensions (e.g., *Apex Trading Inc* vs *Apex*). Introducing `fuzz.partial_ratio` would boost recall on shortened enterprise names.
* **What Happened:** Macro F0.5 dropped from **74.5% to 73.0%**.
* **Root Cause:**
  `fuzz.partial_ratio(s1, s2)` identifies the best-matching substring of the longer string matching the shorter string. In business directories, thousands of unrelated enterprises share generic words (e.g., *Holdings*, *Enterprises*, *International*, *Logistics*, *Trading*). When an entity was named *Alpha Holdings*, any target entity having *Holdings* yielded a partial ratio near 1.0.
  Because the global resolution stage uses mutual-best 1-to-1 bipartite matching, these false-positive high scores claimed slots that rightfully belonged to true matches.
* **Fix & Modification:** Purged `fuzz.partial_ratio` completely. Enforced normalized token-sort ratio and length-penalized Levenshtein core matching.

### 2. The CatBoost GPU Tabular Model Collapse (57.0% F0.5)
* **Hypothesis:** A Gradient Boosted Decision Tree (GBDT) with 2,500 trees trained on 1.5 million hard-mined negative pairs with 32 hand-engineered features would discover non-linear interactions superior to hand-crafted heuristic rules.
* **What Happened:** Macro F0.5 collapsed to **57.0%** on the 50k evaluation benchmark (a 17.5% drop from the V2 rule baseline).
* **Root Cause:**
  1. **Lack of Dense Semantic Pretrained Embeddings:** Tabular models split solely on scalar feature values (edit distances, token counts, jaccard indices). Without dense neural embeddings (SBERT/E5), GBDTs cannot learn semantic synonyms, DBA aliases, or lexical translations.
  2. **Invariant Violations:** Symbolic rules strictly enforce negative constraints (e.g., conflicting city or country concordance). GBDT trees learn soft combinations where a slightly higher name score overrides a hard geographic contradiction, leaking false merges.
  3. **Inference Latency & OOM Bottleneck:** Running 32-feature extraction across 40M+ candidate pairs on 16 GB RAM required massive data transfers between host RAM and GPU VRAM, stalling the machine and causing extreme swap thrashing.
* **Key Learning:** On massive open-set entity resolution tasks lacking dense embeddings, a carefully calibrated symbolic engine with strict negative constraints and bipartite matching decisively beats shallow tabular GBDTs.

### 3. The V5 Calibrated Regression: The Empty Target Address Gate Trap (67.4% F0.5)
* **Hypothesis:** Real-world entities are predominantly physical businesses. If a candidate target entity has no address information, require a higher similarity score (`sc >= 0.95`) before accepting it, otherwise prune it into a singleton to protect precision.
* **What Happened:** Leaderboard score plunged from **74.5% to 67.4%**.
* **Root Cause:**
  Data auditing of the 8.46M target corpus revealed that **~3.2% of ground-truth target entities have empty address strings** (holding companies, online providers, international subsidiaries). True business name variations across datasets rarely score >= 0.95 due to legal suffixes, punctuation, and DBA forms (typical legitimate matches score 0.85 - 0.93).
  This single condition caused the pipeline to prune **~74,000 legitimate matches** into singletons, triggering severe penalties on both precision and recall.
* **Fix & Modification:** Removed the `not has_ta` pruning condition. Returned to composite name score and address concordance gates (`sc < 0.82 and a_sim < 0.60`).

### 4. CPU V6 Bottleneck -> V7 GPU Rebuild
* **Failure Analysis:** V6 achieved comprehensive matching across 1.73M queries, but executed on single-threaded CPU with mechanical HDD swap thrashing, taking 21 hours.
* **V7 GPU Acceleration:** Rewrote blocking to utilize PyTorch Sparse Tensor Matrix Multiplication (SpMM) on RTX 3060 (12GB VRAM), shrinking candidate retrieval from 21 hours down to under 45 minutes.

