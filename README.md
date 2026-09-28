# Amazon ML Challenge 2026: Large-Scale Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hardware: RTX 3060](https://img.shields.io/badge/CUDA-PyTorch%20GPU-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Metric: Macro F0.5](https://img.shields.io/badge/Metric-Macro%20F0.5-orange.svg)](https://en.wikipedia.org/wiki/F-score)

Production-grade, memory-bounded entity resolution pipeline resolving **1,732,544 query business entities** against an open-set corpus of **8,468,089 target reference entities** under the **Macro F0.5** evaluation metric with singleton evaluation.

---

## 📑 Table of Contents
- [Benchmark & Leaderboard Evolution](#-benchmark--leaderboard-evolution)
- [Key Architectural Highlights](#-key-architectural-highlights)
- [Repository Structure](#-repository-structure)
- [In-Depth Post-Mortems of Failure Modes & Architectural Fixes](#-in-depth-post-mortems-of-failure-modes--architectural-fixes)
- [Ground-Truth Data Audit & Reference Anchors](#-ground-truth-data-audit--reference-anchors)
- [Installation & Quick Start](#-installation--quick-start)
- [Running Benchmarks & Diagnostics](#-running-benchmarks--diagnostics)
- [Generating Submission Files](#-generating-submission-files)
- [Validation & Compliance](#-validation--compliance)

---

## 📊 Benchmark & Leaderboard Evolution

| Version / Model | Leaderboard / Eval F0.5 | Precision | Recall | Singleton % | Match Count | Runtime | Status & Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **V1 Baseline (Uncalibrated GBDT)** | 0.310 (LB) | 24.2% | 45.1% | 59.33% | 1,393,727 | ~9 min (CPU) | ❌ **Failure**: Threshold sweep pushed to $\tau=0.999$, zeroing out 54% of non-singletons |
| **V2 Production Baseline** | **0.745** (LB) | **78.2%** | **63.1%** | 4.12% | 6,120,440 | 14 hrs (CPU) | 🏆 **Gold Standard**: 8-route blocking, mutual-best 1-to-1 bipartite resolution |
| **V3 Fast Stream** | 0.741 (est) | 77.4% | 63.8% | 4.85% | 5,840,110 | 4.5 hrs (CPU) | In-process streaming with category conflict guard |
| **V4 Lean Pipeline** | 0.730 (LB) | 75.8% | 64.0% | 7.30% | 5,696,249 | 2.5 hrs (CPU) | ⚠️ **Regression (-1.5%)**: Substring false positives via `fuzz.partial_ratio` |
| **CatBoost GPU Tabular GBDT** | 0.570 (50k) | 61.3% | 44.7% | 8.90% | 4,110,230 | 45 min (GPU) | ❌ **Severe Failure (-17.5%)**: Tabular splits failed on surface text without dense semantics |
| **V5 Calibrated Pipeline** | 0.674 (LB) | 69.1% | 61.8% | 12.12% | 4,270,746 | 18 min (CPU) | ⚠️ **Regression (-7.1%)**: Aggressive pruning on empty target addresses dropped 74k matches |
| **V6 ULTIMATE (CPU Stream)** | **0.752** (est) | **80.4%** | **65.2%** | **3.78%** | **6,761,573** | 21 hrs (CPU) | 🚀 **Peak Accuracy**: 12-route blocking, tuned gate (`sc < 0.82`), 104.5 MB output |
| **V7 ULTIMATE (GPU PyTorch)** | **Target >0.76** | **High** | **High** | **~4.0%** | **~6.8M** | **~35 min (GPU)** | ⚡ **100% GPU Acceleration**: PyTorch Sparse Tensor SpMM + CUDA batch scoring |

---

## 🚀 Key Architectural Highlights

1. **12-Route Candidate Union Blocker ($\ge 98\%$ Recall)**:
   Combines exact core name, alphabetical token-sorted core, 4-gram prefix keys, soundex/double-metaphone phonetic hashing, address cluster keys, postal/number anchors, and sparse character TF-IDF to eliminate recall bottlenecks before pairwise classification.
2. **Open-Set France Generalization**:
   Zero closed-set hardcoded country rules. Relational country features (`+1, 0, -1`), accent folding (`café` $\to$ `cafe`), and universal legal suffix expansion (`SARL, SAS, SA, EURL, SCI, SNC` alongside `Inc, LLC, Pvt, Ltd`).
3. **1-to-1 Bipartite Mutual-Best Resolution**:
   Enforces target exclusivity by solving maximum-weight bipartite matching, ensuring each target ID in Source 2 and Source 3 is owned strictly by its highest-affinity query claimant.
4. **Calibrated Singleton Protection Gate**:
   Directly exploits Macro $F_{0.5}$ metric dynamics: predicts empty `""` on ambiguous singletons to lock in a 1.0 score and prevent the catastrophic 0.0 penalty of false positive merges.
5. **GPU-Accelerated Inference (V7 Ultimate)**:
   Vectorizes all 8.47M reference targets into GPU sparse matrices (PyTorch CUDA float16), executing batch SpMM blocking and vectorized candidate scoring in ~35 minutes on an NVIDIA RTX 3060.

---

## 📁 Repository Structure

```
AMAZON-ML/
├── README.md                               # High-level architecture, benchmarks & reproduction
├── SOLUTION_DOCUMENTATION.md               # Complete 10-section competition methodology report
├── Documentation_template.md               # Official submission methodology document
├── best_plan.md                            # Strategic master plan and architectural roadmap
├── requirements.txt                        # Python dependencies
│
├── run_gpu_v7_ultimate.py                  # 🚀 V7 ULTIMATE: 100% GPU-accelerated PyTorch SpMM pipeline
│
├── src/                                    # Core source modules
│   ├── baseline_rule_matcher.py            # Initial rule baseline
│   ├── data_audit_and_split.py             # Data auditing & connected-component CV fold builder
│   ├── feature_extractor.py                # 35+ pairwise similarity, discrepancy & context features
│   ├── metrics.py                          # Official Macro F0.5 per-entity metric implementation
│   ├── multi_route_blocker.py              # 12-route high-recall union blocking engine
│   ├── run_production_v6_ultimate.py       # V6 ULTIMATE CPU streaming production engine
│   ├── run_production_v5_calibrated.py     # V5 Calibrated pipeline
│   ├── run_robust_production_v4_lean.py    # V4 Lean pipeline
│   ├── run_robust_production_v3_fast.py    # V3 Fast streaming pipeline
│   └── evaluate_fixed_benchmark.py         # Standardized 50,000-entity benchmark harness
│
├── diagnostics/                            # Standalone audit & verification tools
│   ├── compute_real_gt_stats.py            # Exact 2.2M ground truth statistics & anchors
│   ├── verify_splits_leakage.py            # Zero-leakage connected-component validation
│   ├── analyze_france_breakdown.py         # Country candidate & match coverage audit
│   ├── run_calibrated_cv_sweep.py          # Calibrated GBDT holdout grid sweep
│   ├── sweep_precision_rules.py            # Rule engine threshold optimization
│   ├── test_high_recall_blocker.py         # Multi-route blocking recall diagnostic
│   └── test_tfidf.py                       # Sparse TF-IDF candidate retrieval test
│
├── docs/                                   # Documentation & archived benchmark logs
│   └── logs/                               # Historical execution logs from Day 1 to Day 3
│       ├── v6_run.log                      # Complete 21-hour V6 production execution log
│       ├── v5_calibrated_run.log           # V5 Calibrated execution log
│       ├── v4_lean_run.log                 # V4 Lean execution log
│       ├── v3_fast_run.log                 # V3 Fast stream execution log
│       ├── tune_50k_results.txt            # Parameter sweep on 50k benchmark
│       └── calibrated_sweep_summary.json   # Calibrated GBDT grid sweep output
│
└── utils/
    └── validate_submission.py              # Official competition format & constraint validator
```

---

## 🔬 In-Depth Post-Mortems of Failure Modes & Architectural Fixes

### 1. The V1 Calibration & Singleton Collapse (0.31 Leaderboard Score)
* **Hypothesis:** Artificial class balancing (`scale_pos_weight = neg / 2*pos`) would help GBDTs learn from rare positive candidate pairs.
* **What Happened:** Leaderboard score collapsed to **0.310**.
* **Root Cause:**
  Artificially inflating positive weights caused the model to output severely inflated probabilities for negative pairs. In an attempt to suppress false positives under $F_{0.5}$, the threshold optimizer drove $\tau$ to **0.999**. At $\tau = 0.999$, the model predicted **59.33% of all test entities as singletons** (1,027,955 empty rows), missing true matches for over 930,000 non-singletons. Under the competition metric, *a missed non-singleton scores flat 0.0*, automatically zeroing out 54% of all rows.
* **The Fix:** Removed artificial weighting (`scale_pos_weight = 1.0`), calibrated raw probabilities using unweighted logloss, and swept thresholds across a fine-grained grid `[0.10 ... 0.90]`.

### 2. The V4 Lean Regression: The Toxic Substring Trap (`fuzz.partial_ratio`)
* **Hypothesis:** Business names frequently contain appended corporate extensions (e.g., *Apex Trading Inc* vs *Apex*). Adding `fuzz.partial_ratio` would boost recall on shortened enterprise names.
* **What Happened:** Macro $F_{0.5}$ dropped from **74.5% to 73.0%**.
* **Root Cause:**
  `fuzz.partial_ratio(s1, s2)` identifies the best-matching substring of the longer string matching the shorter string. In enterprise directories, thousands of unrelated companies share generic corporate terms (e.g., *Holdings*, *Enterprises*, *International*, *Logistics*, *Trading*). An entity named *Alpha Holdings* matched any target entity containing *Holdings* with similarity near 1.0. Because the bipartite resolver selects the single highest-scoring link per target, these false-positive high scores hijacked slots belonging to true matches.
* **The Fix:** Purged `fuzz.partial_ratio` completely. Enforced normalized token-sort ratio and length-penalized Levenshtein core matching.

### 3. The CatBoost GPU Tabular Model Collapse (57.0% $F_{0.5}$)
* **Hypothesis:** A Gradient Boosted Decision Tree (GBDT) with 2,500 trees trained on 1.5 million hard-mined negative pairs with 32 hand-engineered features would discover non-linear interactions superior to hand-crafted heuristic rules.
* **What Happened:** Macro $F_{0.5}$ collapsed to **57.0%** on the 50k evaluation benchmark (a 17.5% drop from the V2 rule baseline).
* **Root Cause:**
  1. **Lack of Dense Semantic Pretrained Embeddings:** Tabular models split solely on scalar feature values (edit distances, token counts, jaccard indices). Without dense neural embeddings, GBDTs cannot learn semantic synonyms, DBA aliases, or lexical translations.
  2. **Invariant Violations:** Symbolic rules strictly enforce negative constraints (e.g., conflicting city or country concordance). GBDT trees learn soft combinations where a slightly higher name score overrides a hard geographic contradiction, leaking false merges.
  3. **Inference Latency & OOM Bottleneck:** Running 32-feature extraction across 40M+ candidate pairs on 16 GB RAM caused extreme swap thrashing.
* **The Fix:** Promoted the calibrated structural rule engine as the primary inference backbone.

### 4. The V5 Calibrated Regression: The Empty Target Address Gate Trap (67.4% $F_{0.5}$)
* **Hypothesis:** Real-world entities are predominantly physical businesses. If a candidate target entity has no address information, require a higher similarity score (`sc >= 0.95`) before accepting it, otherwise prune it into a singleton to protect precision.
* **What Happened:** Leaderboard score plunged from **74.5% to 67.4%**.
* **Root Cause:**
  Data auditing revealed that **~3.2% of ground-truth target entities have empty address strings** (holding companies, online providers, international subsidiaries). True business name variations across datasets rarely score $\ge 0.95$ due to legal suffixes, punctuation, and DBA forms (typical legitimate matches score 0.85 - 0.93). This single condition pruned **~74,000 legitimate matches** into singletons.
* **The Fix:** Removed the blank address pruning condition. Returned to composite name score and address concordance gates (`sc < 0.82 and a_sim < 0.60`).

### 5. CPU V6 Bottleneck $\to$ V7 GPU Rebuild
* **Failure Analysis:** V6 achieved optimal accuracy (6.76M matches, 3.78% singletons), but executed on single-threaded CPU with mechanical HDD swap thrashing, taking 21.2 hours.
* **V7 GPU Acceleration:** Rewrote blocking to utilize PyTorch Sparse Tensor Matrix Multiplication (SpMM) on RTX 3060 (12GB VRAM), shrinking candidate retrieval from 21 hours down to under 45 minutes.

---

## 📈 Ground-Truth Data Audit & Reference Anchors

Computed directly from all **2,206,821 Source-1 training records** (`diagnostics/compute_real_gt_stats.py`):

```
============================================================================================
 GROUND TRUTH EXACT BENCHMARK STATS (2,206,821 TRAINING ENTITIES)
============================================================================================
Total S1 Entities:                     2,206,821
True Singletons (0 matches):          123,247 (5.58%)
True Non-Singletons (>=1 match):      2,083,574 (94.42%)
Total True Match Links:               7,638,365
Avg True Matches per Non-Singleton:   3.666
Avg True Matches per ALL Entities:    3.461
Match Distribution (Non-Sing):        Median=4, p90=6, Max=11
Target Exclusivity Violations:        0 (Strictly 0 targets shared across multiple S1)

============================================================================================
 GROUND TRUTH STATS BY COUNTRY (US vs INDIA)
============================================================================================
Country      | Total S1   | Singletons | Sing %   | Total Matches  | Matches/Non-Sing | S2/S3 Ratio 
--------------------------------------------------------------------------------------------
US           | 1,323,633  | 73,896     |   5.58%  | 4,578,522      |            3.664 | 48.3% / 51.7%
India        |   883,188  | 49,351     |   5.59%  | 3,059,843      |            3.670 | 48.4% / 51.6%
```

---

## 🛠️ Installation & Quick Start

```bash
# 1. Clone repository
git clone https://github.com/gnshx/amazon-ml-2026.git
cd amazon-ml-2026

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🔬 Running Benchmarks & Diagnostics

```bash
# 1. Run exact ground-truth distribution audit
python3 diagnostics/compute_real_gt_stats.py

# 2. Verify zero-leakage connected-component cross-validation folds
python3 diagnostics/verify_splits_leakage.py

# 3. Run standardized 50,000-entity benchmark harness
python3 src/evaluate_fixed_benchmark.py

# 4. Run calibrated GBDT holdout grid sweep
python3 diagnostics/run_calibrated_cv_sweep.py
```

---

## ⚡ Generating Submission Files

### Option A: GPU-Accelerated Pipeline (Recommended, ~35-45 minutes)
```bash
python3 run_gpu_v7_ultimate.py
```

### Option B: CPU Streaming Pipeline (V6 Ultimate)
```bash
python3 src/run_production_v6_ultimate.py
```

Both pipelines output:
- `output/matching_results.tsv`: Source-1 to matched Source-2/3 IDs (~104 MB).
- `output/candidate_pairs.tsv`: Source-1 to blocked candidate IDs (~3.2 GB).

---

## ✅ Validation & Compliance

Verify generated submission files against official constraints before upload:
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```

Expected output:
```
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (65406 empty, 1667138 non-empty).
  candidate_pairs.tsv: 1732544 rows (0 empty, 1732544 non-empty).

PASS — no blocking issues found. Safe to submit.
```
