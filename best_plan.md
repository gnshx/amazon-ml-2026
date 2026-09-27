# The Battle-Tested Master Plan: Amazon ML Challenge 2026 (Business Entity Resolution)
## Precision-First, High-Recall Union Blocking & Calibrated Macro $F_{0.5}$ Optimization

---

## 1. Executive Summary & Core Philosophy

The Amazon Business Entity Resolution Challenge requires linking Source 1 (S1) records to zero, one, or multiple records across Source 2 (S2) and Source 3 (S3). Scored via **macro-averaged per-Source 1 $F_{0.5}$**, precision carries twice the weight of recall, and **singletons (entities with zero matches) are worth a full 1.0 if predicted empty, but plunge straight to 0.0 if even a single false link is made**.

### The Winning Sequence:
$$\text{Trivial Baseline (Hour 1)} \longrightarrow \text{Blocking Recall } \ge 98\% \longrightarrow \text{Calibrated GBDT} \longrightarrow \text{Dedicated Singleton Gate} \longrightarrow \text{Macro } F_{0.5} \text{ Thresholds}$$

### Critical Adjustments from Review & Bottleneck Analysis:
1. **Submit a Trivial Baseline in Hour 1–2**: Do NOT wait for complex models to make the first submission. An exact core-name + postal-code rule-based submission locks in an early leaderboard timestamp (ties are broken by submission time), validates the entire formatting and evaluation loop, and provides a true public-LB anchor for local cross-validation.
2. **Demote 1-to-1 / Mutual-Best from Fact to Hypothesis**: The challenge explicitly states Source 1 can match **many** records across Source 2 and Source 3. Do not enforce mutual-best conflict resolution blindly; first audit the ground truth to test whether target IDs are exclusive. If target records match multiple S1 entities, enforcing mutual-best will delete valid matches and damage $F_{0.5}$.
3. **No Hardcoded Post-Processing Rules**: Avoid arbitrary rules like `NameSim ≥ 0.92 AND ...`. Stacking manual cutoffs on top of calibrated probabilities cuts recall on legitimate fuzzy or transliterated matches. Keep similarity metrics as model features and let probability cutoffs optimize the precision/recall trade-off.
4. **Push Blocking Recall to $\ge 98\%$**: A 95% ceiling means dropping 5% of possible points before classification even starts. Use score-based or per-route top-$K$, not a rigid 50–150 candidate cap.
5. **Dedicated Singleton Gate**: Train a specialized entity-level classifier (*"Does this S1 entity have any match at all?"*) based on aggregated pool signals. This provides the highest ROI for $F_{0.5}$.
6. **Prioritize GBDT over Heavy Transformers**: A feature-rich LightGBM/CatBoost ensemble will capture >90% of achievable performance. DeBERTa/cross-encoders are deferred to Day 2 and are the first to be cut if time runs short.

---

## 2. Metric Dynamics: Exploiting Macro $F_{0.5}$

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Strategic Directives:
* **Asymmetric Penalty**: In $F_{0.5}$, false positives are penalized twice as heavily as false negatives.
* **When in Doubt, Predict Empty (`""`)**:
  * Predicting empty for a true singleton yields **1.0**.
  * Adding a single false link to a true singleton yields **0.0**.
  * Missing a match on an entity with 1 true match yields **0.0**.
  * Guessing on low-confidence singletons is mathematically negative-sum. Abstention on ambiguous pairs protects precision.
* **Threshold Tuning Against True Metric**: Never use the default 0.5 decision boundary. Sweep decision thresholds on out-of-fold validation sets to directly maximize macro-averaged $F_{0.5}$.

---

## 3. Data Auditing & The France Open-Set Strategy

### 3.1 Ground Truth & Exclusivity Audit
Before building models, run an automated data audit:
* Measure singleton percentage in training ground truth.
* **Exclusivity Check**: Count how many S2 and S3 IDs appear in multiple Source 1 ground-truth match sets. If $>0$, strict 1-to-1 matching is mathematically invalid.
* Verify ID prefixes (`S1-`, `S2-`, `S3-`) and column names:
  * Matching results: `source1_entity_id`, `matched_entity_ids`
  * Candidate pairs: `source1_entity_id`, `candidate_entity_ids`

### 3.2 Zero-Shot France Generalization
* The training data contains **US and India** records; test data includes **France**.
* **Never one-hot encode country** and never write conditional logic like `if country == "India"`.
* Treat country as an open-set comparison:
  * `country_match = 1` if both present and identical.
  * `country_match = 0` if present and conflicting.
  * `country_match = -1` if either is missing or novel.
* **Accent Folding (`café` $\to$ `cafe`)**: Built directly into text normalizers to handle French diacritics.
* **Universal Legal Suffix & Street Dictionary**:
  * Normalize US, Indian, and French legal forms together:
    * US: `inc, llc, corp, ltd, co, dba, lp`
    * India: `pvt, private, ltd, limited, llp, opc`
    * France: `sarl, sas, sa, eurl, sci, snc`
  * Normalize street types: `st/street, rd/road, ave/avenue, blvd/boulevard, ste/suite, rue, av, bd, all`.
* Always retain both **Raw** and **Core** (suffix-stripped, punctuation-collapsed) text.

---

## 4. Candidate Generation (Target: $\ge 98\%$ Recall)

Build a multi-route union blocker and evaluate candidate recall on held-out validation sets:

1. **Exact & Prefix Core Name**: Exact core-name equality, plus first 4–5 characters when country matches.
2. **Character $n$-gram TF-IDF Cosine**: Character 3–5 grams top-$K$ via sparse matrix cosine similarity (handles typos, spelling noise, transliteration).
3. **Phonetic Keys**: Double Metaphone / Soundex on primary tokens (robust to Indian/French phonetic spelling variants).
4. **Address Structural Keys**: Exact postal code (5-digit US/FR, 6-digit India) + first street number.
5. **Token-Sorted Exact Key**: Neutralizes word-order permutations (*"Apex Healthcare Services"* $\leftrightarrow$ *"Healthcare Services Apex"*).
6. **Rare-Token Overlap**: Inverted index matching tokens with high Inverse Document Frequency (IDF).
7. **Dense Semantic Embeddings**: Pretrained multilingual sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-m3`, Apache-2.0 / MIT) indexed via FAISS.

*Blocking Rule*: Use score-based thresholds or per-route top-$K$ rather than a rigid global candidate cap. Export the exact candidate pool fed to the classifier as `candidate_pairs.tsv`.

---

## 5. Pairwise Feature Engineering (High-ROI Signals)

For every candidate pair `(S1, S2/S3)`:

### 5.1 Name Similarity Signals
* RapidFuzz metrics: `token_sort_ratio`, `token_set_ratio`, `levenshtein_ratio`, `jaro_winkler`, `partial_ratio`.
* Character 3–5 gram TF-IDF cosine similarity.
* Core name exact match boolean.
* Acronym / initials match indicator.
* **Token IDF Rarity / Generic Name Penalty**: Average and minimum IDF of shared tokens (distinguishes rare brand names from common words like *"General Store"*).

### 5.2 Address Similarity Signals
* Normalized address `token_set_ratio` and Jaccard similarity.
* **Postal Code Prefix Match**: Exact match (`1.0`), 3/4-digit prefix match (`0.5`), mismatch (`0.0`), missing (`-1.0`).
* **Numeric Street Number Verification**: Exact match boolean and absolute numerical difference $|num_1 - num_2|$.
* Component-wise Jaccard overlap (city, state, road type).

### 5.3 Joint, Cross & Context Signals
* Evidence interaction: $\text{NameSim} \times \text{AddrSim}$ and $\min(\text{NameSim}, \text{AddrSim})$.
* Discrepancy flags:
  * `Strong_Name_Weak_Addr`: High name match ($>0.9$) + low address match ($<0.3$) $\to$ flags distinct chain branches.
  * `Weak_Name_Strong_Addr`: Low name match + identical address $\to$ flags collocated different businesses.
* Country match indicator (`+1, 0, -1`).
* Source indicator (`S2` vs `S3`).
* Candidate pool rank and score delta: $(p_{\text{best}} - p_{\text{second\_best}})$.

---

## 6. Modeling Architecture

### 6.1 Backbone: Calibrated GBDT Ensemble
* Blend of **LightGBM** and **CatBoost** (both MIT/Apache-2.0).
* **Negative Sampling & Calibration**: Train on candidate pairs. When downsampling negatives to speed up training, perform probability calibration (Platt Scaling or Isotonic Regression) on an **unsampled validation candidate set** to avoid distorting predicted probabilities.

### 6.2 The Dedicated Singleton Gate (Highest ROI for $F_{0.5}$)
Train a specialized entity-level classifier: **"Does this Source 1 entity have ANY match?"**
* **Entity-Level Features**:
  * $\max_{j} P(S1, \text{Cand}_j)$ (maximum pair probability)
  * Mean and sum of top-3 pair probabilities
  * Score gap: $p_{(1)} - p_{(2)}$
  * Number of candidates passing a loose threshold ($p > 0.3$)
  * Source 1 name token IDF rarity
  * Best address similarity score across all candidates
* **Decision**: If Singleton Probability $> \tau_{\text{singleton}}$, immediately output `""`. This guarantees protecting true singletons.

### 6.3 Inference & Threshold Optimization
* **Per-Source Asymmetric Thresholds**: Tune $\tau_{S2}$ and $\tau_{S3}$ independently on out-of-fold validation predictions using 2D grid search.
* **Country-Agnostic Thresholding for France**: For test records where `country_match == -1` (unseen country / France), apply a slightly more conservative threshold to protect precision under distribution shift.
* **Mutual-Best Conflict Resolution (Ablation Only)**: If ground truth audit confirms that target IDs are exclusive across S1 entities, apply greedy highest-probability assignment to resolve target collisions. If audit shows non-exclusivity, skip this step.

---

## 7. Leak-Free Validation Strategy

### 7.1 Connected-Component Grouped 5-Fold CV
* Construct an undirected graph connecting every Source 1 ID to its ground-truth Source 2 and Source 3 matches.
* Compute connected components (singletons form 1-node components).
* Partition the data into **5 folds grouped strictly by Connected Component ID** to prevent entity identity leakage.

### 7.2 Country Holdout Diagnostic
* Train on US data $\to$ Evaluate on India (and vice versa).
* Use as a diagnostic stress test to detect reliance on memorized country-specific vocabulary.

### 7.3 The 4-Metric Evaluation Dashboard
On every experiment, log:
1. **Macro $F_{0.5}$** (the competition metric)
2. **Singleton Accuracy** (% of true singletons correctly assigned `""`)
3. **Non-Singleton Precision and Recall**
4. **Candidate Blocking Recall** (% of true links present in `candidate_pairs`)

---

## 8. Compressed 72-Hour Execution Roadmap

```
Hour 0–2:   [Data Audit] ──► [CV Split] ──► [Trivial Baseline Submission] (Lock timestamp & test LB)
Hour 2–8:   [Multi-Route Blocking] ──► Audit Recall (Target: ≥ 98%)
Hour 8–20:  [Feature Extraction] ──► [Calibrated GBDT] ──► [Macro F0.5 Threshold Sweep] ──► Submit #2
Hour 20–36: [Dedicated Singleton Gate] ──► [Per-Source Cutoffs] ──► [Mutual-Best Ablation] ──► Submit #3
Hour 36–54: [Dense Embeddings & Model Blending] ──► [Country Stress Test] ──► Submit #4
Hour 54–66: Freeze Code ──► Retrain on 100% Data ──► Test Inference ──► Final Submission #5
Hour 66–72: Run Validator ──► Write Methodology Doc ──► Package Code & Dependencies
```

---

## 9. Submission Hygiene & Disqualification Safeguards

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

### Non-Negotiable Checks:
- [x] **Zero External Lookups**: No Google Maps, Nominatim, OpenCorporates, or external geocoding APIs (instant disqualification).
- [x] **Official Column Names**:
  - `matching_results.tsv`: `source1_entity_id`, `matched_entity_ids`
  - `candidate_pairs.tsv`: `source1_entity_id`, `candidate_entity_ids`
- [x] **Strict Tab Separation**: `sep="\t"`, no quotes around lists.
- [x] **Full S1 Entity Coverage**: Every test Source 1 entity appears exactly once, including all French records.
- [x] **Clean Singletons**: Empty strings `""` for singletons — never output `NaN`, `null`, or `"None"`.
- [x] **No Spaces in Lists**: `S2-101,S3-204` (never `S2-101, S3-204`).
- [x] **Candidate Subset Assertion**: $\text{matched\_entity\_ids} \subseteq \text{candidate\_entity\_ids}$ for every row.
- [x] **Permitted Licenses**: Models strictly MIT / Apache 2.0 and $\le 8\text{B}$ parameters (avoid Llama).
