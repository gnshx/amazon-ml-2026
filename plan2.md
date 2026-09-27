To secure a top rank in the **Amazon ML Challenge 2026 (Business Entity Resolution)**, you need a high-precision strategy tailored specifically to the $F_{0.5}$ metric. Since $F_{0.5}$ penalizes false positives (incorrectly merged entities) twice as harshly as false negatives, your solution must prioritize **high precision**, **rigorous singleton handling**, and **robust multi-stage candidate generation**.

---

### Strategy 1: Metric-Driven Optimization ($F_{0.5}$ & Singletons)

The macro-averaged $F_{0.5}$ score rewards precision twice as much as recall:


$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

1. **Prioritize High Precision over High Recall:**
* A false match on a singleton entity tanks that row's score from $1.0$ straight to $0.0$.


* It is better to predict an empty list (no match) when uncertain than to risk a low-confidence false positive.




2. **Optimize Thresholds Directly for Macro $F_{0.5}$:**
* Do not rely on default $0.5$ decision boundaries. Sweep threshold cutoffs ($t \in [0.50, 0.95]$) on your local cross-validation to maximize the macro-averaged $F_{0.5}$ score.



---

### Strategy 2: High-Recall Multi-Pass Candidate Generation (Blocking)

Your candidate set (`candidate_pairs.tsv`) defines the theoretical upper bound of your recall. Build a multi-pass blocking pipeline combining lexical, fuzzy, and semantic retrievals:

1. **Character $n$-gram TF-IDF & BM25 Search:**
* Build TF-IDF vectorizers (char $n$-grams from 2 to 5) separately on `business_name` and `business_address`.
* Retrieve the top-$K$ ($K \approx 50\text{--}100$) nearest neighbors using cosine similarity or BM25 index per `country`.


2. **Dense Semantic Embeddings (Bi-Encoder):**
* Use an open-source model under 8B parameters (e.g., `BAAI/bge-large-en-v1.5`, `sentence-transformers/all-mpnet-base-v2`, or `Qwen2.5-Coder-7B`).


* Concatenate `[country] + name + address` and index vectors with **FAISS** or **HNSW** to pull top-$K$ dense neighbors.


3. **Locality-Sensitive Hashing (LSH) / MinHash:**
* Apply MinHash LSH on tokenized entity strings to quickly cluster records with spelling variations, typos, and token transpositions.




4. **Union & Deduplication:**
* Combine candidates from all blocking passes. Keep candidate size per entity manageable (e.g., top 50–200 candidates per S1 entity) to maintain high pipeline speed and low false-positive surface area.



---

### Strategy 3: Advanced Feature Engineering & Pairwise Scoring

Once candidates are generated, extract rich pairwise similarity features and train a two-stage classification/ranking model:

#### 1. Detailed Feature Engineering

* **String & Fuzzy Metrics:** RapidFuzz token-sort ratio, token-set ratio, Levenshtein distance, Jaccard similarity, and Jaro-Winkler distance computed separately on `business_name` and `business_address`.


* **Numerical & Digit Matching:** Extract house numbers, PIN/ZIP codes, and phone numbers. Compute exact/partial matches on numeric substrings (crucial for distinguishing branches of the same chain).
* **Token Overlap:** Ratio of shared non-common words vs. common legal terms (e.g., strip or lower-weight standard terms like *Corp*, *Pvt*, *Ltd*, *Street*, *Road*).


* **Embedding Similarities:** Cosine, Euclidean, and dot-product distances of dense name/address embeddings.

#### 2. Classification Engine (GBDT + Cross-Encoder Reranker)

* **Stage 1 (Fast GBDT Ranker):** Train LightGBM / CatBoost / XGBoost on candidate pairs using the extracted feature set to produce match probabilities.
* **Stage 2 (Deep Cross-Encoder Fine-Tuning):**
* Fine-tune a Transformer model (e.g., `microsoft/deberta-v3-large` or an 8B open-source LLM cross-encoder).


* Pass pair input directly: `Source 1: {name}, {address} [SEP] Source 2/3: {name}, {address}`.
* Train using binary cross-entropy or focal loss with heavy negative sampling.



---

### Strategy 4: Strict Pipeline & Open-Set Handling

1. **Dynamic Country Handling (Open-Set Rule):**
* The training set contains US and India, but the test set includes France.


* **Do not hardcode or one-hot encode country names**. Treat country as an open string or process candidate search dynamically per country group.




2. **Zero External Data / Geocoding:**
* External APIs, geocoding lookup, or external data fetching are **strictly prohibited** and lead to immediate disqualification. All features must be derived purely from the provided dataset.




3. **Local Validation Scheme:**
* Create a local group-wise validation split (e.g., 20% holdout of S1 entities with their ground truth matches).


* Mirror the test distribution including singletons. Never judge model quality without computing the exact macro $F_{0.5}$ metric on the full validation split.





---

### Strategy 5: Post-Processing & Submission Checklist

1. **Confidence Thresholding:**
* Apply an aggressive threshold (e.g., probability $> 0.85$) on model scores to drop marginal candidates, ensuring high precision.




2. **Formatting & Constraint Verification:**
* Every S1 test entity must appear exactly once in `matching_results.tsv` and `candidate_pairs.tsv`.


* `matched_entity_ids` must strictly be a subset of `candidate_entity_ids`.


* Ensure tab separation (`sep="\t"`) without quotes around comma-separated lists.




3. **Run Validation Script Local Test:**
* Use the provided helper script before consuming any of your 5 daily submissions:


```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```[cite: 1]

```




4. **Reproducibility Package:**
* Keep all code clean and modular under `code/business_entity_resolution/src/`.


* Pin all dependencies in `requirements.txt` and fill out `Documentation_template.md` thoroughly.