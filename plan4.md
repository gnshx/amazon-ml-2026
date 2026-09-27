Nobody can promise Top 1. The teams that get there on this challenge will not have a fancier 7B model. They will **not over-merge**, **not fail France**, and **not waste Day 1 on architecture**. F₀.₅ + macro + singletons is a precision contest with a recall ceiling set by blocking.

Here is the actual playbook.

---

## What the metric actually rewards

Macro F₀.₅, β = 0.5, **per Source 1 entity, then averaged**.

That means:

- A singleton predicted empty = **1.0**
- A singleton with even one false match = **0.0**
- Precision is weighted **2×** recall
- One loud false merge on a singleton hurts as much as missing several true links elsewhere

Most of the field will over-match. Top 1 will be **aggressively conservative** on gray pairs and almost perfect on “no match.”

Also: **ties go to earlier submission time**. Get a valid, high-precision file on the board in the first few hours.

---

## 72-hour clock (do not improvise this)

| Window | Job | Output |
|---|---|---|
| **Hour 0–1** | Read data, write validator + scorer, dump a dummy submission | Format never bites you later |
| **Hour 1–6** | Data autopsy + normalization + multi-key blocking | Blocking recall ≥ 95% on a holdout |
| **Hour 6–14** | Features + LightGBM/CatBoost + threshold for F₀.₅ | First real leaderboard score |
| **Night 1** | Train while you sleep. One more submission if it is clearly better | Don’t burn all 5 |
| **Day 2** | Embeddings + cross-encoder on hard pairs, 1–1 assignment, France stress test | Second big jump |
| **Day 3 morning** | Freeze. Ensemble only if val F₀.₅ goes up. Write docs + zip | Stop touching the model |

5 submissions/day. Use them as **A/B tests**, not as hope.

---

## Hour 0–6: the autopsy that decides rank

Do this before any neural net.

**Counts**
- Match rate: what % of S1 have 0 / 1 / 2+ matches
- Mean matches per non-singleton
- Can one S2/S3 ID appear on multiple S1 rows? (if yes, your matcher must not allow it)
- Missing name / address / country by source
- Country mix (train = US + India only)

**True pairs vs random pairs**
- Token Jaccard on name
- Token Jaccard on address
- Exact postal code rate
- Street-number agreement
- Country disagreement rate (should be ~0)

You will find that **country + a few cheap keys already recover most true pairs**. That tells you blocking is solvable. Matching is a precision problem.

**Holdout correctly:** split by **S1 entity**, never by pair. Also hold out **one entire country** (e.g. train on US, val on India) to simulate **France**. If your pipeline dies on the held-out country, it will die on test.

---

## Normalization (do this once, share everywhere)

Do **not** hard-code `{US, India}`. Country is an open string.

1. Unicode NFKC, lower, strip punctuation, `&` → `and`
2. Collapse whitespace
3. Accent fold (`café` → `cafe`) — this is free France insurance
4. Expand / strip legal suffixes into a **core name**
   - US: inc, llc, corp, ltd, lp, pll c, dba, co
   - IN: pvt, private, ltd, limited, llp, opc
   - FR (must handle even though unseen): sarl, sas, sa, eurl, sci, snc
5. Address: `rd↔road`, `st↔street`, `ave↔avenue`, `blvd↔boulevard`, `hwy↔highway`, `ste↔suite`, `n↔north`, etc.
6. Extract, don’t throw away:
   - postal-like token (5-digit, 6-digit, `NNN NNN`, `NNNNN`)
   - first street number
   - significant tokens (drop `the, of, and, near, opp, opposite`)

Keep both **raw** and **core** forms. Matching uses both.

---

## Blocking: this is your recall ceiling

They even collect `candidate_pairs.tsv` to measure it. Target **≥95% pair completeness**, then cap candidates per S1 (50–150).

**Union** of independent keys (a pair is a candidate if **any** key hits):

1. Same country (if both present) **and** first 4–5 chars of core name
2. Sorted name tokens (catches word-order swaps)
3. Double Metaphone / Soundex of first content token
4. Exact postal code
5. Street number + first street token
6. Rare-token blocking (tokens that appear in few records — high precision blocks)
7. Char n-gram TF-IDF (3–5) cosine top-K via FAISS / sparse cosine
8. Dense ANN top-K on `core_name + " | " + address` with a **multilingual** embedder (`BAAI/bge-m3` or `paraphrase-multilingual-mpnet-base-v2`, MIT)

Do **not** block on city/state alone. “Acme” in two cities is a false-merge factory.

Measure on holdout:

```text
blocking_recall = true_pairs_in_candidates / all_true_pairs
reduction_ratio = 1 - |candidates| / |S1|×|S2∪S3|
```

If recall < 93%, add a key. If candidates explode, tighten only the dense/TF-IDF K, not the exact keys.

`candidate_pairs.tsv` = **the last list the model actually scores**, not an early pass.

---

## Matcher: the stack that actually wins 72h

Do **not** start with a 7B LLM. Winners in past Amazon ML Challenges that jumped to big models without a metric-tuned classical stack lost to teams with better post-processing.

### Stage A — features + gradient boosting (Day 1, this is your floor)

Per (S1, S2/S3) candidate:

| Family | Features |
|---|---|
| Name | Jaro-Winkler, token_set_ratio, token_sort_ratio, Levenshtein ratio, Jaccard, overlap coeff, prefix, core-name equality |
| Address | same suite + numeric-token Jaccard + postal exact/prefix + street-number match |
| Rarity | shared rare tokens, IDF-weighted overlap |
| Embed | TF-IDF char-ngram cosine, dense cosine |
| Rank | is_best_for_S1, is_best_for_cand, gap to 2nd best |
| Meta | country match, source (S2 vs S3), length ratios, missing-field flags |

Train **LightGBM or CatBoost** with:

- Negatives = non-matches **inside your candidate set** (not random worldwide negatives)
- Class weight or subsample so you don’t drown in easy negatives
- Optimize **binary logloss**, then **sweep the threshold on macro F₀.₅**

That last sentence is the whole contest.

### Stage B — transformer only on hard pairs (Day 2)

Serialize like Ditto:

```text
[COL] name [VAL] acme corp [COL] addr [VAL] 12 main st ... [COL] ctry [VAL] US
```

Fine-tune a small **MIT/Apache** cross-encoder:

- `microsoft/deberta-v3-base` (MIT)
- `BAAI/bge-reranker-base` / MiniLM cross-encoder

Train on labeled candidates. Use it as a **feature** in the booster, or as a second-stage score fused with LightGBM (`0.6·lgbm + 0.4·ce`).

### Stage C — optional 7B only if val is stuck

License trap: **Llama 3.1 is not MIT/Apache. Do not use it.**

Safe: **Qwen2.5-7B-Instruct (Apache 2.0)**, **Mistral-7B (Apache 2.0)**, **Phi-3 (MIT)**. ≤8B.

Use the LLM **only as a reranker on the top 3 ambiguous candidates**, not as the whole pipeline. You will not finish 72h if every pair goes through 7B.

---

## The precision tricks that separate Top 1 from Top 50

These matter more than model class.

**1. Predict empty when unsure.**  
If best candidate score < τ, output `""`. τ is chosen to max val F₀.₅. Start high (precision-first), then lower until F₀.₅ peaks.

**2. One S2/S3 → at most one S1.**  
S1 is a deduplicated reference. The same `S2-…` matching two S1s is almost always a false merge. After scoring, run **greedy mutual-best** (or Hungarian): assign each candidate ID to its highest-scoring S1, and drop the rest.

**3. Per-source thresholds.**  
S2 noise ≠ S3 noise. Sweep `τ_S2` and `τ_S3` independently.

**4. Require two weak signals or one strong one.**  
Example rule that saves singletons:

- name_jw ≥ 0.93 **and** (postal match **or** street-number match **or** addr_jaccard ≥ 0.5)
- or name_core exact **and** country match **and** not a generic name (`general store`, `pharmacy`)

Generic / high-DF names need **stricter** address agreement.

**5. Never emit a match that was not a candidate.** Validator will flag it; it also means your pipeline is lying.

---

## France (the private-LB bomb)

Train has US + India. Test adds France. Teams that one-hot country or write `if country == "India"` will get wrecked on private.

Do this instead:

- Country is just a string equality feature, not a vocabulary
- Multilingual embeddings
- Accent folding + French legal suffixes in the **same** suffix table as US/IN
- Address parser that only knows “number / postal-shaped token / rest”, not “PIN vs ZIP vs CP”
- **Stress test:** train on one country, evaluate on the other. If F₀.₅ collapses, you are not ready for France
- Every test S1 row must appear in the file, France included

---

## Validation you can trust

Implement the official metric yourself on hour 0:

```text
for each S1:
  P = |pred ∩ gold| / |pred|   (1 if both empty)
  R = |pred ∩ gold| / |gold|   (1 if both empty)
  if pred nonempty and gold empty: P=0, R=0  → F=0
  F0.5 = 1.25PR / (0.25P + R)
macro = mean(F0.5)
```

Track four numbers every experiment:

1. Macro F₀.₅ (the score)
2. Singleton accuracy
3. Precision / recall on non-singletons
4. Blocking recall

If (2) drops, you over-matched. If (4) is low, no matcher can save you.

**Do not chase public LB on Day 3.** Public is a subset. Private has more France and more singletons than you think.

---

## Submission hygiene (instant DQ if you skip)

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Must be true:

- Tab-separated, exact column names
- One row per test S1, including France
- Empty string for no matches, not `NaN` / `"None"`
- Only S2-/S3- IDs that exist
- No duplicate IDs in a list, no duplicate S1 rows
- Final matches ⊆ candidates

Forbidden: OpenCorporates, Google Maps, Nominatim, any business-registry lookup. That is **disqualification**, not a clever trick.

---

## What to submit Hour 6 vs Hour 70

**First scored submission (Hour 6–10):**  
High-precision rules + blocking. Goal: valid file, F₀.₅ that is “good not great.” You now have a floor and a timestamp.

**Best Day-1 submission:** LightGBM + threshold sweep + 1–1 assignment.

**Best Day-2:** add cross-encoder + country-holdout-tuned threshold.

**Final:** the val-max F₀.₅ model, **slightly more conservative** than the public-best (raise τ a notch). Precision errors on private are more expensive than a bit of missed recall.

---

## Team split (2–4 people)

- **A:** data + blocking + `candidate_pairs.tsv`
- **B:** features + LightGBM + threshold / assignment
- **C:** embeddings + cross-encoder
- **D (or A at night):** scorer, validator, docs, zip, submission discipline

One person owns “is the TSV legal.” That person never trains models.

---

## The honest ranking of impact

```text
correct empty-list on singletons     ████████████
blocking recall                      ███████████
1-to-1 assignment + high threshold   ██████████
normalization / suffix / postal      █████████
LightGBM features                    ████████
France-safe features                 ███████
cross-encoder                        █████
7B LLM rerank                        ██
fancy architecture                   █
```

If you want, next message I can write the **exact Day-1 pipeline** (normalizer, blocking keys, feature list, LightGBM training loop, F₀.₅ threshold search, TSV writers) as copy-paste code against `dataset/train/*.tsv`. That’s the fastest path from this plan to a scored file.