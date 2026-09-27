Top 1 in this challenge is less about one clever model and more about a **precision-first, high-recall blocking + calibrated pairwise classifier** pipeline. The metric is macro F0.5 per Source 1 entity, and singletons are included. That means:

- False merges are very expensive.
- Predicting an empty list for a true singleton gives you a full 1.0.
- Missing a match gives recall 0 for that entity.
- Blocking determines your recall ceiling — if the true match never appears in `candidate_pairs.tsv`, your model can never find it.

Here is a practical blueprint that can compete for top 1.

---

## 1. Treat the metric as the objective

F0.5 per entity:

\[
F_{0.5} = \frac{1.25 \times P \times R}{0.25P + R}
\]

Then macro-average over all S1 entities.

Implications:
- Use a **high confidence threshold**.
- Build a **singleton classifier**: “does this S1 have any match at all?”
- If singleton probability is high, output empty.
- Only include candidate matches above a tuned threshold.
- Tune everything on a validation split using the exact macro F0.5.

Do **not** optimize accuracy or F1. Optimize F0.5 directly.

---

## 2. Build a blocking stage with very high recall

Your `candidate_pairs.tsv` is audited. It should be the exact set fed to your final model. Use a union of multiple blocking strategies so true matches are almost never missed:

- Normalized name token blocking
- Sorted name token blocking
- Phonetic keys: Soundex / Metaphone / Double Metaphone
- Address tokens: PIN/ZIP, city, state, locality
- TF-IDF cosine top-K on name + address
- Character n-gram TF-IDF top-K, especially 3–5 grams
- MinHash/LSH for scalability
- Embedding top-K using a small Apache/MIT model like `all-MiniLM-L6-v2`

Union all candidates. Do not be afraid of a large candidate set — the model will filter it. But keep `candidate_pairs.tsv` as the final blocking set before scoring.

Country is an open set. Do **not** hard-code `{US, India}`. Use country equality as a feature. France will appear in test.

---

## 3. Normalize aggressively, but not destructively

Create normalized versions for matching, not for display:

- Lowercase, Unicode NFKC
- Remove punctuation
- Expand common abbreviations:
  - `Corp` → `Corporation`, `Ltd` → `Limited`, `Pvt` → `Private`
  - `Rd` → `Road`, `St` → `Street`, `Ave` → `Avenue`
  - For France: `SARL`, `SAS`, `SA`, `Rue`, `Av`, `Bd`
- Strip legal suffixes for a separate “core name”
- Extract:
  - India PIN: 6 digits
  - US ZIP: 5 or 9 digits
  - France postal code: 5 digits
- Extract state/city/landmark tokens
- Keep original fields for character-level similarity

Typos and transliterations are expected. Character n-grams and edit-distance features handle this better than exact tokens.

---

## 4. Feature engineering is where most of the score comes from

For each candidate pair `(S1, S2/S3)`, compute:

### Name features
- Token Jaccard
- TF-IDF cosine on word n-grams
- TF-IDF cosine on char 3–5 grams
- Levenshtein ratio
- Jaro-Winkler
- Sorted token ratio
- Partial ratio
- Acronym/initials match
- Similarity after legal suffix removal
- IDF-weighted token overlap

### Address features
- Same string similarity metrics on address
- Exact match on PIN/ZIP
- Exact match on state/city
- Numeric token overlap
- House/building number equality
- Road-type token match
- Landmark token match
- Component-wise similarity after reordering

### Cross features
- Country equality
- Source pair: S1–S2 vs S1–S3
- Candidate rank in blocking
- Which blocking key matched
- Number of candidates for this S1
- Max similarity across all candidates
- Name frequency / token IDF

### Optional collective features
- If an S2 and S3 record both match the same S1, and S2–S3 are similar, boost.
- Graph-based propagation can help, but keep it simple first.

---

## 5. Model: pairwise classifier + singleton gate

A strong baseline:

- LightGBM / XGBoost / CatBoost binary classifier on pair features.
- Train on all candidate pairs from training, labeled by ground truth.
- Use 5-fold CV grouped by S1 entity to avoid leakage.
- Calibrate probabilities with Platt scaling or isotonic regression.

Then add a **singleton gate**:

1. For each S1, compute max match probability and number of candidates above a low threshold.
2. Train a small model to predict “has any match or not”.
3. If singleton probability > threshold, output empty.
4. Otherwise, include candidates with match probability > match threshold.

This directly protects the precision-heavy F0.5.

You can also try a cross-encoder transformer, but with a 3-day window, a well-featured GBDT ensemble is usually faster and stronger.

---

## 6. Threshold tuning is critical

Do not use 0.5. Tune the threshold on validation to maximize macro F0.5.

Consider:
- Global threshold
- Per source-pair threshold: S1–S2 vs S1–S3
- Per country threshold if validation has enough data
- Separate singleton threshold

For France, which is unseen in training:
- Use country equality as a feature.
- Avoid country-specific hard rules.
- Use generic normalization and character n-grams.
- If validation cannot cover France, keep the model country-agnostic and use a slightly higher threshold for unseen countries.

---

## 7. Validation and submissions

Split training by S1 entity, not by pairs. Compute macro F0.5 exactly as the leaderboard does.

Track:
- Candidate recall: are all true matches in `candidate_pairs.tsv`?
- Precision/recall curves
- Singleton accuracy
- Per-country F0.5

Submission strategy with 5/day:
1. Baseline: simple blocking + GBDT + default threshold.
2. Add stronger features.
3. Add singleton gate.
4. Tune thresholds.
5. Ensemble seeds/models.
6. Final validated submission.

Always run:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Every S1 test entity must have exactly one row. No duplicates. Only S2/S3 IDs. Empty list for singletons.

---

## 8. Top-1 edge tactics

- Ensemble multiple GBDT seeds and model types.
- Use calibrated probabilities, not raw scores.
- Use collective inference: if S2 and S3 both match S1, and S2–S3 are similar, increase confidence.
- Use pseudo-labeling only if very careful; high-confidence test pairs can augment training, but false labels are dangerous.
- Optimize blocking recall first, then precision.
- Keep the pipeline reproducible and documented.
- Final model must be MIT/Apache 2.0 and ≤8B parameters.
- No external data lookup, geocoding APIs, government databases, or commercial ER services.

---

## Winning formula

**High-recall union blocking** → **rich string/address/country features** → **calibrated GBDT ensemble** → **singleton gate** → **F0.5-tuned thresholds** → **strict validation** → **clean submission package**.

If you execute that carefully, you give yourself a real shot at the top. The biggest mistakes are over-merging, ignoring singletons, and letting blocking miss true matches.