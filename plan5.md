Winning this one comes down to three things: blocking recall, feature quality, and precision-tuned thresholding — in that order of leverage. Here's how I'd approach it end to end.

**1. Nail the blocking stage first (it caps your recall ceiling)**

Since F0.5 macro-averages per S1 entity and singletons matter, you cannot afford to drop true matches at the candidate stage — but you also can't generate huge candidate sets (kills precision downstream). Use a hybrid, not a single method:

- **Lexical blocking**: sorted-token n-grams, character 3-grams + TF-IDF cosine top-k, and phonetic keys (Soundex/Double Metaphone) on normalized business names — this catches typos and word-order swaps.
- **Semantic blocking**: embed names+addresses with a small open-source sentence embedding model (e.g. MiniLM, Apache-2.0 licensed) and do approximate nearest-neighbor retrieval (FAISS). This is what saves you on transliteration variants and generalizes to France even though it's unseen in training — it doesn't rely on country-specific string rules.
- **Address blocking**: postal code exact/fuzzy match, city-name blocking, first-line-of-address token overlap.
- Take the **union** of candidates from these methods, then measure recall on a held-out validation split (fraction of true matches present in candidate_pairs). Aim for >95% candidate recall before you even think about the matching model — you can't recover a match the blocker never surfaced.

**2. Normalize aggressively, but generically (not with hardcoded US/India rules)**

- Strip/standardize legal suffixes (Inc/Corp/Ltd/Pvt/LLC), punctuation (& vs "and"), casing.
- Expand common address abbreviations (St/Rd/Ave), strip landmark phrases ("Near SBI ATM") into a separate feature rather than deleting them outright.
- Keep normalization rule-based but *pattern-driven* (regex on common suffix/abbreviation lists), not conditioned on the `country` field — since France appears only at test time, anything keyed to `country == 'US'/'India'` will silently fail there. Let embeddings absorb the cross-language generalization.

**3. Feature engineering for the pairwise matcher**

For each (S1, candidate) pair, compute:
- Name: Jaccard/token-sort ratio, Levenshtein ratio, Jaro-Winkler, TF-IDF cosine, acronym/initials match, suffix-stripped similarity, embedding cosine similarity.
- Address: postal code match, city/state match, street-number match, token overlap, embedding cosine similarity on the address strings.
- Structural: length ratios, token-count differences, country match/mismatch (as a feature, not a filter).

**4. Matching model — optimize for precision, not just AUC**

Given the license/size constraint (MIT/Apache-2.0, ≤8B params), the strongest combo is usually:
- A **gradient boosted tree** (LightGBM/XGBoost/CatBoost) trained on the pairwise features above, using true matches from `train_ground_truth.tsv` as positives and other candidates from the same blocking pass as negatives. GBMs are fast, robust on tabular similarity features, and easy to threshold precisely — often beating end-to-end neural matchers on this kind of task.
- Optionally add a small cross-encoder's match probability (e.g. a lightweight Apache/MIT-licensed BERT variant fine-tuned on name+address pairs) as *one additional feature* into the GBM, rather than using it standalone — this gives you semantic robustness without giving up the calibration precision of trees.
- Calibrate probabilities (Platt/isotonic) so your threshold search is meaningful.

**5. Threshold tuning is where F0.5 is actually won**

- Because F0.5 weights precision 2× recall, tune your decision threshold on the validation split by directly maximizing macro F0.5 (grid search the probability cutoff), not by optimizing F1 or accuracy.
- Explicitly handle singletons: if no candidate clears the threshold for an S1 entity, output an empty match list — this is worth a full 1.0, so don't force a match just because a low-confidence candidate exists.
- Resolve conflicts where one S2/S3 record scores high against multiple S1 entities by keeping only its best-scoring match (mutual best-match constraint) — this protects precision.

**6. Validate like the real scorer**

Hold out a slice of `train_source1`/ground truth, run your full pipeline on it, and compute the exact macro F0.5 formula given. Check performance split by country to sanity-check you're not overfitting to US/India patterns — that's your best proxy for how you'll do on the unseen France entities.

**7. Process/logistics that actually affect ranking**

- Run `utils/validate_submission.py` locally before every leaderboard upload — you only get 5 submissions/day, don't burn one on a format rejection.
- Keep candidate_pairs.tsv genuinely equal to what you feed the model at inference (it's audited).
- Start the methodology write-up early and update it as you iterate — top teams' packages get reviewed in detail.
- Never touch external lookups/geocoding APIs — instant disqualification, not worth the risk even "just to validate."

If you want, I can help you sketch the actual blocking + LightGBM pipeline code once you've got the data loaded — that's usually the fastest way to get a working baseline you can then iterate on toward the top of the leaderboard.