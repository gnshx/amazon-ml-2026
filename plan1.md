# A Competition Plan for the Amazon Business Entity Resolution Challenge

## Executive recommendation

The strongest practical strategy is a **high-recall, multi-route candidate generator followed by a precision-calibrated pair classifier and a strict per-entity abstention rule**. Do not solve this as one global nearest-neighbor lookup. A Source 1 business may match zero, one, or several records across Sources 2 and 3, and the score is macro-averaged over Source 1 entities. The model must therefore learn both which candidate pairs are credible and when to return an empty list.

No plan can guarantee first place without seeing the data, competing submissions, or hidden labels. The plan below is designed to maximize expected private-leaderboard performance while avoiding the common failure modes most likely to erase a strong score: low candidate recall, false merges, singleton errors, validation leakage, overfitting the public leaderboard, and mishandling France.

## What matters most in this challenge

The task is to link every Source 1 record to zero or more Source 2 and Source 3 records. The target is not a single best match. The scored output is a set of linked IDs for each Source 1 ID.

The evaluation uses macro-averaged per-Source-1 F0.5. Precision matters more than recall, and correctly returning an empty list for a true singleton earns a full score for that entity. A plausible but wrong link to a singleton can turn that entity's score from 1 to 0. This makes **conservative, calibrated linking** more valuable than maximizing raw pair recall at inference.

The challenge also specifies that test data includes France, which is absent from training. Treat country as an open-set text field. A country label can be used as a pairwise consistency feature, but must not be a filter or a closed-set one-hot assumption that drops France.

## Phase 1: Audit the data before modeling

Read every TSV with an explicit tab delimiter. Check column names, row counts, duplicate IDs, blank values, country labels, and whether the source prefixes agree with their files. Parse each ground-truth comma-separated list into a set, with an empty cell represented by an empty set. Assert that every matched ID exists in Source 2 or Source 3 and that no Source 1 ID is linked to itself.

Measure the quantities that determine the modeling strategy: the fraction of Source 1 entities that are singletons; the distribution of match counts per entity; positive-link counts by target source; exact duplicate and near-duplicate rates; field missingness; and the number of records by country and source. Inspect random positive pairs and difficult negatives. In particular, look for repeated business names, chain locations, same-address buildings, transliterations, and records with only one useful field.

Keep all training and test records in their original text form. Do not query the web, use geocoding, call entity-resolution APIs, or augment the records with external business data. That is both prohibited by the statement and a major audit risk.

## Phase 2: Build a candidate generator whose recall is measurable

Candidate generation is the recall ceiling: a matching model cannot recover a true link that blocking never presents. Build a **union of complementary blocking routes**, and measure candidate recall on held-out entities before spending time tuning the classifier.

For each Source 1 record, search Source 2 and Source 3 separately. Add candidates from several routes:

1. Exact match on lightly normalized business name, and exact match on normalized address where available.

1. Shared rare name tokens or token prefixes, using inverse-document-frequency weights so common words such as “store” do not dominate.

1. Character n-gram TF-IDF nearest neighbors for names. This is robust to punctuation, spelling noise, abbreviations, and modest transliteration differences.

1. Word/token TF-IDF nearest neighbors for names and addresses, preserving a complementary signal to character n-grams.

1. Address routes based on shared distinctive tokens, numeric components, and postal-code-like strings when present. Treat these as useful evidence, not mandatory requirements.

1. A generous top-k nearest-neighbor route for name and address representations, with separate limits by target source.

Normalize conservatively. Lowercase; normalize Unicode; standardize punctuation and whitespace; and expand only high-confidence common abbreviations. Preserve both raw and normalized forms. Avoid aggressive removal of legal suffixes or location tokens: these can erase useful distinctions between branches or businesses with similar names.

Do not require country equality during blocking. Use it as a soft feature. If both records have a country value, equality can increase confidence; inequality can reduce it. Missing or unseen values must not eliminate a candidate. This keeps French test records eligible and avoids brittle country-specific logic.

For each validation entity, compute candidate recall as the fraction of its true linked IDs present in the candidate set. Also report candidate recall separately for Source 2 and Source 3, single-field-only records, countries, and match-count groups. Track average and high-percentile candidate counts and the reduction ratio. The aim is to increase recall until the extra candidates have a manageable downstream cost. Do not select a blocker by candidate count alone.

The `candidate_pairs.tsv` output must record the final candidate set actually scored by the final model, after every blocking step and before pair classification. Every predicted link must be included there.

## Phase 3: Train a pairwise match model

Create one example per candidate pair `(Source 1, candidate Source 2/3)`. Label the pair positive if the candidate ID is in that Source 1 entity's ground-truth set and negative otherwise. Retain the target-source indicator because Source 2 and Source 3 may have different noise patterns.

Start with a gradient-boosted tree model such as **LightGBM (MIT)** or **CatBoost (Apache-2.0)**, after verifying the exact package/model license used in the final package. These are compact, reproducible models that fit the stated license and size constraints. A linear model on TF-IDF similarities is a useful baseline, not necessarily the final model. Avoid a foundation model unless its exact license, parameter limit, reproducibility, and competition-rule compliance are clearly established; the challenge can be won with record-derived features and a supervised pair model.

Useful pair features include:

- Name similarity: normalized edit similarity, token-sort and token-set similarity, Jaro-Winkler or equivalent, character n-gram cosine, word TF-IDF cosine, and weighted token overlap.

- Address similarity: the same lexical features, plus overlap of distinctive numbers and tokens. Keep components such as unit numbers and postal-code-like strings separately where extraction is reliable.

- Joint evidence: name and address similarities together, their minimum or product, and indicators for “strong name / weak address” or the reverse. These help distinguish a true partial match from a name collision.

- Record quality: missingness, character/token counts, number of shared rare tokens, and whether the match depends on a single generic token.

- Context: target source, country equality/inequality/missingness, and whether a value is unseen in training. Encode country as an open-set comparison, not fixed US/India categories.

Train on realistic candidate pairs, including hard negatives that share a name token, address token, or common business name. If negative sampling is needed for scale, retain the sampling probability and correct the resulting probability calibration, or use sampling only for model fitting and calibrate on an unsampled validation set. Random easy negatives alone will make the classifier look impressive and then fail on real ambiguities.

A useful enhancement is to compare one boosted-tree model against a calibrated linear similarity model, then blend their out-of-fold probabilities if the blend improves the entity-level validation metric. The second model is worthwhile only if it adds complementary errors rather than merely increasing complexity.

## Phase 4: Validate without leaking entity identity

Do not tune only on random candidate-pair splits. Pairs from the same underlying business can appear on both sides, creating an unrealistically easy validation score.

Construct identity groups from the training ground truth: connect each Source 1 record to every matched Source 2 and Source 3 record, then take connected components. A Source 1 singleton is its own component. Use grouped folds so a component is not split between pair-model training and validation. In every fold, run the complete pipeline: candidate generation, feature computation, model training, probability calibration, and threshold selection.

For validation, score the actual set of predicted IDs per Source 1 entity, including entities for which the model predicts nothing. Compute the challenge's per-entity F0.5 and macro-average it. Also report singleton accuracy, false-link rate on true singletons, precision and recall for non-singletons, and candidate recall. Pairwise AUC or F1 is only diagnostic; it is not the competition objective.

Because full-pool ranking can differ from a smaller validation pool, add a second validation view that preserves realistic candidate competition as closely as possible. Make sure the same construction is applied consistently across folds, and document it. Prefer stable performance across folds over a spectacular score from one favorable split.

Use out-of-fold predictions for threshold selection. The final threshold must not be chosen on the public leaderboard. Treat public score changes as noisy feedback and make a change only when offline folds support it.

## Phase 5: Optimize the actual F0.5 decision rule

At inference, score every candidate pair, then decide independently for each Source 1 entity which candidate IDs to retain. Never force a best match. A low-confidence best candidate should still yield an empty list.

Tune decision thresholds against the complete macro per-entity F0.5 on out-of-fold predictions. Search a compact grid over a global probability threshold and, if supported by enough validation examples, separate thresholds for Source 2 and Source 3. Source-specific thresholds often help when source noise differs, but should be shrunk toward a shared threshold when one source has little validation data.

Evaluate an optional two-part decision: first, determine whether the entity has any match; second, retain individual candidates above a calibrated threshold. The first stage can use the maximum pair score, the gap between top candidates, the number of credible candidates, and the strongest combined name/address evidence. Keep it only if cross-validation shows a consistent gain. It is especially useful for singleton control, but a badly calibrated gate can suppress valid multi-match entities.

Allow multiple matches when several candidates independently exceed the evidence threshold. Do not impose a one-to-one assignment: the problem statement explicitly permits many Source 2 or Source 3 records to match one Source 1 entity. Tune for the set metric, not for an assumed one-to-one database relationship.

Check threshold stability by fold and by entity subgroup. If tiny threshold changes swing the score substantially, calibration or validation sample size is inadequate. Prefer a stable, slightly more conservative choice to a fragile public-board peak.

## Phase 6: Make France and distribution shift boring

France is not a reason to invent country-specific rules. Build a pipeline that uses general string features and treats country as open text. Confirm that every test Source 1 row is emitted, including all French entities. Confirm that France is never rejected merely because its category was unseen in training.

Run stress tests on the training data by holding out one country at a time for validation where feasible. This is not a perfect proxy for France, but it reveals dependence on memorized country labels or country-specific token patterns. Compare a model with country features against one without them. If country features help only when train and validation share labels but hurt held-out-country performance, remove or downweight them.

## Phase 7: Ensembling and finalization

Use an ensemble only after the single-model pipeline and thresholds are strong. A sensible low-risk ensemble is an average of out-of-fold-calibrated probabilities from a tree model and a complementary lexical model. Tune the blend and thresholds on out-of-fold predictions. Do not ensemble models whose errors are nearly identical, and do not use the public leaderboard as the blend optimizer.

Freeze the full pipeline after validation. Retrain on all labeled training data, generate the test candidate pairs, score them, apply the selected thresholds, and write exactly one row for every test Source 1 ID. Use only valid Source 2 and Source 3 test IDs. Use comma-separated IDs without spaces inside the list, no duplicates, and an empty cell for no match. Keep the candidate file synchronized with the final model input.

Run the provided `utils/validate_submission.py` against both output files and the test directory. Then independently check that every predicted ID is in its row's candidate list, every row is unique, and the row count equals the number of test Source 1 records. Package the runnable code, pinned dependencies, both output files, README instructions, and a precise methodology write-up. Save random seeds, fold assignments, feature definitions, model parameters, calibration procedure, and threshold values so the result can be reproduced.

## Recommended order of work

| Priority | Work | Decision gate |
| --- | --- | --- |
| 1 | Data audit and ground-truth integrity checks | No malformed labels or ID inconsistencies |
| 2 | Grouped validation and challenge-metric scorer | Reproducible fold scores before tuning |
| 3 | Multi-route candidate generation | High recall at a feasible candidate count |
| 4 | Similarity-feature baseline | Strong, interpretable first reference score |
| 5 | Boosted pair classifier with hard negatives | Improvement in entity-level F0.5 across folds |
| 6 | Calibration and empty-list thresholding | Better macro F0.5 and fewer singleton false links |
| 7 | Source-specific or two-stage decisions | Kept only when stable across folds |
| 8 | Optional complementary ensemble | Consistent out-of-fold improvement |
| 9 | Full-test run, validator, and reproducibility package | Every format and audit check passes |

## Common approaches that are likely to lose points

A single top-1 nearest neighbor misses legitimate multiple matches and can turn a singleton into a false link. Exact name blocking misses typos, abbreviations, and transliteration. Address-only blocking misses records with incomplete or landmark-based addresses. Requiring exact country equality can erase valid candidates under missing or novel labels. Tuning for pairwise accuracy ignores macro averaging and the value of correct empty predictions. Random pair splits leak entity identity. Aggressive normalization can merge distinct businesses. Finally, trying to maximize the public leaderboard can overfit the public subset and reduce the private score.

## Bottom line

The highest-value sequence is **measure candidate recall, build hard-negative pair features, validate by identity group, and tune abstention on the exact per-entity F0.5 metric**. Keep the country logic open-set, return multiple matches only when evidence supports them, and treat an empty prediction as a deliberate model decision rather than a fallback. These choices directly target the challenge's scoring rules and its most consequential failure modes.

## References

[1]: /home/ubuntu/upload/6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf "Business Entity Resolution Challenge problem statement"