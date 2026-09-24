# amazon-ml-2026

## Amazon ML Challenge 2026 — Business Entity Resolution Challenge

End-to-end machine learning pipeline for large-scale multi-source business entity resolution.

### Architecture
- **High-Recall Multi-Route Blocking**: 7 complementary routes (Exact Core, Sorted Key, Prefix, Char $n$-gram TF-IDF, Rare Tokens, Postal/Street Structure, Phonetic Soundex) targeting $\ge 98\%$ candidate recall.
- **Pairwise Feature Engineering**: 30+ similarity and discrepancy signals.
- **Calibrated GBDT Classifier**: LightGBM/CatBoost with probability calibration.
- **Dedicated Singleton Gate**: Entity-level classification protecting the 1.0 macro $F_{0.5}$ singleton score.
- **1-to-1 Mutual-Best Conflict Resolution**: Mathematically verified target exclusivity constraint.

### Structure
```
├── dataset/
│   ├── train/
│   └── test/
├── src/
│   ├── metrics.py
│   ├── data_audit_and_split.py
│   ├── generate_baseline_submission.py
│   ├── multi_route_blocker.py
│   ├── feature_extractor.py
│   └── train_gbdt_and_singleton_gate.py
├── utils/
│   └── validate_submission.py
├── best_plan.md
├── Documentation_template.md
├── package_submission.py
└── run_pipeline.py
```
