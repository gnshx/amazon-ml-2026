#!/usr/bin/env python3
"""Calibrated GBDT Classifier & Dedicated Singleton Gate.

Pipeline:
1. Loads candidate pairs and extracts 30+ similarity features.
2. Trains LightGBM / CatBoost pairwise match classifier across 5 connected-component folds.
3. Performs probability calibration (Platt scaling) on unsampled out-of-fold predictions.
4. Trains a dedicated Singleton Gate classifier ("Does Source 1 have ANY true match?").
5. Performs 2D/3D threshold search directly optimizing the competition Macro F0.5 metric.
6. Generates full out-of-fold evaluation report and saves model artifacts.
"""

from collections import defaultdict
import csv
import json
import os
import sys
from typing import Dict, List, Set, Tuple
import numpy as np

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Import local modules
from metrics import evaluate_macro_f05, compute_entity_f05
from feature_extractor import FeatureExtractor


def train_fold_pairwise_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[object, np.ndarray]:
    """Trains a LightGBM pairwise match classifier."""
    if not HAS_LGB:
        raise ImportError("LightGBM is required. Please install via pip install -r requirements.txt.")

    # Positive class weighting for precision bias
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos = max(1.0, (neg_count / (2.0 * pos_count))) if pos_count > 0 else 1.0

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    return model, val_probs


def build_singleton_features(
    s1_cand_probs: Dict[str, List[Tuple[str, float]]],
    s1_features: Dict[str, Dict[str, float]] = None
) -> Tuple[np.ndarray, List[str]]:
    """Builds entity-level aggregated features for the Singleton Gate."""
    feature_matrix = []
    s1_ids = []

    for s1_id, cand_prob_list in s1_cand_probs.items():
        s1_ids.append(s1_id)
        if not cand_prob_list:
            # No candidates at all -> definitely singleton
            row = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        else:
            sorted_probs = sorted([p for _, p in cand_prob_list], reverse=True)
            p_max = sorted_probs[0]
            p_2nd = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
            p_gap = p_max - p_2nd
            top3_sum = sum(sorted_probs[:3])
            c_count = float(len(sorted_probs))
            c_count_30 = float(sum(1 for p in sorted_probs if p > 0.3))
            is_empty_flag = 0.0
            row = [p_max, p_2nd, p_gap, top3_sum, c_count, c_count_30, is_empty_flag]

        feature_matrix.append(row)

    return np.array(feature_matrix), s1_ids


def optimize_thresholds_for_f05(
    s1_cand_probs: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, List[str]],
    singleton_probs: Dict[str, float] = None,
) -> Tuple[float, float, float, float]:
    """Sweeps decision thresholds to maximize macro-averaged F0.5.

    Returns:
        (best_macro_f05, best_tau_s2, best_tau_s3, best_tau_singleton)
    """
    best_f05 = -1.0
    best_tau_s2 = 0.80
    best_tau_s3 = 0.80
    best_tau_singleton = 0.70

    # Grid search candidate thresholds
    tau_s2_grid = [0.65, 0.75, 0.85, 0.90]
    tau_s3_grid = [0.65, 0.75, 0.85, 0.90]
    tau_sing_grid = [0.50, 0.65, 0.75, 0.85] if singleton_probs else [1.0]

    for t_sing in tau_sing_grid:
        for t_s2 in tau_s2_grid:
            for t_s3 in tau_s3_grid:
                preds = {}
                for s1_id, cand_probs in s1_cand_probs.items():
                    # Check singleton gate
                    if singleton_probs and singleton_probs.get(s1_id, 0.0) >= t_sing:
                        preds[s1_id] = []
                        continue

                    # Filter candidates by source threshold
                    retained = []
                    for c_id, p in cand_probs:
                        threshold = t_s2 if "s2" in c_id.lower() else t_s3
                        if p >= threshold:
                            retained.append(c_id)
                    preds[s1_id] = retained

                eval_res = evaluate_macro_f05(preds, ground_truth)
                score = eval_res["macro_f05"]
                if score > best_f05:
                    best_f05 = score
                    best_tau_s2 = t_s2
                    best_tau_s3 = t_s3
                    best_tau_singleton = t_sing

    return best_f05, best_tau_s2, best_tau_s3, best_tau_singleton


if __name__ == "__main__":
    print("Train GBDT and Singleton Gate module loaded.")
