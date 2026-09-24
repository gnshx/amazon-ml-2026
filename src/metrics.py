"""Official Metric Implementation: Macro-averaged per-Source-1 F0.5.

Metric Specification:
- Precision is weighted 2x recall (beta = 0.5):
  F0.5 = (1 + 0.5^2) * (P * R) / (0.5^2 * P + R) = 1.25 * P * R / (0.25 * P + R)
- If both prediction and ground-truth are empty (true singleton correctly predicted empty): F0.5 = 1.0
- If ground-truth is empty and prediction is non-empty (false match on singleton): F0.5 = 0.0
- If ground-truth is non-empty and prediction is empty (missed match): F0.5 = 0.0
- If P + R == 0: F0.5 = 0.0
- Final score: Macro-average of F0.5 across all Source 1 entities.
"""

from typing import Dict, List, Set, Union


def compute_entity_f05(pred_set: Set[str], gold_set: Set[str]) -> float:
    """Computes F0.5 for a single Source 1 entity."""
    pred_len = len(pred_set)
    gold_len = len(gold_set)

    # Both empty: true singleton correctly predicted empty -> 1.0
    if pred_len == 0 and gold_len == 0:
        return 1.0

    # Prediction non-empty but gold empty (false positive on singleton) -> 0.0
    if pred_len > 0 and gold_len == 0:
        return 0.0

    # Prediction empty but gold non-empty (false negative) -> 0.0
    if pred_len == 0 and gold_len > 0:
        return 0.0

    # Both non-empty: standard precision and recall
    intersection = len(pred_set.intersection(gold_set))
    if intersection == 0:
        return 0.0

    precision = intersection / pred_len
    recall = intersection / gold_len

    denominator = 0.25 * precision + recall
    if denominator == 0:
        return 0.0

    return (1.25 * precision * recall) / denominator


def evaluate_macro_f05(
    predictions: Dict[str, Union[Set[str], List[str]]],
    ground_truth: Dict[str, Union[Set[str], List[str]]],
) -> Dict[str, float]:
    """Computes macro-averaged F0.5, singleton accuracy, and precision/recall diagnostics.

    Args:
        predictions: Dict mapping source1_entity_id -> set/list of predicted target IDs.
        ground_truth: Dict mapping source1_entity_id -> set/list of true target IDs.

    Returns:
        Dict containing:
            - macro_f05: The primary competition score.
            - singleton_accuracy: Accuracy on true singletons (target empty).
            - non_singleton_f05: F0.5 evaluated solely on non-singletons.
            - singleton_count: Total count of true singletons.
            - non_singleton_count: Total count of non-singletons.
    """
    all_s1_ids = list(ground_truth.keys())
    entity_f05_scores = []
    singleton_scores = []
    non_singleton_scores = []

    for s1_id in all_s1_ids:
        gold = set(ground_truth.get(s1_id, []))
        pred = set(predictions.get(s1_id, []))

        score = compute_entity_f05(pred, gold)
        entity_f05_scores.append(score)

        if len(gold) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

    mean_f05 = sum(entity_f05_scores) / len(entity_f05_scores) if entity_f05_scores else 0.0
    mean_singleton = sum(singleton_scores) / len(singleton_scores) if singleton_scores else 0.0
    mean_non_singleton = sum(non_singleton_scores) / len(non_singleton_scores) if non_singleton_scores else 0.0

    return {
        "macro_f05": float(mean_f05),
        "singleton_accuracy": float(mean_singleton),
        "non_singleton_f05": float(mean_non_singleton),
        "singleton_count": len(singleton_scores),
        "non_singleton_count": len(non_singleton_scores),
        "total_entities": len(all_s1_ids),
    }


if __name__ == "__main__":
    # Unit tests for edge cases
    test_cases = [
        ("Singleton correct", set(), set(), 1.0),
        ("Singleton false positive", {"S2_1"}, set(), 0.0),
        ("Missed match", set(), {"S2_1"}, 0.0),
        ("Exact match (1/1)", {"S2_1"}, {"S2_1"}, 1.0),
        ("1 true + 1 false match", {"S2_1", "S2_2"}, {"S2_1"}, 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0)),  # 0.625 / 1.125 = 0.5555...
    ]
    print("Running metric unit tests (Standard Library)...")
    for name, p, g, expected in test_cases:
        score = compute_entity_f05(p, g)
        assert abs(score - expected) < 1e-4, f"Failed on {name}: got {score}, expected {expected}"
        print(f"  [PASS] {name}: {score:.4f}")
    print("All metric unit tests passed successfully!")
