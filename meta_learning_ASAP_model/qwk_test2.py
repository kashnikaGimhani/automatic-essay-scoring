#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import cohen_kappa_score


def denormalize_score(score_norm: float, min_score: float, max_score: float) -> float:
    return score_norm * (max_score - min_score) + min_score


def clamp_and_round_score(score_raw: float, min_score: float, max_score: float) -> int:
    return int(np.clip(np.round(score_raw), min_score, max_score))


def compute_qwk(y_true, y_pred) -> float:
    if len(y_true) == 0:
        return float("nan")

    unique_labels = sorted(set(y_true) | set(y_pred))
    if len(unique_labels) <= 1:
        return 1.0

    return float(
        cohen_kappa_score(
            y_true,
            y_pred,
            labels=unique_labels,
            weights="quadratic",
        )
    )


def evaluate_qwk_per_task(records):
    """
    records: list of dicts with keys:
        essay_set
        trait
        target_norm
        pred_norm
        min_score
        max_score
    """
    task_to_true = {}
    task_to_pred = {}

    for r in records:
        essay_set = int(r["essay_set"])
        trait = str(r["trait"])
        target_norm = float(r["target_norm"])
        pred_norm = float(r["pred_norm"])
        min_s = float(r["min_score"])
        max_s = float(r["max_score"])

        true_raw = denormalize_score(target_norm, min_s, max_s)
        pred_raw = denormalize_score(pred_norm, min_s, max_s)

        true_int = clamp_and_round_score(true_raw, min_s, max_s)
        pred_int = clamp_and_round_score(pred_raw, min_s, max_s)

        task_key = (essay_set, trait)
        task_to_true.setdefault(task_key, []).append(true_int)
        task_to_pred.setdefault(task_key, []).append(pred_int)

    qwk_per_task = {}
    qwk_values = []

    for task_key in sorted(task_to_true.keys()):
        y_true = task_to_true[task_key]
        y_pred = task_to_pred[task_key]
        qwk = compute_qwk(y_true, y_pred)
        qwk_per_task[task_key] = qwk
        if not np.isnan(qwk):
            qwk_values.append(qwk)

    mean_qwk = float(np.mean(qwk_values)) if len(qwk_values) > 0 else float("nan")

    return qwk_per_task, mean_qwk


def print_case(title, records):
    print("=" * 80)
    print(title)
    print("-" * 80)

    qwk_per_task, mean_qwk = evaluate_qwk_per_task(records)

    for task_key, qwk in qwk_per_task.items():
        print(f"Task {task_key}: QWK = {qwk:.6f}")

    print(f"Mean QWK: {mean_qwk:.6f}")
    print()


def main():
    # ------------------------------------------------------------------
    # CASE 1: Perfect prediction
    # Expected: QWK = 1.0 for all tasks
    # ------------------------------------------------------------------
    perfect_records = [
        {"essay_set": 1, "trait": "content", "target_norm": 0.0, "pred_norm": 0.0, "min_score": 1, "max_score": 6},   # 1
        {"essay_set": 1, "trait": "content", "target_norm": 0.2, "pred_norm": 0.2, "min_score": 1, "max_score": 6},   # 2
        {"essay_set": 1, "trait": "content", "target_norm": 0.4, "pred_norm": 0.4, "min_score": 1, "max_score": 6},   # 3
        {"essay_set": 1, "trait": "content", "target_norm": 0.6, "pred_norm": 0.6, "min_score": 1, "max_score": 6},   # 4
        {"essay_set": 1, "trait": "content", "target_norm": 0.8, "pred_norm": 0.8, "min_score": 1, "max_score": 6},   # 5
        {"essay_set": 1, "trait": "content", "target_norm": 1.0, "pred_norm": 1.0, "min_score": 1, "max_score": 6},   # 6

        {"essay_set": 8, "trait": "organization", "target_norm": 0.0, "pred_norm": 0.0, "min_score": 2, "max_score": 12},  # 2
        {"essay_set": 8, "trait": "organization", "target_norm": 0.3, "pred_norm": 0.3, "min_score": 2, "max_score": 12},  # 5
        {"essay_set": 8, "trait": "organization", "target_norm": 0.5, "pred_norm": 0.5, "min_score": 2, "max_score": 12},  # 7
        {"essay_set": 8, "trait": "organization", "target_norm": 0.8, "pred_norm": 0.8, "min_score": 2, "max_score": 12},  # 10
        {"essay_set": 8, "trait": "organization", "target_norm": 1.0, "pred_norm": 1.0, "min_score": 2, "max_score": 12},  # 12
    ]

    # ------------------------------------------------------------------
    # CASE 2: Slightly imperfect prediction
    # Expected: high positive QWK, but less than 1.0
    # ------------------------------------------------------------------
    slight_error_records = [
        {"essay_set": 1, "trait": "content", "target_norm": 0.0, "pred_norm": 0.1, "min_score": 1, "max_score": 6},
        {"essay_set": 1, "trait": "content", "target_norm": 0.2, "pred_norm": 0.2, "min_score": 1, "max_score": 6},
        {"essay_set": 1, "trait": "content", "target_norm": 0.4, "pred_norm": 0.5, "min_score": 1, "max_score": 6},
        {"essay_set": 1, "trait": "content", "target_norm": 0.6, "pred_norm": 0.6, "min_score": 1, "max_score": 6},
        {"essay_set": 1, "trait": "content", "target_norm": 0.8, "pred_norm": 0.7, "min_score": 1, "max_score": 6},
        {"essay_set": 1, "trait": "content", "target_norm": 1.0, "pred_norm": 0.9, "min_score": 1, "max_score": 6},

        {"essay_set": 8, "trait": "organization", "target_norm": 0.0, "pred_norm": 0.1, "min_score": 2, "max_score": 12},
        {"essay_set": 8, "trait": "organization", "target_norm": 0.3, "pred_norm": 0.25, "min_score": 2, "max_score": 12},
        {"essay_set": 8, "trait": "organization", "target_norm": 0.5, "pred_norm": 0.6, "min_score": 2, "max_score": 12},
        {"essay_set": 8, "trait": "organization", "target_norm": 0.8, "pred_norm": 0.75, "min_score": 2, "max_score": 12},
        {"essay_set": 8, "trait": "organization", "target_norm": 1.0, "pred_norm": 0.9, "min_score": 2, "max_score": 12},
    ]

    # ------------------------------------------------------------------
    # CASE 3: Bad / reversed prediction
    # Expected: low or negative QWK
    # ------------------------------------------------------------------
    bad_records = [
        {"essay_set": 1, "trait": "content", "target_norm": 0.0, "pred_norm": 1.0, "min_score": 1, "max_score": 6},
        {"essay_set": 1, "trait": "content", "target_norm": 0.2, "pred_norm": 0.8, "min_score": 1, "max_score": 6},
        {"essay_set": 1, "trait": "content", "target_norm": 0.4, "pred_norm": 0.6, "min_score": 1, "max_score": 6},
        {"essay_set": 1, "trait": "content", "target_norm": 0.6, "pred_norm": 0.4, "min_score": 1, "max_score": 6},
        {"essay_set": 1, "trait": "content", "target_norm": 0.8, "pred_norm": 0.2, "min_score": 1, "max_score": 6},
        {"essay_set": 1, "trait": "content", "target_norm": 1.0, "pred_norm": 0.0, "min_score": 1, "max_score": 6},

        {"essay_set": 8, "trait": "organization", "target_norm": 0.0, "pred_norm": 1.0, "min_score": 2, "max_score": 12},
        {"essay_set": 8, "trait": "organization", "target_norm": 0.3, "pred_norm": 0.8, "min_score": 2, "max_score": 12},
        {"essay_set": 8, "trait": "organization", "target_norm": 0.5, "pred_norm": 0.5, "min_score": 2, "max_score": 12},
        {"essay_set": 8, "trait": "organization", "target_norm": 0.8, "pred_norm": 0.2, "min_score": 2, "max_score": 12},
        {"essay_set": 8, "trait": "organization", "target_norm": 1.0, "pred_norm": 0.0, "min_score": 2, "max_score": 12},
    ]

    # ------------------------------------------------------------------
    # CASE 4: Clamp behavior
    # Expected: predictions outside range get clamped correctly
    # ------------------------------------------------------------------
    clamp_records = [
        {"essay_set": 7, "trait": "style", "target_norm": 0.0, "pred_norm": -0.5, "min_score": 0, "max_score": 6},
        {"essay_set": 7, "trait": "style", "target_norm": 0.2, "pred_norm": 0.2,  "min_score": 0, "max_score": 6},
        {"essay_set": 7, "trait": "style", "target_norm": 0.4, "pred_norm": 0.4,  "min_score": 0, "max_score": 6},
        {"essay_set": 7, "trait": "style", "target_norm": 0.6, "pred_norm": 0.6,  "min_score": 0, "max_score": 6},
        {"essay_set": 7, "trait": "style", "target_norm": 0.8, "pred_norm": 1.4,  "min_score": 0, "max_score": 6},
        {"essay_set": 7, "trait": "style", "target_norm": 1.0, "pred_norm": 2.0,  "min_score": 0, "max_score": 6},
    ]

    print_case("CASE 1: PERFECT PREDICTION", perfect_records)
    print_case("CASE 2: SLIGHTLY IMPERFECT PREDICTION", slight_error_records)
    print_case("CASE 3: BAD / REVERSED PREDICTION", bad_records)
    print_case("CASE 4: CLAMP BEHAVIOR", clamp_records)


if __name__ == "__main__":
    main()