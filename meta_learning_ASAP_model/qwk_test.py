import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def denormalize_score(score_norm: float, min_score: float, max_score: float) -> float:
    """
    Convert normalized score back to original score range.
    """
    return score_norm * (max_score - min_score) + min_score


def restore_valid_score(score_norm: float, min_score: float, max_score: float) -> int:
    """
    Convert normalized score to a valid integer score:
    1. denormalize
    2. clamp to valid range
    3. round to nearest integer
    """
    raw_score = denormalize_score(score_norm, min_score, max_score)
    raw_score = max(min_score, min(max_score, raw_score))
    return int(round(raw_score))


def compute_qwk_from_records(records):
    """
    Compute QWK separately for each (essay_set, trait) group,
    then return mean QWK across groups.
    """
    if len(records) == 0:
        return {
            "mean_qwk": 0.0,
            "group_qwks": {},
        }

    df_tmp = pd.DataFrame(records)

    group_qwks = {}
    qwk_values = []

    for (essay_set, trait), group_df in df_tmp.groupby(["essay_set", "trait"]):
        min_score = float(group_df["min_score"].iloc[0])
        max_score = float(group_df["max_score"].iloc[0])

        y_true = [
            restore_valid_score(v, min_score, max_score)
            for v in group_df["true_norm"].astype(float).tolist()
        ]
        y_pred = [
            restore_valid_score(v, min_score, max_score)
            for v in group_df["pred_norm"].astype(float).tolist()
        ]

        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        if pd.isna(qwk):
            qwk = 0.0

        key = f"prompt_{essay_set}_{trait}"
        group_qwks[key] = float(qwk)
        qwk_values.append(float(qwk))

        print(f"\nGroup: {key}")
        print(f"  Score range: [{min_score}, {max_score}]")
        print(f"  True raw scores: {y_true}")
        print(f"  Pred raw scores: {y_pred}")
        print(f"  QWK: {qwk:.6f}")

    mean_qwk = float(np.mean(qwk_values)) if len(qwk_values) > 0 else 0.0

    return {
        "mean_qwk": mean_qwk,
        "group_qwks": group_qwks,
    }


def run_test_case(case_name, records):
    print("\n" + "=" * 80)
    print(f"TEST CASE: {case_name}")
    print("=" * 80)

    results = compute_qwk_from_records(records)

    print("\nSummary")
    print(f"  Mean QWK: {results['mean_qwk']:.6f}")
    print(f"  Group QWKs: {results['group_qwks']}")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # TEST 1: Perfect predictions
    # Expected: every group QWK should be 1.0
    # ------------------------------------------------------------------
    perfect_records = [
        # prompt 1, content, range 1-6
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 0.0, "pred_norm": 0.0},   # 1
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 0.2, "pred_norm": 0.2},   # 2
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 0.4, "pred_norm": 0.4},   # 3
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 0.6, "pred_norm": 0.6},   # 4
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 0.8, "pred_norm": 0.8},   # 5
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 1.0, "pred_norm": 1.0},   # 6

        # prompt 3, language, range 0-3
        {"essay_set": 3, "trait": "language", "min_score": 0, "max_score": 3, "true_norm": 0.0, "pred_norm": 0.0},   # 0
        {"essay_set": 3, "trait": "language", "min_score": 0, "max_score": 3, "true_norm": 1/3, "pred_norm": 1/3},   # 1
        {"essay_set": 3, "trait": "language", "min_score": 0, "max_score": 3, "true_norm": 2/3, "pred_norm": 2/3},   # 2
        {"essay_set": 3, "trait": "language", "min_score": 0, "max_score": 3, "true_norm": 1.0, "pred_norm": 1.0},   # 3
    ]

    # ------------------------------------------------------------------
    # TEST 2: Slightly imperfect predictions
    # Expected: QWK should be high, but less than 1.0
    # ------------------------------------------------------------------
    slight_error_records = [
        # prompt 1, content, range 1-6
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 0.0, "pred_norm": 0.0},   # true 1, pred 1
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 0.2, "pred_norm": 0.4},   # true 2, pred 3
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 0.4, "pred_norm": 0.4},   # true 3, pred 3
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 0.6, "pred_norm": 0.6},   # true 4, pred 4
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 0.8, "pred_norm": 0.6},   # true 5, pred 4
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 1.0, "pred_norm": 1.0},   # true 6, pred 6

        # prompt 3, language, range 0-3
        {"essay_set": 3, "trait": "language", "min_score": 0, "max_score": 3, "true_norm": 0.0, "pred_norm": 0.0},      # 0 -> 0
        {"essay_set": 3, "trait": "language", "min_score": 0, "max_score": 3, "true_norm": 1/3, "pred_norm": 2/3},      # 1 -> 2
        {"essay_set": 3, "trait": "language", "min_score": 0, "max_score": 3, "true_norm": 2/3, "pred_norm": 2/3},      # 2 -> 2
        {"essay_set": 3, "trait": "language", "min_score": 0, "max_score": 3, "true_norm": 1.0, "pred_norm": 2/3},      # 3 -> 2
    ]

    # ------------------------------------------------------------------
    # TEST 3: Bad predictions
    # Expected: QWK should be low or possibly negative
    # ------------------------------------------------------------------
    bad_records = [
        # prompt 1, content, range 1-6
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 0.0, "pred_norm": 1.0},   # 1 -> 6
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 0.2, "pred_norm": 0.8},   # 2 -> 5
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 0.4, "pred_norm": 0.6},   # 3 -> 4
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 0.6, "pred_norm": 0.4},   # 4 -> 3
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 0.8, "pred_norm": 0.2},   # 5 -> 2
        {"essay_set": 1, "trait": "content", "min_score": 1, "max_score": 6, "true_norm": 1.0, "pred_norm": 0.0},   # 6 -> 1

        # prompt 3, language, range 0-3
        {"essay_set": 3, "trait": "language", "min_score": 0, "max_score": 3, "true_norm": 0.0, "pred_norm": 1.0},   # 0 -> 3
        {"essay_set": 3, "trait": "language", "min_score": 0, "max_score": 3, "true_norm": 1/3, "pred_norm": 2/3},   # 1 -> 2
        {"essay_set": 3, "trait": "language", "min_score": 0, "max_score": 3, "true_norm": 2/3, "pred_norm": 1/3},   # 2 -> 1
        {"essay_set": 3, "trait": "language", "min_score": 0, "max_score": 3, "true_norm": 1.0, "pred_norm": 0.0},   # 3 -> 0
    ]

    # ------------------------------------------------------------------
    # TEST 4: Check clamp behavior
    # Expected: predictions outside [0,1] should be clamped to valid raw range
    # ------------------------------------------------------------------
    clamp_records = [
        {"essay_set": 5, "trait": "content", "min_score": 0, "max_score": 4, "true_norm": 0.0, "pred_norm": -0.5},  # below min -> clamp to 0
        {"essay_set": 5, "trait": "content", "min_score": 0, "max_score": 4, "true_norm": 0.5, "pred_norm": 0.5},   # normal
        {"essay_set": 5, "trait": "content", "min_score": 0, "max_score": 4, "true_norm": 1.0, "pred_norm": 1.5},   # above max -> clamp to 4
    ]

    run_test_case("Perfect predictions (expect QWK = 1.0)", perfect_records)
    run_test_case("Slightly imperfect predictions (expect high QWK < 1.0)", slight_error_records)
    run_test_case("Bad predictions (expect low/negative QWK)", bad_records)
    run_test_case("Clamp behavior check", clamp_records)