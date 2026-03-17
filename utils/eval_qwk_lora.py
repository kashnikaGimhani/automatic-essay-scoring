import pandas as pd
import numpy as np
import re
import logging
from sklearn.metrics import confusion_matrix, cohen_kappa_score, mean_squared_error
from six import string_types
import json
import ast
from typing import Optional


class Evaluator:
    """
    Parse model outputs (string -> dict of trait scores) and compute QWK/RMSE.

    Supported prediction/target formats (any of these):
      - JSON dict: {"overall": 5, "content": 4, ...}
      - Python dict: {'overall': 5, 'content': 4, ...}
      - Key-value pairs without braces: overall: 5, content: 4, ...

    Missing / unparseable trait values become -1.
    """

    def __init__(self, traits):
        self.traits = traits

    # ---------- Metrics ----------
    def kappa(self, y_true, y_pred, weights=None, allow_off_by_one=False):
        logger = logging.getLogger(__name__)
        assert len(y_true) == len(y_pred)

        try:
            y_true = [int(np.round(float(y))) for y in y_true]
            y_pred = [int(np.round(float(y))) for y in y_pred]
        except ValueError as e:
            logger.error(
                "For kappa, labels should be ints or strings convertible to ints (e.g., '4.0', '3')."
            )
            raise e

        min_rating = min(min(y_true), min(y_pred))
        max_rating = max(max(y_true), max(y_pred))

        y_true = [y - min_rating for y in y_true]
        y_pred = [y - min_rating for y in y_pred]

        num_ratings = max_rating - min_rating + 1
        observed = confusion_matrix(y_true, y_pred, labels=list(range(num_ratings)))
        num_scored_items = float(len(y_true))

        if isinstance(weights, string_types):
            wt_scheme = weights
            weights = None
        else:
            wt_scheme = ""

        if weights is None:
            weights = np.empty((num_ratings, num_ratings))
            for i in range(num_ratings):
                for j in range(num_ratings):
                    diff = abs(i - j)
                    if allow_off_by_one and diff:
                        diff -= 1
                    if wt_scheme == "linear":
                        weights[i, j] = diff
                    elif wt_scheme == "quadratic":
                        weights[i, j] = diff**2
                    elif not wt_scheme:
                        weights[i, j] = bool(diff)
                    else:
                        raise ValueError(f"Invalid weight scheme specified for kappa: {wt_scheme}")

        hist_true = np.bincount(y_true, minlength=num_ratings)[:num_ratings] / num_scored_items
        hist_pred = np.bincount(y_pred, minlength=num_ratings)[:num_ratings] / num_scored_items
        expected = np.outer(hist_true, hist_pred)

        observed = observed / num_scored_items

        k = 1.0
        if np.count_nonzero(weights):
            k -= (np.sum(weights * observed) / np.sum(weights * expected))

        return k

    def calc_kappa(self, pred, original, weight="quadratic"):
        return self.kappa(original, pred, weights=weight)

    def calc_kappa_cohen(self, pred, original, weight="quadratic"):
        return cohen_kappa_score(original, pred, weights=weight)

    def calc_rmse(self, pred, original):
        return mean_squared_error(original, pred) ** 0.5

    # ---------- Parsing ----------
    def read_results(self, pred):
        """Parse list/series of strings into a dataframe of trait columns."""
        results = pd.DataFrame(columns=self.traits)

        for i in range(len(pred)):
            text = str(pred[i]).strip()

            # Normalize bare nan tokens
            safe = re.sub(r"\bnan\b", "None", text, flags=re.IGNORECASE)

            # If looks like key:value pairs but missing braces, wrap it
            if ":" in safe and not safe.lstrip().startswith("{"):
                safe = "{" + safe + "}"

            obj = None
            try:
                obj = json.loads(safe)
            except Exception:
                try:
                    obj = ast.literal_eval(safe)
                except Exception:
                    obj = None

            trait_scores = []
            if isinstance(obj, dict):
                for trait in self.traits:
                    v = obj.get(trait)

                    # allow underscore/space variants
                    if v is None:
                        alt = trait.replace("_", " ") if "_" in trait else trait.replace(" ", "_")
                        v = obj.get(alt, None)

                    if v is None:
                        trait_scores.append(-1)
                    else:
                        try:
                            trait_scores.append(float(v))
                        except Exception:
                            trait_scores.append(-1)
            else:
                # regex fallback
                for trait in self.traits:
                    m = re.search(
                        rf"{re.escape(trait)}\s*['\"]?\s*:\s*(nan|-?\d+(?:\.\d+)?)",
                        text,
                        flags=re.IGNORECASE,
                    )
                    if (m is None) or (m.group(1).lower() == "nan"):
                        trait_scores.append(-1)
                    else:
                        trait_scores.append(float(m.group(1)))

            assert len(self.traits) == len(trait_scores), "trait length error!"
            results.loc[i] = trait_scores

        return results.astype("float").fillna(-1)

    # ---------- Evaluation ----------
    def evaluate_notnull(self, pred, target, weight="quadratic"):
        """
        Trait-wise QWK, skipping rows where target is missing (-1).
        """
        results = self.read_results(pred)
        targets = self.read_results(target)

        qwk_results = {}
        for key in results.columns:
            trait_tgt = targets[key][targets[key] != -1]
            if len(trait_tgt) == 0:
                qwk_results[key] = np.nan
                continue
            trait_pred = results[key][trait_tgt.index]
            qwk_results[key] = self.calc_kappa(trait_pred, trait_tgt, weight=weight)

        return qwk_results


# ============================================================
# Trait mapping helpers (UPDATED to match your new pipeline)
# ============================================================

# Small dataset schema -> ASAP schema (your requested mapping)
SMALL_TO_ASAP = {
    "ideas": "content",
    "flow": "sentence_fluency",
    "coherence": "organization",
    "vocab": "word_choice",
    "grammar": "conventions",
}

# These are the ONLY traits we care about in LoRA fine-tuning + evaluation
ASAP_TRAITS_5 = ["content", "sentence_fluency", "organization", "word_choice", "conventions"]

# During evaluation we also include overall, but it is COMPUTED, not predicted during training
ASAP_TRAITS_WITH_OVERALL = ["overall"] + ASAP_TRAITS_5


def compute_overall_from_5(d: dict) -> Optional[int]:
    """
    overall = round(mean of the five mapped trait scores)
    Returns None if any of the 5 traits missing.
    """
    vals = []
    for k in ASAP_TRAITS_5:
        v = d.get(k, None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        vals.append(float(v))
    return int(round(sum(vals) / len(vals)))


def build_asap_target_strings_from_rows(df: pd.DataFrame) -> pd.Series:
    """
    Build target strings for Evaluator in ASAP schema + computed overall.

    Works for:
    - Your small dataset columns: ideas, flow, coherence, vocab, grammar (overall_score optional)
    - Or already-unified ASAP columns: content, sentence_fluency, organization, word_choice, conventions

    IMPORTANT:
    - We ALWAYS compute overall from the 5 traits to match your rule.
    - We do NOT depend on df['overall_score'] even if it exists.
    """
    has_small_schema = all(c in df.columns for c in ["ideas", "flow", "coherence", "vocab", "grammar"])
    has_asap_schema = all(c in df.columns for c in ASAP_TRAITS_5)

    out = []
    for _, row in df.iterrows():
        d = {}

        # (1) Fill 5 ASAP traits from whichever schema exists
        if has_asap_schema:
            for k in ASAP_TRAITS_5:
                d[k] = row.get(k)
        elif has_small_schema:
            for src, tgt in SMALL_TO_ASAP.items():
                d[tgt] = row.get(src)
        else:
            raise ValueError(
                "Could not detect schema. Expected either:\n"
                "  - small schema: ideas/flow/coherence/vocab/grammar\n"
                "  - ASAP schema: content/sentence_fluency/organization/word_choice/conventions"
            )

        # (2) Compute overall from those 5 traits (rounded mean)
        overall = compute_overall_from_5(d)
        d["overall"] = overall if overall is not None else None

        # (3) Convert to string; Evaluator.read_results() will parse dict-like strings
        out.append(str(d))

    return pd.Series(out)