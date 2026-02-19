import pandas as pd
import numpy as np
import re
import logging
from sklearn.metrics import confusion_matrix, cohen_kappa_score, mean_squared_error
from six import string_types
import json
import ast


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
        assert len(y_true) == len(y_pred) # check the length of predictions match the length of the target values

        try:
            y_true = [int(np.round(float(y))) for y in y_true]
            y_pred = [int(np.round(float(y))) for y in y_pred] # convert predicted and target trait values to int or else throw and error if can not convert
        except ValueError as e:
            logger.error(
                "For kappa, labels should be ints or strings convertible to ints (e.g., '4.0', '3')."
            )
            raise e

        min_rating = min(min(y_true), min(y_pred)) # get the min and max values from predicted and target trait scores
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
        results = pd.DataFrame(columns=self.traits) # create an empty dataframe with ASAP traits as columns

        for i in range(len(pred)):
            text = str(pred[i]).strip() # loop though each prediction and convert prediction value to a string

            # Normalize bare nan tokens
            safe = re.sub(r"\bnan\b", "None", text, flags=re.IGNORECASE) # replace any occurrence of 'nan' with 'None' to handle missing values

            # If looks like key:value pairs but missing braces, wrap it
            if ":" in safe and not safe.lstrip().startswith("{"): #add braces to the string if not found
                safe = "{" + safe + "}"

            obj = None
            try:                # convert the prediction string to a dict/oblect
                obj = json.loads(safe)
            except Exception:
                try:
                    obj = ast.literal_eval(safe)
                except Exception:
                    obj = None

            trait_scores = []
            if isinstance(obj, dict): # if obj is a dict
                for trait in self.traits:
                    v = obj.get(trait)

                    # allow underscore/space variants
                    if v is None:
                        alt = trait.replace("_", " ") if "_" in trait else trait.replace(" ", "_") # get the value for the ASAP trait, check for alternated trait names with underscore/space if prior found none
                        v = obj.get(alt, None)

                    if v is None:
                        trait_scores.append(-1)
                    else:
                        try:
                            trait_scores.append(float(v)) #if no value found append -1 for the trait score list elase covert the score to float and append to the trait score list
                        except Exception:
                            trait_scores.append(-1)
            else:               # if couldnt convert to string use regex to find trait score combinations
                # regex fallback
                for trait in self.traits:
                    m = re.search(
                        rf"{re.escape(trait)}\s*['\"]?\s*:\s*(nan|-?\d+(?:\.\d+)?)",
                        text,
                        flags=re.IGNORECASE,
                    )
                    if (m is None) or (m.group(1).lower() == "nan"): # if no regex match found or match is 'nan' append -1 else convert to float and append
                        trait_scores.append(-1)
                    else:
                        trait_scores.append(float(m.group(1)))

            assert len(self.traits) == len(trait_scores), "trait length error!" # check if the number of traits matches the number of trait scores parsed, if not raise an error
            results.loc[i] = trait_scores       # add ith row to the results df and assign the trait scores to the corresponding columns

        return results.astype("float").fillna(-1)   # convert all values in result df to float and fill missing values with -1

    # ---------- Evaluation ----------
    def evaluate_notnull(self, pred, target, weight="quadratic"):
        """
        Trait-wise QWK, skipping rows where target is missing (-1).
        """
        results = self.read_results(pred)
        targets = self.read_results(target)

        qwk_results = {}
        for key in results.columns:  # loop through each trail column in the results df
            trait_tgt = targets[key][targets[key] != -1]  # get the target score values for the trait skipping rows where target is missing (-1)
            if len(trait_tgt) == 0:
                qwk_results[key] = np.nan # if no target trait values found for the trait set value to NaN in the qwk results dict
                continue
            trait_pred = results[key][trait_tgt.index] # else get the trait scores from results for the same rows as the target traits where target traits not missing
            qwk_results[key] = self.calc_kappa(trait_pred, trait_tgt, weight=weight)

        return qwk_results


# ---------- Trait mapping helpers ----------
# New dataset schema -> unified ASAP schema
IDEAS_SCHEMA_TO_UNIFIED = {
    "ideas": "content",
    "coherence": "organization",
    "flow": "sentence_fluency",
    "vocab": "word_choice",
    "grammar": "conventions",
    "overall_score": "overall",
}

# Unified trait list + overall used by evaluator
UNIFIED_TRAITS_WITH_OVERALL = [
    "overall",
    "content",
    "organization",
    "word_choice",
    "sentence_fluency",
    "conventions",
    "prompt_adherence",
    "language",
    "narrativity",
    "style",
    "voice",
]


def build_unified_target_strings_from_rows(df: pd.DataFrame) -> pd.Series:
    """
    Build target strings in unified schema for each row.

    If df already has unified columns, uses them.
    Else if df has ideas/flow/coherence/vocab/grammar/overall_score, maps them.

    Missing unified traits are set to None (=> Evaluator will interpret as -1).

    This creates the target string value series from the original dataset with the trait names from ASAP.
    """
    has_unified = (
        all(c in df.columns for c in ["content", "organization", "word_choice", "sentence_fluency", "conventions"])
        and ("overall" in df.columns or "overall_score" in df.columns)
    )                                                                                                                   # check whether dataset columns has ASAP fdataset column names
    has_ideas_schema = all(c in df.columns for c in ["ideas", "flow", "coherence", "vocab", "grammar", "overall_score"]) # check is the daatset columns has the default column names of the dataset

    out = []
    for _, row in df.iterrows(): #iterate through all the rows in the dataframe, return row values and index of the row
        d = {k: None for k in UNIFIED_TRAITS_WITH_OVERALL} # create a dict/object with key value pairs with all ASAP traits and set values to None

        if has_unified:
            d["overall"] = row.get("overall", row.get("overall_score")) #row.get(key, default) this returns the value for key if exists else return the defualt value defined
            for k in d.keys():
                if k == "overall":
                    continue
                if k in df.columns:
                    d[k] = row.get(k) # sets values for the ASAP trait names, exclude overall column

        elif has_ideas_schema:
            for src, tgt in IDEAS_SCHEMA_TO_UNIFIED.items(): # iterate through the dict and get the key value pairs, here get the mapping asap trait for each original data trait
                d[tgt] = row.get(src) #get values for trait from original data and assign to the asap trait in the dict d

        else:
            raise ValueError(
                "Could not detect schema. Expected either unified traits "
                "or ideas/flow/coherence/vocab/grammar + overall_score."
            )

        out.append(str(d)) #covert the dict to a string and append to the list/array creating a array/list of strings

    return pd.Series(out) # converted to a pandas series so that it can be used as a column in dataframes
