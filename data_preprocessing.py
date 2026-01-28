"""
This script combine ASAP dataset with the promptwise trait scores from the ASAP++ project and 
create a unified TSV file which consist of essay text, overall scores, and trait scores.

NOTE: The trait score for pompt 7 and 8 are derived from the rater scores in the original ASAP dataset 
and trait scores for prompt 1-6 are taken from the provided trait score files from ASAP++.

Data files used:
- training_set_rel3.tsv (full training set with essay text and scores; includes rater traits for prompts 7–8 from ASAP)
- trait-score files for prompts 1–6 (one file per prompt) from ASAP++ project

This script:
1) Loads training_set_rel3.tsv
2) Adds unified trait columns
3) Merges prompt 1–6 trait files by essay_id
4) Resolves prompt 7 traits as rater1+rater2
5) Resolves prompt 8 traits as (rater1+rater2) OR (rater3*2 if available)
6) Writes a new TSV
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------

UNIFIED_TRAITS: List[str] = [
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


@dataclass
class PromptTraitFile:
    prompt_id: int
    path: str
    # mapping from file columns -> unified trait names
    # must include an essay id column mapping to "essay_id"
    colmap: Dict[str, str]
    file_type: str = "csv"  # "csv" or "tsv"
    encoding: Optional[str] = None


def load_asap_training_tsv(asap_path: str) -> pd.DataFrame:
    """
    Loads the ASAP training_set_rel3.tsv (latin-1 encoding is common for this file).
    Expected columns include: essay_id, essay_set, essay, domain1_score, domain2_score,
    and for prompts 7–8: raterX_traitY columns.
    """
    df = pd.read_csv(asap_path, sep="\t", encoding="latin-1")
    return df


def load_trait_file(spec: PromptTraitFile) -> pd.DataFrame:
    if spec.file_type.lower() == "csv":
        df = pd.read_csv(spec.path, encoding=spec.encoding or "utf-8")
    elif spec.file_type.lower() == "tsv":
        df = pd.read_csv(spec.path, sep="\t", encoding=spec.encoding or "utf-8")
    else:
        raise ValueError(f"Unsupported file_type: {spec.file_type}")

    df = df.rename(columns=spec.colmap) # Rename columns to unified names

    if "essay_id" not in df.columns: # Ensure essay_id exists
        raise ValueError(
            f"Trait file for prompt {spec.prompt_id} must map an id column to 'essay_id'"
        )

    df["essay_id"] = df["essay_id"].astype(int) # Normalize essay_id type
    df = df.drop_duplicates(subset=["essay_id"]) # Drop duplicate essay_ids if any

    keep = ["essay_id"] + [c for c in UNIFIED_TRAITS if c in df.columns]
    df = df[keep].copy() # Create a new df with only essay_id and trait columns

    return df


def init_base_output(asap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the base output dataframe with essay text, prompt id, and overall scores.
    Adds empty unified trait columns.
    """
    out = asap_df[["essay_id", "essay_set", "essay", "domain1_score", "domain2_score"]].copy()
    out = out.rename(columns={"domain1_score": "score_domain1", "domain2_score": "score_domain2"})

    # Ensure types
    out["essay_id"] = out["essay_id"].astype(int)
    out["essay_set"] = out["essay_set"].astype(int)

    # Add empty unified trait columns
    for trait in UNIFIED_TRAITS:
        out[trait] = np.nan

    return out


def merge_prompt_traits(
    base_df: pd.DataFrame,
    trait_df: pd.DataFrame,
    prompt_id: int,
) -> pd.DataFrame:
    """
    Merges a prompt's trait file onto the base by essay_id and fills traits ONLY for that prompt.
    """
    # Merge and keep original columns; trait cols come in with suffix "_src"
    merged = base_df.merge(trait_df, on="essay_id", how="left", suffixes=("", "_src"))

    mask = merged["essay_set"] == prompt_id

    # For each trait that exists in trait_df, fill base unified trait
    for trait in UNIFIED_TRAITS:
        src_col = f"{trait}_src"
        if src_col in merged.columns:
            merged.loc[mask, trait] = merged.loc[mask, src_col]

    # Drop the merged-in src columns
    drop_cols = [c for c in merged.columns if c.endswith("_src")]
    merged = merged.drop(columns=drop_cols)

    return merged


# -----------------------------
# Prompt 7/8 resolving
# -----------------------------

import numpy as np
import pandas as pd

def resolve_prompt7_traits(asap_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prompt 7:
      trait1..4 = content, organization, style, conventions
      resolved = rater1 + rater2
    Uses essay_id join to avoid index mismatch.
    """
    p7 = asap_df[asap_df["essay_set"] == 7].copy()

    r1 = p7[["rater1_trait1","rater1_trait2","rater1_trait3","rater1_trait4"]].to_numpy(dtype=float)
    r2 = p7[["rater2_trait1","rater2_trait2","rater2_trait3","rater2_trait4"]].to_numpy(dtype=float)
    resolved = r1 + r2

    resolved_df = pd.DataFrame({
        "essay_id": p7["essay_id"].astype(int).to_numpy(),
        "content": resolved[:, 0],
        "organization": resolved[:, 1],
        "style": resolved[:, 2],
        "conventions": resolved[:, 3],
    })

    # merge back; overwrite prompt-7 rows only
    out = base_df.merge(resolved_df, on="essay_id", how="left", suffixes=("", "_p7"))
    mask7 = out["essay_set"] == 7
    for col in ["content", "organization", "style", "conventions"]:
        out.loc[mask7, col] = out.loc[mask7, f"{col}_p7"]
    out.drop(columns=[c for c in out.columns if c.endswith("_p7")], inplace=True)

    return out


def resolve_prompt8_traits(asap_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prompt 8:
      trait1..6 = content, organization, voice, word_choice, sentence_fluency, conventions
      resolved = rater1 + rater2
      if rater3_trait* exists -> use rater3*2 (same scale as sum of two raters)
    Uses essay_id join to avoid index mismatch.
    """
    p8 = asap_df[asap_df["essay_set"] == 8].copy()

    r1 = p8[[f"rater1_trait{i}" for i in range(1, 7)]].to_numpy(dtype=float)
    r2 = p8[[f"rater2_trait{i}" for i in range(1, 7)]].to_numpy(dtype=float)
    r3 = p8[[f"rater3_trait{i}" for i in range(1, 7)]].to_numpy(dtype=float)

    resolved = r1 + r2
    has_r3 = ~np.isnan(r3)
    resolved[has_r3] = r3[has_r3] * 2

    resolved_df = pd.DataFrame({
        "essay_id": p8["essay_id"].astype(int).to_numpy(),
        "content": resolved[:, 0],
        "organization": resolved[:, 1],
        "voice": resolved[:, 2],
        "word_choice": resolved[:, 3],
        "sentence_fluency": resolved[:, 4],
        "conventions": resolved[:, 5],
    })

    out = base_df.merge(resolved_df, on="essay_id", how="left", suffixes=("", "_p8"))
    mask8 = out["essay_set"] == 8
    for col in ["content", "organization", "voice", "word_choice", "sentence_fluency", "conventions"]:
        out.loc[mask8, col] = out.loc[mask8, f"{col}_p8"]
    out.drop(columns=[c for c in out.columns if c.endswith("_p8")], inplace=True)

    return out


# -----------------------------
# Pipeline
# -----------------------------

def build_combined_dataset(
    asap_path: str,
    prompt_trait_specs: List[PromptTraitFile],
) -> pd.DataFrame:
    """
    Full pipeline that returns a combined DataFrame.
    """
    asap_df = load_asap_training_tsv(asap_path)
    base_df = init_base_output(asap_df)

    # Merge prompt 1–6 traits from provided files
    for spec in prompt_trait_specs:
        print(f"Processing trait file for prompt {spec.prompt_id} ...")

        trait_df = load_trait_file(spec)
        base_df = merge_prompt_traits(base_df, trait_df, spec.prompt_id)

        print(f"✔ Finished merging traits for prompt {spec.prompt_id}")


    # Resolve prompt 7–8 traits from rater columns in ASAP TSV
    print("Resolving traits for prompt 7 ...")
    base_df = resolve_prompt7_traits(asap_df, base_df)
    print("✔ Finished resolving traits for prompt 7")

    print("Resolving traits for prompt 8 ...")
    base_df = resolve_prompt8_traits(asap_df, base_df)
    print("✔ Finished resolving traits for prompt 8")

    return base_df


def save_tsv(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    df.to_csv(out_path, sep="\t", index=False)


# -----------------------------
# Example usage
# -----------------------------

# if __name__ == "__main__":
#     ASAP_TSV = "data/training_set_rel3.tsv"

#     # Example: Prompt 1 trait file like the one you uploaded:
#     # columns: EssayID, Content, Organization, Word Choice, Sentence Fluency, Conventions
#     prompt_trait_specs = [
#         PromptTraitFile(
#             prompt_id=1,
#             path="data/ASAP++/Prompt-1.csv",
#             file_type="csv",
#             colmap={
#                 "EssayID": "essay_id",
#                 "Content": "content",
#                 "Organization": "organization",
#                 "Word Choice": "word_choice",
#                 "Sentence Fluency": "sentence_fluency",
#                 "Conventions": "conventions",
#             },
#         ),
#         PromptTraitFile(
#             prompt_id=2,
#             path="data/ASAP++/Prompt-2.csv",
#             file_type="csv",
#             colmap={
#                 "Essay ID": "essay_id",
#                 "Content": "content",
#                 "Organization": "organization",
#                 "Word Choice": "word_choice",
#                 "Sentence Fluency": "sentence_fluency",
#                 "Conventions": "conventions",
#             },
#         ),
#         PromptTraitFile(
#             prompt_id=3,
#             path="data/ASAP++/Prompt-3.csv",
#             file_type="csv",
#             colmap={
#                 "Essay ID": "essay_id",
#                 "Content": "content",
#                 "Prompt Adherence": "prompt_adherence",
#                 "Language": "language",
#                 "Narrativity": "narrativity"
#             },
#         ),
#         PromptTraitFile(
#             prompt_id=4,
#             path="data/ASAP++/Prompt-4.csv",
#             file_type="csv",
#             colmap={
#                 "Essay ID": "essay_id",
#                 "Content": "content",
#                 "Prompt Adherence": "prompt_adherence",
#                 "Language": "language",
#                 "Narrativity": "narrativity"
#             },
#         ),
#         PromptTraitFile(
#             prompt_id=5,
#             path="data/ASAP++/Prompt-5.csv",
#             file_type="csv",
#             colmap={
#                 "Essay ID": "essay_id",
#                 "Content": "content",
#                 "Prompt Adherence": "prompt_adherence",
#                 "Language": "language",
#                 "Narrativity": "narrativity"
#             },
#         ),
#         PromptTraitFile(
#             prompt_id=6,
#             path="data/ASAP++/Prompt-6.csv",
#             file_type="csv",
#             colmap={
#                 "Essay ID": "essay_id",
#                 "Content": "content",
#                 "Prompt Adherence": "prompt_adherence",
#                 "Language": "language",
#                 "Narrativity": "narrativity"
#             },
#         )
#     ]

#     combined = build_combined_dataset(ASAP_TSV, prompt_trait_specs)
#     save_tsv(combined, "data/asap_train_with_all_traits.tsv")
#     print("Saved:", "data/asap_train_with_all_traits.tsv")
