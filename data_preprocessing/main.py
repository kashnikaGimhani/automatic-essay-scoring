"""
main.py

Runs:
1) Data preprocessing / trait combination
2) 5-fold cross-validation split creation
"""

from data_preprocessing.data_preprocessing import (
    build_combined_dataset,
    save_tsv,
    PromptTraitFile
)

from create_folds import create_5fold_cv


# -----------------------------
# CONFIG
# -----------------------------

ASAP_TSV_PATH = "data/training_set_rel3.tsv"
COMBINED_OUTPUT_PATH = "data/asap_train_with_all_traits.tsv"
FOLDS_OUTPUT_DIR = "data/5fold_cv"


PROMPT_TRAIT_SPECS = [
        PromptTraitFile(
            prompt_id=1,
            path="data/ASAP++/Prompt-1.csv",
            file_type="csv",
            colmap={
                "EssayID": "essay_id",
                "Content": "content",
                "Organization": "organization",
                "Word Choice": "word_choice",
                "Sentence Fluency": "sentence_fluency",
                "Conventions": "conventions",
            },
        ),
        PromptTraitFile(
            prompt_id=2,
            path="data/ASAP++/Prompt-2.csv",
            file_type="csv",
            colmap={
                "Essay ID": "essay_id",
                "Content": "content",
                "Organization": "organization",
                "Word Choice": "word_choice",
                "Sentence Fluency": "sentence_fluency",
                "Conventions": "conventions",
            },
        ),
        PromptTraitFile(
            prompt_id=3,
            path="data/ASAP++/Prompt-3.csv",
            file_type="csv",
            colmap={
                "Essay ID": "essay_id",
                "Content": "content",
                "Prompt Adherence": "prompt_adherence",
                "Language": "language",
                "Narrativity": "narrativity"
            },
        ),
        PromptTraitFile(
            prompt_id=4,
            path="data/ASAP++/Prompt-4.csv",
            file_type="csv",
            colmap={
                "Essay ID": "essay_id",
                "Content": "content",
                "Prompt Adherence": "prompt_adherence",
                "Language": "language",
                "Narrativity": "narrativity"
            },
        ),
        PromptTraitFile(
            prompt_id=5,
            path="data/ASAP++/Prompt-5.csv",
            file_type="csv",
            colmap={
                "Essay ID": "essay_id",
                "Content": "content",
                "Prompt Adherence": "prompt_adherence",
                "Language": "language",
                "Narrativity": "narrativity"
            },
        ),
        PromptTraitFile(
            prompt_id=6,
            path="data/ASAP++/Prompt-6.csv",
            file_type="csv",
            colmap={
                "Essay ID": "essay_id",
                "Content": "content",
                "Prompt Adherence": "prompt_adherence",
                "Language": "language",
                "Narrativity": "narrativity"
            },
        )
    ]


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def main():
    print("▶ Starting data preprocessing...")

    combined_df = build_combined_dataset(
        asap_path=ASAP_TSV_PATH,
        prompt_trait_specs=PROMPT_TRAIT_SPECS
    )

    save_tsv(combined_df, COMBINED_OUTPUT_PATH)
    print(f"✔ Combined dataset saved to {COMBINED_OUTPUT_PATH}")

    print("\n▶ Creating 5-fold cross-validation splits...")

    create_5fold_cv(
        df=combined_df,
        output_dir=FOLDS_OUTPUT_DIR,
        seed=42
    )

    print("\n✅ Pipeline completed successfully")


if __name__ == "__main__":
    main()
