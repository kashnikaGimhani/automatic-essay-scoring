import pandas as pd
from typing import Dict, List, Tuple, Optional

TRAIT_COLUMNS = [
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

SCORE_RANGES: Dict[Tuple[int, str], Tuple[float, float]] = {

    # -------------------------
    # Prompt 1
    # -------------------------
    (1, "content"): (1, 6),
    (1, "organization"): (1, 6),
    (1, "word_choice"): (1, 6),
    (1, "sentence_fluency"): (1, 6),
    (1, "conventions"): (1, 6),

    # -------------------------
    # Prompt 2
    # -------------------------
    (2, "content"): (1, 6),
    (2, "organization"): (1, 6),
    (2, "word_choice"): (1, 6),
    (2, "sentence_fluency"): (1, 6),
    (2, "conventions"): (1, 6),

    # -------------------------
    # Prompt 3
    # -------------------------
    (3, "content"): (0, 3),
    (3, "prompt_adherence"): (0, 3),
    (3, "language"): (0, 3),
    (3, "narrativity"): (0, 3),

    # -------------------------
    # Prompt 4
    # -------------------------
    (4, "content"): (0, 3),
    (4, "prompt_adherence"): (0, 3),
    (4, "language"): (0, 3),
    (4, "narrativity"): (0, 3),

    # -------------------------
    # Prompt 5
    # -------------------------
    (5, "content"): (0, 4),
    (5, "prompt_adherence"): (0, 8),
    (5, "language"): (0, 8),
    (5, "narrativity"): (0, 8),

    # -------------------------
    # Prompt 6
    # -------------------------
    (6, "content"): (0, 8),
    (6, "prompt_adherence"): (0, 4),
    (6, "language"): (0, 4),
    (6, "narrativity"): (0, 4),

    # -------------------------
    # Prompt 7
    # -------------------------
    (7, "content"): (0, 6),
    (7, "organization"): (0, 6),
    (7, "style"): (0, 6),
    (7, "conventions"): (0, 6),

    # -------------------------
    # Prompt 8
    # -------------------------
    (8, "content"): (2, 12),
    (8, "organization"): (2, 12),
    (8, "voice"): (2, 12),
    (8, "word_choice"): (2, 12),
    (8, "sentence_fluency"): (2, 12),
    (8, "conventions"): (2, 12),
}

TRAIT_RUBRICS = {
    "content": "quality of ideas, relevance to the prompt, depth of explanation, and supporting details",
    "organization": "logical sequencing of ideas, paragraph structure, transitions, and clear beginning and ending",
    "word_choice": "appropriate vocabulary, precision of word usage, and variety of expressions",
    "sentence_fluency": "sentence flow, readability, smoothness, and natural phrasing",
    "conventions": "grammar accuracy, spelling, punctuation, capitalization, and sentence correctness",
    "prompt_adherence": "how well the essay addresses the prompt and stays on topic",
    "language": "overall language control, correctness, and clarity of expression",
    "narrativity": "storytelling quality, narrative development, sequencing of events, and engagement",
    "style": "effectiveness of expression, tone, and rhetorical quality",
    "voice": "writer presence, individuality, and distinctiveness of expression",
    "overall": "overall quality of the essay considering content, structure, language, and effectiveness",
}

def create_traitwise_records_df():
    
    df = pd.read_csv('../data/asap_train_with_all_traits.tsv', sep='\t')
    available_traits = [trait for trait in TRAIT_COLUMNS if trait in df.columns]
    records = []
    for _, row in df.iterrows():
        for trait in available_traits:
            if pd.notna(row[trait]):
                sample = {
                    "essai_id": row["essay_id"],
                    "essay_set": int(row["essay_set"]),
                    "essay_text": str(row["essay"]),
                    "trait": trait,
                    "raw_score": float(row[trait]),
                }
                records.append(sample)
    print(f"Trait wise samples: {len(records)}")
    return pd.DataFrame(records)

def get_score_ranges(essay_set: int, trait: str) -> Tuple[float, float]:
    return SCORE_RANGES[(essay_set, trait)]


def main():
    df_new = create_traitwise_records_df()
    # print("Shape of new DataFrame: ", df_new.shape)
    # print("Columns of new DataFrame: ", df_new.columns)
    # print("First 5 rows of new DataFrame: ", df_new.head())

if __name__ == "__main__":
    main()