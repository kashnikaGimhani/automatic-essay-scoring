import torch
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import argparse
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from utils.eval_vuw_data import Evaluator, UNIFIED_TRAITS_WITH_OVERALL, build_unified_target_strings_from_rows

RUBRIC_SCHEMA_PROMPT = """You are an expert writing assessor.

Score the essay independently for each trait using integer scores from 1 to 6.
Follow the rubric below strictly. Do not guess or infer information not present in the essay.

=== RUBRIC GUIDELINES ===

CONTENT
1 = ideas are unclear, missing, or irrelevant
2 = some relevant ideas but minimal development
3 = main ideas present but only partially developed
4 = relevant ideas with reasonable development
5 = well-developed and clearly supported ideas
6 = fully developed, insightful, and convincing ideas

ORGANIZATION
1 = no clear organization; ideas disconnected
2 = weak organization; relationships unclear
3 = some logical structure but inconsistent flow
4 = clear overall structure with minor lapses
5 = well-organized with effective connections
6 = highly cohesive and well-controlled organization

SENTENCE_FLUENCY
1 = very difficult to read; flow severely disrupted
2 = uneven flow; reading requires effort
3 = generally readable but inconsistent
4 = mostly smooth with minor disruptions
5 = smooth and natural flow
6 = effortless, highly fluent reading

WORD_CHOICE
1 = extremely limited vocabulary; frequent misuse
2 = limited range with frequent inaccuracies
3 = adequate vocabulary but repetitive or imprecise
4 = some variety and generally appropriate use
5 = varied and accurate vocabulary
6 = precise, flexible, and sophisticated vocabulary

CONVENTIONS
1 = frequent severe errors that obscure meaning
2 = frequent errors that disrupt understanding
3 = noticeable errors but meaning mostly clear
4 = minor errors that rarely interfere
5 = strong grammatical control with few errors
6 = high level of grammatical accuracy and control

=== OVERALL SCORE GUIDELINES ===

The overall score is a holistic judgment derived from the trait scores.
It is reached by averaging the 5 individual trait scores and then rounding.

=== INSTRUCTIONS ===
- Score each trait independently.
- Use only integer values from 1 to 6.
- Base scores solely on the essay content.
- Output your scores as a JSON dictionary with exactly these keys:

{
  "overall": <int>,
  "content": <int>,
  "organization": <int>,
  "sentence_fluency": <int>,
  "word_choice": <int>,
  "conventions": <int>
}

=== ESSAY ===
"""


def main():
    parser = argparse.ArgumentParser(description="Evaluate T5 for AES (single-schema dataset)")
    parser.add_argument("--model_path", type=str, required=True, help="model checkpoint directory")
    parser.add_argument("--data_file", type=str, required=True, help="TSV/CSV file with essays + trait columns")
    parser.add_argument("--output_path", type=str, default="output_eval", help="prefix for saving outputs")
    parser.add_argument("--max_input_len", type=int, default=512, help="max input length")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="max new tokens to generate")
    parser.add_argument("--test_batch", type=int, default=4, help="batch size for generation")
    parser.add_argument("--weight", type=str, default="quadratic", choices=["quadratic", "linear", ""], help="QWK weights")
    args = parser.parse_args()

    # --- Load raw table (pandas) to build target strings + keep alignment ---
    if args.data_file.endswith(".tsv"):
        df_raw = pd.read_csv(args.data_file, sep="\t")
    else:
        df_raw = pd.read_csv(args.data_file)

    # Build unified-schema target strings (dict-like string per row)
    target_strings = build_unified_target_strings_from_rows(df_raw) 

    # --- Load as HF dataset for tokenization + batching ---
    ds = load_dataset(                                          # loads the dataset into a huggingface dataset object, in each key value pair, key is the column name and value the list of values for that column
        "csv",
        data_files=args.data_file,
        delimiter="\t" if args.data_file.endswith(".tsv") else ",",
        split="train",
    )

    # Pick essay column
    essay_col = "essay" if "essay" in ds.column_names else ("content_text" if "content_text" in ds.column_names else None) #check is essay column is present in the dataset
    if essay_col is None:
        raise ValueError("Could not find essay text column. Expected 'essay' or 'content_text'.")

    tokenizer = T5Tokenizer.from_pretrained(args.model_path)            # load the tokenizer and the trained checkpoint model
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")   # choose the best available hardware and move the model to that hardware
    model = model.to(device)

    # Add targets into dataset as a column so preprocess can read it
    ds = ds.add_column("unified_target", target_strings.tolist())  # convert the target string series again to a list so that it can be added as a column to the huggingface dataset object then add to the huggingface dataset object

    def preprocess_function(examples):
        inputs, targets = [], []
        for i in range(len(examples[essay_col])):   # loop through each essay in essay column, check if essay exists, convert it to a string, then add the rubric prompt and essay text together as the input, append to the input array and add the target string to the targets list
            if examples[essay_col][i] is None:
                continue

            essay = str(examples[essay_col][i])
            prompt = RUBRIC_SCHEMA_PROMPT + "\n" + essay
            inputs.append(prompt)
            # inputs.append("score the essay: " + str(examples[essay_col][i]))
            targets.append(str(examples["unified_target"][i]))

        model_inputs = tokenizer(       #tokenize the model inputs list
            inputs,
            max_length=args.max_input_len,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(         #tokenize the target strings list
            text_target=targets,
            max_length=args.max_new_tokens,
            padding="max_length",
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"] # add the tokenized label input ids to the model inputs
        return model_inputs

    ds_tok = ds.map(preprocess_function, batched=True, remove_columns=ds.column_names)  # apply the preprocess function to the huggingface dataset object, process in batches of rows, return tokenized dataset rows

    # Torch dataloader
    ds_tok = ds_tok.with_format("torch", columns=["input_ids", "attention_mask", "labels"]) # convert the columns to tensors
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True) # create a collator function that will dynamically pad the inputs and labels to the maximum length in the batch, this is more efficient than padding to max length in the preprocess function
    loader = DataLoader(ds_tok, batch_size=args.test_batch, shuffle=False, collate_fn=collator) # create a iterator function over batches of the tokenized dataset, with the specified batch size, no shuffling, and using the collator function to pad the batches

    preds, tars = [], []
    model.eval()        # set the model for evaluation mode, this turns off dropout and other training-specific layers.
    with torch.no_grad(): # disable gradient calculation, this reduces memory usage and speeds up computation since we are only doing inference
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()} # move the batch tensors to the same device as the model (GPU or CPU)
            generated = model.generate(     # generate predictions from the model
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                do_sample=False,
                use_cache=True,
            )
            preds.extend(tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)) # convert the generated tokens back to text and add them to the list
            tars.extend(tokenizer.batch_decode(batch["labels"], skip_special_tokens=True, clean_up_tokenization_spaces=True)) #convert labels pack to text and add them to a new list

    # Save raw outputs
    out_df = pd.DataFrame({"prediction": preds, "target": tars})
    out_df.to_csv(args.output_path + "_preds.csv", index=False)

    # QWK evaluation (single run, no prompt loop)
    evaluator = Evaluator(UNIFIED_TRAITS_WITH_OVERALL)
    qwk = evaluator.evaluate_notnull(preds, tars, weight=args.weight)
    qwk_df = pd.DataFrame([qwk])
    qwk_df.to_csv(args.output_path + "_qwk.csv", index=False)

    # Optional: range sanity check based on target min/max (data-driven)
    pred_parsed = evaluator.read_results(preds)
    tar_parsed = evaluator.read_results(tars)

    ranges = {}
    for col in pred_parsed.columns:
        valid = tar_parsed[col][tar_parsed[col] != -1]
        if len(valid) == 0:
            continue
        ranges[col] = (float(valid.min()), float(valid.max()))

    over_rows = []
    for col, (mn, mx) in ranges.items():
        bad = pred_parsed[(pred_parsed[col] != -1) & ((pred_parsed[col] < mn) | (pred_parsed[col] > mx))].copy()
        if len(bad):
            bad["trait"] = col
            bad["min_allowed"] = mn
            bad["max_allowed"] = mx
            over_rows.append(bad[["trait", col, "min_allowed", "max_allowed"]])

    if over_rows:
        pd.concat(over_rows, axis=0).to_csv(args.output_path + "_out_of_range.csv", index=False)

    print("Saved:")
    print(" -", args.output_path + "_preds.csv")
    print(" -", args.output_path + "_qwk.csv")
    if over_rows:
        print(" -", args.output_path + "_out_of_range.csv")


if __name__ == "__main__":
    main()
