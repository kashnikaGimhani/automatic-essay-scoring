import torch  # PyTorch: tensors, device placement (CPU/GPU), and model execution
import pandas as pd  # Pandas: tabular data handling + saving CSV outputs
import numpy as np  # NumPy: numerical utilities (not strictly needed here, but often used)
from transformers import T5Tokenizer, T5ForConditionalGeneration  # T5 tokenizer + seq2seq model (generate score text)
from datasets import load_dataset  # HuggingFace Datasets: load TSV/CSV into Dataset objects and map preprocessing
from utils.eval_qwk import Evaluator  # Your custom evaluator for QWK and trait parsing
import argparse  # Read command-line arguments like --model_path, --data_path, etc.

def main():  # Main function to run evaluation end-to-end
    parser = argparse.ArgumentParser(description="Evaluate T5 for AES")  # Create CLI parser with a description
    parser.add_argument('--max_tgt_len', type=int, default=162, help='max target length')  # Max length for generated/label tokens
    parser.add_argument('--test_batch', type=int, default=2, help='test_batch')  # Intended evaluation batch size (NOTE: currently not used)
    parser.add_argument('--model_path', type=str, default='results/checkpoint-5190/', help='model checkpoint directory name')  # Path to saved checkpoint folder
    parser.add_argument('--data_path', type=str, default='data', help='data directory name')  # Folder containing fold files (train/dev/test TSVs)
    parser.add_argument('--output_path', type=str, default='output', help='output directory name')  # Prefix/path for saving prediction CSV

    args = parser.parse_args()  # Parse CLI args into "args" object

    test_set_list = load_dataset(  # Load test data using HuggingFace Datasets "csv" loader
        "csv",  # Use CSV loader (it can also read TSV if delimiter is provided)
        data_files=args.data_path + "/test.tsv",  # The test file path (TSV file)
        delimiter="\t",  # Tell parser to split columns by TAB (TSV)
        split=[f'train[{k}%:{k+20}%]' for k in range(0, 100, 20)],  # Split test into 5 chunks of 20% each to reduce memory spikes
    )

    tokenizer = T5Tokenizer.from_pretrained(args.model_path)  # Load tokenizer from the checkpoint (ensures same vocab/settings)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)  # Load trained T5 model weights from checkpoint
    # tokenizer.pad_token = tokenizer.eos_token  # (Optional) Set pad token (not needed for T5 usually)

    def preprocess_function(examples):  # Converts raw rows (essay, target, prompt) into token IDs for model input/labels
        inputs, targets, prompts = [], [], []  # Lists to collect clean examples for this batch
        for i in range(len(examples['essay'])):  # Loop through each row in the current batch
            if examples['essay'][i] and examples['target'][i]:  # Keep only rows with non-missing essay and score
                inputs.append(examples['essay'][i])  # Add the essay text as model input content
                targets.append(str(examples['target'][i]))  # Add the score as STRING (T5 outputs text; score becomes text label)
                prompts.append(examples['essay_set'][i])  # Add the prompt id (essay_set) to condition the instruction

        inputs = [  # Build T5 "instruction-style" inputs for each essay
            "score the essay of the prompt " + str(prompts[i]) + ": " + inp  # e.g., "score the essay of the prompt 3: <essay>"
            for (i, inp) in enumerate(inputs)  # Loop with index to match prompts list
        ]
        model_inputs = tokenizer(  # Tokenize the input strings into token IDs
            inputs,  # List of instruction+essay strings
            max_length=512,  # Truncate/pad essay inputs to 512 tokens
            padding='max_length',  # Pad every example to exactly 512 tokens (simple but slower than dynamic padding)
            truncation=True  # Truncate if longer than max_length
        )

        labels = tokenizer(  # Tokenize target score strings into label token IDs
            text_target=targets,  # This tells tokenizer these are target texts (labels) for seq2seq
            max_length=args.max_tgt_len,  # Max token length for label sequence (score text is short; this is usually overkill)
            padding='max_length',  # Pad labels to max_tgt_len (again simple but may be inefficient)
            truncation=True  # Truncate labels if too long (unlikely for numeric scores)
        )
        model_inputs["labels"] = labels["input_ids"]  # Attach label token IDs so Trainer/your eval can decode them later
        return model_inputs  # Return dict with input_ids, attention_mask, labels

    test_set_list = [  # Apply preprocessing to each of the 5 test chunks
        test_set.map(preprocess_function, batched=True)  # "batched=True" feeds a batch dict (lists) into preprocess_function
        for test_set in test_set_list  # Loop over each split chunk
    ]

    #model = model.cuda()  # Move model to NVIDIA GPU (CUDA). (NOTE: this will crash on Mac unless you have CUDA GPU.)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)


    def test(tokenizer, device, test_set):  # Runs generation on one preprocessed Dataset chunk and returns decoded preds/targets
        preds = model.generate(  # Generate output token IDs from the model (seq2seq decoding)
            input_ids=torch.tensor(test_set["input_ids"]).to(device),  # Convert stored input_ids list -> tensor and move to device
            attention_mask=torch.tensor(test_set["attention_mask"]).to(device),  # Convert attention_mask list -> tensor and move to device
            max_length=args.max_tgt_len  # Maximum length of generated sequence (labels are padded to this too)
        )

        predictions = preds  # Store generated sequences (token IDs) as predictions
        label_ids = test_set['labels']  # Get label token IDs (padded to max_tgt_len)

        assert len(predictions) == len(test_set)  # Sanity check: number of preds equals number of examples
        assert len(predictions) == len(label_ids)  # Sanity check: preds length equals labels length

        predictions = torch.tensor(predictions).to(device, dtype=torch.long)  # Ensure predictions are long tensors on device
        label_ids = torch.tensor(label_ids).to(device, dtype=torch.long)  # Ensure labels are long tensors on device

        results = [  # Decode each generated token sequence into text (e.g., "4")
            tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)  # Remove <pad>, </s>, etc.
            for g in predictions  # Loop over each predicted sequence
        ]
        actuals = [  # Decode each label token sequence back into text (e.g., "4")
            tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for t in label_ids  # Loop over each label sequence
        ]
        return results, actuals  # Return decoded predictions and decoded ground-truth scores (as strings)

    predictions = []  # List to store decoded predictions across all chunks
    targets = []  # List to store decoded targets across all chunks
    for list in test_set_list:  # Loop over each preprocessed chunk (20% splits)
        pred, target = test(tokenizer, device, list)  # Run generation/eval on this chunk (device string 'cuda')
        predictions.extend(pred)  # Append this chunk's predictions to global list
        targets.extend(target)  # Append this chunk's targets to global list

    outputs = pd.DataFrame({'predictions': predictions, 'targets': targets})  # Build a dataframe of text preds and text targets
    outputs.to_csv(args.output_path + '.csv')  # Save prediction vs target pairs to CSV (e.g., output.csv)

    qwk_results = []  # Will store QWK results (per prompt) returned by Evaluator
    over_range = []  # Will store info about predictions that fall outside valid score ranges per trait

    trait_1 = ['overall', 'content', 'word choice', 'organization', 'sentence fluency', 'conventions']  # Trait names for prompts using rubric set 1
    trait_2 = ['overall', 'content', 'prompt adherence', 'language', 'narrativity']  # Trait names for rubric set 2
    trait_3 = ['overall', 'content', 'organization', 'conventions', 'style']  # Trait names for rubric set 3
    trait_4 = ['overall', 'content', 'word choice', 'organization', 'sentence fluency', 'conventions', 'voice']  # Trait names for rubric set 4

    trait_sets = [trait_1, trait_1, trait_2, trait_2, trait_2, trait_2, trait_3, trait_4]  # Map prompt 1..8 -> which trait list to use
    test_target = pd.read_csv(args.data_path + "/test.tsv", delimiter="\t")  # Load a CSV that contains prompt_id for each test example (NOTE: path/filename must match your data)
    total_data = pd.DataFrame({  # Create a table aligning predictions/targets with prompt IDs
        'pred': predictions,  # Model predicted score strings
        'target': targets,  # Ground-truth score strings
        'prompt': test_target['essay_set']  # Prompt IDs from the CSV (must align row-by-row with predictions order)
    })

    for i in range(8):  # Loop through prompt indices 0..7 (representing prompts 1..8)
        print('Prompt ', i + 1, ' results!')  # Print which prompt is being evaluated

        traits = trait_sets[i]  # Get trait list that corresponds to this prompt
        QWK_EVAL = Evaluator(traits)  # Create an evaluator configured for these traits
        min_max = QWK_EVAL.get_min_max_scores()  # Get allowed min/max score ranges for each trait per prompt

        preds = total_data[total_data['prompt'] == (i + 1)]['pred'].reset_index()['pred']  # Get preds for this prompt only
        tars = total_data[total_data['prompt'] == (i + 1)]['target'].reset_index()['target']  # Get targets for this prompt only

        result = QWK_EVAL.evaluate_notnull(preds, tars)  # Compute QWK (and possibly trait-wise QWK) ignoring nulls
        pred_df = QWK_EVAL.read_results(preds)  # Parse predicted strings into structured columns for each trait (depends on your Evaluator format)

        over = {}  # Dict to store out-of-range predictions for each trait
        for t in traits:  # Check each trait’s predicted values for validity
            df = pred_df[  # Select rows where predicted trait value is outside valid min/max range for that prompt
                (pred_df[t] > min_max[i + 1][t][1]) | (pred_df[t] < min_max[i + 1][t][0])
            ]
            over[t] = df  # Store out-of-range rows for this trait

        qwk_results.append(result)  # Save this prompt's QWK results
        over_range.append(over)  # Save out-of-range info for this prompt
        print(result)  # Print QWK results for this prompt

    pd.DataFrame(over_range).to_csv(args.model_path + '/over_range.csv')  # Save out-of-range predictions info to CSV in model folder
    pd.DataFrame(qwk_results).to_csv(args.model_path + '/qwk_results.csv')  # Save QWK results (per prompt) to CSV in model folder

if __name__ == '__main__':  # Standard Python entry point: run main() only when executed as a script
    main()  # Execute evaluation
