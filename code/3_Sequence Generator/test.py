import os
import pandas as pd
import torch.cuda
import wandb
from datetime import datetime
from torch.optim import AdamW
from config import set_args
from torch.nn import CrossEntropyLoss
from tokenizer import T5PegasusTokenizer
from torch.utils.data import DataLoader
from data_helper import MT5Dataset, load_data, collate_fn
from transformers import get_linear_schedule_with_warmup
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from bert_score import score
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import csv
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import jieba


def set_device():
    """Set the computing device (GPU if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_data(data, train_size=0.8, val_size=0.1, random_state=42):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        data (DataFrame): The dataset to be split.
        train_size (float): Proportion of training data.
        val_size (float): Proportion of validation data.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: DataFrames for training, validation, and test sets.
    """
    test_size = 1 - train_size - val_size
    train_data, temp_data = train_test_split(data, test_size=(val_size + test_size), random_state=random_state)
    val_data, test_data = train_test_split(temp_data, test_size=(test_size / (val_size + test_size)),
                                           random_state=random_state)
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    return train_data, val_data, test_data


def beamsearch_sample_decode(input_ids, model, tokenizer, device):
    """
    Generate a summary using beam search and sampling.

    Args:
        input_ids (torch.Tensor): Token IDs of the input.
        model (MT5ForConditionalGeneration): Pretrained MT5 model.
        tokenizer (T5PegasusTokenizer): Tokenizer.
        device (torch.device): Computing device.

    Returns:
        str: Generated summary.
    """
    max_length = 200
    repetition_penalty = 1.1
    num_beams = 3
    temperature = 1.0
    topk = 3
    topp = 0.95

    input_ids = input_ids.to(device)

    output_beamsearch_random = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        num_beams=num_beams,
        temperature=temperature,
        top_k=topk,
        top_p=topp,
        repetition_penalty=repetition_penalty,
        decoder_start_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    res = tokenizer.decode(output_beamsearch_random[0], skip_special_tokens=True)
    res = res.replace(' ', '')  # Remove spaces (for Chinese text processing)
    return res


def generate_predictions(model, dataloader, tokenizer, args):
    """
    Generate summaries for given input texts.

    Args:
        model (MT5ForConditionalGeneration): Pretrained MT5 model.
        dataloader (DataLoader): DataLoader for test dataset.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer.
        args (Namespace): Configuration arguments.

    Returns:
        tuple: Lists of input texts, generated summaries, and reference summaries.
    """
    device = set_device()
    model.to(device)
    generated_summaries = []
    reference_summaries = []
    input_texts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Summaries"):
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = batch
            input_ids_cpu = input_ids.cpu()

            # Decode input texts
            decoded_inputs = tokenizer.batch_decode(input_ids_cpu, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)
            decoded_inputs = [input_text.replace(' ', '') for input_text in decoded_inputs]
            input_texts.extend(decoded_inputs)

            # Generate summaries
            for i in range(input_ids.size(0)):
                single_input_ids = input_ids[i].unsqueeze(0)
                summary = beamsearch_sample_decode(single_input_ids, model, tokenizer, device)
                generated_summaries.append(summary)

            # Decode reference summaries
            decoded_references = tokenizer.batch_decode(decoder_input_ids.cpu(), skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)
            decoded_references = [ref.replace(' ', '') for ref in decoded_references]
            reference_summaries.extend(decoded_references)

    return input_texts, generated_summaries, reference_summaries


def compute_bert_score(predictions, references, lang="zh"):
    """Compute BERTScore for text evaluation."""
    P, R, F1 = score(predictions, references, lang=lang, verbose=True)
    return P, R, F1


def compute_bleurt_score(predictions, references, checkpoint="bleurt-base-128"):
    """
    Compute BLEURT Score.

    Args:
        predictions (list of str): List of generated summaries.
        references (list of str): List of reference summaries.
        checkpoint (str): Path to the BLEURT model checkpoint.

    Returns:
        float: Average BLEURT score.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()
    inputs = tokenizer(
        references,
        predictions,
        truncation=True,
        padding=True,
        return_tensors='pt',
        max_length=128
    )

    with torch.no_grad():
        scores = model(**inputs)[0].squeeze()
    avg_score = scores.mean().item()
    return scores


def tokenize(sentence):
    """Tokenize a sentence using Jieba."""
    return list(jieba.cut(sentence))


def compute_bleu(predictions, references):
    """Compute BLEU score."""
    bleu_scores = []
    smooth = SmoothingFunction().method1  # Smoothing function
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = [tokenize(ref)]
        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)
        bleu_scores.append(score)
    return bleu_scores


def compute_rouge(predictions, references):
    """Compute ROUGE scores."""
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge_scores.append({
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        })
    return rouge_scores


def main():
    """Main function to run the summarization model evaluation."""
    args = set_args()
    all_df = load_data(args.data_path)

    checkpoint = "output/best_model_mt5_4"
    tokenizer = T5PegasusTokenizer.from_pretrained(checkpoint)
    model = MT5ForConditionalGeneration.from_pretrained(checkpoint)

    train_df, val_df, test_df = split_data(all_df, train_size=0.8, val_size=0.19, random_state=42)
    print(f"Training size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

    test_dataset = MT5Dataset(test_df, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    input_texts, generated_summaries, reference_summaries = generate_predictions(model, test_dataloader, tokenizer,
                                                                                 args)

    df = pd.DataFrame(zip(input_texts, generated_summaries, reference_summaries),
                      columns=["input", "prediction", "reference"])

    df['bleurt_score'] = compute_bleurt_score(df['prediction'].tolist(), df['reference'].tolist())
    df['bleu_score'] = compute_bleu(df['prediction'].tolist(), df['reference'].tolist())
    rouge_results = compute_rouge(df['prediction'].tolist(), df['reference'].tolist())
    df['rouge1'] = [r['rouge1'] for r in rouge_results]
    df['rouge2'] = [r['rouge2'] for r in rouge_results]
    df['rougeL'] = [r['rougeL'] for r in rouge_results]

    print(f"BLEURT Average Score: {df['bleurt_score'].mean()}")
    print(f"BLEU Average Score: {df['bleu_score'].mean()}")
    print(f"ROUGEL Average Score: {df['rougeL'].mean()}")

    output_dir = os.path.join(os.path.dirname(args.output_dir), "detailed_results")
    output_file_path = os.path.join(output_dir, "predictions_mt5_3.csv")
    df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    main()
