from pathlib import Path 

import numpy as np
from transformers import GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu

from constants import *

PREDICTIONS_DIR = "./predictions/"
RESULTS_DIR = "./results/"

def bleu_scoring(predictions, targets):
    bleu_scores = []
    for i in range(len(predictions)):
        score = sentence_bleu([targets[i]], predictions[i])
        bleu_scores.append(score)

    return bleu_scores


def read_dataset(filename):
    # Open source
    with open(Path(DATASET_DIR, filename), 'r') as f:
        dataset = []
        for sentence in f:
            dataset.append([int(x) for x in sentence.split(' ')[:-1]])

    return dataset


if __name__ == '__main__':

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('./models/tokenizer')

    # Open target
    target_test_filename = "wmt14_fr_test.trg"
    target_test_dataset = read_dataset(target_test_filename)
    targets = []
    for sentence in target_test_dataset:
        sentence = [token for token in sentence if (token != tokenizer.bos_token_id and token != tokenizer.eos_token_id)]
        sentence = tokenizer.decode(sentence)
        targets.append(sentence.split(" "))

    # Open predictions
    predictions_file = "wmt14_en_fr_llm.txt"
    with open(Path(PREDICTIONS_DIR, predictions_file), 'r') as f:
        predictions = []
        for sentence in f:
            tokenized_sentece = [x for x in sentence.split(' ')[:-1]]
            predictions.append(tokenized_sentece)

    # Compute BLEU scores
    bleu_scores = bleu_scoring(predictions, targets)

    # Compute Statistics
    bleu_mean = np.mean(bleu_scores)
    bleu_std = np.std(bleu_scores)

    # Print results
    print(f"BLEU score: {bleu_mean}")
    print(f"BLEU score std: {bleu_std}")

    # Store results
    with open(Path(RESULTS_DIR, 'llm_results.txt'), 'w') as f:
        f.write(f"BLEU Mean: {bleu_mean}\n")
        f.write(f"BLEU Std: {bleu_std}\n")