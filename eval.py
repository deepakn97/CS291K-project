import sys
from pathlib import Path 
import argparse
from tqdm import tqdm
import numpy as np
from transformers import GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu
import torch

def bleu_scoring(predictions, targets):
    bleu_scores = []
    for i in range(len(predictions)):
        score = sentence_bleu([targets[i]], predictions[i])
        bleu_scores.append(score)

    return bleu_scores

def read_dataset(filename):
    # Open source
    with open(filename, 'r') as f:
        dataset = []
        for sentence in f:
            dataset.append([int(x) for x in sentence.split(' ')[:-1]])

    return dataset

def predict(dataset, model, tokenizer, strategy, top_p, top_k, temperature, seq_length, device):
    predictions = []
    print("Generating predictions...")
    for source in tqdm(dataset):
        source = source = torch.LongTensor([source]).to(device)
        source_mask = torch.ones(1, 1, source.size(1)).to(device)
        prediction = model.module.generate(source, source_mask, seq_length, tokenizer.bos_token_id, tokenizer.eos_token_id, top_p, top_k, temperature, strategy)
        txt_prediction = tokenizer.decode(prediction[0])
        txt_prediction = [word for word in txt_prediction.split(" ") if (word != tokenizer.bos_token and word != tokenizer.eos_token)]
        predictions.append(txt_prediction)
    return predictions

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', default='./models/', help='Model directroy to the tokenizer and model from')
    parser.add_argument('--model', default='wmt14_en_fr_Dec_03_2022_2010', help='Model to load')
    parser.add_argument('--dataset_dir', default='./datasets/wmt14_en_fr', help='Dataset directory. The filenames should be in the form train.src, train.trg and so on.')
    parser.add_argument('--eval_split', default='test', help='Dataset split to evaluate on') 
    parser.add_argument('--device', default='cuda:4', help='Device to use for evaluation. Default is cuda:4')
    parser.add_argument('--max_seq_length', default=1000, help='Maximum sequence length to generate', type=int)
    parser.add_argument('--pred_file', default=None, help='File to read predictions from. If not provided, prediction will be generated according to other arguments')
    parser.add_argument('--strategy', default='greedy', help='Strategy to use for generation. Default is greedy. Use sample for top_p, top_k and temperature sampling', choices=['greedy', 'sample'])
    parser.add_argument('--top_p', default=0.9, help='Top_p value to use for sampling', type=float)
    parser.add_argument('--top_k', default=50, help='Top_k value to use for sampling', type=int)
    parser.add_argument('--temperature', default=1.0, help='Temperature value to use for sampling', type=float)
    parser.add_argument('--debug', action='store_true', help='Debug mode. If set, only 5 samples will be used from target to generate predictions and printed. No bleu score will be computed')

    return parser

def main():
    # Load tokenizer
    args = argparser().parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(Path(args.models_dir, 'tokenizer'))
    # Read targets 
    target_dataset = read_dataset(Path(args.dataset_dir, f"{args.eval_split}.trg"))
    targets = []
    for sentence in target_dataset:
        sentence = [token for token in sentence if (token != tokenizer.bos_token_id and token != tokenizer.eos_token_id)]
        sentence = tokenizer.decode(sentence)
        targets.append(sentence.split(" "))

    if args.pred_file is None:
        model = torch.load(Path(args.models_dir, f'{args.model}/model.pt'))

        # Open source and target
        source_dataset = read_dataset(Path(args.dataset_dir, f"{args.eval_split}.src"))

        # get predictions
        if args.debug:
            predictions = predict(source_dataset[:5], model, tokenizer, args.strategy, args.top_p, args.top_k, args.temperature, args.max_seq_length, args.device)
            for source, target, prediction in zip(source_dataset[:5], targets[:5], predictions):
                print(f"Source: {tokenizer.decode(source)}")
                print(f"Target: {target}")
                print(f"Prediction: {prediction} \n")
            sys.exit()
        else:
            predictions = predict(source_dataset, model, tokenizer, args.strategy, args.top_p, args.top_k, args.temperature, args.max_seq_length, args.device)
            # Open predictions
            predictions_file = Path(args.models_dir, args.model, f"predictions_{args.eval_split}_{args.strategy}_{args.top_p}_{args.top_k}_{args.temperature}.txt")
            with open(predictions_file, 'w') as f:
                for prediction in predictions:
                    f.write(" ".join(prediction) + '\n')
    else:
        predictions = []
        with open(args.pred_file, 'r') as f:
            predictions = []
            for sentence in f:
                predictions.append(sentence.split(' ')[:-1])

    # Compute BLEU scores
    bleu_scores = bleu_scoring(predictions, targets)

    # Compute Statistics
    bleu_mean = np.mean(bleu_scores)
    bleu_std = np.std(bleu_scores)

    # Print results
    print(f"BLEU score: {bleu_mean}")
    print(f"BLEU score std: {bleu_std}")

if __name__ == '__main__':
    main()