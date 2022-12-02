'''
Inference code should be written in this file.
'''
from pathlib import Path
from tqdm import tqdm
import json
import torch
import argparse

from transformers import GPT2Tokenizer

from models import Transformer
from constants import *

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./models/wmt14_en_fr_Dec_02_2022_0026', help='Model directroy to load the model from')
    parser.add_argument('--predict', action='store_true', help='Generate Translations and print')
    parser.add_argument('--inp_file', default='./datasets/wmt14_en_fr/test.src', help='Input file to translate')
    
    return parser

def main():
    args = argparser().parse_args()

    # Open test source
    source_test_dataset = ('wmt14_en_test.src')

    with open("config.json") as f:
      config = json.load(f)

    embed_dim = config.get('EMBED_DIM', 256)
    tokenizer = GPT2Tokenizer.from_pretrained('./models/tokenizer')
    vocab_size = tokenizer.vocab_size
    special_tokens = len(tokenizer.special_tokens_map)
    num_encoder_layers = config.get('NUM_ENCODER_LAYERS', 2)
    num_decoder_layers = config.get('NUM_DECODER_LAYERS', 2)
    n_heads = config.get('NUM_ATTENTION_HEADS', 4)
    ffn_hidden_dim = config.get('FFN_HIDDEN_DIM', 512)

    # Load the model 
    model = Transformer(
      embed_dim=embed_dim,
      src_vocab_size=vocab_size + special_tokens,
      trg_vocab_size=vocab_size + special_tokens,
      num_layers_enc=num_encoder_layers,
      num_layers_dec=num_decoder_layers,
      n_head=n_heads,
      hidden_size=ffn_hidden_dim
    )

    # Predict
    predictions = []
    pad_idx = tokenizer.pad_token_id
    print("Running predictions...")
    for source in tqdm(source_test_dataset):
      source = source = torch.LongTensor([source])
      source_mask = torch.ones(1, 1, source.size(1))
      prediction = model.generate_greedy(source, source_mask, MAX_SEQ_LENGTH, tokenizer.bos_token_id, tokenizer.eos_token_id)
      txt_prediction = tokenizer.decode(prediction[0])
      txt_prediction = [word for word in txt_prediction.split(" ") if (word != tokenizer.bos_token and word != tokenizer.eos_token)]
      txt_prediction = " ".join(txt_prediction)
      predictions.append(txt_prediction)

    # Store results
    with open(Path(PREDICTIONS_DIR, 'wmt14_en_fr_transformer.txt'), 'w') as f:
      for prediction in predictions:
        f.write(prediction + '\n')

if __name__ == "__main__":
  main()
  