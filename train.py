'''
Training code should be written in this file.
'''
import argparse
import json
import os
import math
import time
from utils import LabelSmoothing, rate, MultiGPULossCompute, Batch
from models import Transformer
from data import load_vocabulary, get_dataset
from transformers import GPT2Tokenizer
import torch
from torch import nn
from constants import *
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(model, train_loader, val_loader, vocab_size, embedding_dim, devices, steps_per_epoch, n_epochs, pad_idx, lr, model_output_dir):
  # set current cuda device to first gpu in devices
  torch.cuda.set_device(devices[0])

  model.cuda()

  criterion = LabelSmoothing(vocab_size, pad_idx, smoothing=0.0)
  criterion.cuda()


  model_par = nn.DataParallel(model, device_ids=devices)

  optimizer = torch.optim.Adam(
      model.parameters(), lr=lr, betas=(0.9, 0.98), eps = 1e-9
  )
  lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer=optimizer,
      lr_lambda=lambda step: rate(step, embedding_dim, factor=1.0, warmup=4000)
  )

  for epoch in range(n_epochs):
    model_par.train()
    train_loss = run_epoch(
      train_loader,
      model_par,
      MultiGPULossCompute(model.generator, criterion, devices=devices, opt=optimizer, scheduler=lr_scheduler),
      steps_per_epoch,
      epoch,
      n_epochs,
      pad_idx=pad_idx,
      model_output_dir=model_output_dir
    )
    model_par.eval()
    eval_loss = run_epoch(
      val_loader,
      model_par, 
      MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None),
      steps_per_epoch,
      epoch,
      n_epochs,
      pad_idx=pad_idx,
      model_output_dir=model_output_dir
    )

    print(f"Epoch: {epoch}/{n_epochs} | Eval Loss: {eval_loss}")
    with open(os.path.join(model_output_dir, 'eval_losses.txt'), 'a') as f:
      f.write(str(eval_loss))
      f.write('\n')
  
def run_epoch(data_iter, model, loss_compute, steps_per_epoch, epoch, num_epochs, pad_idx, model_output_dir):
  "Standard Training and Logging Function"
  start = time.time()
  total_tokens = 0
  total_loss = 0
  tokens = 0
  for i, batch in enumerate(data_iter):
      batch = Batch(
        source=batch[0],
        target=batch[1],
        pad_idx=pad_idx
      )
      out = model.forward(batch.source, batch.target, 
                          batch.source_mask, batch.target_mask)
      loss = loss_compute(out, batch.target_y, batch.ntokens)
      total_loss += loss
      total_tokens += batch.ntokens
      tokens += batch.ntokens
      if loss_compute.opt is not None:
        print_freq = steps_per_epoch // 5
        if i % 50 == 0:
            elapsed = time.time() - start
            print("Step: %6d/%d Loss: %6.2f Tokens per Sec: %7.1f Time per batch: %7.1f Learning Rate: %6.1e" %
                    (steps_per_epoch * (epoch) + i, steps_per_epoch * num_epochs, loss / batch.ntokens, tokens / elapsed, elapsed, loss_compute.opt.param_groups[0]['lr']))
            start = time.time()
            tokens = 0

            with open(os.path.join(model_output_dir, 'learning_rates.txt'), 'a') as f:
              f.write(str(loss_compute.opt.param_groups[0]["lr"]))
              f.write('\n')
            with open(os.path.join(model_output_dir, 'train_losses.txt'), 'a') as f:
              loss_ = (loss/batch.ntokens).data
              f.write(str((loss / batch.ntokens).item()))
              f.write('\n')
      if ( steps_per_epoch * (epoch) + i ) % 1000 == 0:
        torch.save(model, os.path.join(model_output_dir,'model.pt'))
  
  return total_loss / total_tokens

def argparser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
  parser.add_argument('--dataset', type=str, default='wmt14_en_fr', help='Dataset name')
  parser.add_argument('--eval_split', type=str, default='validation', choices=['validation', 'test'], help='Evaluation split')
  return parser

def main():
  # Load data function
  args = argparser().parse_args()
  with open(args.config) as f:
    config = json.load(f)

  # get all these params from the config file
  batch_size = config.get('BATCH_SIZE', 64)
  embed_dim = config.get('EMBEDDING_DIM', 256)
  n_epochs = config.get('NUM_EPOCHS', 10)
  devices = config.get('DEVICES', [0, 1, 2, 3])
  lr = config.get('LEARNING_RATE', 0.5)
  num_encoder_layers = config.get('NUM_ENCODER_LAYER', 2)
  num_decoder_layers = config.get('NUM_DECODER_LAYER', 2)
  n_heads = config.get('NUM_ATTENTION_HEADS', 4)
  ffn_hidden_dim = config.get('FFN_HIDDEN_DIM', 512)
  datasets_dir = config.get('DATASET_DIR', './datasets/')
  timestamp = time.strftime('%b_%d_%Y_%H%M', time.localtime())
  model_output_dir = os.path.join(config.get('MODELS_DIR', './models'), f'{args.dataset}_{timestamp}')
  data_dir = os.path.join(datasets_dir, args.dataset)
  logger.info(f'Config loaded from {args.config}')
  print(f'Batch Size: {batch_size}\nEmbedding Dim: {embed_dim}\nNumber of Epochs: {n_epochs}\nLearning Rate: {lr}\nEncoder Layers: {num_encoder_layers}\nDecoder Layers: {num_decoder_layers}\nAttention Heads: {n_heads}\nFFN Hidden Size: {ffn_hidden_dim}')

  # first make sure all the output directory exists, if not create it and save config file
  if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)
  with open(os.path.join(model_output_dir, 'config.json'), 'w') as f:
    json.dump(config, f)

  logger.info(f'Loading Vocabulary...')
  tokenizer = GPT2Tokenizer.from_pretrained('./models/tokenizer')
  vocab_size = tokenizer.vocab_size
  special_tokens = len(tokenizer.special_tokens_map)
  pad_idx = tokenizer.pad_token_id
  del tokenizer

  logger.info(f'Loading Dataset...')
  # load dataset
  data_size, train_loader, val_loader = get_dataset(data_dir, args.eval_split, batch_size, pad_idx)
  steps_per_epoch = math.ceil(data_size / batch_size)

  
  # define model
  logger.info(f'Initializing Model...')
  model = Transformer(
    embed_dim=embed_dim, #what shold it be?
    src_vocab_size=vocab_size + special_tokens,
    trg_vocab_size=vocab_size + special_tokens,
    num_layers_enc=num_encoder_layers,
    num_layers_dec=num_decoder_layers,
    n_head=n_heads,
    hidden_size=ffn_hidden_dim
    )
  # call train model
  train(model, train_loader, val_loader, vocab_size + special_tokens, embed_dim, devices, steps_per_epoch, n_epochs, pad_idx, lr, model_output_dir)

if __name__ == "__main__":
  main()