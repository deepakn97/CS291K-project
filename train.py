'''
Training code should be written in this file.
'''
from tqdm import tqdm
from time import time
from utils import create_mask, TrainState, LabelSmoothing, rate, SimpleLossCompute, MultiGPULossCompute, MyIterator, DummyScheduler, DummyOptimizer, rebatch, batch_size_fn, set_device
from models import Transformer
from data import CustomDataset, load_vocabulary
import torch
from torch import nn
from constants import *
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(model, train_data, val_data, vocabulary, embedding_dim, devices, batch_size, n_epochs):
  # get appropriate device for computation
  device = set_device()

  model.to(device)

  criterion = LabelSmoothing(len(vocabulary), 0, smoothing=0.0)
  criterion.to(device)

  train_iter = MyIterator(
    train_data, batch_size=batch_size, device=devices[0], 
    repeat=False, sort_key=lambda x: (len(x.soure), len(x.target)),
    batch_size_fn=batch_size_fn, train=True
  )
  val_iter = MyIterator(
    val_data, 
    batch_size=batch_size, device=devices[0], 
    repeat=False, sort_key=lambda x: (len(x.source), len(x.target)),
    batch_size_fn=batch_size_fn, train=False
  )

  model_par = nn.DataParallel(model, device_ids=devices)

  optimizer = torch.optim.Adam(
      model.parameters(), lr=0.5, betas=(0.9, 0.98), eps = 1e-9
  )
  lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer=optimizer,
      lr_lambda=lambda step: rate(step, embedding_dim, factor=1.0, warmup=400)
  )

  for epoch in tqdm(range(n_epochs)):
    model_par.train()
    run_epoch(
      (rebatch(len(vocabulary), b) for b in train_iter),
      model_par,
      MultiGPULossCompute(model, criterion, devices=devices, opt=optimizer),
      optimizer,
      lr_scheduler,
      mode='train'
    )
    model_par.eval()
    loss = run_epoch(
      (rebatch(len(vocabulary), b) for b in val_iter),
      model_par, 
      MultiGPULossCompute(model, criterion, devices=devices, opt=None),
      DummyOptimizer(),
      DummyScheduler(),
    )

    if n_epochs % 100 == 0:
      torch.save(model_par, MODEL_STORE_PATH)
    print(f"Epoch: {epoch}/{n_epochs} | Eval Loss: {loss}")
  
    

def run_epoch(data_iter, model, loss_func, optimizer, scheduler, mode='train', accum_iter = 1, train_state=TrainState()):
  """Run training for one epoch
  :param data_iter: one batch of data
  :param model: instance of a trainable model
  :param loss_func: loss computation function
  :param optimizer: choice of optimizer
  :param scheduler: learning rate schedular
  :param mode: can take values 'train', 'train_log' and 'val'
  :param accum_iter:
  :param train_state: instance of the TrainState class
  """
  start = time()
  total_tokens = 0
  total_loss = 0
  tokens = 0
  number_accumulation_steps = 0

  for i, batch in enumerate(tqdm(data_iter)):
    out = model(batch.source, batch.target, batch.source_mask, batch.target_mask)
    loss, loss_node = loss_func(out, batch.target_y, batch.ntokens)

    # If in training mode, update loss and training states, apply optimizer and scheduler step
    if mode == 'train' or mode == 'train_log':
      loss_node.backward()
      train_state.step += 1
      train_state.samples += batch.source.shape[0]
      train_state.tokens += batch.ntokens
      if i % accum_iter == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        number_accumulation_steps += 1
        train_state.accumulation_step += 1
      scheduler.step()

    total_loss += loss
    total_tokens += batch.ntokens
    tokens += batch.ntokens

    if i % 40 == 1 and (mode == 'train' or mode == 'train_log'):
      learning_rate = optimizer.param_groups[0]["lr"]
      elapsed = time() - start
      print(
        (
        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
      ) % (i, number_accumulation_steps, loss / batch.ntokens, tokens / elapsed, learning_rate)
      )

      start = time()
      tokens = 0

    del loss
    del loss_node
  return total_loss / total_tokens, train_state


def main():
  # Load data function
  logger.info(f'Loading Dataset...')
  train_data = CustomDataset("wmt14_en_validation.src", "wmt14_fr_validation.trg")
  val_data = CustomDataset("wmt14_en_test.src", "wmt14_fr_test.trg")
  
  # define model
  logger.info(f'Loading Vocabulary...')
  vocabulary = load_vocabulary()
  embed_dim = 512
  logger.info(f'Initializing Model...')
  model = Transformer(
    embed_dim=embed_dim, #what shold it be?
    src_vocab_size=len(vocabulary),
    trg_vocab_size=len(vocabulary),
    seq_length=None
    )
  # call train model
  n_epochs = 2
  devices = [0,1]
  batch_size = 12000
  train(model, train_data, val_data, vocabulary, embed_dim, devices, batch_size, n_epochs)

if __name__ == "__main__":
  main()