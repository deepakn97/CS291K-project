'''
Training code should be written in this file.
'''
import tqdm
from time import time
from models import Transformer

def train(model, data):
  pass

def run_epoch(data_iter, model, loss, optimizer, scheduler, mode='train', accum_iter = 1, train_state=TrainState()):
  """Run training for one epoch
  :param data_iter: one batch of data
  :param model: instance of a trainable model
  :param loss: loss computation function
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

  for i, batch in tqdm(data_iter):
    out = model(batch.source, batch.target, batch.source_mask, batch.target_mask)


def main():
  # Load data function
  # define model
  # call train model
  # predict on validation data
  pass

if __name__ == "__main__":
  main()