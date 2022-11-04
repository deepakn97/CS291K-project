'''
Training code should be written in this file.
'''
from tqdm import tqdm
from time import time
from utils import create_mask, TrainState
from models import Transformer

def train(model, data):
  pass

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
  # define model
  # call train model
  # predict on validation data
  pass

if __name__ == "__main__":
  main()