import torch 
import numpy as np 


def create_mask(size):
  """
  :param size: size of the square matrix
  :return mask: masked matrix
  """
  attention_shape = (1, size, size)
  mask = torch.triu(torch.ones(attention_shape), diagonal=1).type(torch.uint8)
  return mask == 0

class Batch:
  """This class loads a batch of the data"""
  def __init__(self, source, target=None, pad=0):
    """
    :param source: source sequence
    :param target: target sequence
    :param pad: padding token
    """
    self.source = source
    self.source_mask = (source != pad).unsqueeze(-2)
    self.target = target
    if target is not None:
        self.target = target[:, :-1]
        self.target_y = target[:, 1:]
        self.target_mask = \
            self.make_mask(self.target, pad)
        self.ntokens = (self.target_y != pad).data.sum()
    
  @staticmethod
  def make_mask(target, pad):
      "Create a mask to hide padding and future words."
      target_mask = (target != pad).unsqueeze(-2)
      target_mask = target_mask & torch.autograd.Variable(
          create_mask(target.size(-1)).type_as(target_mask.data))
      return target_mask

def data_gen(V, batch, nbatches):
  "Generate random data for a src-tgt copy task."
  for i in range(nbatches):
      data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
      data[:, 0] = 1
      src = torch.autograd.Variable(data, requires_grad=False)
      tgt = torch.autograd.Variable(data, requires_grad=False)
      yield Batch(src, tgt, 0)

class TrainState:
  """Track training parameters - steps, examples and tokens processed"""
  step = 0 # Steps in the current epochs
  accumulation_step = 0 # Number of gradient accumulation steps
  samples = 0 # total number of examples used
  tokens = 0 # total number of tokens processed


