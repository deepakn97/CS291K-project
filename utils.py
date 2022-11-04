import torch 
import torch.nn as nn
import numpy as np 


def create_mask(size):
  """
  :param size: size of the square matrix
  :return mask: masked matrix
  """
  attention_shape = (1, size, size)
  mask = torch.triu(torch.ones(attention_shape), diagonal=1).type(torch.uint8)
  return mask == 0

def data_gen(V, batch, nbatches):
  "Generate random data for a src-tgt copy task."
  for i in range(nbatches):
      data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
      data[:, 0] = 1
      src = torch.autograd.Variable(data, requires_grad=False)
      tgt = torch.autograd.Variable(data, requires_grad=False)
      yield Batch(src, tgt, 0)

def loss(x, criterion):
  d = x + 3 * 1
  predict = torch.FloatTensor([[0, x / d, 1/ d, 1 / d, 1 / d]])
  return criterion(predict.log(), torch.LongTensor([1])).data

def rate(step, model_size, factor, warmup):
  """This function implements the learning rate scheduler as defined in the Attention is all you need paper.
  :param step:
  :param model_size:
  :param factor:
  :param warmup:
  """
  # default the step to 1 to avoid 0 raising to negative power
  if step == 0:
    step = 1
  step_num_term = step ** (-0.5)
  warmup_term = (step * warmup) ** (-1.5)
  model_size_term = model_size ** (-0.5)
  return model_size_term * min(step_num_term, warmup_term)

class DummyOptimizer(torch.optim.Optimizer):
  def __init__(self):
    self.param_groups = [{'lr': 0}]
    None
  
  def step(self):
    None
  
  def zero_grad(self, set_to_none=False):
    None

class DummyScheduler:
  def step(self):
    None

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

class TrainState:
  """Track training parameters - steps, examples and tokens processed"""
  step = 0 # Steps in the current epochs
  accumulation_step = 0 # Number of gradient accumulation steps
  samples = 0 # total number of examples used
  tokens = 0 # total number of tokens processed


class LabelSmoothing(nn.Module):
  """Implement Label Smoothing
  This hurts perplexity of the model but improves accuracy and BLEU score.
  Label smoothing is implemented using KL Divergence loss.
  How do you explain this tradeoff?"""
  def __init__(self, size, pad, smoothing=0.0):
    """
    :param size:
    :param pad: padding token
    :param smoothing:
    """
    super(LabelSmoothing, self).__init__()
    self.criterion = nn.KLDivLoss(reduction="sum")
    self.padding_idx = pad
    self.confidence = 1 - smoothing
    self.smoothing = smoothing
    self.size = size
    self.true_dist = None
  
  def forward(self, x, target):
    assert x.size(1) == self.size, f"Dimension of input should mactch label smoothing size, but found input dim: {x.size(1)} and size: {self.size}"
    true_dist = x.data.clone()
    true_dist.fill_(self.smoothing / (self.size - 2))
    true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
    true_dist[:, self.padding_idx] = 0
    mask = torch.nonzero(target.data == self.padding_idx)
    if mask.dim() > 0:
      true_dist.index_fill_(0, mask.squeeze(), 0.0)
    self.true_dist = true_dist
    return self.criterion(x, true_dist.clone().detach())

class SimpleLossCompute:
  def __init__(self, criterion):
    self.criterion = criterion
  
  def __call__(self, x, y, norm):
    sloss = (
      self.criterion(
        x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
      ) / norm
    )
    return sloss.data * norm, sloss
