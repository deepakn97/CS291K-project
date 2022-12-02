import torch 
import logging
import torch.nn as nn
import numpy as np 
from torchtext.legacy import data as d
from torch.autograd import Variable

def set_device():
  if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info('Setting device to cuda')
  elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    logging.info('Setting device to mps')
  else:
    device = torch.device("cpu")
    logging.info('Setting device to cpu')
  
  return device

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
  warmup_term = step * warmup ** (-1.5)
  model_size_term = model_size ** (-0.5)
  return factor * model_size_term * min(step_num_term, warmup_term)


class Batch:
  """This class loads a batch of the data"""
  def __init__(self, source, target=None, pad_idx=-100):
    """
    :param source: source sequence
    :param target: target sequence
    :param pad: padding token
    """
    self.source = source
    self.source_mask = (source != pad_idx).unsqueeze(-2)
    self.target = target
    if target is not None:
        self.target = target[:, :-1]
        self.target_y = target[:, 1:]
        self.target_mask = \
            self.make_mask(self.target, pad_idx)
        self.ntokens = (self.target_y != pad_idx).data.sum()
    
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

# We don't need this currently.
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

class MyIterator(d.Iterator):
  def create_batches(self):
      if self.train:
          def pool(data, random_shuffler):
              for p in d.batch(data, self.batch_size * 100):
                  p_batch = d.batch(
                      sorted(p, key=self.sort_key),
                      self.batch_size, self.batch_size_fn)
                  for b in random_shuffler(list(p_batch)):
                      yield b
          self.batches = pool(self.data(), self.random_shuffler)
          
      else:
          self.batches = []
          for b in d.batch(self.data(), self.batch_size,
                                        self.batch_size_fn):
              self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.source.transpose(0, 1), batch.target.transpose(0, 1)
    return Batch(src, trg, pad_idx)


# Multi GPU loss compute for one epoch
class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, scheduler=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, 
                                               devices=devices)
        self.opt = opt
        self.scheduler = scheduler
        self.devices = devices
        self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, 
                                                devices=self.devices)
        out_scatter = nn.parallel.scatter(out, 
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, 
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
            l = l.sum() / normalize
            total += l.data

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.            
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, 
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.scheduler.step()
        return total * normalize

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new[0]))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new[1]) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
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