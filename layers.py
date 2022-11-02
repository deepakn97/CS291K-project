'''
All the model layers should be defined in this file as a nn.Module subclass.
'''

import re
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
  '''
  This class implements multi head attention for a transformer.
  '''
  def __init__(self, embed_dim=512, n_head=8) -> None:
    """
    Args:
      embed_dim: dimension of the word embeddings vector
      n_heads: number of self-attention heads
    """
    #TODO(deepakn97): consider adding dropout, bias for input and output, and bias for key and value weights
    super(MultiHeadAttention).__init__()

    # Assign default parameters as class parameters
    self.embed_dim = embed_dim
    self.n_head = n_head
    self.embed_dim_single_head = (self.embed_dim // self.n_head)
    assert self.n_head * self.embed_dim_single_head == self.embed_dim, "embed_dim must be divisible by n_head"

    # Initialize query, key, value and out matrices 
    self.W_q = nn.Linear(self.embed_dim_single_head, self.embed_dim_single_head, bias=False)
    self.W_v = nn.Linear(self.embed_dim_single_head, self.embed_dim_single_head, bias=False)
    self.W_k = nn.Linear(self.embed_dim_single_head, self.embed_dim_single_head, bias=False)
    self.out = nn.Linear(self.embed_dim, self.embed_dim)

  def forward(self, key, query, value, mask=None):
    # dimensions = batch_size x seq_len x embed_dim
    """
    Args:
      key: key vector
      query: query vector
      value: value vector
      mask: for masked attention in decoder

    Returns:
      output vector from multihead attention
    """

    batch_size = key.size(0)
    seq_length = key.size(1)

    # query dimension might be different while decoding
    query_seq_length = query.size(1)

    # Change dims from (batch_size, seq_len, embed_dim) to (batch_size, seq_len, n_head, embed_per_head)
    key = key.view(batch_size, seq_length, self.n_heads, self.embed_dim_single_head)
    query = query.view(batch_size, query_seq_length, self.n_heads, self.embed_dim_single_head)
    value = value.view(batch_size, seq_length, self.n_heads, self.embed_dim_single_head)

    # Multiply the inputs with corresponding weight matrices to get final key, value and query
    
    Q = self.W_q(query)
    K = self.W_k(key)
    V = self.W_v(value)

    # Change the dims to (batch_size, n_head, seq_len, embed_per_head) to facilitate matrix multiplication
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)

    # compute attention
    product = torch.matmul(Q, K.transpose(-1, -2)) # (batch_size, n_head, seq_len, seq_len)

    # add mask for decoder -> fill with arbitrary small number instead of zero to avoid zero division
    if mask is not None:
      product = product.maksed_fill(mask == 0, -1e10)

    # scaling by 1/sqrt(dk) to improve gradient propogation
    product = product / math.sqrt(self.embed_dim_single_head)

    # apply softmax
    scores = F.softmax(product, dim=-1)

    # multiple with value matrix
    scores = torch.matmul(scores, v) # (batch_size, n_head, seq_length, embed_per_head)

    # concatenate outputs
    concat = scores.transpose(1, 2).contiguous().view(batch_size, query_seq_length, self.embed_dim)

    output = self.out(concat) # (batch_size, n_head, seq_length)

    return output


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Word embedding. Represents each word in the sentence as an embedding vector.
        :param  vocab_size: size of the vocabulary
        :param  embed_dim:  embedding dimentiosn 
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        """
        Forward pass
        :param  x:  Input vector to embed
        :return:    Embedding
        """
        out = self.embed(x)
        return out


class PositionalEmbedding(nn.Module):
  def __init__(self,max_len,embed_dim):
      """
      Introduces information to the moodel about the relative or 
      absolute position of the tokens in the sequence
      :param  max_len:    Max Sequence Length
      :param  embed_dim:  Embedding dimension
      """
      super(PositionalEmbedding, self).__init__()
      self.embed_dim = embed_dim

      embedding = torch.zeros(max_len,self.embed_dim)
      for pos in range(max_len):
          for i in range(0,self.embed_dim,2):
              embedding[pos, i] = \
                  math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
              embedding[pos, i + 1] = \
                  math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
      
      embedding = embedding.unsqueeze(0)
      self.register_buffer('positional_embedding', embedding)


  def forward(self, x):
      """
      :param  x:  Input vector
      :return:    Embedding
      """

      # make embeddings larger
      x = x * math.sqrt(self.embed_dim)

      #add constant to embedding
      seq_len = x.size(1)
      x = x + torch.autograd.Variable(
          self.positional_embedding[:,:seq_len], requires_grad=False
          )
      return x