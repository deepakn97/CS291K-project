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
    :param embed_dim: dimension of the word embeddings vector
    :param n_heads: number of self-attention heads
    """
    #TODO(deepakn97): consider adding dropout, bias for input and output, and bias for key and value weights
    super(MultiHeadAttention, self).__init__()

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
    :param key: key vector
    :param query: query vector
    :param value: value vector
    :param mask: for masked attention in decoder

    :return: output vector from multihead attention
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
  
class TransformerBlock(nn.Module):
  def __init__(self, embed_dim, hidden_size=2048, n_head=8, dropout=0.2):
    """
    :param embed_dim: dimension of embeddings
    :param hidden_size: hidden layer dimension for feed forward layer
    :param n_head: number of multihead attention heads
    :param dropout: dropout value

    :return out: output of the encoder
    """
    super(TransformerBlock, self).__init__()

    self.attention = MultiHeadAttention(embed_dim, n_head)

    self.layer_norm1 = nn.LayerNorm(embed_dim)
    self.layer_norm2 = nn.LayerNorm(embed_dim)
    
    self.feed_forward = nn.Sequential(
      nn.Linear(embed_dim, hidden_size)
      nn.ReLU(),
      nn.Linear(hidden_size, embed_dim),
    )

    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

    def forward(self, key, query, value):
      """
      :param key: key vector
      :param query: query vector
      :param value: value vector
      :param norm2_out: output of transformer block
      """

      attention_output = self.attention(key, query, value)
      # add residual connection here
      attention_residual_output = attention_output + value
      
      layer_norm1_output = self.dropout1(self.layer_norm1(attention_residual_output))
      
      feed_forward_output = self.feed_forward(layer_norm1_out)
      # add residual connection here
      feed_forward_residual_output = feed_forward_output + layer_norm1_output

      layer_norm2_out = self.dropout2(self.layer_norm2(feed_forward_residual_output))

      return layer_norm2_out
    
class TransformerEncoder(nn.Module):
  def __init__(self, seq_length, vocab_size, embed_dim, num_layers, hidden_size, n_head, dropout) -> None:
    """
    :param seq_length: input sequence length
    :param vocab_size: size of the vocabulary
    :param embed_dim: dimension of embeddings
    :param num_layers: number of encoder layers
    :param hidden_size: hidden layer dimension for feed forward layer
    :param n_head: number of multihead attention heads
    :param dropout: dropout for encoder layers

    :return out: output of the encoder
    """
    super(TransformerEncoder, self).__init__()
    self.embedding_layer = Embedding(vocab_size, embed_dim)
    self.positional_encoder = PositionalEmbedding(seq_length, embed_dim)

    self.enc_layers = nn.ModuleList([TransformerBlock(embed_dim, hidden_size, n_head, dropout) for i in range(num_layers)])

  def forward(self, x):
    embedded_input = self.embedding_layer(x)
    embedded_input = self.positional_encoder(embedded_input)
    for layer in self.layers:
      out = layer(embedded_input, embedded_input, embedded_input)
    
    return out



