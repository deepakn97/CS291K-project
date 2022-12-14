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
  def __init__(self, embed_dim, n_head) -> None:
    """
    :param embed_dim: dimension of the word embeddings vector
    :param n_head: number of self-attention heads
    """
    #TODO(deepakn97): consider adding dropout, bias for input and output, and bias for key and value weights
    super(MultiHeadAttention, self).__init__()

    # Assign default parameters as class parameters
    self.embed_dim = embed_dim
    self.n_head = n_head
    self.embed_dim_single_head = (self.embed_dim // self.n_head)
    assert self.n_head * self.embed_dim_single_head == self.embed_dim, "embed_dim must be divisible by n_head"
    self.attention = None

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

    # Same mask is applied to all attention heads
    if mask is not None:
      mask = mask.unsqueeze(1)

    # Change dims from (batch_size, seq_len, embed_dim) to (batch_size, seq_len, n_head, embed_per_head)
    key = key.view(batch_size, seq_length, self.n_head, self.embed_dim_single_head)
    query = query.view(batch_size, query_seq_length, self.n_head, self.embed_dim_single_head)
    value = value.view(batch_size, seq_length, self.n_head, self.embed_dim_single_head)

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
      product = product.masked_fill(mask == 0, -1e10)

    # scaling by 1/sqrt(dk) to improve gradient propogation
    self.attention = product / math.sqrt(self.embed_dim_single_head)

    # apply softmax
    self.attention = F.softmax(self.attention, dim=-1) 

    # multiply with value matrix
    scores = torch.matmul(self.attention, V) # (batch_size, n_head, seq_length, embed_per_head)

    # concatenate outputs
    concat = scores.transpose(1, 2).contiguous().view(batch_size, query_seq_length, self.embed_dim)

    output = self.out(concat) # (batch_size, n_head, seq_length)

    return output
    
class TransformerBlock(nn.Module):
  def __init__(self, embed_dim, hidden_size, n_head, dropout=0.2):
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
      nn.Linear(embed_dim, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, embed_dim),
    )

    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, key, query, value, mask=None):
    """
    :param key: key vector
    :param query: query vector
    :param value: value vector
    :param norm2_out: output of transformer block
    """

    # mask here is used to mask the padded tokens
    attention_output = self.attention(key, query, value, mask)

    # add residual connection here
    attention_residual_output = attention_output + query
    layer_norm1_output = self.dropout1(self.layer_norm1(attention_residual_output))
    
    feed_forward_output = self.feed_forward(layer_norm1_output)
    # add residual connection here
    feed_forward_residual_output = feed_forward_output + layer_norm1_output

    layer_norm2_out = self.dropout2(self.layer_norm2(feed_forward_residual_output))

    return layer_norm2_out
    
class TransformerEncoder(nn.Module):
  def __init__(self, embed_dim, num_layers, hidden_size, n_head, dropout) -> None:
    """
    :param vocab_size: size of the vocabulary
    :param embed_dim: dimension of embeddings
    :param num_layers: number of encoder layers
    :param hidden_size: hidden layer dimension for feed forward layer
    :param n_head: number of multihead attention heads
    :param dropout: dropout for encoder layers

    :return out: output of the encoder
    """
    super(TransformerEncoder, self).__init__()
    self.positional_encoder = PositionalEmbedding(embed_dim)

    self.layers = nn.ModuleList([TransformerBlock(embed_dim, hidden_size, n_head, dropout) for i in range(num_layers)])

  def forward(self, x, mask=None):
    embedded_input = self.positional_encoder(x)
    for layer in self.layers:
      out = layer(embedded_input, embedded_input, embedded_input, mask)
    
    return out

class Generator(nn.Module):
  def __init__(self, vocab_size, embed_dim):
    super(Generator, self).__init__()
    self.proj = nn.Linear(embed_dim, vocab_size)

  def forward(self, x):
    return F.log_softmax(self.proj(x), dim=-1)

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
  def __init__(self, embed_dim, max_len=1200):
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


class DecoderBlock(nn.Module):
  def __init__(self, embed_dim, hidden_size=512, n_head=4, dropout=0.2):
    super(DecoderBlock, self).__init__()
    """
    Converts the internal embeddings to ouput embeddings
    :param  embed_dim:    embedding dimension
    :param  hidden_size:  hidden size in the decoder
    :param  n_head:      number of attention heads
    :param  dropout:      drouput value
    """
    
    self.masked_attention = MultiHeadAttention(embed_dim, n_head=n_head)
    self.norm = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(dropout)
    self.transformer_block = TransformerBlock(embed_dim, hidden_size, n_head, dropout)

  def forward(self, key, query, value, source_mask, target_mask):
    """
    :param  key:    key vector 
    :param  query:  query vector -> in Decoder query comes from embedding target sequence
    :param  value:      value vector
    :param  mask:   mask for multi-head attention
    :return:        block transformer output
    """
    masked_attention = self.masked_attention(query, query, query, mask=target_mask) #32x10x512
    query = self.dropout(self.norm(masked_attention + query))
    # print('Key shape in DecoderBlock: ', key.shape)
    # print('Query shape in DecoderBlock: ', query.shape)
    # print('Value shape in DecoderBlock: ', value.shape)
    # print('Attention Shape in Decoder Block: ', attention.shape)
    # print('Decoder block ends here\n')
      
    out = self.transformer_block(key, query, value, source_mask)
    return out


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_layers=2, hidden_size=512, n_head=4, dropout=0.2):
        super(TransformerDecoder, self).__init__()
        """  
        Transforms the code 
        :param  target_vocab_size:  target vocabulary size
        :param  embed_dim:          embedding dimension
        :param  seq_len:            input sequence length
        :param  num_layers:         number of encoder layers
        :param  hidden_size:        hidden size in the decoder
        :param  n_head:            number of heads for multihead attention
        :param  droupout:           drouput value
        """

        self.position_embedding = PositionalEmbedding(embed_dim)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_dim, hidden_size, n_head) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, enc_out, source_mask, target_mask):
        """
        :param  x:          Input vector from target  
        :param  enc_out:    Output from encoder layer
        :param source_mask: Padding mask for source sequence
        :param target_mask: Padding and subsequent mask for target sequence
        :return:            Output vector
        """
        x = self.position_embedding(x) #batch_size x seq_length x embed_dim
        x = self.dropout(x)
     
        for layer in self.layers:
            # print('Query Shape being sent to decoder block: ', x.shape)
            x = layer(enc_out, x, enc_out, source_mask, target_mask) # 

        return self.norm(x)
