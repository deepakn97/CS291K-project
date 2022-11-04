'''
All the different variations of the model should be defined here.
Note: This file has a dependency on the layers.py file.
'''

from layers import *
from utils import create_mask

class Transformer(nn.Module):
  def __init__(self, embed_dim, src_vocab_size, trg_vocab_size, seq_length, num_layers_enc=2, num_layers_dec=3, hidden_size=2048, n_head=8, enc_dropout=0.2, dec_dropout=0.2):
    """
    :param embed_dim: embedding dimensions
    :param src_vocab_size: vocabulary size for input
    :param trg_vocab_size: vocabulary size for output
    :param seq_lenght: length of the input sequence
    :param num_layers_enc: number of encoder layers
    :param num_layers_dec: number of decoder layers
    :param hidden_size: hidden layer dimension for feed forward neural network (both encoder and decoder)
    :param n_head: number of heads for Multi Head Attention
    :param enc_dropout: dropout for encoder layer
    :param dec_dropout: dropout for decoder layer
    """
    super(Transformer, self).__init__()

    self.out_vocab_size = trg_vocab_size
    self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers_enc, hidden_size, n_head, enc_dropout)
    self.decoder = TransformerDecoder(trg_vocab_size, embed_dim, seq_length, num_layers_dec, hidden_size, n_head, dec_dropout)

    # Inititalize parameters with Glorot / fan_avg
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  # TODO(deepakn97): Modify the function to create target and target_mask
  def generate_greedy(self, source, target, source_mask, target_mask, max_seq_length):
    """
    :param source: input of encoder
    :param target: input of decoder

    :return out_prob: returns final prediction of sequence
    """
    enc_out = self.encoder(source, source_mask)

    #TODO(deepakn97): check for <eos> token and end early
    for i in range(max_seq_length):
      out = self.decoder(target, enc_out, source_mask, target_mask) # batch_size x seq_length x vocab_size
      prob = out[:, -1]
      _, next_word = torch.max(prob, dim=1)
      next_word = next_word.data[0]
      target = torch.cat(
        [target, torch.zeros(1, 1).type_as(source.data).fill_(next_word)], dim=1
      )

    return target

  def encode(self, source, source_mask):
    return self.encoder(source, source_mask)
  
  def decode(self, target, enc_out, source_mask, target_mask):
    return self.decoder(target, enc_out, source_mask, target_mask)
  
  def forward(self, source, target, source_mask, target_mask):
    """
    :param source: input to encoder
    :param target: input to decoder
    
    :return out: final vector which returns probabilites of each target word
    """
    encoder_out = self.encoder(source, source_mask)
    outputs = self.decoder(target, encoder_out, source_mask, target_mask)
    return outputs

