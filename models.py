'''
All the different variations of the model should be defined here.
Note: This file has a dependency on the layers.py file.
'''

from layers import *
from utils import create_mask

class Transformer(nn.Module):
  def __init__(self, embed_dim, src_vocab_size, trg_vocab_size, num_layers_enc=2, num_layers_dec=2, hidden_size=512, n_head=4, enc_dropout=0.2, dec_dropout=0.2):
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
    self.word_embedding = Embedding(src_vocab_size, embed_dim)
    self.encoder = TransformerEncoder(embed_dim, num_layers_enc, hidden_size, n_head, enc_dropout)
    self.decoder = TransformerDecoder(embed_dim, num_layers_dec, hidden_size, n_head, dec_dropout)
    self.generator = Generator(trg_vocab_size, embed_dim)

    # Inititalize parameters with Glorot / fan_avg
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  # TODO(deepakn97): Modify the function to create target and target_mask
  def generate_greedy(self, source, source_mask, max_seq_length, bos_token, eos_token):
    """
    :param source: input of encoder
    :param target: input of decoder

    :return out_prob: returns final prediction of sequence
    """
    embedded_inp = self.word_embedding(source)
    enc_out = self.encoder(embedded_inp, source_mask)
    target = torch.ones(source.shape[0],1).fill_(bos_token).type_as(source.data)
    target_mask = create_mask(target.shape[1]).type_as(target.data)

    #TODO(deepakn97): check for <eos> token and end early
    for i in range(max_seq_length):
      # TODO(dnathani): 
      embedded_target = self.word_embedding(target)
      out = self.decoder(embedded_target, enc_out, source_mask, target_mask) # batch_size x seq_length x vocab_size
      prob = self.generator(out[:, -1]) # batch_size x vocab_size
      _, next_word = torch.max(prob, dim=1)
      next_word = next_word.data[0] # 1
      if next_word == eos_token:
        break
      target = torch.cat(
        [target, torch.zeros(1,1).fill_(next_word).type_as(source.data)], dim=1
      )

    return target

  def encode(self, source, source_mask):
    embedded_inp = self.word_embedding(source)
    return self.encoder(embedded_inp, source_mask)
  
  def decode(self, target, enc_out, source_mask, target_mask):
    embedded_target = self.word_embedding(target)
    return self.decoder(embedded_target, enc_out, source_mask, target_mask)
  
  def forward(self, source, target, source_mask, target_mask):
    """
    :param source: input to encoder
    :param target: input to decoder
    
    :return out: final vector which returns probabilites of each target word
    """
    embedded_source = self.word_embedding(source)
    embedded_target = self.word_embedding(target)
    encoder_out = self.encoder(embedded_source, source_mask)
    outputs = self.decoder(embedded_target, encoder_out, source_mask, target_mask)
    return outputs

