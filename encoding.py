# register buffer in Pytorch ->
# If you have parameters in your model, which should be saved and restored in the state_dict,
# but not trained by the optimizer, you should register them as buffers.


import torch
import torch.nn as nn

import math

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