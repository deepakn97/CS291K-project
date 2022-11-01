import torch
import torch.nn as nn

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
