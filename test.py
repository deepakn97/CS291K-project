from layers import Transformer
import torch

def main():
    
    src_vocab_size = 11
    target_vocab_size = 11
    num_layers = 6
    seq_length= 12


    # let 0 be sos token and 1 be eos token
    src = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 9, 10, 1], 
                        [0, 2, 8, 7, 3, 4, 5, 6, 7, 2, 10, 1]])
    target = torch.tensor([[0, 1, 7, 4, 3, 5, 9, 2, 8, 10, 9, 1], 
                        [0, 1, 5, 6, 2, 4, 7, 6, 2, 8, 10, 1]])

    print(src.shape,target.shape)
    model = Transformer(embed_dim=512, src_vocab_size=src_vocab_size, 
                        target_vocab_size=target_vocab_size, seq_length=seq_length,
                        num_layers=num_layers, expansion_factor=4, n_heads=8)
    model

if __name__ == "__main__":

    main
