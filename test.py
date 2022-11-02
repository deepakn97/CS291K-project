import torch
import numpy as np

from layers import Transformer

def main():
    
    source_vocab_size = 20
    target_vocab_size = 20
    num_layers = 5
    seq_length = 10

    
    source = torch.tensor(np.random.randint(0,source_vocab_size,(3,seq_length)))
    target = torch.tensor(np.random.randint(0,target_vocab_size,(3,seq_length)))

    print(f"Shape of source vector: {source.shape} \n \
        Shape of output vector: {target.shape}")

    model = Transformer(embed_dim=512, src_vocab_size=source_vocab_size, 
                        trg_vocab_size=target_vocab_size, seq_length=seq_length,
                        num_layers_enc=num_layers, num_layers_dec=num_layers
                        hidden_size=248, n_heads=8, enc_dropout=0.2, dec_drouput=0.2)
    model

if __name__ == "__main__":

    main()
