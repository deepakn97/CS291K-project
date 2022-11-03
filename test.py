import torch
import numpy as np
from utils import create_mask

from models import Transformer

def main():
    
    source_vocab_size = 20
    target_vocab_size = 20
    num_layers = 5
    seq_length = 10

    
    source = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    source_mask = torch.ones(1, 1, 10)
    target = torch.zeros(1, 1).type_as(source)
    target_mask = create_mask(target.size(1)).type_as(source.data)

    print(f"Shape of source vector: {source.shape} \n \
        Shape of output vector: {target.shape}")

    model = Transformer(embed_dim=512, src_vocab_size=source_vocab_size, 
                        trg_vocab_size=target_vocab_size, seq_length=seq_length,
                        num_layers_enc=num_layers, num_layers_dec=num_layers,
                        hidden_size=2048, n_head=8, enc_dropout=0.2, dec_dropout=0.2, max_seq_length=10)
    
    print("Model loaded")
    print(model)

    out = model(source, target, source_mask, target_mask)
    print(out.shape)

    # inference test
    source = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 1]])
    target = torch.tensor([[0]])
    print(source.shape, target.shape)
    out = model.decode(source, target, source_mask, target_mask)
    print(out)

if __name__ == "__main__":
    main()
