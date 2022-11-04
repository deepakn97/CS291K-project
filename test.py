# %%
import torch
from tqdm import tqdm
import numpy as np
from utils import create_mask, LabelSmoothing, rate, data_gen
from utils import SimpleLossCompute, DummyScheduler, DummyOptimizer
from train import run_epoch
from models import Transformer

source_vocab_size = 20
target_vocab_size = 20
num_layers = 2
seq_length = 10
# %%

source = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
source_mask = torch.ones(1, 1, 10)
target = torch.zeros(1, 1).type_as(source)
target_mask = create_mask(target.size(1)).type_as(source.data)

print(f"Shape of source vector: {source.shape} \n \
    Shape of output vector: {target.shape}")

model = Transformer(embed_dim=512, src_vocab_size=source_vocab_size, 
                    trg_vocab_size=target_vocab_size, seq_length=seq_length,
                    num_layers_enc=num_layers, num_layers_dec=num_layers,
                    hidden_size=2048, n_head=8, enc_dropout=0.2, dec_dropout=0.2)
print("Model loaded")
# print(model)

# %%
# inference test
source = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 1]])
target = torch.tensor([[0]])
out = model.generate_greedy(source, target, source_mask, target_mask, 10)
print(out)
# %%

def example_simple_model(model, embed_dim):
    vocab = 20
    criterion = LabelSmoothing(vocab, 0, smoothing=0.0)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps = 1e-9
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, embed_dim, factor=1.0, warmup=400)
    )
    batch_size = 80
    for epoch in tqdm(range(5)):
        model.train()
        run_epoch(
            data_gen(vocab, batch_size, 20),
            model,
            SimpleLossCompute(criterion),
            optimizer,
            lr_scheduler,
            mode='train'
        )
        model.eval()
        run_epoch(
            data_gen(vocab, batch_size, 5),
            model,
            SimpleLossCompute(criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode='eval'
        )
    
    model.eval()
    source = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_seq_len = 10
    source_mask = torch.ones(1, 1, source.size(1))
    target = torch.tensor([[0]])
    target_mask = create_mask(target.size(1)).type_as(source.data)
    print(model.generate_greedy(source, target, source_mask, target_mask, max_seq_len))
    
example_simple_model(model, 512)
# %%
