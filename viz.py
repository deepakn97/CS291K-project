# %%
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer

MODEL_DIR = "./models/wmt14_en_fr_Dec_03_2022_1537"
def draw(data, x, y, ax):
  sns.heatmap(data, xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, cbar=False, ax=ax)

# %%
# load model from directory
model = torch.load(f'{MODEL_DIR}/model.pt').to("cpu")
tokenizer = GPT2Tokenizer.from_pretrained("./models/tokenizer")
sent = [tokenizer.convert_ids_to_tokens(x) for x in tokenizer.encode("<|bos|> hello my name is luigi <|eos|>")]
src = torch.LongTensor([tokenizer.encode("<|bos|> hello my name is luigi <|eos|>")])
src_mask = (src != tokenizer.pad_token_id).unsqueeze(-2)
tgt_sent = "<|bos|> bonjour je m'appelle luigi <|eos|>"

# %%
model.eval()
out = model.module.generate_greedy(src, src_mask, 10, tokenizer.bos_token_id, tokenizer.eos_token_id)

# %%
for layer in range(2):
  fig, axs = plt.subplots(1, 4, figsize=(20, 10))
  print("Encoder Layer", layer + 1)
  for h in range(4):
    draw(model.module.encoder.layers[layer].attention.attention[0, 2*h].data, sent, sent if h == 0 else [], ax = axs[h])
  plt.show()

# %%
tgt_sent = tokenizer.convert_ids_to_tokens(out[0])
for layer in range(2):
  fig, axs = plt.subplots(1, 4, figsize=(20, 10))
  print("Decoder Layer", layer + 1)
  for h in range(4):
    draw(model.module.decoder.layers[layer].masked_attention.attention[0, 2*h].data[:len(tgt_sent), :len(tgt_sent)], tgt_sent, tgt_sent if h == 0 else [], ax = axs[h])
  plt.show()

  print("Decoder Src Layer", layer+1)
  fig, axs = plt.subplots(1, 4, figsize=(20,10))
  for h in range(4):
    draw(model.module.decoder.layers[layer].transformer_block.attention.attention[0, h].data[:len(tgt_sent), :len(sent)], sent, tgt_sent if h == 0 else [], ax = axs[h])
  plt.show()

# %%
