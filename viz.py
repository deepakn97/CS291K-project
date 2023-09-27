# %%
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer

MODEL_DIR = "/home/dnathani/CS291K-project/models/wmt14_en_fr_Dec_03_2022_2010"
def draw(data, x, y, ax):
  sns.heatmap(data, xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, cbar=False, ax=ax)

# %%
# load model from directory
model = torch.load(f'{MODEL_DIR}/model.pt').to("cpu")
tokenizer = GPT2Tokenizer.from_pretrained("./models/tokenizer")
sent = "hello my name is luigi"
sent = [tokenizer.convert_ids_to_tokens(x) for x in tokenizer.encode(sent)]
src = torch.LongTensor([tokenizer.encode(sent)])
src_mask = (src != tokenizer.pad_token_id).unsqueeze(-2)
tgt_sent = "<|bos|> bonjour je m'appelle luigi <|eos|>"

# %%
model.eval()
out = model.module.generate(src, src_mask, 512, tokenizer.bos_token_id, tokenizer.eos_token_id)
translation = tokenizer.decode(out[0])
print(translation)

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
    draw(model.module.decoder.layers[layer].masked_attention.attention[0, 2*h].data[:len(tgt_sent)+1, :len(tgt_sent)+1], tgt_sent, tgt_sent if h == 0 else [], ax = axs[h])
  plt.show()

  print("Decoder Src Layer", layer+1)
  fig, axs = plt.subplots(1, 4, figsize=(20,10))
  for h in range(4):
    draw(model.module.decoder.layers[layer].transformer_block.attention.attention[0, h].data[:len(tgt_sent), :len(sent)], sent, tgt_sent if h == 0 else [], ax = axs[h])
  plt.show()

# %%

# Plot trainig and eval loss
train_loss_file = f"{MODEL_DIR}/train_losses.txt"
eval_loss_file = f"{MODEL_DIR}/eval_losses.txt"
learning_rate_file = f"{MODEL_DIR}/learning_rates.txt"

with open(train_loss_file, 'r') as f:
  train_loss = [float(x) for x in f.readlines()]
with open(eval_loss_file, 'r') as f:
  eval_loss = [float(x) for x in f.readlines()]
with open(learning_rate_file, 'r') as f:
  learning_rates = [float(x) for x in f.readlines()]

train_steps = [i*50 for i in range(len(train_loss))]
eval_steps = [i*1953 for i in range(len(eval_loss))]
lr_steps = [i*50 for i in range(len(learning_rates))]

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.plot(train_steps, train_loss, label="Train Loss")
plt.plot(eval_steps, eval_loss, label="Eval Loss")
plt.title("Training and Eval Loss Curves")
plt.legend()
plt.show()

plt.xlabel("Steps")
plt.ylabel("Learning Rate")
plt.plot(lr_steps, learning_rates, label="Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()
plt.show()
# %%