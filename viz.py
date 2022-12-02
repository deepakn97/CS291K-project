# %%
import torch
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_DIR = "./models/wmt14_en_fr_Dec_02_2022_0026"
def draw(data, x, y, ax):
  sns.heatmap(data, xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, cbar=False, ax=ax)
# %%

# load model from directory
model = torch.load(f'{MODEL_DIR}/model.pt').to("cuda:0")

# %%
for layer in range(2):
  fig, axs = plt.subplots(1, 4, figsize=(20, 10))
  print("Encoder Layer", layer + 1)
  for h in range(4):
    draw(model.module.encoder.layers[layer].attention[0, h].data, sent, sent if h == 0 else [], ax = axs[h])
  plt.show()

for layer in range(2):
  fig, axs = plt.subplots(1, 4, figsize=(20, 10))
  print("Decoder Layer", layer + 1)
  for h in range(4):
    draw(model.module.decoder.layers[layer].attention[0, h].data, sent, sent if h == 0 else [], ax = axs[h])
  plt.show()
