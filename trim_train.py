# %%
import numpy as np
with open('./datasets/wmt14_en_fr/large_train.src', 'r') as f:
  src = f.readlines()

with open('./datasets/wmt14_en_fr/large_train.trg', 'r') as f:
  trg = f.readlines()

trim_srcs = []
trim_trgs = []
data_size = len(src)
# %%

selected_indices = np.random.randint(0, data_size, 500000)

for i in selected_indices:
  if len(src[i].split()) <= 1000 and len(trg[i].split()) <= 1000:
    trim_srcs.append(src[i])
    trim_trgs.append(trg[i])

# %%
with open('./datasets/wmt14_en_fr/train.src', 'w') as f:
  f.writelines(trim_srcs)

with open('./datasets/wmt14_en_fr/train.trg', 'w') as f:
  f.writelines(trim_trgs)
# %%
