'''
Inference code should be written in this file.
'''
import argparse
import torch
from models import Transformer
import numpy as np 

def predict(model, data):
  
  model.eval()
  for batch_x, batch_y in data: # take into account batches
    output = model.decode( batch_x, torch.tensor(np.zeros()) )



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('saved_model', type=str)
  parser.add_argument('dataset', type=str)
  args = parser.parse_args()
  
  model = torch.load(args.saved_model)
  # Load data
  data = None

  predict(
    model = model,
    data = data
  )