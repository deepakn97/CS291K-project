'''
Inference code should be written in this file.
'''
from pathlib import Path

from constants import *

def read_dataset(filename):
    # Open source
    with open(Path(DATASET_DIR, filename), 'r') as f:
        dataset = []
        for sentence in f:
            dataset.append([int(x) for x in sentence.split(' ')[:-1]])

    return dataset



if __name__ == "__main__":
  
  # Open test source
  source_test_dataset = read_dataset('wmt14_en_test.src')

  # Load the model 
  