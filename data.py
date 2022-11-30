# You will need to install the follwing packages
# !pip3 install torch torchvision torchaudio
# !pip install transformers
# !pip install datasets
import os
import json
from tqdm import tqdm
from pathlib import Path
import torch
from torch.utils.data import Dataset
import linecache
from data import CustomDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import GPT2Tokenizer
from constants import *
from datasets import load_dataset

def process_dataset(dataset_name="wmt14", dataset_languages="fr-en"):
    languages = dataset_languages.split('-')

    # Create dataset dir if it does not exist yet
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_languages)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2",
        unk_token="<|unk|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>", 
        )

    # Select from ["train", "validation", "test"]
    dataset_splits = ["train", "validation", "test"]

    for split in dataset_splits:
        print(f"Processing {split}...")
        filename_source = dataset_name + "_" + languages[1] + "_" + split + ".src"
        filename_target = dataset_name + "_" + languages[0] + "_" + split + ".trg"

        f_source = open(Path(DATASET_DIR, filename_source), "w+")
        f_target = open(Path(DATASET_DIR, filename_target), "w+")

        for sentence in tqdm(dataset[split]):
            sentence =  sentence["translation"]

            source_sentence = "<|bos|>" + sentence[languages[1]] + "<|eos|>"
            target_sentence = "<|bos|>" + sentence[languages[0]] + "<|eos|>"

            # source_sentence_tokenized = tokenizer.tokenize(source_sentence)
            # target_sentence_tokenized = tokenizer.tokenize(target_sentence)
            source_sentence_tokenized = tokenizer.encode(source_sentence)
            target_sentence_tokenized = tokenizer.encode(target_sentence)

            # f_source.write(' '.join(source_sentence_tokenized) + '\n')
            # f_target.write(' '.join(target_sentence_tokenized) + '\n')
            for token in source_sentence_tokenized:
                f_source.write(str(token) + ' ')
            f_source.write('\n')

            for token in target_sentence_tokenized:
                f_target.write(str(token) + ' ')
            f_target.write('\n')

        print(f"Finished with {split}. Stored in: {filename_source}, {filename_target}")
        f_source.close()
        f_target.close()
    
    print("Saving vocabulary...")
    filename_vocabulary = Path(DATASET_DIR, "vocab.json")
    tokenizer.save_vocabulary(save_directory=DATASET_DIR)
    print("FINISHED!")

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    
    #__call__: a default method
    ##   First the obj is created using MyCollate(pad_idx) in data loader
    ##   Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        #get all source indexed sentences of the batch
        source = [item[0] for item in batch] 
        #pad them using pad_sequence method from pytorch. 
        source = pad_sequence(source, batch_first=False, padding_value = self.pad_idx) 
        
        #get all target indexed sentences of the batch
        target = [item[1] for item in batch] 
        #pad them using pad_sequence method from pytorch. 
        target = pad_sequence(target, batch_first=False, padding_value = self.pad_idx)
        return source, target

def load_data(dataset_name, dataset_languages, split):
    """
    Loads the tokenized data
    :param  dataset_name: name of the dataset [wmt14]
    :param  dataset_languages: languages of the dataset [fr-en]
    :param  split: selected split [train/validation/test]
    :return: (source_sentences, target_sentences) tuple of source, target tokens
    """
    languages = dataset_languages.split('-')

    filename_source = dataset_name + "_" + languages[1] + "_" + split + ".src"
    filename_target = dataset_name + "_" + languages[0] + "_" + split + ".trg"

    try:
        with open(Path(DATASET_DIR, filename_source), 'r') as f_source:
            source_sentences = [line for line in f_source]

        with open(Path(DATASET_DIR, filename_target), 'r') as f_target:
            target_sentences = [line for line in f_target] 

    except FileNotFoundError:
        print("Dataset file does not exist. Check parameter of load_data function.")

    return source_sentences, target_sentences


class CustomDataset(Dataset):
    def __init__(self, source_filename, target_filename):
        self.source_filename = source_filename
        self.target_filename = target_filename
        self.num_pairs = sum(1 for _ in open(Path(DATASET_DIR, source_filename), 'r'))

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        source_sentence = linecache.getline(str(Path(DATASET_DIR, self.source_filename)), idx+1)
        source_sentence = [int(x) for x in source_sentence.split(' ')[:-1]]
        target_sentence = linecache.getline(str(Path(DATASET_DIR, self.target_filename)), idx+1)
        target_sentence = [int(x) for x in target_sentence.split(' ')[:-1]]
        return torch.Tensor(source_sentence), torch.Tensor(target_sentence)

def get_train_loader(dataset, batch_size, num_workers=0, shuffle=True, pin_memory=True): #increase num_workers according to CPU
    #get pad_idx for collate fn
    pad_idx = dataset.source_vocab.stoi['<PAD>']
    #define loader
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=shuffle,
                       pin_memory=pin_memory, collate_fn = MyCollate(pad_idx=pad_idx)) #MyCollate class runs __call__ method by default
    return loader

def load_vocabulary():
    filename_vocabulary = Path(DATASET_DIR, "vocab.json")
    vocabulary = json.load(open(filename_vocabulary, 'r'))
    return vocabulary

if __name__ == "__main__":
    process_dataset()
  