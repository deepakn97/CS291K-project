# You will need to install the follwing packages
# !pip3 install torch torchvision torchaudio
# !pip install transformers
# !pip install datasets
import os
import json
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
import linecache

from transformers import GPT2Tokenizer
from datasets import load_dataset

DATASET_DIR = "./datasets/"

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
    # Update vocabulary with extra tokens
    vocabulary = json.load(open(filename_vocabulary, 'r'))
    extra_vocabulary = tokenizer.get_added_vocab()
    new_vocabulary= {**vocabulary, **extra_vocabulary}
    json.dump(
        new_vocabulary, 
        open(filename_vocabulary,'w', encoding="utf-8"),
        ensure_ascii=False
    )
        
    print("FINISHED!")

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


class Wmt14Dataset(Dataset):
    def __init__(self, src_filename, trg_filename):

        self.src_filename = src_filename
        self.trg_filename = trg_filename
        self.num_pairs = sum(1 for _ in open(Path(DATASET_DIR, src_filename), 'r'))

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        source_sentence = linecache.getline(str(Path(DATASET_DIR,self.src_filename)), idx)
        source_sentence = source_sentence.split(' ')[:-1]
        target_sentence = linecache.getline(str(Path(DATASET_DIR, self.src_filename)), idx)
        target_sentence = target_sentence.split(' ')[:-1]
        return source_sentence, target_sentence


def load_vocabulary():
    filename_vocabulary = "vocab.json"
    vocabulary = json.load(open(filename_vocabulary, 'r'))
    return vocabulary

if __name__ == "__main__":
    process_dataset()
  