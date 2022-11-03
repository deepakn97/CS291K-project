# You will need to install the follwing packages
# !pip3 install torch torchvision torchaudio
# !pip install transformers
# !pip install datasets

from transformers import GPT2Tokenizer
from datasets import load_dataset

def process_dataset(dataset_name="wmt14", dataset_languages="fr-en"):
  languages = dataset_languages.split('-')

  # Load dataset
  dataset = load_dataset(dataset_name, dataset_languages)

  # Load tokenizer
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


  dataset_splits = ["train"]
  for split in dataset_splits:
    print(f"Processing {split}...")
    filename_source = dataset_name + "_" + languages[1] + "_" + split + ".src"
    filename_target = dataset_name + "_" + languages[0] + "_" + split + ".trg"

    f_source = open(filename_source, "w+")
    f_target = open(filename_target, "w+")

    for sentence in dataset[split]:
      sentence = sentence["translation"]

      source_sentence = sentence[languages[1]]
      target_sentence = sentence[languages[0]]

      source_sentence_tokenized = tokenizer.tokenize(source_sentence)
      target_sentence_tokenized = tokenizer.tokenize(target_sentence)

      f_source.write(' '.join(source_sentence_tokenized) + '\n')
      f_target.write(' '.join(target_sentence_tokenized) + '\n')

    print(f"Finished with {split}. Stored in: {filename_source}, {filename_target}")
    f_source.close()
    f_target.close()
    
if __name__ == "__main__":
  process_dataset()
  