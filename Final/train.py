import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader, random_split

# Below are installed using pip command
# pip install datasets
# pip install tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel # Tokenizer of the type WordLevel
from tokenizers.trainers import WordLevelTrainer # Train the tokenizer
from tokenizers.pre_tokenizers import Whitespace # to perform tokenization


# stuff with paths
from pathlib import Path


def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]


# config --- is a configuration map
# ds -- dataset
# lang -- tokenizer of which lang
def get_or_build_tokenizer(config, ds, lang):
    # String file path    
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path): # means tokenizer file does not exist
        # instantiating tokenizer of type WordLevel
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        
        # pre-tokenizer
        tokenizer.pre_tokenizer = Whitespace()

        #trainer will set the special tokens and will only train on
        # words having frequency of atleast 2
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]","[SOS]","[EOS]"], min_frequency = 2)

        # train_from_iterator
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)

        # save to the tokenizer path
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer    


def get_ds(config):

    ds_raw = load_dataset("opus_books", f"{config["lang_src"]}-{config["lang_tgt"]}",split="train")

    # Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw,config["lang_src"])
    tokenizer_tgt= get_or_build_tokenizer(config, ds_raw,config["lang_tgt"])

    # split into train and test 90 to 10, train to validation
    train_ds_size = int(0.9 *len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw,[train_ds_size, val_ds_size])






