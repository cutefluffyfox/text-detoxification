import gc
from collections import Counter

import pandas as pd
from torchtext.vocab import vocab

from src.modules import data_manager


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

SPECIALS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


def count_words(data: pd.DataFrame) -> Counter:
    data = data.copy()

    counter = Counter()
    for tokens in data.message:
        for token in tokens:
            counter[token] += 1

    return counter


def delete_low_frequency_inplace(counter: Counter, threshold: int = 10):
    for (elem, freq) in list(counter.items()):
        if freq < threshold:
            del counter[elem]

    gc.collect()
    return counter


def create_vocab(counter: Counter):
    tox_vocab = vocab(counter, specials=SPECIALS)
    tox_vocab.set_default_index(tox_vocab[UNK_TOKEN])
    return tox_vocab


def create_and_save_vocab(dataset, min_freq_threshold: int):
    min_freq_threshold = int(min_freq_threshold)

    # determine path's
    loader = data_manager.Load(dataset, 'tokenized')
    saver = data_manager.Save(dataset, 'vocab')

    dataframes = [loader.read_dataframe(file) for file in loader.get_all_files()]
    whole_dataset = pd.concat(dataframes)

    word_counter = count_words(whole_dataset)
    delete_low_frequency_inplace(word_counter, threshold=min_freq_threshold)
    word_vocab = create_vocab(word_counter)

    files = saver.save_vocab([word_vocab], [f'all-words-vocab-{min_freq_threshold}.pth'])

    return f'Successfully created vocab with {len(word_vocab)} tokens and saved it to {files}'

