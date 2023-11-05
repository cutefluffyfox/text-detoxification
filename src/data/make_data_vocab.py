import gc
from collections import Counter

import pandas as pd
from torchtext.vocab import vocab

from src.modules import path_manager


# special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

SPECIALS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


def count_words(data: pd.DataFrame) -> Counter:
    # copy data not to modify it
    data = data.copy()

    # count # of times each token appeared
    counter = Counter()
    for tokens in data.message:
        for token in tokens:
            counter[token] += 1

    # return counter
    return counter


def delete_low_frequency_inplace(counter: Counter, threshold: int = 10):
    # delete from counter all low-frequency keys
    for (elem, freq) in list(counter.items()):
        if freq < threshold:
            del counter[elem]

    # make sure deleted items didn't stash in garbage collector
    gc.collect()
    return counter


def create_vocab(counter: Counter):
    # use torchtext vocab() function to create vocab, and set default index to unknown
    tox_vocab = vocab(counter, specials=SPECIALS)
    tox_vocab.set_default_index(tox_vocab[UNK_TOKEN])
    return tox_vocab


def create_and_save_vocab(dataset, min_freq_threshold: int):
    # define threshold
    min_freq_threshold = int(min_freq_threshold)

    # determine path's
    loader = path_manager.LoadData(dataset, 'tokenized')
    saver = path_manager.SaveData(dataset, 'vocab')

    # load dataframes and combine them into one
    dataframes = [loader.read_dataframe(file, apply_ast_to='message') for file in loader.get_all_data()]
    whole_dataset = pd.concat(dataframes)

    # get word counter, delete low frequency and create vocab
    word_counter = count_words(whole_dataset)
    delete_low_frequency_inplace(word_counter, threshold=min_freq_threshold)
    word_vocab = create_vocab(word_counter)

    # save vocab (for future use)
    files = saver.save_vocab([word_vocab], [f'all-words-vocab-{min_freq_threshold}.pth'])

    return f'Successfully created vocab with {len(word_vocab)} tokens and saved it to {files}'

