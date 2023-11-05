import torch
import pandas as pd
from torch.utils.data import Dataset
from torchtext.vocab.vocab import Vocab


class ToxicDataset(Dataset):
    """
    Dataset for any toxic-formatted dataset
    """

    def __init__(self,
                 tokenized_tox_data: pd.DataFrame,
                 tokenized_non_tox_data: pd.DataFrame,
                 vocab: Vocab,
                 max_size: int = 150
                 ):
        # save parameters
        self.max_size = max_size
        self.tox_data = tokenized_tox_data
        self.non_tox_data = tokenized_non_tox_data
        self.vocab = vocab

    def _get_sentence(self, index: int, is_toxic: bool) -> list[int]:
        # retrieves sentence from dataset by index
        if is_toxic:
            sent = ['<SOS>'] + self.tox_data.message[index] + ['<EOS>']
        else:
            sent = ['<SOS>'] + self.non_tox_data.message[index] + ['<EOS>']

        # pads/slice if required
        if len(sent) <= self.max_size:
            sent.extend(['<PAD>'] * (self.max_size - len(sent)))
        else:
            sent = sent[:self.max_size - 1] + ['<EOS>']

        # return vocab_id's os sentence
        return self.vocab(sent)

    def __getitem__(self, index) -> tuple[list[int], list[int]]:
        return self._get_sentence(index, is_toxic=True), self._get_sentence(index, is_toxic=False)

    def __len__(self) -> int:
        return self.tox_data.shape[0]


def collate_toxic_batch(batch: list):
    """
    Custom collate batch for toxic_datasets working with DSkBart
    """

    # tmp lists
    toxic, non_toxic = [], []

    # for each instance transform to tensor
    for (tox, non_tox) in batch:
        toxic.append(torch.tensor(tox))
        non_toxic.append(torch.tensor(non_tox))

    # stack and return
    return torch.stack(toxic), torch.stack(non_toxic)
