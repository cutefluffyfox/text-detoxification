import torch
import pandas as pd
from torchtext.vocab.vocab import Vocab


class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self,
                 tokenized_tox_data: pd.DataFrame,
                 tokenized_non_tox_data: pd.DataFrame,
                 vocab: Vocab,
                 max_size: int = 150
                 ):
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

        if len(sent) <= self.max_size:
            sent.extend(['<PAD>'] * (self.max_size - len(sent)))
        else:
            sent = sent[:self.max_size - 1] + ['<EOS>']

        return self.vocab(sent)

    def __getitem__(self, index) -> tuple[list[int], list[int]]:
        return self._get_sentence(index, is_toxic=True), self._get_sentence(index, is_toxic=False)

    def __len__(self) -> int:
        return self.tox_data.shape[0]


def collate_toxic_batch(batch: list):
    toxic, non_toxic = [], []
    for (tox, non_tox) in batch:
        toxic.append(torch.tensor(tox))
        non_toxic.append(torch.tensor(non_tox))
    return torch.stack(toxic), torch.stack(non_toxic)
