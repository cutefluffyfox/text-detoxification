import os

import torch
import pandas as pd
from torchtext.vocab import Vocab


class Save:
    def __init__(self, dataset: str, *extras, dir_type: str = 'intermediate'):
        self.dataset = dataset
        self.dir_type = dir_type
        self.extras = extras
        self.dir = os.path.join(os.getcwd(), 'data', dir_type, dataset, *extras)
        os.makedirs(self.dir, exist_ok=True)

    def save_dataframes(self, files: list[pd.DataFrame], names: list[str]) -> list[str]:
        save_func = lambda csv_file, file_name: csv_file.to_csv(file_name)
        return self.__save(files, names, save_func)

    def save_vocab(self, files: list[Vocab], names: list[str]):
        save_func = lambda vocab_file, file_name: torch.save(vocab_file, file_name)
        return self.__save(files, names, save_func)

    def __save(self, files: list, names: list, save_func):
        if len(files) != len(names):
            raise ValueError('Length of names should be the save as length of files')

        paths = []
        for data, file_name in zip(files, names):
            paths.append(os.path.join(self.dir, file_name))
            save_func(data, paths[-1])

        return paths


class Load:
    def __init__(self, dataset: str, *extras, dir_type: str = 'intermediate'):
        self.dataset = dataset
        self.dir_type = dir_type
        self.extras = extras
        self.dir = os.path.join(os.getcwd(), 'data', dir_type, dataset, *extras)

    def read_dataframe(self, file_name: str):
        return pd.read_csv(os.path.join(self.dir, file_name), index_col=0)

    def get_all_files(self):
        return os.listdir(self.dir)


def read_dir_type(dir_type: str, *extras):
    if dir_type not in ['raw', 'intermediate', 'external']:
        raise NotImplementedError('This code supports only `raw`, `intermediate` and `external` directory types')
    full_path = os.path.join(os.getcwd(), 'data', dir_type, *extras)
    return os.listdir(full_path) if os.path.exists(full_path) else []


