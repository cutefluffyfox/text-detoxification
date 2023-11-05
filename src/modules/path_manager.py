import os
import ast

import pandas as pd
import torch
from torchtext.vocab import Vocab


class Loader:
    dir = os.getcwd()

    def get_all_data(self):
        return os.listdir(self.dir)


class Saver:
    dir = os.getcwd()


class SaveData(Saver):
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


class LoadData(Loader):
    def __init__(self, dataset: str, *extras, dir_type: str = 'intermediate'):
        self.dataset = dataset
        self.dir_type = dir_type
        self.extras = extras
        self.dir = os.path.join(os.getcwd(), 'data', dir_type, dataset, *extras)

    def read_dataframe(self, file_name: str, apply_ast_to: str = None):
        df = pd.read_csv(os.path.join(self.dir, file_name), index_col=0)
        if apply_ast_to is not None:
            df[apply_ast_to] = df[apply_ast_to].apply(ast.literal_eval)
        return df

    def load_vocab(self, file_name: str):
        return torch.load(os.path.join(self.dir, file_name))


class LoadModelCheckpoint(Loader):
    def __init__(self, model_type: str, *extras, dir_type: str = 'checkpoints'):
        self.model_type = model_type
        self.extras = extras
        self.dir = os.path.join(os.getcwd(), dir_type, model_type, *extras)

    def load_pytorch(self, model_name: str):
        return torch.load(os.path.join(self.dir, model_name))

    def load_model_dict(self, model, model_name: str):
        return model.load_state_dict(torch.load(os.path.join(self.dir, model_name)))


class SaveModelCheckpoint(Saver):
    def __init__(self, model_type: str, *extras, dir_type: str = 'checkpoints'):
        self.model_type = model_type
        self.extras = extras
        self.dir = os.path.join(os.getcwd(), dir_type, model_type, *extras)
        os.makedirs(self.dir, exist_ok=True)

    def save_pytorch(self, model, model_name: str):
        torch.save(model, os.path.join(self.dir, model_name))

    def save_vocab(self, vocab_file: Vocab, file_name: str):
        torch.save(vocab_file, os.path.join(self.dir, file_name))


class GradioReaders:
    @staticmethod
    def read_dir_type(dir_type: str, *extras):
        full_path = os.path.join(os.getcwd(), 'data', dir_type, *extras)
        return os.listdir(full_path) if os.path.exists(full_path) else []

    @staticmethod
    def vocab_readers(dataset: str):
        full_path = os.path.join(os.getcwd(), 'data', 'intermediate', dataset, 'vocab')
        return os.listdir(full_path) if os.path.exists(full_path) else []

    @staticmethod
    def checkpoint_readers(model_type: str, dir_type: str = 'checkpoints'):
        full_path = os.path.join(os.getcwd(), dir_type, model_type)
        return os.listdir(full_path) if os.path.exists(full_path) else []



