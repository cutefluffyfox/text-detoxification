import os
import ast

import pandas as pd
import torch
from torchtext.vocab import Vocab


class Loader:
    """
    Base class for any Loader
    """
    dir = os.getcwd()

    def get_all_data(self):
        """
        returns list of all files in self.dir (main) directory
        """
        return os.listdir(self.dir)


class Saver:
    """
    Base class for any Saver
    """
    dir = os.getcwd()


class SaveData(Saver):
    """
    Save pandas DataFrames, torchtext Vocabs
    """
    def __init__(self, dataset: str, *extras, dir_type: str = 'intermediate'):
        # save each part
        self.dataset = dataset
        self.dir_type = dir_type
        self.extras = extras

        # combine into one main path
        self.dir = os.path.join(os.getcwd(), 'data', dir_type, dataset, *extras)

        # make directory if not exist
        os.makedirs(self.dir, exist_ok=True)

    def save_dataframes(self, files: list[pd.DataFrame], names: list[str]) -> list[str]:
        save_func = lambda csv_file, file_name: csv_file.to_csv(file_name)
        return self.__save(files, names, save_func)

    def save_vocab(self, files: list[Vocab], names: list[str]):
        save_func = lambda vocab_file, file_name: torch.save(vocab_file, file_name)
        return self.__save(files, names, save_func)

    def __save(self, files: list, names: list, save_func):
        """
        Tricky function that save all files by `save_func`
        """
        # check that each file have name
        if len(files) != len(names):
            raise ValueError('Length of names should be the save as length of files')

        paths = []

        # for each pair save it & add full path to paths list
        for data, file_name in zip(files, names):
            paths.append(os.path.join(self.dir, file_name))
            save_func(data, paths[-1])

        return paths


class LoadData(Loader):
    """
    Load pandas DataFrames, torchtext Vocabs
    """
    def __init__(self, dataset: str, *extras, dir_type: str = 'intermediate'):
        # save each part of a path
        self.dataset = dataset
        self.dir_type = dir_type
        self.extras = extras

        # combine parts into one path
        self.dir = os.path.join(os.getcwd(), 'data', dir_type, dataset, *extras)

    def read_dataframe(self, file_name: str, apply_ast_to: str = None):
        # read dataframe
        df = pd.read_csv(os.path.join(self.dir, file_name), index_col=0)

        # if some column supposed to have list type but is string, parse it
        if apply_ast_to is not None:
            df[apply_ast_to] = df[apply_ast_to].apply(ast.literal_eval)

        # return dataframe
        return df

    def load_vocab(self, file_name: str):
        return torch.load(os.path.join(self.dir, file_name))


class LoadModelCheckpoint(Loader):
    """
    Load pytorch models/vocabs
    """
    def __init__(self, model_type: str, *extras, dir_type: str = 'checkpoints'):
        # save each part of a path
        self.model_type = model_type
        self.extras = extras
        self.dir_type = dir_type

        # combine parts into one path
        self.dir = os.path.join(os.getcwd(), dir_type, model_type, *extras)

    def load_pytorch(self, model_name: str):
        return torch.load(os.path.join(self.dir, model_name))

    def load_model_dict(self, model, model_name: str):
        return model.load_state_dict(torch.load(os.path.join(self.dir, model_name)))


class SaveModelCheckpoint(Saver):
    """
    Save pytorch models/vocabs
    """
    def __init__(self, model_type: str, *extras, dir_type: str = 'checkpoints'):
        # save each part of a path
        self.model_type = model_type
        self.extras = extras
        self.dir_type = dir_type

        # combine parts into one path
        self.dir = os.path.join(os.getcwd(), dir_type, model_type, *extras)

        # make directory if not exist
        os.makedirs(self.dir, exist_ok=True)

    def save_pytorch(self, model, model_name: str):
        torch.save(model, os.path.join(self.dir, model_name))

    def save_vocab(self, vocab_file: Vocab, file_name: str):
        torch.save(vocab_file, os.path.join(self.dir, file_name))


class GradioReaders:
    """
    Main class for all path's Gradio may require
    """
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



