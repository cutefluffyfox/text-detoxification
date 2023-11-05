import spacy
import pandas as pd
from gradio import Progress

from src.modules import path_manager


def tokenize_df(df: pd.DataFrame, tokenize_fn, progress_bar):
    tokenized, tox_scores = [], []
    for message, tox_score in progress_bar.tqdm(df.values, desc='Loading from df'):
        tokenized.append([str(token) for token in tokenize_fn(message.lower())])
        tox_scores.append(tox_score)

    new_df = pd.DataFrame(columns=['message', 'tox_score'])
    new_df.message = tokenized
    new_df.tox_score = tox_scores

    return new_df


def tokenize_dataset(dataset: str, spacy_tokenizer: str, progress_bar: Progress = Progress()):
    loader = path_manager.LoadData(dataset, 'split')
    saver = path_manager.SaveData(dataset, 'tokenized')

    tokenize_fn = spacy.load(spacy_tokenizer)

    for file_name in progress_bar.tqdm(['tox.csv', 'non_tox.csv'], desc='Loading new .csv file'):
        df = loader.read_dataframe(file_name)

        new_data = tokenize_df(df, tokenize_fn, progress_bar)

        saver.save_dataframes(files=[new_data], names=['tokenized_' + file_name])

    return 'Successfully tokenized `tox.csv` and `non_tox.csv`'
