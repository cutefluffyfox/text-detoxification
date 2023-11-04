import spacy
import pandas as pd
import gradio as gr

from src.modules import data_manager


def tokenize_dataset(dataset: str, spacy_tokenizer: str, progress_bar: gr.Progress = gr.Progress()):
    loader = data_manager.Load(dataset, 'split')
    saver = data_manager.Save(dataset, 'tokenized')

    tokenize_fn = spacy.load(spacy_tokenizer)

    for file_name in progress_bar.tqdm(['tox.csv', 'non_tox.csv'], desc='Loading new .csv file'):
        df = loader.read_dataframe(file_name)

        tokenized, tox_scores = [], []
        for message, tox_score in progress_bar.tqdm(df.values, desc='Loading from list'):
            tokenized.append([str(token) for token in tokenize_fn(message.lower())])
            tox_scores.append(tox_score)

        new_data = pd.DataFrame(columns=['message', 'tox_score'])
        new_data.message = tokenized
        new_data.tox_score = tox_scores

        saver.save_dataframes(files=[new_data], names=['tokenized_' + file_name])

    return 'Successfully tokenized `tox.csv` and `non_tox.csv`'
