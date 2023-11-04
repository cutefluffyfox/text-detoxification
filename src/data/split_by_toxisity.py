import os
import pandas as pd
from src.modules import data_manager


def split_by_toxicity(raw_dataset: str, toxicity_difference: float, similarity_rate: float):
    # determine path's
    cwd = os.getcwd()
    tsv_file = data_manager.read_dir_type('raw', raw_dataset)[0]
    dataset_path = os.path.join(cwd, 'data', 'raw', raw_dataset, tsv_file)

    # read raw dataset
    df = pd.read_table(dataset_path, index_col=0)

    # split words to toxic & non-toxic based on rate [1.1]
    tox_queries = []
    ntox_queries = []

    for tox_query, ntox_query, sim, len_diff, tox, ntox in df.values:
        if tox < ntox:
            tox, ntox = ntox, tox
            tox_query, ntox_query = ntox_query, tox_query

            # add thresholds on toxicity_difference [1.2] and similarity rate [1.3]
        if (tox - ntox) >= toxicity_difference and sim >= similarity_rate:
            tox_queries.append((tox_query, tox))
            ntox_queries.append((ntox_query, ntox))

    # convert processed data to dataframes
    tox = pd.DataFrame(tox_queries, columns=['message', 'tox_score'])
    non_tox = pd.DataFrame(ntox_queries, columns=['message', 'tox_score'])

    # save dataframes
    save = data_manager.Save(raw_dataset, 'split')
    paths = save.save_dataframes([tox, non_tox], names=['tox.csv', 'non_tox.csv'])

    return (f'Queries passed: {round(100 * len(tox_queries) / df.shape[0], 2)}%\n\n'
            f'Successfully saved splits to {paths}')
