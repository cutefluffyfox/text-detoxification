import os
from zipfile import ZipFile

import gradio as gr


def download_dataset_from_zip(file_obj) -> dict:
    # determine path to save dataset
    cwd = os.getcwd()
    file_name = file_obj.name.split('/')[-1]
    dir_path = os.path.join(cwd, 'data', 'raw', file_name.strip('.zip'))
    file_path = os.path.join(dir_path, file_name)

    # make directory for dataset
    if os.path.exists(dir_path):
        gr.Warning(f'Dataset {file_name.strip(".zip")} already exists, overriding existing one')
    os.makedirs(dir_path, exist_ok=True)

    # read tmp file and save binary content
    with open(file_obj.name, 'rb') as file:
        content = file.read()

    # save binary content to needed path
    with open(file_path, 'wb') as file:
        file.write(content)

    # unzip dataset
    if not file_path.endswith('.zip'):
        gr.Warning('You uploaded not a .zip file, errors may occur')

    with ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(dir_path)

    # check that .tsv or .csv file exists
    if not any(file.endswith('.tsv') or file.endswith('csv') for file in os.listdir(dir_path)):
        gr.Warning('Failed to find any .tsv or .csv file in the uploaded dataset, may result in future problems')

    # delete archive file
    os.remove(file_path)

    return {'result': 'success'}


load_interface = gr.Interface(download_dataset_from_zip, inputs="file", outputs="json")


demo = gr.TabbedInterface([load_interface], ["upload dataset"])
demo.queue()

if __name__ == '__main__':

    demo.launch(inbrowser=True, show_error=True)
