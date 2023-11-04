import gradio as gr

from src.modules import data_manager
from src.data import parse_zip, split_by_toxisity, tokenize_dataset, make_data_vocab


def update_dropdown(*args, **kwargs):
    new_choices = gr.update(choices=data_manager.read_dir_type('raw'))
    return new_choices


# Preprocess dataset tabs
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("Top text (idk what to add here)")

    # preprocessing tab
    with gr.Tab("Preprocess dataset"):

        # load zip archive
        gr.Markdown('Load & Unzip dataset')
        with gr.Row():
            with gr.Column():
                zip_file = gr.File()
            with gr.Column():
                zip_output = gr.Text(label='Status of unzip')
        zip_btn = gr.Button("Load archive")
        zip_btn.click(parse_zip.download_dataset_from_zip, inputs=[zip_file], outputs=[zip_output])

        # split dataset by toxicity
        gr.Markdown('Split dataset by toxicity')
        with gr.Row():
            with gr.Column():
                update_btn = gr.Button('Refresh dataset list')
                raw_dataset = gr.Dropdown(data_manager.read_dir_type('raw'), label="Chose dataset", info="Chose what dataset to preprocess", interactive=True)
                tox_diff = gr.Slider(0, 1, value=0.75, label="Toxicity difference", info="Threshold on minimum toxicity difference")
                sim_rate = gr.Slider(0, 1, value=0.65, label="Similarity rate", info="Threshold on minimum similarity rate")
            with gr.Column():
                split_output = gr.Text(label='Status of split')
        split_btn = gr.Button('Split by toxicity')
        split_btn.click(split_by_toxisity.split_by_toxicity, inputs=[raw_dataset, tox_diff, sim_rate], outputs=[split_output])
        update_btn.click(update_dropdown, inputs=[], outputs=[raw_dataset])

        # Tokenize dataset
        gr.Markdown('Tokenize dataset')
        with gr.Row():
            with gr.Column():
                spacy_tokenizer = gr.Textbox(value='en_core_web_sm', label='Spacy tokenizer', info='Tokenizer from spacy python library to use')
            with gr.Column():
                tokenize_output = gr.Text(label='Status of tokenization')
        tokenize_btn = gr.Button('Tokenize data')
        tokenize_btn.click(tokenize_dataset.tokenize_dataset, inputs=[raw_dataset, spacy_tokenizer], outputs=[tokenize_output])

        # Create vocab
        gr.Markdown('Create words vocab')
        with gr.Row():
            with gr.Column():
                min_freq_numb = gr.Number(value=10, minimum=0, label='Frequency threshold', info='Minimum amount of word occurences in the whole dataset in order to be present in vocab')
            with gr.Column():
                vocab_output = gr.Text(label='Status of counter/vocab')
        vocab_btn = gr.Button('Generate vocab')
        vocab_btn.click(make_data_vocab.create_and_save_vocab, inputs=[raw_dataset, min_freq_numb], outputs=[vocab_output])

demo.queue()
if __name__ == '__main__':
    demo.launch(inbrowser=True, show_error=True)
