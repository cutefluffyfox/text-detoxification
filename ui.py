import gradio as gr

from src.modules.path_manager import GradioReaders
from src.data import parse_zip, split_by_toxisity, tokenize_dataset, make_data_vocab
from src.models.seq_to_seq import train_model, predict_model


def update_dropdown(func):
    def update_func_dropdown(*args, **kwargs):
        return gr.update(choices=func())
    return update_func_dropdown


def update_raw_dropdown(*args, **kwargs):
    new_choices = gr.update(choices=GradioReaders.read_dir_type('raw'))
    return new_choices


def update_vocab(dataset):
    vocab_choices = gr.update(choices=GradioReaders.vocab_readers(dataset))
    return vocab_choices


def update_intermediate_checkpoint_dropdown(*args, **kwargs):
    intermediate_choices = gr.update(choices=GradioReaders.read_dir_type('intermediate'))
    checkpoint_choices = gr.update(choices=['random'] + GradioReaders.checkpoint_readers('seq_to_seq'))
    return intermediate_choices, checkpoint_choices


def update_inf_checkpoint_dropdown(*args, **kwargs):
    intermediate_choices = gr.update(choices=GradioReaders.read_dir_type('intermediate'))
    checkpoint_choices = gr.update(choices=GradioReaders.checkpoint_readers('seq_to_seq'))
    return intermediate_choices, checkpoint_choices


# Preprocess dataset tabs
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("Detox **seq-to-seq** pipeline - Made by @cutefluffyfox")

    # preprocessing tab
    with gr.Tab("Preprocess dataset"):

        # load zip archive
        gr.Markdown('LoadData & Unzip dataset')
        with gr.Row():
            with gr.Column():
                zip_file = gr.File()
            with gr.Column():
                zip_output = gr.Text(label='Status of unzip')
        zip_btn = gr.Button("LoadData archive", variant='primary')
        zip_btn.click(parse_zip.download_dataset_from_zip, inputs=[zip_file], outputs=[zip_output])

        # split dataset by toxicity
        gr.Markdown('Split dataset by toxicity')
        with gr.Row():
            with gr.Column():
                raw_dataset = gr.Dropdown(GradioReaders.read_dir_type('raw'), label="Chose dataset", info="Chose what dataset to preprocess", interactive=True)
                update_raw_btn = gr.Button('Refresh dataset list')
                tox_diff = gr.Slider(0, 1, value=0.75, label="Toxicity difference", info="Threshold on minimum toxicity difference")
                sim_rate = gr.Slider(0, 1, value=0.65, label="Similarity rate", info="Threshold on minimum similarity rate")
            with gr.Column():
                split_output = gr.Text(label='Status of split')
        split_btn = gr.Button('Split by toxicity', variant='primary')
        split_btn.click(split_by_toxisity.split_by_toxicity, inputs=[raw_dataset, tox_diff, sim_rate], outputs=[split_output])
        update_raw_btn.click(update_raw_dropdown, inputs=[], outputs=[raw_dataset])

        # Tokenize dataset
        gr.Markdown('Tokenize dataset')
        with gr.Row():
            with gr.Column():
                spacy_tokenizer = gr.Textbox(value='en_core_web_sm', label='Spacy tokenizer', info='Tokenizer from spacy python library to use')
            with gr.Column():
                tokenize_output = gr.Text(label='Status of tokenization')
        tokenize_btn = gr.Button('Tokenize data', variant='primary')
        tokenize_btn.click(tokenize_dataset.tokenize_dataset, inputs=[raw_dataset, spacy_tokenizer], outputs=[tokenize_output])

        # Create vocab
        gr.Markdown('Create words vocab')
        with gr.Row():
            with gr.Column():
                min_freq_numb = gr.Slider(value=10, minimum=0, maximum=1000, label='Frequency threshold', info='Minimum amount of word occurences in the whole dataset in order to be present in vocab')
            with gr.Column():
                vocab_output = gr.Text(label='Status of counter/vocab')
        vocab_btn = gr.Button('Generate vocab', variant='primary')
        vocab_btn.click(make_data_vocab.create_and_save_vocab, inputs=[raw_dataset, min_freq_numb], outputs=[vocab_output])

    # Training tab
    with gr.Tab("Train seq-to-seq"):
        gr.Markdown('Split data to train/test')
        with gr.Row():
            with gr.Column():

                train_dataset = gr.Dropdown(GradioReaders.read_dir_type('intermediate'), label="Chose dataset", info="Chose what dataset to preprocess", interactive=True)
                train_vocab = gr.Dropdown([], label='Chose vocab', info='Chose what vocab to be trained on', interactive=True)
                train_checkpoints = gr.Dropdown(['random'] + GradioReaders.checkpoint_readers('seq_to_seq'), value='random', label='Checkpoints', info='Checkpoint to load model from (if `random` chosen, initialize from random)', interactive=True)
                update_train_btn = gr.Button('Refresh dropdown lists')
                train_size = gr.Slider(0, 1, value=0.9, label="Train size", info="Ratio of train in dataset train/val split")
                train_dataset_split = gr.Number(value=400000, minimum=0, label='Dataset split', info='It takes a long time to train on whole dataset, so this parameter controls dataset split on max amount of elements')
                train_max_size = gr.Number(value=75, minimum=3, label='Max sentence size', info='Max number of tokens model can generate')
                train_batch_size = gr.Slider(value=64, minimum=1, maximum=1024, label='Batch size', info='Number of sentences in one batch')
                train_n_epoch = gr.Slider(minimum=1, maximum=100, value=8, label='Number of epochs', info='Number of epochs to train model on')
                train_device = gr.Radio(['auto', 'cuda', 'cpu'], value='auto', label='Chose device to move model to', info='Device on which model will be trained on (cuda highly recommended)', show_label=True)
                save_model_name = gr.Textbox(value='seq_to_seq_model    .pt', label='Checkpoint name', info='Name of model checkpoint file')
            with gr.Column():
                train_load_output = gr.Text(label='Status of training', max_lines=200)
        load_btn = gr.Button('Start training', variant='primary')
        load_btn.click(train_model.train_validate_for_n_epochs, inputs=[train_dataset, train_vocab, train_size, train_dataset_split, train_max_size, train_batch_size, train_n_epoch, train_device, train_checkpoints, save_model_name], outputs=[train_load_output])
        update_train_btn.click(update_intermediate_checkpoint_dropdown, inputs=[], outputs=[train_dataset, train_checkpoints])
        train_dataset.change(update_vocab, inputs=[train_dataset], outputs=[train_vocab])

    # Inference tab
    with gr.Tab("Inference"):
        gr.Markdown('Tab to check how model works yourself!')
        with gr.Row():
            with gr.Column():
                inf_text = gr.Textbox(value='Your toxic message here', label='Textbox for inference messages', info='Textbox for 1 message to inference')
                inf_dataset = gr.Dropdown(GradioReaders.read_dir_type('intermediate'), label="Chose dataset model is based on", info="Chose what dataset model was trained on", interactive=True)
                inf_vocab = gr.Dropdown([], label='Chose vocab', info='Chose what vocab model was trained on', interactive=True)
                inf_checkpoints = gr.Dropdown(
                    GradioReaders.checkpoint_readers('seq_to_seq'),
                    label='Checkpoints', info='Checkpoint to load model from',
                    interactive=True
                )
                update_inf_btn = gr.Button('Refresh dropdown lists')
                inf_spacy_tokenizer = gr.Textbox(value='en_core_web_sm', label='Spacy tokenizer', info='Tokenizer from spacy python library to use')
                inf_max_size = gr.Number(value=75, minimum=3, label='Max sentence size', info='Max number of tokens model can generate')
                inf_device = gr.Radio(['auto', 'cuda', 'cpu'], value='auto', label='Chose device to move model to', info='Device on which model will be inferenced on (cuda highly recommended)', show_label=True)
            with gr.Column():
                inf_output = gr.Text(label='status of inference')
        inf_btn = gr.Button('Start inference', variant='primary')
        inf_btn.click(predict_model.inference,
                       inputs=[inf_text, inf_checkpoints, inf_dataset, inf_vocab, inf_spacy_tokenizer, inf_max_size, inf_device],
                       outputs=[inf_output])
        update_inf_btn.click(update_inf_checkpoint_dropdown, inputs=[], outputs=[inf_dataset, inf_checkpoints])
        inf_dataset.change(update_vocab, inputs=[inf_dataset], outputs=[inf_vocab])


demo.queue()
if __name__ == '__main__':
    demo.launch(inbrowser=True, show_error=True)
