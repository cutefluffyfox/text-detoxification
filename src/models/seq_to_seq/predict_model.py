import torch
import spacy
import pandas as pd
from gradio import Progress
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader

from src.data.tokenize_dataset import tokenize_df
from src.modules import path_manager
from src.modules.toxic_dataset import ToxicDataset, collate_toxic_batch
from src.modules.seq_to_seq_model import DSkBart


def from_vocab(text: list[list[int]], vocab: Vocab):
    for line in text:
        yield vocab.lookup_tokens(line)


def inference(text: str, checkpoint: str, dataset: str, vocab_file: str, spacy_tokenizer: str, max_size: int, device_type: str, progress: Progress = Progress()):
    text_df = pd.DataFrame(columns=['message', 'tox_score'])
    text_df.message = [text]
    text_df.tox_score = 0.5

    vocab: Vocab = path_manager.LoadData(dataset, 'vocab').load_vocab(vocab_file)

    tokenize_fn = spacy.load(spacy_tokenizer)
    tokenized_text = tokenize_df(text_df, tokenize_fn, progress)
    text_dataset = ToxicDataset(tokenized_text, tokenized_text, vocab, max_size=int(max_size))
    text_dataloader = DataLoader(dataset=text_dataset, batch_size=64, shuffle=False, collate_fn=collate_toxic_batch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = device if (device_type == 'auto' or device is None) else torch.device(device_type)

    checkpoint_loader = path_manager.LoadModelCheckpoint('seq_to_seq')
    if checkpoint.endswith('weight.pt'):
        model = DSkBart(len(vocab))
        checkpoint_loader.load_model_dict(model, checkpoint)
    else:
        model: DSkBart = checkpoint_loader.load_pytorch(checkpoint)
    model.device = device
    model.to(device)

    model.eval()

    words = []

    with torch.no_grad():
        for i, batch in enumerate(text_dataloader):
            tox, non_tox = batch
            tox, non_tox = tox.to(device), non_tox.to(device)
            tox, non_tox = tox.T, non_tox.T

            output = model(tox, non_tox, 0)  # turn off teacher forcing

            words.extend(output.argmax(dim=2).T.cpu().detach().tolist())

    return list(from_vocab(words, vocab))



