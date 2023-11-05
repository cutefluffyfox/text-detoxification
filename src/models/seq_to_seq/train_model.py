import math

import torch
from torch import nn
from gradio import Progress
from torch.optim import Adam
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.modules import path_manager
from src.modules.seq_to_seq_model import DSkBart
from src.modules.toxic_dataset import ToxicDataset, collate_toxic_batch


def split_dataset(tokenizer_loader: path_manager.LoadData, dataset_lim: int, train_size: float):
    dataset_lim = int(dataset_lim)
    tox_df = tokenizer_loader.read_dataframe('tokenized_tox.csv', apply_ast_to='message')[:dataset_lim]
    non_tox_df = tokenizer_loader.read_dataframe('tokenized_non_tox.csv', apply_ast_to='message')[:dataset_lim]

    tox_train, tox_val, non_tox_train, non_tox_val = train_test_split(tox_df, non_tox_df, train_size=train_size)

    tox_train = tox_train.reset_index(drop=True)
    tox_val = tox_val.reset_index(drop=True)
    non_tox_train = non_tox_train.reset_index(drop=True)
    non_tox_val = non_tox_val.reset_index(drop=True)

    return tox_train, tox_val, non_tox_train, non_tox_val


def train(model, iterator, optimizer, criterion, clip, device, progress):
    model.train()

    epoch_loss = 0
    for i, batch in enumerate(progress.tqdm(iterator, desc='Training model')):
        tox, non_tox = batch
        tox, non_tox = tox.to(device), non_tox.to(device)
        tox, non_tox = tox.T, non_tox.T

        optimizer.zero_grad()
        output = model(tox, non_tox)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        non_tox = non_tox[1:].flatten()

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, non_tox)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device, progress: Progress):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(progress.tqdm(iterator, desc='Evaluating model')):
            tox, non_tox = batch
            tox, non_tox = tox.to(device), non_tox.to(device)
            tox, non_tox = tox.T, non_tox.T

            output = model(tox, non_tox, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            non_tox = non_tox[1:].flatten()

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, non_tox)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def train_validate_for_n_epochs(dataset, vocab_file, train_size, dataset_split, max_sentence_size, batch_size, n_epoch, device_type, checkpoint, save_file_name, progress: Progress = Progress()):
    tokenizer_loader = path_manager.LoadData(dataset, 'tokenized')
    vocab: Vocab = path_manager.LoadData(dataset, 'vocab').load_vocab(vocab_file)

    tox_train, tox_val, non_tox_train, non_tox_val = split_dataset(tokenizer_loader, dataset_split, train_size)

    max_sentence_size = int(max_sentence_size)
    train_dataset = ToxicDataset(tox_train, non_tox_train, vocab, max_size=max_sentence_size)
    test_dataset = ToxicDataset(tox_val, non_tox_val, vocab, max_size=max_sentence_size)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_toxic_batch)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_toxic_batch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = device if (device_type == 'auto' or device is None) else torch.device(device_type)

    model = DSkBart(len(vocab), device=device).to(device)
    if checkpoint != 'random':
        checkpoint_loader = path_manager.LoadModelCheckpoint('seq_to_seq')
        model: DSkBart = checkpoint_loader.load_pytorch(checkpoint)
        model.device = device
        model = model.to(device)

    optimizer = Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])

    best_valid_loss = float('inf')
    output_message = "Starting training\n\n"

    model_saver = path_manager.SaveModelCheckpoint('seq_to_seq')

    for epoch in progress.tqdm(range(n_epoch), desc='Starting new epoch'):
        train_loss = train(model, train_dataloader, optimizer, criterion, 2, device, progress)
        valid_loss = evaluate(model, test_dataloader, criterion, device, progress)

        output_message += f'Epoch: {epoch + 1:02}\n'
        output_message += f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n'
        output_message += f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}\n\n'

        if valid_loss < best_valid_loss:
            output_message += f'!! New best valid_loss found, saving model to `{save_file_name}` on epoch {epoch + 1}\n\n'
            best_valid_loss = valid_loss

            model_saver.save_pytorch(model, save_file_name)

    output_message += f'Training for {n_epoch} finished successfully\n'
    return output_message






