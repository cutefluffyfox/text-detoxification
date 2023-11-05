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
    # load tox_df and non_tox df
    dataset_lim = int(dataset_lim)
    tox_df = tokenizer_loader.read_dataframe('tokenized_tox.csv', apply_ast_to='message')[:dataset_lim]
    non_tox_df = tokenizer_loader.read_dataframe('tokenized_non_tox.csv', apply_ast_to='message')[:dataset_lim]

    # split to train/val
    tox_train, tox_val, non_tox_train, non_tox_val = train_test_split(tox_df, non_tox_df, train_size=train_size)

    # update indexes
    tox_train = tox_train.reset_index(drop=True)
    tox_val = tox_val.reset_index(drop=True)
    non_tox_train = non_tox_train.reset_index(drop=True)
    non_tox_val = non_tox_val.reset_index(drop=True)

    # return
    return tox_train, tox_val, non_tox_train, non_tox_val


def train(model, iterator: DataLoader, optimizer, criterion, clip: float, device: torch.device, progress: Progress):
    model.train()

    epoch_loss = 0
    for i, batch in enumerate(progress.tqdm(iterator, desc='Training model')):
        # unpack batch
        tox, non_tox = batch
        tox, non_tox = tox.to(device), non_tox.to(device)
        tox, non_tox = tox.T, non_tox.T

        # zero_grad the optimizer
        optimizer.zero_grad()

        # make predictions
        output = model(tox, non_tox)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        output = output[1:].view(-1, output_dim)
        non_tox = non_tox[1:].flatten()

        # calculate loss and make backward pass
        loss = criterion(output, non_tox)
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # make a step
        optimizer.step()

        # update epoch loss
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device, progress: Progress):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(progress.tqdm(iterator, desc='Evaluating model')):
            # unpack batch
            tox, non_tox = batch
            tox, non_tox = tox.to(device), non_tox.to(device)
            tox, non_tox = tox.T, non_tox.T

            # make predictions
            output = model(tox, non_tox, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            output = output[1:].view(-1, output_dim)
            non_tox = non_tox[1:].flatten()

            # calculate loss and update total loss
            loss = criterion(output, non_tox)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def train_validate_for_n_epochs(
        dataset: str,
        vocab_file: str,
        train_size: float,
        dataset_split: int,
        max_sentence_size: int,
        batch_size: int,
        n_epoch: int,
        device_type: str,
        checkpoint: str,
        save_file_name: str,
        progress: Progress = Progress()
        ):
    # load tokenizer & vocab
    tokenizer_loader = path_manager.LoadData(dataset, 'tokenized')
    vocab: Vocab = path_manager.LoadData(dataset, 'vocab').load_vocab(vocab_file)

    # split dataset to train/val
    tox_train, tox_val, non_tox_train, non_tox_val = split_dataset(tokenizer_loader, dataset_split, train_size)

    # convert pandas DataFrame to ToxicDataset
    max_sentence_size = int(max_sentence_size)
    train_dataset = ToxicDataset(tox_train, non_tox_train, vocab, max_size=max_sentence_size)
    test_dataset = ToxicDataset(tox_val, non_tox_val, vocab, max_size=max_sentence_size)

    # define DataLoaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_toxic_batch)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_toxic_batch)

    # determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = device if (device_type == 'auto' or device is None) else torch.device(device_type)

    # define model
    model = DSkBart(len(vocab), device=device).to(device)
    if checkpoint != 'random':
        checkpoint_loader = path_manager.LoadModelCheckpoint('seq_to_seq')

        # load either weights or full model itself
        if checkpoint.endswith('weight.pt'):
            model = DSkBart(len(vocab))
            checkpoint_loader.load_model_dict(model, checkpoint)
        else:
            model: DSkBart = checkpoint_loader.load_pytorch(checkpoint)

        # move model to device
        model.device = device
        model = model.to(device)

    # define optimizer and loss
    optimizer = Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])

    # define output parameters
    best_valid_loss = float('inf')
    output_message = "Starting training\n\n"

    # define savers
    checkpoint_model_saver = path_manager.SaveModelCheckpoint('seq_to_seq')
    models_model_saver = path_manager.SaveModelCheckpoint('seq_to_seq', save_file_name.rstrip('.pt'), dir_type='models')
    models_model_saver.save_vocab(vocab, vocab_file)

    # start training for `n_epoch`s
    for epoch in progress.tqdm(range(n_epoch), desc='Starting new epoch'):
        # train and validate for 1 epoch, get loss
        train_loss = train(model, train_dataloader, optimizer, criterion, 2, device, progress)
        valid_loss = evaluate(model, test_dataloader, criterion, device, progress)

        # format output nicely
        output_message += f'Epoch: {epoch + 1:02}\n'
        output_message += f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n'
        output_message += f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}\n\n'

        # save model if new best val_loss achieved
        if valid_loss < best_valid_loss:
            output_message += f'!! New best valid_loss found, saving model to `{save_file_name}` on epoch {epoch + 1}\n\n'
            best_valid_loss = valid_loss

            checkpoint_model_saver.save_pytorch(model, save_file_name)
            models_model_saver.save_pytorch(model, save_file_name)

    output_message += f'Training for {n_epoch} finished successfully\n'
    return output_message






