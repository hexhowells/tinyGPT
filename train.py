"""
Trains a character-level language model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from tinyGPT.model import GPT
from tinyGPT.trainer import Trainer
from tinyGPT.utils import set_seed, load_config


class CharDataset(Dataset):
    """
    Emits batches of characters
    """
    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config['data']['block_size']

    def __len__(self):
        return len(self.data) - self.config['data']['block_size']

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config['data']['block_size'] + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = load_config()
    print(config)

    # construct the training dataset
    text = open('tiny-shakespear.txt', 'r').read()
    train_dataset = CharDataset(config, text)

    # construct the model
    config['vocab_size'] = train_dataset.get_vocab_size()
    config['block_size'] = train_dataset.get_block_size()
    model = GPT(config)

    # construct the trainer object
    trainer = Trainer(config['trainer'], model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = "O God, O God!"
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(
                    f'\n[Sample at iter: {trainer.iter_num}]:'
                    f'\n{"-"*10}\n'
                    f'{completion}'
                    f'\n{"-"*10}\n'
                    )
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config['system']['work_dir'], "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()