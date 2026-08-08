import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from transformers import AutoTokenizer

from tinyGPT.model import GPT
from tinyGPT.utils import set_seed, load_config
from dataloader import FineWebDataset


# load config
set_seed(101)
config = load_config()

if config['trainer']['device'] == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = config['trainer']['device']
print(f'Running on device {device}')


# load dataset
folder = Path(config['system']['work_dir'])
folder.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.model_max_length = int(1e30)  # override max-length to prevent seq length warning

dataset = FineWebDataset(
    data_dir=config['system']['data_dir'], 
    seq_len=1024,
    tokenizer=tokenizer
)
    
loader = DataLoader(
    dataset,
    batch_size=config['trainer']['batch_size'],
    num_workers=config['trainer']['num_workers'],
    pin_memory=True,
    persistent_workers=(config['trainer']['num_workers'] > 0),
)


# construct the model
config['vocab_size'] = len(tokenizer)
config['block_size'] = config['context_size']
model = GPT(config).to(device)

optimiser = model.configure_optimizers(config['trainer'])
model = torch.compile(model) 

# begin training
for step, batch in enumerate(loader):
    batch = [t.to(device) for t in batch]
    x, y = batch

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, loss = model(x, y)
        print(loss)

    model.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(model.parameters(), config['trainer']['grad_norm_clip'])
    optimiser.step()
