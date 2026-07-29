import torch

from model import GPT
from utils import load_config
from bpe import BPETokenizer

import time


config = load_config()

model = GPT.from_pretrained(model_type=config['model_type'])
model.eval()

def generate(prompt='', num_samples=10, steps=20, do_sample=True):
    tokenizer = BPETokenizer()
    if prompt == '':
        # to create unconditional samples...
        # manually create a tensor with only the special <|endoftext|> token
        # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
        x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long)
    else:
        x = tokenizer(prompt)
    
    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)
    
    for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        print('-'*80)
        print(out)

start = time.perf_counter()
generate(prompt="Once upon a time, there was a", num_samples=1, steps=200)
print(f'Took {(time.perf_counter() - start):.2f} seconds to run.')
