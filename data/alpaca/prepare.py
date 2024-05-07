import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset


enc = tiktoken.get_encoding("gpt2")

num_proc = 4

alpaca_prompt = """Question : {} {} \nAnswer : {}"""



# EOS_TOKEN = enc.eot_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) #+ EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
# dataset = load_dataset("yahma/alpaca-cleaned",  split = "train", cache_dir='datasets/')
dataset = load_dataset("mylesgoose/alpaca-cleaned-gpt4-turbo",  split = "train", cache_dir='datasets/')

dataset = dataset.map(formatting_prompts_func, batched = True)

print(dataset)

dataset = dataset.remove_columns(['output', 'input', 'instruction'])

print(dataset)

split_dataset = dataset.train_test_split(test_size=0.005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

print(split_dataset)
# exit()
# dataset = dataset.map(formatting_prompts_func, batched = True)

def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

vocab_size = enc.max_token_value
print('max_token_value:', enc.max_token_value)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 200

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()
