import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

num_proc = 8  # This should be roughly half the number of CPU cores available

enc = tiktoken.get_encoding("gpt2")

print_iter = 0

def process(example):
    global print_iter 
    ids = enc.encode_ordinary(example['text'])
    if print_iter < 1:
        print(vars(enc))
        print(print_iter)
        print_iter +=1

    
    ids.append(enc.eot_token)  # Ensure this token is correctly used, whether appended or prepended
    return {'ids': ids, 'len': len(ids)}

if __name__ == '__main__':
    # Load WikiText-2 dataset, raw version
    dataset = load_dataset("wikitext", "wikitext-2-v1", num_proc=num_proc, cache_dir='datasets/')

    dataset.pop('test')

    # Need to be named val
    dataset['val'] = dataset['validation']
    del dataset['validation']

    # print(dataset['train']['text'])

    # Since WikiText-2 already has train, validation, and test splits, no need for manual splitting
    # We directly tokenize the dataset using our defined process function
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing the splits",
        num_proc=num_proc,
    )

    

    vocab_size = vars(enc)['max_token_value']
    print('max_token_value:', vocab_size)

    # Concatenate all the ids in each dataset split into a single large file for training purposes
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # Safe since GPT-2 encoding uses a max token value of 50256
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()