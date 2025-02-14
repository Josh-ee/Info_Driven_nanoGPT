"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

input_file_path = os.path.join(os.path.dirname(__file__), 'out_numbers.txt')

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
split_idx = int(n * 0.8)

# Find the nearest newline after the
while split_idx < n and data[split_idx] != '\n':
    split_idx += 1

# Perform the split at the newline
train_data = data[:split_idx]
val_data = data[split_idx+1:]

def print_plus_minus_stats(dataset, name):
    lines = [line for line in dataset.splitlines() if line.strip()]  # get non-empty lines
    add_count = 0
    sub_count = 0
    for line in lines:
        # Check if the line has enough characters.
        # Our equations are in the form "<i+ j=...>" or "<i- j=...>"
        if len(line) >= 4:
            op = line[2]  # The operator should be at position 2
            if op == '+':
                add_count += 1
            elif op == '-':
                sub_count += 1
    total = add_count + sub_count
    if total > 0:
        add_percent = add_count / total
        sub_percent = sub_count / total
    else:
        add_percent = sub_percent = 0
    print(f"{name} stats:")
    print(f"  '+' count: {add_count} ({add_percent:.2%})")
    print(f"  '-' count: {sub_count} ({sub_percent:.2%})")
    
print_plus_minus_stats(train_data, "Train data")
print_plus_minus_stats(val_data, "Validation data")

print("---------------")
print("train_data_start", train_data[:8])
print("train_data_end", train_data[-8:])
print("---------------")
print("val_data_start", val_data[:8])
print("val_data_end", val_data[-8:])
print("---------------")

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
