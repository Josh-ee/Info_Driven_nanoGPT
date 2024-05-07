# import time

"""
python train.py config/train_alpaca.py

python sample_kan.py --out_dir=out-alpaca --start="### Instruction:\nList 5 different animals"

python sample.py --out_dir=out-alpaca --start="Question : List 5 different animals \nAnswer :"

"""

out_dir = 'out-alpaca'

eval_interval = 100 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 10 # don't print too too often

wandb_log = False # feel free to turn on
wandb_project = 'alpaca'
wandb_run_name = 'alpaca-run'

dataset = 'alpaca'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# Must have: n_embd % n_head == 0
n_layer = 6
n_head = 6
n_embd = 192
dropout = 0

bias = False
block_size = 1024 # context len


# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
batch_size = 16
gradient_accumulation_steps = 1 # gradient_accumulation * GPUs

learning_rate = 1e-2 # with baby networks can afford to go a bit higher
max_iters = 100000
lr_decay_iters = 100000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially
