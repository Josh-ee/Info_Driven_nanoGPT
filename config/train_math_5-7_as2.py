# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

"""
python data/math_5-7-as/prepare.py

python train.py config/train_math_5-7_as2.py

python sample.py --out_dir=out-math-5-7-as_2-1 --start=FILE:prompt_75.txt
python sample.py --out_dir=out-math-5-7-as_2-1 --start=FILE:prompt_57.txt

python sample.py --out_dir=out-math-5-7-as_2-1 --start=FILE:prompt_75-.txt
python sample.py --out_dir=out-math-5-7-as_2-1 --start=FILE:prompt_57-.txt

python sample.py --out_dir=out-math-5-7-as_2-1 --start=FILE:prompt.txt

"""

dataset = 'math_5-7-as'
"""
BEST MODEL: out-math-5-7-as_2
25:75 split

n_layer = 1
n_head = 1
n_embd = 4
dropout = 0.0
mlp_expansion = 1

eval_interval = 1000 #1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 #100 # don't print too too often

gradient_accumulation_steps = 1
batch_size = 1024
block_size = 9 # context of up to 256 previous characters


learning_rate = 0.03 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 8000 # make equal to max_iters usually
min_lr = 0.002# learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

-------------------------------------------------
out_dir = 'out-math-5-7-as_2' is correct
step 9000: train loss 0.6892, val loss 0.7426
saving checkpoint to out-math-5-7-as_2
"""


out_dir = 'out-math-5-7-as_2-1'
eval_interval = 1000 #1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 #100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'math-char'
wandb_run_name = 'mini-gpt'


gradient_accumulation_steps = 1
batch_size = 1024
block_size = 9 # context of up to 256 previous characters


learning_rate = 0.03 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 8000 # make equal to max_iters usually
min_lr = 0.002# learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small



"""
number of parameters: 0.17K
best loss: step 5000: train loss 0.5249, val loss 0.5338
"""
n_layer = 1
n_head = 1
n_embd = 4
dropout = 0.0
mlp_expansion = 1

# learning_rate = 0.01 # with baby networks can afford to go a bit higher
# max_iters = 2000000
# lr_decay_iters = 2000 # make equal to max_iters usually
# min_lr = 0.001# learning_rate / 10 usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small



"""
Wrong large
number of parameters: 0.67M
<5+7=11>
best loss: 
step 1200: train loss 0.4868, val loss 0.4865
step 10000: train loss 0.4859, val loss 0.4872
"""
# n_layer = 6
# n_head = 6
# n_embd = 192
# dropout = 0.1
# mlp_expansion = 4

# learning_rate = 0.005 # with baby networks can afford to go a bit higher
# max_iters = 1000000
# lr_decay_iters = 8000 # make equal to max_iters usually
# min_lr = 0.0005# learning_rate / 10 usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
