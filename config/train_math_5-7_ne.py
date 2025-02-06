# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

# python data/math_5-7/prepare.py

# python train.py config/train_math_5-7_ne.py
# python sample.py --out_dir=out-math-5-7-ne --start=FILE:prompt.txt




out_dir = 'out-math-5-7-ne'
eval_interval = 1000 #1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 #100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'math-char'
wandb_run_name = 'mini-gpt'

dataset = 'math_5-7'
gradient_accumulation_steps = 1
batch_size = 1024
block_size = 9 # context of up to 256 previous characters


learning_rate = 0.03 # with baby networks can afford to go a bit higher
max_iters = 100000
lr_decay_iters = 20000 # make equal to max_iters usually
min_lr = 0.0003# learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# learning_rate = 0.04 # with baby networks can afford to go a bit higher
# max_iters = 7000
# lr_decay_iters = 2500 # make equal to max_iters usually
# min_lr = 0.004# learning_rate / 10 usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

"""
number of parameters: 0.33M
<5+7=12>
best loss: step 1800: train loss 0.4890, val loss 0.4897
step 3400: train loss 0.4863, val loss 0.4875
"""
n_layer = 6
n_head = 6
n_embd = 96
dropout = 0.0
mlp_expansion = 1

"""
number of parameters: 0.67M
<5+7=11>
best loss: step 1200: train loss 0.4868, val loss 0.4865
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
