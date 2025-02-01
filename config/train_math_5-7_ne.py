# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

# python data/math_5-7/prepare.py

# python train.py config/train_math_5-7_ne.py
# python sample.py --out_dir=out-math-5-7-ne --start=FILE:prompt.txt




out_dir = 'out-math-5-7-ne'
eval_interval = 100 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'math-char'
wandb_run_name = 'mini-gpt'

dataset = 'math_5-7'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 8 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 96
dropout = 0.0
mlp_expansion = 1

learning_rate = 0.0001 # with baby networks can afford to go a bit higher
max_iters = 7000
lr_decay_iters = 1000 # make equal to max_iters usually
min_lr = 0.00001# learning_rate / 10 usually
beta2 = 0.95 # make a bit bigger because number of tokens per iter is small

warmup_iters = 10 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
