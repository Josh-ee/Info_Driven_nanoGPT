# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

# python data/math_char/prepare.py

# python train.py config/train_math.py
# python sample.py --out_dir=out-math --start="<9+7=?"

"""
python sample.py \
    --out_dir=out-math \
    --start="<9+7=?" \
    --num_samples=5 --max_new_tokens=10

"""


out_dir = 'out-math-2x'
eval_interval = 100 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'math-char'
wandb_run_name = 'mini-gpt'

dataset = 'math_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 8 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 96
dropout = 0.0

learning_rate = 0.04 # with baby networks can afford to go a bit higher
max_iters = 4000
lr_decay_iters = 2500 # make equal to max_iters usually
min_lr = 0.004# learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 0 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
