# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

"""
python data/math_5-7-as_alt/prepare.py

python train.py config/train_math_5-7-as_alt.py

--------------------------------------------------------------------------

python sample.py --out_dir=out-math-5-7-as_alt --start=FILE:prompt_75.txt
python sample.py --out_dir=out-math-5-7-as_alt --start=FILE:prompt_57.txt

python sample.py --out_dir=out-math-5-7-as_alt --start=FILE:prompt_75-.txt
python sample.py --out_dir=out-math-5-7-as_alt --start=FILE:prompt_57-.txt

python sample.py --out_dir=out-math-5-7-as_alt --start=FILE:prompt.txt

--------------------------------------------------------------------------
python sample.py --out_dir=out-math-5-7-as_alt_bal --start=FILE:prompt_75.txt
python sample.py --out_dir=out-math-5-7-as_alt_bal --start=FILE:prompt_57.txt

python sample.py --out_dir=out-math-5-7-as_alt_bal --start=FILE:prompt_75-.txt
python sample.py --out_dir=out-math-5-7-as_alt_bal --start=FILE:prompt_57-.txt

python sample.py --out_dir=out-math-5-7-as_alt_bal --start=FILE:prompt.txt



-----------------------------
learning_rate = 0.03 # with baby networks can afford to go a bit higher
max_iters = 2000
lr_decay_iters = 1000 # make equal to max_iters usually
min_lr = 0.003# learning_rate / 10 usually
beta2 = 0.98 # make a bit bigger because number of tokens per iter is small

step 1000: train loss 0.7259, val loss 0.7866
saving checkpoint to out-math-5-7-as_alt
"""

dataset = 'math_5-7-as_alt'


out_dir = 'out-math-5-7-as_alt'
eval_interval = 500 #1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'math-char'
wandb_run_name = 'mini-gpt'


gradient_accumulation_steps = 1
batch_size = 1024
block_size = 9 


# learning_rate = 0.008 # with baby networks can afford to go a bit higher
# max_iters = 5000
# lr_decay_iters = 3000 # make equal to max_iters usually
# min_lr = 0.0002# learning_rate / 10 usually
# beta2 = 0.95 # make a bit bigger because number of tokens per iter is small

#0.69 ~50k
learning_rate = 0.01 # with baby networks can afford to go a bit higher
max_iters = 100000
lr_decay_iters = 50000 # make equal to max_iters usually
min_lr = 0.0005# learning_rate / 10 usually
beta2 = 0.98 # make a bit bigger because number of tokens per iter is small


learning_rate = 0.009 
max_iters = 1000000
lr_decay_iters = 500000 # make equal to max_iters usually
min_lr = 0.0001# learning_rate / 10 usually
beta2 = 0.98 # make a bit bigger because number of tokens per iter is small

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

warmup_iters = 10 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
