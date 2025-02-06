# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

"""
python data/math_5-7/prepare.py

python train.py config/train_math_5-7_ne_small.py

python sample.py --out_dir=out-math-5-7-ne-small --start=FILE:prompt_75.txt
python sample.py --out_dir=out-math-5-7-ne-small --start=FILE:prompt_57.txt

python sample.py --out_dir=out-math-5-7-ne-small --start=FILE:prompt.txt
"""





out_dir = 'out-math-5-7-ne-small'
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


learning_rate = 0.04 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 2500 # make equal to max_iters usually
min_lr = 0.004# learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small


"""
number of parameters: 0.06M
best loss: step 4000: train loss 0.4860, val loss 0.4871
<7+5=12>
<5+7=12>
"""
n_layer = 4
n_head = 4
n_embd = 48
dropout = 0.0
mlp_expansion = 1



"""
number of parameters: 0.01M
best loss: step 4000: train loss 0.4859, val loss 0.4868
<7+5=12>
<5+7=12>
"""
n_layer = 4
n_head = 4
n_embd = 24
dropout = 0.0
mlp_expansion = 1


"""
number of parameters: 0.00M
best loss: step 6000: train loss 0.4861, val loss 0.4869
<7+5=12>
<5+7=12>
"""
n_layer = 4
n_head = 4
n_embd = 12
dropout = 0.0
mlp_expansion = 1


"""
number of parameters: 0.01M
best loss: step 4000: train loss 0.4859, val loss 0.4868
<7+5=12>
<5+7=12>
"""
n_layer = 4
n_head = 4
n_embd = 24
dropout = 0.0
mlp_expansion = 1


"""
number of parameters: 1.98K
best loss: step 4000: train loss 0.4858, val loss 0.4878
<7+5=12>
<5+7=12>
"""
n_layer = 2
n_head = 2
n_embd = 12
dropout = 0.0
mlp_expansion = 1



"""
number of parameters: 1.09K
best loss: step 8000: train loss 0.4862, val loss 0.4862
<7+5=12>
<5+7=12>
"""
n_layer = 1
n_head = 1
n_embd = 12
dropout = 0.0
mlp_expansion = 1


"""
number of parameters: 0.33K
best loss: step 4000: train loss 0.4890, val loss 0.4881
<7+5=12>
<5+7=12>
"""
n_layer = 1
n_head = 1
n_embd = 6
dropout = 0.0
mlp_expansion = 1


"""
number of parameters: 0.11K
best loss: 
saving checkpoint to out-math-5-7-ne-small
<7+5=11>
<5+7=16>
"""
n_layer = 1
n_head = 1
n_embd = 3
dropout = 0.0
mlp_expansion = 1

"""
number of parameters: 0.17K
best loss: step 9000: train loss 0.5532, val loss 0.5543
<7+5=12>
<5+7=12>
"""
n_layer = 1
n_head = 1
n_embd = 4
dropout = 0.0
mlp_expansion = 1


"""
number of parameters: 0.17K
best loss: step 10000: train loss 0.5044, val loss 0.5037
"""
n_layer = 1
n_head = 1
n_embd = 4
dropout = 0.0
mlp_expansion = 1

learning_rate = 0.05 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 2000 # make equal to max_iters usually
min_lr = 0.005# learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small



"""
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
