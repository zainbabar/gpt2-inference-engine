import torch.nn as nn
from transformers import GPT2LMHeadModel
import torch

hf = GPT2LMHeadModel.from_pretrained("gpt2")
sd = hf.state_dict()
print(hf.config)
# wte weight is the token table, 1 row for each token  (50257, 768)
# lm head weight is the output head, takes each tokens final 786 and produces
# 1 score per possible next token , shape (50257, 768)
print(torch.equal(sd["lm_head.weight"], sd["transformer.wte.weight"]))
# are literally the same thing in memory
print(sd["lm_head.weight"].data_ptr() == sd["transformer.wte.weight"].data_ptr())

linear = nn.Linear(768, 2304)
print(linear.weight.shape)
for k, v in hf.state_dict().items():
    print(k, tuple(v.shape))


# legend
# transformer        main body (everything except lm_head)
# h.0 ... h.11       the 12 transformer blocks ("h" = hidden layers)
# attn               multi-head attention part of a block
# mlp                feed-forward part of a block
#
# wte                word token embedding: token table, (50257, 768)
# wpe                word position embedding: position table, (1024, 768)
# ln_1               LayerNorm before attention
# ln_2               LayerNorm before the MLP
# ln_f               final LayerNorm, after all 12 blocks
# c_attn             makes Q, K, V: 768 -> 2304            ("c" = conv, really a linear layer)
# attn.c_proj        W_O, mixes the heads: 768 -> 768
# c_fc               MLP expand: 768 -> 3072               ("fc" = fully connected)
# mlp.c_proj         MLP shrink: 3072 ->

# flow is like this:
# wte (token table) + wpe (position table) to get inputs
# go thru 12 blocks, each has c_attn that builds query, key, value matricies.
# then add residual after attention, then mlp and add residualadds residual to input
# final layer norm, then lm head (output head) takes each tokesn final 768 numbers
# and outputs 50257 scores
# 1 per possible next token

# flow:
# x = wte[token ids] + wpe[positions]          -> (B, T, 768)
# 12 blocks, each:
#   x = x + attn(ln_1(x))   # c_attn makes Q,K,V; c_proj = W_O mixes heads
#   x = x + mlp(ln_2(x))    # c_fc 768->3072, GELU, c_proj 3072->768
# x = ln_f(x)                                  # final layer norm
# logits = lm_head(x)                          -> (B, T, 50257)

# note that
# - c_attn, attn.c_proj, c_fc, mlp.c_proj are stored (in, out): transpose them
#   (attn.c_proj is square, so a missed transpose won't raise an error)
# - lm_head.weight IS wte.weight (weight tying), no separate copy
