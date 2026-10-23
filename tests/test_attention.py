# tests for the Attention module by itself
# 1. shapes: output shape should match the input shape for diff batch sizes/lengths
# 2. causality: changing the last token shouldnt change any earlier token's output
# 3. correctness: load hf's real block 0 attn weights, outputs should match theirs

import argparse

import torch
from transformers import GPT2LMHeadModel

from gpt2_engine.model import Attention, GPT2Config

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
args = parser.parse_args()

torch.manual_seed(0)  # same random inputs every run
config = GPT2Config()
attn = Attention(config).to(args.device).eval()

# 1. shapes
# attention should give back the same shape it takes in, (B, T, 768)
with torch.no_grad():
    for shape in [(1, 1, 768), (1, 5, 768), (1, 7, 768), (2, 5, 768)]:
        outputs = attn(torch.randn(shape, device=args.device))
        assert outputs.shape == shape, f"input {shape} -> output {tuple(outputs.shape)}"
print("1. shapes: ok")

# 2. causality
# if the mask works, each token can only see itself and the tokens before it
# so changing the last token should only change the last output row
with torch.no_grad():
    inputs = torch.randn(1, 6, 768, device=args.device)
    outputs = attn(inputs)

    changed_inputs = inputs.clone()
    changed_inputs[:, -1] = torch.randn(
        768, device=args.device
    )  # only change the last token
    changed_outputs = attn(changed_inputs)

# biggest change in any row except the last, should be 0
diff = (outputs[:, :-1] - changed_outputs[:, :-1]).abs().max().item()
if diff < 1e-6:
    status = "ok"
else:
    status = "FAIL: an earlier token can see the last one"
print(f"2. causality: earlier rows changed by {diff:.2e}  {status}")

assert diff < 1e-6, "causality check failed"

# 3. compare against hf's block 0 attention w/ the real gpt2 weights
hf = GPT2LMHeadModel.from_pretrained("gpt2").to(args.device).eval()
hf_attn = hf.transformer.h[0].attn

# run the whole hf model once and save what goes into and comes out of block 0's attn
# easier than calling hf_attn directly b/c its forward args change between hf versions
captured = {}


def save_attn_io(_module, args, kwargs, output):
    # depending on the hf version the input comes in as a positional or keyword arg
    if args:
        captured["inputs"] = args[0]
    else:
        captured["inputs"] = kwargs["hidden_states"]
    # hf's attn can return a tuple (output, attn weights, ...), we only want output
    if isinstance(output, tuple):
        captured["outputs"] = output[0]
    else:
        captured["outputs"] = output


# with_kwargs=True so the hook gets the keyword args too
hook = hf_attn.register_forward_hook(save_attn_io, with_kwargs=True)
with torch.no_grad():
    hf(
        torch.randint(0, config.vocab_size, (1, 8), device=args.device)
    )  # 8 random tokens
hook.remove()  # only needed it for that one run

# copy block 0's attn weights into ours
# hf stores them as (in, out) b/c it uses Conv1D, nn.Linear wants (out, in) so .t()
with torch.no_grad():
    attn.c_attn.weight.copy_(hf_attn.c_attn.weight.t())
    attn.c_attn.bias.copy_(hf_attn.c_attn.bias)
    attn.c_proj.weight.copy_(hf_attn.c_proj.weight.t())
    attn.c_proj.bias.copy_(hf_attn.c_proj.bias)
    my_outputs = attn(captured["inputs"])

# biggest gap between ours and hf's, tiny float differences are fine
diff = (my_outputs - captured["outputs"]).abs().max().item()
if diff < 1e-4:
    status = "ok"
else:
    status = "FAIL"
print(f"3. vs HF block 0: max abs diff {diff:.2e}  {status}")

assert diff < 1e-4, "attention differs from HF"
