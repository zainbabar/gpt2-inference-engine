
import tiktoken
import torch
from transformers import GPT2LMHeadModel

from model import GPT2

PROMPT = "The quick brown fox jumps over the lazy dog because"
TOL = 1e-3  
enc = tiktoken.get_encoding("gpt2")
ids = torch.tensor([enc.encode(PROMPT)])
print(f"prompt: {PROMPT!r} ({ids.shape[1]} tokens)\n")

mine = GPT2.from_pretrained().eval()
hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()

captured = {}
hooks = [mine.h[0].register_forward_pre_hook(lambda m, args: captured.__setitem__("embed", args[0]))]
for i, block in enumerate(mine.h):
    hooks.append(block.register_forward_hook(
        lambda m, args, out, i=i: captured.__setitem__(f"h{i}", out)))
hooks.append(mine.ln_f.register_forward_hook(lambda m, args, out: captured.__setitem__("ln_f", out)))

with torch.no_grad():
    my_logits = mine(ids)
    hf_out = hf(ids, output_hidden_states=True)
for h in hooks:
    h.remove()

hf_logits = hf_out.logits
diff = (my_logits - hf_logits).abs().max().item()
match = diff < TOL
print(f"logits shape: mine {tuple(my_logits.shape)}, HF {tuple(hf_logits.shape)}")
print(f"max abs logit difference: {diff:.2e}  {'MATCH' if match else 'MISMATCH'}\n")


def top5(logits):
    probs = torch.softmax(logits[0, -1], dim=-1)
    p, t = probs.topk(5)
    return ", ".join(f"{enc.decode([int(tok)])!r} {float(pr):.3f}" for tok, pr in zip(t, p))


print(f"next token, mine: {top5(my_logits)}")
print(f"next token, HF:   {top5(hf_logits)}")

if not match:
    hs = hf_out.hidden_states
    checks = [("embeddings (into h.0)", captured["embed"], hs[0])]
    checks += [(f"out of h.{i}", captured[f"h{i}"], hs[i + 1]) for i in range(len(mine.h) - 1)]
    checks += [("ln_f output", captured["ln_f"], hs[-1])]
    print("\nwhere they split:")
    for name, a, b in checks:
        d = (a - b).abs().max().item()
        print(f"  {name:24s} {d:.2e}  {'ok' if d < TOL else '<-- first mismatch'}")
        if d >= TOL:
            break