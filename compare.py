# compare my model against hf's gpt2, same input tokens should give the same logits
# if they dont match, find the first step where they split
import tiktoken
import torch
from transformers import GPT2LMHeadModel

from model import GPT2

PROMPT = "The quick brown fox jumps over the lazy dog because"
TOL = 1e-3  # how far apart values can be and still count as matching

enc = tiktoken.get_encoding("gpt2")
ids = torch.tensor([enc.encode(PROMPT)])  # the extra [] makes it shape (1, T)
print(f"prompt: {PROMPT!r} ({ids.shape[1]} tokens)\n")

mine = GPT2.from_pretrained().eval()
hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()

# hooks on my model to save the output after every step
# so if the logits dont match we can see where it first goes wrong
captured = {}


def save_embeddings(_module, args):
    # pre hook on block 0 runs before it, so its input is the embeddings (wte + wpe)
    captured["embed"] = args[0]


def save_output(name: str):
    # makes a hook that saves the module's output under name
    def hook(_module, _args, output):
        captured[name] = output

    return hook


handles = [mine.h[0].register_forward_pre_hook(save_embeddings)]
for i, block in enumerate(mine.h):
    handles.append(block.register_forward_hook(save_output(f"h{i}")))
handles.append(mine.ln_f.register_forward_hook(save_output("ln_f")))

with torch.no_grad():
    my_logits = mine(ids)
    # output_hidden_states=True gives us hf's version of each step to compare against
    hf_out = hf(ids, output_hidden_states=True)
for handle in handles:
    handle.remove()

hf_logits = hf_out.logits
# biggest gap between any of my logits and hf's
diff = (my_logits - hf_logits).abs().max().item()
match = diff < TOL
if match:
    result = "MATCH"
else:
    result = "MISMATCH"
print(f"logits shape: mine {tuple(my_logits.shape)}, HF {tuple(hf_logits.shape)}")
print(f"max abs logit difference: {diff:.2e}  {result}\n")


def top5(logits: torch.Tensor) -> str:
    # top 5 guesses for the next token after the last position, w/ their probs
    probs = torch.softmax(logits[0, -1], dim=-1)  # shape (vocab_size,)
    top_probs, top_ids = probs.topk(5)
    guesses = []
    for token_id, prob in zip(top_ids, top_probs):
        guesses.append(f"{enc.decode([int(token_id)])!r} {float(prob):.3f}")
    return ", ".join(guesses)


print(f"next token, mine: {top5(my_logits)}")
print(f"next token, HF:   {top5(hf_logits)}")

# if they dont match, go thru each step in order and stop at the first one thats off
if not match:
    hidden_states = hf_out.hidden_states
    # hf's hidden_states: [0] is the embeddings, [i + 1] is the output of block i
    # except the last one, hf already ran ln_f on it, so compare it to our ln_f output
    checks = [("embeddings (into h.0)", captured["embed"], hidden_states[0])]
    for i in range(len(mine.h) - 1):
        checks.append((f"out of h.{i}", captured[f"h{i}"], hidden_states[i + 1]))
    checks.append(("ln_f output", captured["ln_f"], hidden_states[-1]))

    print("\nwhere they split:")
    for name, my_step, hf_step in checks:
        d = (my_step - hf_step).abs().max().item()
        if d < TOL:
            print(f"  {name:24s} {d:.2e}  ok")
        else:
            print(f"  {name:24s} {d:.2e}  <-- first mismatch")
            break
