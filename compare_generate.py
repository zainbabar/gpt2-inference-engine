# does my greedy generation give the exact same text as hf's greedy generation
import tiktoken
import torch
from transformers import GPT2LMHeadModel

from generate import generate
from model import GPT2

PROMPTS = [
    "The quick brown fox",
    "Once upon a time,",
    "In linear algebra, an eigenvector is",
]
N = 40  # new tokens per prompt

enc = tiktoken.get_encoding("gpt2")
mine = GPT2.from_pretrained().eval()
hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()

all_match = True
for prompt in PROMPTS:
    # [0] b/c generate returns a list w/ 1 text per sequence in the batch
    my_text = generate(mine, prompt, max_new_tokens=N)[0]

    ids = torch.tensor([enc.encode(prompt)])  # shape (1, T)
    with torch.no_grad():
        # do_sample=False is greedy, always take the top token, same as ours
        # gpt2 has no pad token so we hand it end of text (50256), and the mask
        # is all 1s b/c nothing is padded. both just stop hf from warning
        out = hf.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=N,
            do_sample=False,
            pad_token_id=50256,
        )
    hf_text = enc.decode(out[0].tolist())
    # hf stops if it generates end of text, ours always does all N tokens
    hf_stopped_early = out.shape[1] < ids.shape[1] + N

    match = my_text == hf_text
    if not match:
        all_match = False

    print(f"\nprompt: {prompt!r}")
    if match:
        print("  MATCH")
        print(f"  {my_text!r}")
    else:
        print("  MISMATCH")
        # find the first character where they differ
        # if none do, one is just a longer version of the other, so use where it ends
        i = min(len(my_text), len(hf_text))
        for k in range(i):
            if my_text[k] != hf_text[k]:
                i = k
                break
        print(f"  first difference at character {i}")
        # show 40 characters on each side of it
        print(f"  mine: ...{my_text[max(0, i - 40) : i + 40]!r}")
        print(f"  HF:   ...{hf_text[max(0, i - 40) : i + 40]!r}")
        if hf_stopped_early:
            print("  note: HF stopped early at the end-of-text token (50256)")

if all_match:
    print("\nall prompts match")
else:
    print("\nsome prompts differ")
