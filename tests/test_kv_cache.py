# cached and uncached should do the same thing, just w/ less repeated work
import argparse

import torch

from gpt2_engine.generate import ENCODING, generate
from gpt2_engine.model import GPT2

parser = argparse.ArgumentParser(description="Check cached vs uncached inference")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
args = parser.parse_args()
model = GPT2.from_pretrained().to(args.device).eval()

# 1. compare every position's logits, not just which token wins
# prefill part of a known sequence, then feed the rest 1 token at a time
ids = torch.tensor(
    [ENCODING.encode("The quick brown fox jumps over the lazy dog.")],
    device=args.device,
)
with torch.no_grad():
    model.reset_cache()
    model.set_caching(False)
    expected = model(ids)

    for prompt_len in [1, 4]:
        model.reset_cache()
        model.set_caching(True)
        parts = [model(ids[:, :prompt_len])]
        for i in range(prompt_len, ids.shape[1]):
            parts.append(model(ids[:, i : i + 1]))
        actual = torch.cat(parts, dim=1)
        # same math, but diff matrix sizes can cause tiny float differences
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-5)
        assert model.cache_len == ids.shape[1]
        for block in model.h:
            shape = (1, block.attn.n_head, ids.shape[1], block.attn.head_dim)
            assert block.attn.K_cache.shape == shape
            assert block.attn.V_cache.shape == shape
print("1. logits and cache lengths: ok")

# 2. resetting should clear every layer and the position counter
model.reset_cache()
assert model.cache_len == 0
for block in model.h:
    assert block.attn.K_cache is None
    assert block.attn.V_cache is None
print("2. reset: ok")

# 3. check actual greedy token ids, including calls on the same model
# the hook also records how many positions each forward pass processed
tokens = []
lengths = []


def record_tokens(_module, _args, output):
    tokens.append(output[:, -1, :].argmax(dim=-1).item())
    lengths.append(output.shape[1])


hook = model.lm_head.register_forward_hook(record_tokens)
try:
    for prompt in [
        "The quick brown fox",
        "Once upon a time,",
        "In linear algebra, an eigenvector is",
        "",
    ]:
        tokens.clear()
        lengths.clear()
        uncached = generate(model, prompt, max_new_tokens=20, cache=False)
        expected_tokens = tokens.copy()
        prompt_len = len(ENCODING.encode(prompt)) or 1  # empty prompt uses BOS
        assert lengths == list(range(prompt_len, prompt_len + 20))

        # do it twice so stale history from a previous call gets caught too
        for _ in range(2):
            tokens.clear()
            lengths.clear()
            cached = generate(model, prompt, max_new_tokens=20, cache=True)
            assert tokens == expected_tokens, "cached greedy tokens differ"
            assert cached == uncached, "cached decoded text differs"
            assert lengths == [prompt_len] + [1] * 19
        print(
            f"3. greedy tokens, one-token calls, repeated generation: ok for {prompt!r}"
        )
finally:
    hook.remove()
    model.reset_cache()
    model.set_caching(False)

print("\nall cache checks pass")
