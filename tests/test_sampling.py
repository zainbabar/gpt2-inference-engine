# checks from sampling (greedy, temp, top-k, top-p)
import argparse

import torch

from gpt2_engine.generate import generate
from gpt2_engine.model import GPT2

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
args = parser.parse_args()

PROMPT = "hello world isn't a beautiful day today? i love studying linear algebra"
N = 20  # new tokens per run

model = GPT2.from_pretrained().to(args.device).eval()

# record every probability tensor passed to torch.multinomial
captured = []
_real_multinomial = torch.multinomial


def spy(probs, *args, **kwargs):
    captured.append(probs.detach().clone())
    return _real_multinomial(probs, *args, **kwargs)


torch.multinomial = spy


def run(seed=None, **kwargs):
    captured.clear()
    if seed is not None:
        torch.manual_seed(seed)
    return generate(model, PROMPT, max_new_tokens=N, **kwargs)[0]


failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


# greedy
greedy = run()
check("greedy (default) is deterministic", run() == greedy)

# top_k=1 leaves one candidate, so sampling must equal greedy
check(
    "top_k=1 at temperature 1.0 equals greedy",
    run(seed=0, temperature=1.0, top_k=1) == greedy,
)

# 3. randomness behaves
a = run(seed=0, temperature=1.0)
b = run(seed=0, temperature=1.0)
c = run(seed=1, temperature=1.0)
check("same seed gives the same sample", a == b)
check("different seeds give different samples", a != c)

# probabilities are valid
run(seed=0, temperature=1.0)
sums = torch.stack([p.sum(dim=-1) for p in captured])
no_nan = not any(torch.isnan(p).any() for p in captured)
check(
    "probabilities sum to 1, no NaN",
    no_nan and torch.allclose(sums, torch.ones_like(sums), atol=1e-4),
)

# top-k really limits the candidates
for k in [5, 50]:
    run(seed=0, temperature=1.0, top_k=k)
    most = max(int((p > 0).sum(dim=-1).max()) for p in captured)
    check(
        f"top_k={k}: at most {k} possible tokens per step",
        most <= k,
        f"max seen {most}",
    )


# temperature changes confidence the right way
def avg_top_prob(t):
    run(seed=0, temperature=t)
    return sum(float(p.max()) for p in captured) / len(captured)


low, high = avg_top_prob(0.5), avg_top_prob(2.0)
check(
    "lower temperature makes the top token more likely",
    low > high,
    f"avg top prob {low:.2f} at 0.5 vs {high:.2f} at 2.0",
)

# top-p
check("tiny top_p equals greedy", run(seed=0, temperature=1.0, top_p=1e-4) == greedy)
run(seed=0, temperature=1.0, top_p=0.9)
kept = [int((p > 0).sum(dim=-1).max()) for p in captured]
check(
    "top_p=0.9: number of kept tokens varies by step",
    len(set(kept)) > 1,
    f"first steps: {kept[:8]}",
)

torch.multinomial = _real_multinomial
print("\nall checks pass" if failures == 0 else f"\n{failures} check(s) failed")

assert failures == 0, f"{failures} sampling check(s) failed"
