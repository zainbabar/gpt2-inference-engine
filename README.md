# GPT-2 Inference Engine

GPT-2 (the 124M model) built from scratch in PyTorch. I load Hugging Face's pretrained weights into my own implementation and check that it produces the same logits as Hugging Face on the same input, so I know the forward pass is actually correct.

## How it works

Token ids get looked up in two learned tables, one for the token (`wte`) and one for its position in the sequence (`wpe`), and the two vectors are added together. That goes through 12 transformer blocks. Each block runs masked self-attention and then an MLP, with a LayerNorm before each one and a residual connection around each one. After the last block there's a final LayerNorm, and the LM head turns each token's vector into a score (logit) for every token in the vocabulary being the next one.

Attention currently loops over the 12 heads one at a time. I wrote it that way to see each head compute its own scores, causal mask, and softmax before batching them together.

Generation is the token-by-token loop on top of the model: run the forward pass, take the scores at the last position, pick the next token, append it, and repeat. Greedy decoding takes the highest score every time. Sampling divides the scores by a temperature, optionally keeps only the top k tokens, and then samples from the resulting probabilities.

## Status

- [x] Forward pass from scratch, loading Hugging Face's weights. Max logit difference vs Hugging Face is 1.22e-04 on a 10-token prompt.
- [x] Greedy generation. Matches Hugging Face's greedy output word for word on 3 prompts, 40 new tokens each.
- [x] Temperature and top-k sampling.
- [ ] KV cache. Store the keys and values each attention layer computes, so each new step only processes the new token instead of the whole sequence again. The cached version should produce the same tokens as the uncached one.
- [ ] Benchmarks.

## Running it

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run python generate.py           # generate text from a prompt (temperature 0.7)
uv run python compare.py            # compare my logits to Hugging Face's, and find where they diverge if they don't match
uv run python compare_generate.py   # compare my greedy generation to Hugging Face's
```

The first run downloads the GPT-2 weights from Hugging Face.

## Files

- `model.py`: the model. Config, attention, MLP, transformer block, the full GPT-2 class, and the loader that copies in Hugging Face's weights.
- `generate.py`: the generation loop, with greedy decoding, temperature, and top-k.
- `compare.py` and `compare_generate.py`: correctness checks against Hugging Face.
- `explore.py`: inspects Hugging Face's state dict to see every weight's name and shape.
