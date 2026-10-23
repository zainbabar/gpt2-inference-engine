import tiktoken
import torch

from gpt2_engine.model import GPT2

ENCODING = tiktoken.get_encoding("gpt2")  # same tokenizer gpt2 was trained w/
CONTEXT_SIZE = 1024  # max tokens the model can see at once (wpe only has 1024 rows)


def generate(
    model: GPT2,
    text: str,
    temperature: float = 0,
    top_k: int | None = None,
    top_p: float | None = None,
    max_new_tokens: int = CONTEXT_SIZE,
    cache: bool = False,
):
    model.reset_cache()  # reset its kv cache if it had one before
    model.set_caching(cache)  # set caching

    # prefill aka first pass for caching
    prefill = True

    # text -> token ids -> tensor, unsqueeze(0) adds the batch dim
    # so shape is (B, T) w/ B = 1
    device = model.wte.weight.device
    token_ids = torch.tensor(
        ENCODING.encode(text), dtype=torch.long, device=device
    ).unsqueeze(0)
    if token_ids.shape[-1] == 0:  # empty prompt, no token ids
        token_ids = torch.tensor(
            [[50256]], dtype=torch.long, device=device
        )  # gpt2's BOS token
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # stop early if we hit the context size, even if we still have tokens to go
            # the model cant take more than CONTEXT_SIZE tokens
            if token_ids.shape[-1] >= CONTEXT_SIZE:
                B = token_ids.shape[0]
                token_ids = token_ids.tolist()
                # decode every sequence in the batch back to text
                generated_texts = [ENCODING.decode(token_ids[i]) for i in range(B)]
                return generated_texts

            # prefill or we arent caching, just do a normal pass
            if prefill or not cache:
                # scores (logits) for the next token at every position
                logits = model(token_ids)  # shape (B, T, vocab_size)
                prefill = False
            # not prefill anymore and we are caching
            else:
                # only feed in the new token 1 at a time
                logits = model(token_ids[:, -1:])  # shape (B, 1, vocab_size)

            # only the last position's scores matter, thats the prediction for
            # the token that comes after everything we have so far
            next_token_scores = logits[:, -1, :]  # shape (B, vocab_size)

            # greedy: pick the highest scoring token for each sequence
            # keepdim so shape is (B, 1) not (B,), lets us cat it onto token_ids
            if temperature == 0:
                new_token_id = next_token_scores.argmax(dim=-1, keepdim=True)
            else:
                # temperature rescales the scores before softmax
                # < 1 sharpens (safer, closer to greedy), > 1 flattens (more random)
                next_token_scores = next_token_scores / temperature

                # filter out any vocab scores not in the top k
                if top_k is not None:
                    values, _ = torch.topk(next_token_scores, k=top_k, dim=-1)
                    # anything >= the last top k value is valid, so cutoff
                    cutoffs = values[..., -1:]  # shape (B, 1), slice to keep the dim
                    mask = next_token_scores < cutoffs  # true means fill
                    next_token_scores = torch.masked_fill(
                        next_token_scores, mask, float("-inf")
                    )

                # softmax over the vocab: scores -> probabilities that sum to 1
                probs = torch.softmax(next_token_scores, dim=-1)  # (B, vocab_size)
                if top_p is not None:
                    # sort high to low; sorted_ids[i] says which token sorted position i came from
                    sorted_probs, sorted_ids = torch.sort(
                        probs, dim=-1, descending=True
                    )  # (B, V) each
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    # remove a token if the total BEFORE it already reached top_p
                    # (keeps the token that crosses top_p, and always keeps the top token)
                    remove = (cumulative - sorted_probs) >= top_p
                    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                    sorted_probs = sorted_probs / sorted_probs.sum(
                        dim=-1, keepdim=True
                    )  # renormalize
                    # sample a sorted position, then look up which real token it is
                    choice = torch.multinomial(sorted_probs, num_samples=1)  # (B, 1)
                    new_token_id = torch.gather(
                        sorted_ids, dim=-1, index=choice
                    )  # (B, 1)
                else:
                    # sample instead of argmax: a weighted die roll, so likely tokens
                    # usually win but not always, which breaks greedy's repetition loops
                    new_token_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # attach the new token to the end of each sequence
            token_ids = torch.cat((token_ids, new_token_id), dim=-1)
            # new shape is (B, T + 1)

    # done generating, hit max_new_tokens
    # decode every sequence in the batch back to text and return the list
    B = token_ids.shape[0]
    token_ids = token_ids.tolist()
    generated_texts = [ENCODING.decode(token_ids[i]) for i in range(B)]
    return generated_texts


if __name__ == "__main__":
    # builds the model and loads hf's weights, uses the GPT2Config defaults
    import argparse

    parser = argparse.ArgumentParser(description="Generate text with GPT-2")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    gpt2 = GPT2.from_pretrained().to(args.device)
    text = (
        "hello world isn't a beautiful day today? "
        "i love studying linear algebra and building models from scratch"
    )
    generated_text = generate(
        gpt2, text, temperature=0.7, max_new_tokens=10, cache=True
    )
    print(generated_text)
