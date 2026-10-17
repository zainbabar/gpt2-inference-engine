from model import GPT2
import torch
import tiktoken

ENCODING = tiktoken.get_encoding("gpt2")  # same tokenizer gpt2 was trained w/
CONTEXT_SIZE = 1024  # max tokens the model can see at once (wpe only has 1024 rows)


def generate(
    model: GPT2,
    text: str,
    temperature: int = 0,
    top_k=None,
    max_new_tokens: int = CONTEXT_SIZE,
):
    # text -> token ids -> tensor, unsqueeze(0) adds the batch dim
    # so shape is (B, T) w/ B = 1
    token_ids = (torch.tensor(ENCODING.encode(text), dtype=torch.long)).unsqueeze(0)
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

            # scores (logits) for the next token at every position
            logits = model(token_ids)  # shape (B, T, vocab_size)

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
                    # anything >= the last top k value is valid, so cutuoff
                    cutoffs = values[..., -1:] # shape (B, 1), slice to keep the dim
                    mask = next_token_scores < cutoffs # true means fill
                    next_token_scores = torch.masked_fill(next_token_scores, mask, float("-inf"))

                # softmax over the vocab: scores -> probabilities that sum to 1
                probs = torch.softmax(next_token_scores, dim=-1)  # (B, vocab_size)
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
    gpt2 = GPT2.from_pretrained()
    text = (
        "hello world isn't a beautiful day today? "
        "i love studying linear algebra and building models from scratch"
    )
    generated_text = generate(gpt2, text, temperature=0.7, max_new_tokens=10)
    print(generated_text)
