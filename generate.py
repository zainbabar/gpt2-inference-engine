from model import GPT2
import torch
import tiktoken

ENCODING = tiktoken.get_encoding("gpt2")
CONTEXT_SIZE = 1024


def generate(model: GPT2, text: str, max_new_tokens: int=CONTEXT_SIZE):
    # convert to tensor of ids, unsqueeze adds the batch dim for (B, T) shape w/ B = 1
    token_ids = (torch.tensor(ENCODING.encode(text), dtype=torch.long)).unsqueeze(0)
    for _ in range(max_new_tokens):
        # break out early if we still have tokens to go but hit our context size
        if token_ids.shape[-1] >= CONTEXT_SIZE:
            B = token_ids.shape[0]
            token_ids = token_ids.tolist()
            generated_texts = [  # build texts for all the batches
                ENCODING.decode(token_ids[i]) for i in range(B)
            ]
            return generated_texts

        # scores for next token at every pos 
        logits = model.forward(token_ids)  # shape (B, T, vocab_size)

        # take scores of last token for all batches
        next_token_scores = logits[:, -1, :] # shape (B, vocab_size)

        # take the highest score per batch, so shape (B, 1)
        new_token_id = next_token_scores.argmax(dim=-1, keepdim=True)

        # attach new token to the end of our current tensor of token ids
        token_ids = torch.cat((token_ids, new_token_id), dim=-1)
        # new shape is (B, T + 1) for the new token

    # finish generating tokens, hit our max new tokens limit
    # return list of texts
    B = token_ids.shape[0]
    token_ids = token_ids.tolist()
    generated_texts = [ENCODING.decode(token_ids[i]) for i in range(B)]
    return generated_texts


if __name__ == "__main__":
    gpt2 = GPT2.from_pretrained()  # builds model, gpt2config is known in model.py
    text = (
        "hello world isn't a beautiful day today? "
        "i love studying linear algebra and building models from scratch"
    )
    generated_text = generate(gpt2, text)
    print(generated_text)
