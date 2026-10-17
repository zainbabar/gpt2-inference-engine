from dataclasses import dataclass
from typing import Self

import torch
import torch.nn as nn


@dataclass
class GPT2Config:
    vocab_size: int = 50257  # num of unique tokens the model knows
    n_positions: int = 1024  # max num of tokens it can handle at once (context)
    n_layer: int = 12  # how many decoder blocks stacked on top of each other
    n_head: int = 12  # how many attention heads per block
    n_embd: int = 768  # how many numbers in each token vector
    layer_norm_epsilon: float = 1e-05  # tiny value added to variance so we never / 0


# the feed forward net (FFN) inside each block, runs right after attention
# attention mixes info between tokens, this processes each token on its own
class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)  # expand 768 -> 3072
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)  # shrink to 768
        self.activation = nn.GELU(approximate="tanh")  # gpt2 uses the tanh approx

    # inputs is the layernormed output of attention
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.activation(self.c_fc(inputs)))


class Attention(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        # one big linear that builds Q, K, V all at once, 768 -> 3 * 768
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  # W_O, mix the heads
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head  # 768 / 12 = 64 per head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # B = batch size, T = # of tokens, every tensor keeps B as dim 0
        # inputs have shape (B, T, 768)
        # c_attn packs Q, K, V side by side along the last dim
        # first 768 cols are Q, next 768 are K, last 768 are V

        qkv = self.c_attn(inputs)  # shape (B, T, 2304)
        # split into chunks of 768 along the last dim, each is (B, T, 768)
        Q, K, V = torch.split(qkv, self.n_embd, dim=-1)

        # 12 heads, each owns a 64 col slice: cols 0-63 are head 0, 64-127 head 1...
        # each head does its own scores, causal mask, and softmax
        # so we split up the heads and do them one at a time

        # loop version of multihead attention first
        multihead_attn = []  # each head's output, cat them at the end
        for head in range(self.n_head):
            # all tokens, just this head's columns of Q, K, V
            # (these are slices of Q/K/V, not weight matrices)
            start, end = head * self.head_dim, (head + 1) * self.head_dim
            W_Q = Q[..., start:end]  # inclusive:exclusive
            W_K = K[..., start:end]
            W_V = V[..., start:end]
            d_k = torch.tensor(self.head_dim)
            # shape here is (B, T, 64)

            # each token gets a score w/ every other token, Q @ K^T
            scores = torch.matmul(W_Q, torch.transpose(W_K, -2, -1))  # shape (B, T, T)
            # scale by 1 / sqrt(d_k), otherwise dot products grow w/ the dim
            # and softmax gets super peaky
            scale = 1 / torch.sqrt(d_k)
            scaled_scores = scale * scores

            # causal mask, set every score above the diagonal to -inf
            # so a token cant look at future tokens, and -inf softmaxes to 0
            # tril gives True on and below the diagonal, same shape as the scores
            mask = torch.tril(torch.ones(scaled_scores.shape, dtype=torch.bool))
            # invert it, masked_fill wants True where we fill (the upper part)
            mask = ~mask
            masked_scores = torch.masked_fill(scaled_scores, mask, float("-inf"))

            # softmax along dim -1 i.e. across each row, so each token's weights
            # over all the tokens it can see sum to 1
            weights = torch.softmax(masked_scores, dim=-1)  # shape (B, T, T), same
            # weighted sum of the value rows, do it all at once via matmul
            # matmul only uses the last 2 dims, B just rides along
            # (B, T, T) x (B, T, 64) = (B, T, 64)
            # each output row is a linear comb of the V rows w/ that token's weights
            head_attn = torch.matmul(weights, W_V)

            multihead_attn.append(head_attn)

        # cat the 12 (B, T, 64) heads along the last dim to get (B, T, 768) back
        multihead_attn = torch.cat(multihead_attn, dim=-1)
        # atp each head's info just sits in its own cols, c_proj mixes them together
        attn = self.c_proj(multihead_attn)

        return attn


# one decoder block: masked self attention + MLP, w/ a layernorm before each
# gpt2 is pre-norm, i.e. ln goes before each piece not after
class Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        # the pieces we run inputs thru
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs + ... is the residual (skip) connection, each piece just adds
        # its update on top of what was already there
        outputs = inputs + self.attn(self.ln_1(inputs))
        outputs = outputs + self.mlp(self.ln_2(outputs))
        return outputs


# before this we take text to token ids w/ tiktoken
# each token id gets looked up into a LEARNED vector
# each position in the sequence also gets its own LEARNED vector, added on top
class GPT2(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        # token table: token id -> vector
        # vocab_size rows (one per token the model knows), each row n_embd long
        self.wte = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.n_embd
        )
        # position table: position -> vector, one row per position up to n_positions
        self.wpe = nn.Embedding(
            num_embeddings=config.n_positions, embedding_dim=config.n_embd
        )

        # h = the stack of n_layer decoder blocks (named h to match hf's keys)
        # list comprehension so each block is its own object w/ its own weights
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # final layernorm, after the last block
        self.ln_f = nn.LayerNorm(config.n_embd)
        # language modelling head, takes each token's vector and turns it into a
        # score (logit) for every token in the vocab being the next token
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying, wte and lm_head do the same job in opposite directions
        # so 1 table is used both ways
        # wte: token -> vector, lm_head: vector -> token
        self.lm_head.weight = self.wte.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # build the embeddings first from tokens
        # token_ids shape is (B, T)
        E = self.wte(token_ids)
        # E shape is (B, T, 768), every token is now a 768 long vector
        # wte looks up by token id, wpe looks up by token POSITION
        # i.e. the token at position i gets wte[token_id] + wpe[i]
        T = token_ids.shape[-1]
        positons = torch.arange(T)  # positions 0 to T-1 for the lookup
        P = self.wpe(positons)
        # P has shape (T, 768)
        embeddings = E + P
        # works b/c pytorch broadcasts P across B
        # position 0 means the same thing for every sequence in the batch

        # pass thru all blocks in order
        attended_embeddings = embeddings
        for block in self.h:
            attended_embeddings = block(attended_embeddings)

        # normalize and run thru lm head to get scores per vocab token
        # shape (B, T, vocab_size)
        scores = self.lm_head(self.ln_f(attended_embeddings))
        return scores

    # loader, load the pretrained weights from hf
    @classmethod  # called on the class itself (GPT2.from_pretrained()), cls is GPT2
    def from_pretrained(cls) -> Self:
        from transformers import GPT2LMHeadModel

        # build the model here and load the weights
        # so when we create a model in main it already has weights
        gpt2 = cls(GPT2Config)  # cls is the class, gpt2 is an object/instance
        hf = GPT2LMHeadModel.from_pretrained("gpt2")
        hf_sd = hf.state_dict()
        sd = gpt2.state_dict()  # our gpt2's state dict
        # need to set our weights, biases, etc to the hf ones
        # hf's gpt2 stores these weights as (in, out) (it uses Conv1D not Linear)
        # but nn.Linear stores them as (out, in), so those need a transpose
        # also hf keys start w/ "transformer." and ours dont, so strip that off

        # the weights we need to transpose for our version to work
        to_transpose = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )

        # track that we have loaded every key in our model
        loaded = set()
        with torch.no_grad():
            for key in hf_sd:
                # does this key need a transpose, check if it ends w/ one of those
                # just the suffix b/c the full key has the layer num before it (h.0.)
                if key.endswith(to_transpose):
                    checkpoint = torch.transpose(hf_sd[key], -2, -1)
                else:
                    checkpoint = hf_sd[key]

                # strip the prefix for lookup in our state dict
                parsed_key = key.removeprefix("transformer.")
                # check shape matches our tensor
                if checkpoint.shape != sd[parsed_key].shape:
                    raise RuntimeError(f"{key} shape does not match")

                # set ours if everything is fine
                # state dict tensors share memory w/ the model's params,
                # so copying in place writes straight into the model
                sd[parsed_key].copy_(checkpoint)
                loaded.add(parsed_key)

        # every key in ours should have been filled, print any we missed
        if set(sd.keys()) != loaded:
            for key in set(sd.keys()):
                if key not in loaded:
                    print(key, " ")
            raise RuntimeError("not all keys were loaded")

        # everything is good, switch to eval mode (only doing inference) and return
        return gpt2.eval()


if __name__ == "__main__":
    from transformers import GPT2LMHeadModel

    model = GPT2.from_pretrained()
    hf_sd = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()

    # plain copy
    print("wte:", torch.equal(model.wte.weight, hf_sd["transformer.wte.weight"]))
    # transposed copies, compare values not just shapes
    # attn c_proj is square (768x768) so the shape check cant catch a missed transpose
    print(
        "h.0 c_attn:",
        torch.equal(
            model.h[0].attn.c_attn.weight,
            hf_sd["transformer.h.0.attn.c_attn.weight"].t(),
        ),
    )
    print(
        "h.5 attn c_proj:",
        torch.equal(
            model.h[5].attn.c_proj.weight,
            hf_sd["transformer.h.5.attn.c_proj.weight"].t(),
        ),
    )
    # tying survived loading
    print("tied:", model.lm_head.weight is model.wte.weight)
