from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class GPT2Config:
    vocab_size: int = 50257  # num of tokens in vocabulary
    n_positions: int = 1024  # num of tokens it can hande at once (context)
    n_layer: int = 12  # how many attention layers
    n_head: int = 12  # how many attention heads
    n_embd: int = 768  # how many numbers in token vector


# our mlp class for the FNN between attention layers
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)  # expand layer
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)  # shrink
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, inputs):  # inputs is outputs of prev attention layer
        return self.c_proj(self.activation(self.c_fc(inputs)))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(
            config.n_embd, config.n_embd * 3
        )  # build Q, K, V from inputs
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  # W_0, mix the heads
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head  # 768 / 12 = 64 per head

    def forward(self, inputs):
        # note that we are working w/ batch size B, every tensor keeps it as dim 0
        # inputs have shape (B, T, 768) w/ T = # of tokens
        # Q, K, V are packed side by side into 1 matrix, wanna split
        # after matrix mult first 768 are Q, etc..., shape (B, T, 768)

        qkv = self.c_attn(inputs)  # project inputs to Q K V, shape (B, T, 2304)
        Q, K, V = torch.split(
            qkv, self.n_embd, dim=-1
        )  # split up into thirds shape (B, T, 768), dim=-1 splits on last dim

        # we have 12 heads here, so basically columns 0-63 head 1, 64-127 2, etc.
        # want to compute the individual score matrix, causal mask, softmax not all tg
        # so split up the heads

        # for multihead attention loop version first
        multihead_attn = []  # empty list of tensors
        for head in range(self.n_head):
            # take all tokesn and the respective columns of Q, K, V corresponding to that head
            start, end = head * self.head_dim, (head + 1) * self.head_dim
            W_Q = Q[..., start:end]  # inclusive:exclusive
            W_K = K[..., start:end]
            W_V = V[..., start:end]
            d_k = torch.tensor(self.head_dim)
            # shape here is (B, T, 64)

            # each token has attn score w/ every other token
            scores = torch.matmul(W_Q, torch.transpose(W_K, -2, -1))  # shape (B, T, T)
            scale = 1 / torch.sqrt(d_k)
            scaled_scores = scale * scores

            # now need to apply causal mask, set each score above the diagonal to -inf
            # so the token cant look at future, and -inf softmaxes to to
            mask = torch.tril(torch.ones(scaled_scores.shape, dtype=torch.bool))  
            # shape should be the same as the scaled scores we are trying to mask
            mask = ~mask # invert it, want true on upper part
            masked_scores = torch.masked_fill(scaled_scores, mask, float("-inf"))

            # softmax over the rows (dim = -1), turns scores row into weights that sum to 1
            # softmaxxing over dim -1 works on the COLUMNS, so moves along a row
            weights = torch.softmax(masked_scores, dim=-1)  # shape (B, T, T), same
            # take weighted sum of entire matrix... do this via matmul
            # only last 2 dim follow the actual matrix mult rule so (T, T) x (T, 64) = (T, 64)
            # (B, T, T) x (B, T, 64) = (B, T, 64)
            head_attn = torch.matmul(
                weights, W_V
            )  # each new row is linear comb of old w/ weights
            # shape is now (B, T, 64) for this head

            multihead_attn.append(head_attn)

        multihead_attn = torch.cat(
            multihead_attn, dim=-1
        )  # cat along the last dim of 64, to get 768 back
        # atp mh attn is back to the orignal shape/size, now need to mix the heads
        attn = self.c_proj(multihead_attn)

        return attn

