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
        self.c_fc = nn.Linear(config.n_embd, 3072)  # expand layer
        self.c_proj = nn.Linear(3072, config.n_embd)  # shrink
        self.activation = nn.GELU(approximate="tanh")

        self.layers = nn.Sequential(  # combine into sequential
            self.c_fc, self.activation, self.c_proj
        )

    def forward(self, inputs):  # inputs is outputs of prev attention layer
        return self.layers(inputs)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(
            config.n_embd, config.n_embd * 3
        )  # build Q, K, V from inputs
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  # W_0, mix the heads
        self.n_head = config.n_head

    def forward(self, inputs):
        # note that we are working w/ batch size 1 so just ignore the B
        # inputs have shape (T, 768) w/ T = # of tokens
        # Q, K, V are packed side by side into 1 matrix, wanna split
        # after matrix mult first 768 are Q, etc..., shape (T, 768)

        # have to transpose it input are (in, out), nn.Linear expects (out, in)
        # since nn.Linear transposes before mulitiplying, so flips inputs
        qkv = self.c_attn(
            torch.transpose(inputs, -2, -1)
        )  # project inputs to Q K V, shape (T, 2304)
        Q, K, V = torch.split(qkv, 3, dim=1)  # split up into thirds shape (T, 768)

        # we have 12 heads here, so basically columns 0-63 head 1, 64-127 2, etc.
        # want to compute the individual score matrix, causal mask, softmax not all tg
        # so split up the heads

        # for multihead attention loop version first
        multihead_attn = torch.empty([])  # empty tensor
        for head in range(self.n_head):
            # take all tokesn and the respective columns of Q, K, V corresponding to that head
            W_Q = Q[:, head * 64 : head * 64 + 64]  # inclusive:exclusive
            W_K = K[:, head * 64 : head * 64 + 64]
            W_V = V[:, head * 64 : head * 64 + 64]
            d_k = 64
            # shape here is (T, 64)

            # each token has attn score w/ every other token
            scores = torch.matmul(W_Q, torch.transpose(W_K, -2, -1))  # shape (T, T)
            scale = torch.sqrt(d_k)
            scaled_scores = scale * scores

            # now need to apply causal mask, set each score above the diagonal to -inf
            # so the token cant look at future, and -inf softmaxes to to
            T = scaled_scores.shape[0]
            mask = torch.tril(
                torch.ones(T)
            )  # lower triang of 1s, works as boolean mask
            masked_scores = torch.masked_fill(scaled_scores, mask, float("-inf"))

            # softmax over the rows (dim = 0) giving (T,), weights, take weighted sum of values
            # weights broadcast i think
            head_attn = torch.softmax(masked_scores, dim=0) * W_V
            # shape is now (T, 64) for this head

            torch.cat((multihead_attn, head_attn), dim=1)

        # atp mh attn is back to the orignal shape, now need to mix the heads
        # transpose again!! and then apply W_O projection output thing
        attn = self.c_proj(torch.transpose(multihead_attn, -2, -1))


if __name__ == "__main__":
    cfg = GPT2Config()
    x = torch.randn(1, 5, cfg.n_embd)
    print(MLP(cfg)(x).shape)  # torch.Size([1, 5, 768])
