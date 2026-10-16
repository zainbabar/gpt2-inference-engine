from dataclasses import dataclass
import torch
import torch.nn as nn
import tiktoken


@dataclass
class GPT2Config:
    vocab_size: int = 50257  # num of tokens in vocabulary
    n_positions: int = 1024  # num of tokens it can hande at once (context)
    n_layer: int = 12  # how many attention layers
    n_head: int = 12  # how many attention heads
    n_embd: int = 768  # how many numbers in token vector
    layer_norm_epsilon = 1e-05 # tiny value added to avoid / 0 if var/sd = 0


# our mlp class for the FNN between attention layers
class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)  # expand layer
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)  # shrink
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, inputs):  # inputs is outputs of prev attention layer
        return self.c_proj(self.activation(self.c_fc(inputs)))


class Attention(nn.Module):
    def __init__(self, config: GPT2Config):
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

# the entire decoder block combining masked self attention, ffn, and layer normalization between
class Block(nn.Module): 
    def __init__(self, config: GPT2Config):
        super().__init__()
        # the pieces we run inputs thru
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, inputs):
        outputs = self.ln_1(inputs)
        outputs = self.attn(outputs)
        outputs = self.ln_2(outputs)
        outputs = self.mlp(outputs)
        return outputs
    
# before this we take text to token ids w/ tiktoken
# we need to take this token id and look up its into a LEARNED vector
# also need to take the positon of the token in the full sentence and add it also learned
class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        # token table, match token id to token
        # vocab size rows i.e. how many unique tokens model knows, each token vector size n_embd
        self.wte = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embd)
        self.wpe = nn.Embedding(num_embeddings=config.n_positions, embedding_dim=config.n_embd)

        # h for hidden, the 12 decorder blocks, list comprehension to do 12 calls, 12 unique blocks
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # final layernorm
        self.ln_f = nn.LayerNorm(config.n_embd)
        # language modelling head, takes each tokens vector and turns it into a 
        # prediction scores for every token in vocab to be the next token
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        # weight typing, b/c wte and lm head both do same job in opposite dir, 1 table used both ways
        # wte: token -> vector, lm_head: vector -> token
        self.lm_head.weight = self.wte.weight 
    
    def forward(self, token_ids):
        # build the embeddings first from tokens
        # token_ids shape is (B, T)
        E = self.wte(token_ids) 
        # embeddings shape is (B, T, 768) now, every token is now a 768 long vector
        # wte is straightforward, but wpe looks up based on token POSITION
        # i.e. token 1 gets token id wte, and then pte[1]
        T = token_ids.shape[-1]
        positons = torch.arange(T) # postions 1  to T-1 for positon lookup
        P = self.wpe(positons)
        # P has shape (T, 768)
        embeddings = E + P 
        # works, b/c pytorch broadcasts it, positon 0 means the same thing for every batch (sentence)

        # pass thru all blocks
        attended_embeddings = embeddings
        for block in self.h:
            attended_embeddings = block(attended_embeddings)

        # normalize and run thru lm head to get scores per vocab token
        scores = self.lm_head(self.ln_f(attended_embeddings))
        return scores
       

if __name__ == "__main__":
    cfg = GPT2Config()
    block = Block(cfg)
    for shape in [(1, 1, 768), (1, 5, 768), (2, 7, 768)]:
        out = block(torch.randn(*shape))
        assert tuple(out.shape) == shape, f"{shape} -> {tuple(out.shape)}"
    print("block shapes ok")
    print(list(block.state_dict().keys()))

    gpt2 = GPT2(cfg)
    token_ids = torch.randint(0, 50257, (1, 5))
    scores = gpt2(token_ids)
    print(scores.shape)
    print(gpt2.lm_head.weight is gpt2.wte.weight)
