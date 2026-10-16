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
    layer_norm_epsilon = 1e-05  # tiny value added to avoid / 0 if var/sd = 0


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
            mask = ~mask  # invert it, want true on upper part
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
        outputs = inputs + self.attn(self.ln_1(inputs))
        outputs = outputs + self.mlp(self.ln_2(outputs))
        return outputs


# before this we take text to token ids w/ tiktoken
# we need to take this token id and look up its into a LEARNED vector
# also need to take the positon of the token in the full sentence and add it also learned
class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        # token table, match token id to token
        # vocab size rows i.e. how many unique tokens model knows, each token vector size n_embd
        self.wte = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.n_embd
        )
        self.wpe = nn.Embedding(
            num_embeddings=config.n_positions, embedding_dim=config.n_embd
        )

        # h for hidden, the 12 decorder blocks, list comprehension to do 12 calls, 12 unique blocks
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # final layernorm
        self.ln_f = nn.LayerNorm(config.n_embd)
        # language modelling head, takes each tokens vector and turns it into a
        # prediction scores for every token in vocab to be the next token
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
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
        positons = torch.arange(T)  # postions 1  to T-1 for positon lookup
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

    # loader, load the pretrained weights from hf
    @classmethod  # class method, works on the actual class not an obj
    def from_pretrained(cls):
        from transformers import GPT2LMHeadModel

        # build the model here, and load the weights
        # so when we create a model in main it comes w/ weights
        gpt2 = cls(GPT2Config)  # cls is the class, gpt2 is an object/ instance
        hf = GPT2LMHeadModel.from_pretrained("gpt2")
        hf_sd = hf.state_dict()
        sd = gpt2.state_dict()  # our gpt2 class state dict
        # need to set our weights, biases, etc to the hf ones
        # need to transpose the in out weights to out in
        # for our nn.Linear
        # also need to parse the transformer. out since we dont use

        # the thing we need to transpose for our version to work
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
                # check does this key need a transform by checking if suffix in transposed
                # check just suffix b/c we have the head num before that
                if key.endswith(to_transpose):
                    checkpoint = torch.transpose(hf_sd[key], -2, -1)
                else:
                    checkpoint = hf_sd[key]

                # parse the key for lookup in our table
                parsed_key = key.removeprefix("transformer.")
                # check shape matches our tensor
                if checkpoint.shape != sd[parsed_key].shape:
                    raise RuntimeError(f"{key} shape does not match")

                # set ours if everything is fine
                sd[parsed_key].copy_(checkpoint)  # copy the checkpoint into thats value
                loaded.add(parsed_key)

        if set(sd.keys()) != loaded:
            for key in set(sd.keys()):
                if key not in loaded:
                    print(key, " ")
            raise RuntimeError("not all keys were loaded")

        # everything is good, just return
        return gpt2.eval()


if __name__ == "__main__":
    from transformers import GPT2LMHeadModel

    model = GPT2.from_pretrained()
    hf_sd = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()

    # plain copy
    print("wte:", torch.equal(model.wte.weight, hf_sd["transformer.wte.weight"]))
    # transposed copies (c_proj is the square one)
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
