import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 23,
        max_length: int = 129,
        context: int = 3,
        embedding_size: int = 32,
        num_hidden: int = 256,
    ):
        super(MLPLanguageModel, self).__init__()

        self.max_length = max_length
        self.context = context

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_size)

        self.l_1 = nn.Linear(in_features=context * embedding_size,
                             out_features=num_hidden)
        self.activation = nn.Tanh()

        self.l_out = nn.Linear(in_features=num_hidden,
                               out_features=vocab_size)

    def forward(self, x, y=None):
        batch_size, length = x.shape

        # Pad on the left so the first token also has a full context
        padding = x.new_zeros(batch_size, self.context - 1)
        x = torch.cat([padding, x], dim=1)

        # The embeddings of the previous `context` tokens, concatenated
        embedded = self.embedding(x)
        x = torch.cat([embedded[:, i:i + length] for i in range(self.context)], dim=-1)

        x = self.activation(self.l_1(x))

        return self.l_out(x)

    def loss(self, x, x_shifted, y=None, ignore_index: int = -1):
        logits = self(x, y)

        return F.cross_entropy(
            logits.transpose(1, 2),
            x_shifted,
            ignore_index=ignore_index,
        )

    @torch.no_grad()
    def generate(
        self,
        n,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        y=None,
        temperature=1.0,
        device=None,
    ):
        self.eval()
        device = device or next(self.parameters()).device

        x = torch.full((n, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)

        for i in range(self.max_length - 1):
            logits = self(x, y)[:, -1, :] / temperature
            logits[:, bos_id] = -math.inf
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1).squeeze(1)

            nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
            x = torch.cat([x, nxt[:, None]], dim=1)

            done = done | (nxt == eos_id)
            if done.all():
                break

        return x
