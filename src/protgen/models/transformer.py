import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T, device=None):
    mask = torch.tril(torch.ones(T, T, device=device))
    return torch.zeros(T, T, device=device).masked_fill(mask == 0, -math.inf)

# for the protain sequences the position is important, so we need to add positional encoding 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)

        # 1 / 10000^(2i / d_model) = exp(-2i * log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 23,
        max_length: int = 129, # from the test file thats the max length of a example of a protein sequence
        dim_model: int = 128, #64, 128, 254
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int | None = None,
    ):
        super().__init__()

        self.max_length = max_length
        self.num_classes = num_classes
        self.d_model = dim_model

        self.token_emb = nn.Embedding(vocab_size, dim_model)
        self.pos_enc = PositionalEncoding(dim_model, dropout=dropout, max_len=max_length)

        self.label_emb = (
            nn.Embedding(num_classes + 1, dim_model)
            if num_classes is not None
            else None
        )

        layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim_model)
        self.head = nn.Linear(dim_model, vocab_size)

    # calls forward methods of the different layers, including the position encoding
    def forward(self, x, y = None):
        B, T = x.shape
        h = self.token_emb(x) * math.sqrt(self.d_model)
        h = self.pos_enc(h)
        h = self.encoder(h, mask=causal_mask(T, device=x.device))

        return self.head(self.norm(h))


    # calculates loss between the predicted next letter and the actual one from the test sequence
    def loss(self, x, x_shifted, y=None, ignore_index: int = -1):
        logits = self(x, y)

        return F.cross_entropy(
            logits.transpose(1, 2), #turns (B, T, 23) into (B, 23, T) same as in the introduction notebook
            x_shifted,
            ignore_index=ignore_index,
        )

    @torch.no_grad()
    def generate(
        self,
        n,
        bos_id=1, #beggining of sequence
        eos_id=2, #emd of sequence
        pad_id=0,
        y=None,
        temperature=1.0,
        device=None,
    ):
        self.eval()
        device = device or next(self.parameters()).device # if device is not specified, use the device of the model parameters

        x = torch.full((n, 1), bos_id, dtype=torch.long, device=device)

        if y is not None and not torch.is_tensor(y):
            y = torch.full((n,), int(y), dtype=torch.long, device=device)
            
        done = torch.zeros(n, dtype=torch.bool, device=device)

        # Generate one token (letter) at a time 
        for i in range(self.max_length - 1):
            logits = self(x, y)[:, -1, :] / temperature
            logits[:, bos_id] = -math.inf
            probs = F.softmax(logits, dim=-1)
            # Sample the next lettter from the prob distribution of possible next letters 
            nxt = torch.multinomial(probs, num_samples=1).squeeze(1)

            nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
            x = torch.cat([x, nxt[:, None]], dim=1)

            done = done | (nxt == eos_id) # done if the next letter is the set end of the sequence token
            if done.all():
                break

        return x
