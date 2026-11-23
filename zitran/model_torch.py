"""ZITRAN v5 — ~50M-parameter Transformer (PyTorch, CPU-trainable).
Same defensive taxonomy / tokenizer as v1-v4; scaled architecture:
token+position embeddings, 15 pre-norm encoder layers (d=512, 8 heads,
ff=2048), MLP head, multi-task outputs (10-class + binary threat).
Exact parameter count is computed and asserted at build time.
"""
import torch
import torch.nn as nn

class ZitranV5(nn.Module):
    def __init__(self, vocab=4000, n_classes=10, max_len=48, d=512, heads=8, ff=2048, layers=15, dropout=0.1):
        super().__init__()
        self.d, self.max_len = d, max_len
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.Wc = nn.Linear(d, n_classes)
        self.Wb = nn.Linear(d, 2)

    def forward(self, x):
        T = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))[None, :, :]
        mask = (x == 0)
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=mask)
        cls = self.lnf(h[:, 0, :])
        hidden = self.head_mlp(cls)
        return self.Wc(hidden), self.Wb(hidden)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

def build_50m(vocab=4000, max_len=48):
    m = ZitranV5(vocab=vocab, max_len=max_len)
    n = m.num_params()
    assert 45_000_000 <= n <= 55_000_000, f"param count off target: {n:,}"
    return m
