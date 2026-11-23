"""ZITRAN-Gen — decoder-only Transformer (causal LM) in PyTorch.
Same BPE vocab as the classifiers; generates text token-by-token with
temperature / top-k sampling. This is the generative brain ('like an LLM',
small and specialist — not a frontier model).
"""
import math, torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d, heads, ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d), nn.Dropout(dropout))
    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))

class ZitranGen(nn.Module):
    def __init__(self, vocab=4000, max_len=96, d=384, heads=6, ff=1536, layers=12, dropout=0.1):
        super().__init__()
        self.d, self.max_len, self.vocab = d, max_len, vocab
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff, dropout) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.MultiheadAttention):
            nn.init.normal_(m.in_proj_weight, std=0.02)
            nn.init.normal_(m.out_proj.weight, std=0.02)

    def forward(self, x):
        T = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))[None, :, :]
        for b in self.blocks: h = b(h)
        return self.head(self.lnf(h))

    def num_params(self): return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, prompt_ids, max_new=60, temperature=0.8, top_k=40, eos_id=None, seed=None):
        self.eval()
        if seed is not None: torch.manual_seed(seed)
        ids = list(prompt_ids)
        for _ in range(max_new):
            ctx = torch.tensor([ids[-self.max_len:]])
            logits = self(ctx)[0, -1] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[-1]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
            ids.append(nxt)
            if eos_id is not None and nxt == eos_id: break
        return ids
