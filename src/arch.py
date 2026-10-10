"""Alternative architectures for the architecture baseline (Section 4.10.11). Same interface as CNNLSTMAE:
forward(x [B,T,78], kg [B,32]) -> (rec [B,T,112], logit [B]), the same early/mid/late fusion of the knowledge-graph vector,
and only layers whose per-example gradients Opacus supports (Linear, LayerNorm; attention built from Linear layers,
fixed sinusoidal positions instead of a trainable parameter)."""
import math, torch, torch.nn as nn
from model import KG_DIM, PAD, IN_CH
def _fuse(x, kg):
    B, Tw = x.shape[:2]
    return torch.cat([x, kg.unsqueeze(1).expand(B, Tw, KG_DIM), x.new_zeros(B, Tw, PAD)], dim=2)   # early fusion [B,T,112]
class MLPAE(nn.Module):
    def __init__(self, T=15, latent=64, hidden=512):
        super().__init__(); d = T * IN_CH; self.T = T
        self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 256), nn.ReLU())
        self.to_latent = nn.Linear(256 + KG_DIM, latent)                                   # mid fusion
        self.head = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 1))
        self.dec = nn.Sequential(nn.Linear(latent + KG_DIM, 256), nn.ReLU(), nn.Linear(256, hidden), nn.ReLU(), nn.Linear(hidden, d))   # late fusion
    def forward(self, x, kg):
        B, Tw = x.shape[:2]
        z = self.to_latent(torch.cat([self.enc(_fuse(x, kg).reshape(B, -1)), kg], dim=1))
        rec = torch.sigmoid(self.dec(torch.cat([z, kg], dim=1))).reshape(B, Tw, IN_CH)
        return rec, self.head(z).squeeze(1)
def _pe(T, d):
    pos = torch.arange(T).unsqueeze(1); div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div); return pe
class _Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h = h; self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
    def forward(self, x):
        B, T, D = x.shape; q, k, v = self.qkv(x).reshape(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(D // self.h), dim=-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, T, D))
class _Block(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.n1 = nn.LayerNorm(d); self.a = _Attn(d, h); self.n2 = nn.LayerNorm(d)
        self.f = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
    def forward(self, x): x = x + self.a(self.n1(x)); return x + self.f(self.n2(x))
class TransAE(nn.Module):
    def __init__(self, T=15, d=64, heads=4, layers=2, latent=64):
        super().__init__(); self.register_buffer('pe', _pe(T, d))
        self.inp = nn.Linear(IN_CH, d); self.enc = nn.ModuleList([_Block(d, heads) for _ in range(layers)]); self.ln = nn.LayerNorm(d)
        self.to_latent = nn.Linear(d + KG_DIM, latent)                                     # mid fusion
        self.head = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 1))
        self.from_latent = nn.Linear(latent, d); self.dec = nn.ModuleList([_Block(d, heads)])
        self.out = nn.Linear(d + KG_DIM, IN_CH)                                             # late fusion
    def forward(self, x, kg):
        B, Tw = x.shape[:2]; h = self.inp(_fuse(x, kg)) + self.pe[:Tw]
        for b in self.enc: h = b(h)
        z = self.to_latent(torch.cat([self.ln(h).mean(dim=1), kg], dim=1))
        d = self.from_latent(z).unsqueeze(1).expand(B, Tw, -1) + self.pe[:Tw]
        for b in self.dec: d = b(d)
        rec = torch.sigmoid(self.out(torch.cat([d, kg.unsqueeze(1).expand(B, Tw, KG_DIM)], dim=2)))
        return rec, self.head(z).squeeze(1)
