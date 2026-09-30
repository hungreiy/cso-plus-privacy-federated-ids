"""Hybrid CNN-LSTM autoencoder-classifier with 3-level KG fusion (thesis Section 3.6).

Input : x  [B, 15, 78]  scaled flow features,  kg [B, 32] Node2Vec window vector
Early fusion : kg tiled over time + 2 zero-padding features -> [B, 15, 112]
Mid fusion   : kg concatenated to the encoder output before the latent layer
Late fusion  : kg tiled and concatenated before the final reconstruction layer
Outputs      : (rec [B,15,112], logit [B])  -- rec = reconstruction, logit = attack score from the latent code
Loss         : mean_i( rec_weight * [y_i = benign] * MSE_i + BCE_i )   (a plain mean of per-example terms,
               which is what DP-SGD / Opacus requires)

DP notes:
  * BatchNorm is NOT valid under DP-SGD (batch statistics mix examples) -> use norm='group'.
  * torch.nn.LSTM is not supported by Opacus -> opacus.layers.DPLSTM is used when dp=True.
"""
import torch, torch.nn as nn
import torch.nn.functional as F

N_FLOW, KG_DIM, PAD, T = 78, 32, 2, 15
IN_CH = N_FLOW + KG_DIM + PAD   # 112

def make_norm(kind, ch):
    if kind == 'batch': return nn.BatchNorm1d(ch)
    if kind == 'group': return nn.GroupNorm(8, ch)
    return nn.Identity()

def make_bilstm(in_size, hidden, dp):
    if dp:
        from opacus.layers import DPLSTM
        return DPLSTM(in_size, hidden, batch_first=True, bidirectional=True)
    return nn.LSTM(in_size, hidden, batch_first=True, bidirectional=True)

class CNNLSTMAE(nn.Module):
    def __init__(self, norm='group', dp=False, latent=64, hidden=128):
        super().__init__()
        self.c1 = nn.Conv1d(IN_CH, 64, 3, padding=1); self.n1 = make_norm(norm, 64)
        self.c2 = nn.Conv1d(64, 128, 3, padding=1);   self.n2 = make_norm(norm, 128)
        self.pool = nn.MaxPool1d(2)
        self.enc_lstm = make_bilstm(128, hidden, dp)
        self.to_latent = nn.Linear(2 * hidden + KG_DIM, latent)      # mid fusion here
        self.from_latent = nn.Linear(latent, 2 * hidden)
        self.dec_lstm = make_bilstm(2 * hidden, hidden, dp)
        self.up1 = nn.Upsample(size=7);  self.d1 = nn.Conv1d(2 * hidden, 128, 3, padding=1); self.m1 = make_norm(norm, 128)
        self.up2 = nn.Upsample(size=T);  self.d2 = nn.Conv1d(128, 64, 3, padding=1);         self.m2 = make_norm(norm, 64)
        self.out = nn.Conv1d(64 + KG_DIM, IN_CH, 1)                    # late fusion here
        self.head = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 1))   # attack classifier on the latent code
        self.act = nn.ReLU()
        self.dp = dp

    def forward(self, x, kg):
        B = x.shape[0]
        kg_t = kg.unsqueeze(1).expand(B, T, KG_DIM)
        pad = x.new_zeros(B, T, PAD)
        h = torch.cat([x, kg_t, pad], dim=2).permute(0, 2, 1)          # early fusion [B,112,15]
        h = self.pool(self.act(self.n1(self.c1(h))))                  # [B,64,7]
        h = self.pool(self.act(self.n2(self.c2(h))))                  # [B,128,3]
        o, _ = self.enc_lstm(h.permute(0, 2, 1))                       # [B,3,256]
        z = self.to_latent(torch.cat([o[:, -1], kg], dim=1))           # [B,64]
        logit = self.head(z).squeeze(1)                                # [B]
        d = self.from_latent(z).unsqueeze(1).expand(B, h.shape[2], -1).contiguous()
        d, _ = self.dec_lstm(d)                                        # [B,3,256]
        d = d.permute(0, 2, 1)
        d = self.act(self.m1(self.d1(self.up1(d))))                   # [B,128,7]
        d = self.act(self.m2(self.d2(self.up2(d))))                   # [B,64,15]
        d = torch.cat([d, kg.unsqueeze(2).expand(B, KG_DIM, T)], dim=1)
        rec = torch.sigmoid(self.out(d)).permute(0, 2, 1)              # [B,15,112]
        return rec, logit

    def rec_error(self, x, kg):
        rec, _ = self.forward(x, kg)
        return ((rec[:, :, :N_FLOW] - x) ** 2).mean(dim=(1, 2))        # flow dims only

    def logit(self, x, kg):
        return self.forward(x, kg)[1]

def combined_loss(rec, logit, x, y, rec_weight=10.0, cls_weight=1.0):
    """Plain mean over the batch of per-example terms (required for DP-SGD)."""
    mse_i = ((rec[:, :, :N_FLOW] - x) ** 2).mean(dim=(1, 2))
    ben = (y == 0).to(mse_i.dtype)
    bce_i = F.binary_cross_entropy_with_logits(logit, y.to(logit.dtype), reduction='none')
    return (rec_weight * ben * mse_i + cls_weight * bce_i).mean()

def get_params(model):
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]

def set_params(model, arrays):
    sd = model.state_dict()
    model.load_state_dict({k: torch.tensor(a) for k, a in zip(sd.keys(), arrays)}, strict=True)
