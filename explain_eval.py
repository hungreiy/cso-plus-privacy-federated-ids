"""Explanation fidelity (removal test) and agreement with a reference model, for the attack logit.
Alerts = attack windows in the test set that the model flags. SHAP (GradientExplainer) attributes the logit to the
78 flow features x 15 flows (and the 32 KG dimensions). The top-k flow features (summed over the 15 flows) are replaced
by their benign-training mean and the drop in the logit is compared with the drop for k random features.
Fidelity ratio = drop(top-k SHAP) / drop(k random)  (> 1: SHAP picks features that matter more than chance).
Optionally reports the Spearman correlation of mean feature importance with a reference model.
Not yet run with your final model: check on the smoke data first (SHAP + LSTM can be fragile)."""
import argparse, json, os
import numpy as np, torch, shap
from scipy.stats import spearmanr
from model import CNNLSTMAE, N_FLOW, T
from evalutil import compute, load_prep

ap = argparse.ArgumentParser()
ap.add_argument('--prep', default='prepared'); ap.add_argument('--weights', required=True)
ap.add_argument('--dp_model', action='store_true'); ap.add_argument('--norm', default='group')
ap.add_argument('--ref_weights'); ap.add_argument('--ref_dp', action='store_true'); ap.add_argument('--ref_norm', default='group')
ap.add_argument('--n_alerts', type=int, default=200); ap.add_argument('--k', type=int, default=5)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
a = ap.parse_args()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
prep = load_prep(a.prep)
def load(path, dp, norm):
    m = CNNLSTMAE(norm=norm, dp=dp); m.load_state_dict(torch.load(path, map_location='cpu')); return m.to(dev).eval()

class Score(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, z): return self.m.logit(z[:, :, :N_FLOW], z[:, 0, N_FLOW:]).unsqueeze(1)
def pack(x, kg): return torch.cat([torch.as_tensor(np.asarray(x)).float(), torch.as_tensor(np.asarray(kg)).float().unsqueeze(1).expand(-1, T, -1)], 2)

def importance(model, x, kg, bg):
    with torch.backends.cudnn.flags(enabled=False):          # cuDNN LSTM cannot be differentiated in eval mode
        sv = shap.GradientExplainer(Score(model), bg).shap_values(pack(x, kg).to(dev))
    sv = np.asarray(sv[0] if isinstance(sv, list) else sv).reshape(len(x), T, -1)
    return np.abs(sv[:, :, :N_FLOW]).sum(axis=1)             # [n, 78] importance of the flow features

net = load(a.weights, a.dp_model, a.norm)
yv = np.asarray(prep['val_y']); _, v_lg = compute(net, prep['val_x'], prep['val_kg'], dev); thr = float(np.percentile(v_lg[yv == 0], 99.5))
_, t_lg = compute(net, prep['test_x'], prep['test_kg'], dev); y = np.asarray(prep['test_y'])
cand = np.where((t_lg > thr) & (y == 1))[0]; fallback = False
if len(cand) < 5:
    fallback = True; att = np.where(y == 1)[0]; cand = att[np.argsort(-t_lg[att])[:a.n_alerts]]
alerts = np.sort(rng.choice(cand, size=min(a.n_alerts, len(cand)), replace=False))
x = np.asarray(prep['test_x'])[alerts]; kg = np.asarray(prep['test_kg'])[alerts]
bi = np.sort(rng.choice(len(prep['train_x']), size=min(200, len(prep['train_x'])), replace=False))
bg = pack(np.asarray(prep['train_x'])[bi], np.asarray(prep['train_kg'])[bi]).to(dev)
imp = importance(net, x, kg, bg)
ben = np.where(np.asarray(prep['train_y']) == 0)[0][:20000]
mean_feat = np.asarray(prep['train_x'][ben]).mean(axis=(0, 1)).astype(np.float32)

def removal_drop(mask):
    xr = (x * (1 - mask[:, None, :]) + mean_feat[None, None, :] * mask[:, None, :]).astype(np.float32)
    _, lg_new = compute(net, xr, kg, dev)
    return t_lg[alerts] - lg_new, lg_new <= thr
top = np.zeros_like(imp, dtype=np.float32)
for i in range(len(imp)): top[i, np.argsort(-imp[i])[:a.k]] = 1
drop_top, flip_top = removal_drop(top)
drops_r, flips_r = [], []
for _ in range(5):
    rm = np.zeros_like(top)
    for i in range(len(top)): rm[i, rng.choice(N_FLOW, size=a.k, replace=False)] = 1
    d, f = removal_drop(rm); drops_r.append(d.mean()); flips_r.append(f.mean())
res = dict(n_alerts=int(len(alerts)), k=a.k, fallback_top_scored_used=fallback, mean_logit_drop_topk_shap=float(drop_top.mean()),
           mean_logit_drop_random=float(np.mean(drops_r)), fidelity_ratio=float(drop_top.mean() / max(1e-12, np.mean(drops_r))),
           alerts_flipped_topk_shap=float(flip_top.mean()), alerts_flipped_random=float(np.mean(flips_r)))
if a.ref_weights:
    ref = load(a.ref_weights, a.ref_dp, a.ref_norm)
    res['spearman_importance_vs_reference'] = float(spearmanr(imp.mean(0), importance(ref, x, kg, bg).mean(0)).correlation)
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); json.dump(res, open(a.out, 'w'), indent=2); print(res)
