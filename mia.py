"""Membership-inference test (threshold attack on the per-window training loss).
Members = training windows the model was trained on; non-members = held-out validation windows (same distribution).
The attacker knows the true label. Reports attack AUC and balanced accuracy (0.5 = no leakage)."""
import argparse, json, os
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from model import CNNLSTMAE
from evalutil import compute, load_prep

ap = argparse.ArgumentParser()
ap.add_argument('--prep', default='prepared'); ap.add_argument('--weights', required=True)
ap.add_argument('--norm', default='group', choices=['batch', 'group', 'none']); ap.add_argument('--dp_model', action='store_true')
ap.add_argument('--rec_weight', type=float, default=10.0); ap.add_argument('--n', type=int, default=5000)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
a = ap.parse_args()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rng = np.random.default_rng(a.seed)
prep = load_prep(a.prep)
net = CNNLSTMAE(norm=a.norm, dp=a.dp_model); net.load_state_dict(torch.load(a.weights, map_location='cpu')); net.to(dev).eval()

def per_window_loss(x, kg, y):
    rec, lg = compute(net, x, kg, dev); y = np.asarray(y).astype(np.float64)
    return a.rec_weight * (y == 0) * rec + (np.logaddexp(0, lg.astype(np.float64)) - y * lg)

k = min(a.n, len(prep['train_x']), len(prep['val_x']))
im = np.sort(rng.choice(len(prep['train_x']), size=k, replace=False)); inn = np.sort(rng.choice(len(prep['val_x']), size=k, replace=False))
s_m = per_window_loss(np.asarray(prep['train_x'])[im], np.asarray(prep['train_kg'])[im], np.asarray(prep['train_y'])[im])
s_n = per_window_loss(np.asarray(prep['val_x'])[inn], np.asarray(prep['val_kg'])[inn], np.asarray(prep['val_y'])[inn])
y = np.r_[np.ones(k), np.zeros(k)]; sc = -np.r_[s_m, s_n]           # lower loss -> more likely a member
auc = float(roc_auc_score(y, sc))
h = k // 2                                                            # threshold chosen on one half, tested on the other
idx = np.r_[np.arange(h), k + np.arange(h)]; idt = np.r_[np.arange(h, k), k + np.arange(h, k)]
cand = np.unique(sc[idx])[::max(1, len(np.unique(sc[idx])) // 500)]
thr = cand[int(np.argmax([((sc[idx] >= t) == y[idx]).mean() for t in cand]))]
acc = float(((sc[idt] >= thr) == y[idt]).mean())
res = dict(mia_auc=auc, mia_accuracy=acc, n_members=int(k), n_nonmembers=int(k), weights=a.weights)
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); json.dump(res, open(a.out, 'w'), indent=2); print(res)
