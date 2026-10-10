"""Stronger membership inference: offline Likelihood Ratio Attack (LiRA) with shadow models (Section 4.6).
Candidates: n training windows of the target (members) and n validation windows (non-members).
K shadow models (centralised, non-private, same architecture and loss as the target) are each trained on a random half of
the training and validation windows. For every candidate, the shadows that did NOT train on it give the distribution of its
log-odds of the true label when it is not a member; the attack scores how far the target's log-odds lie above that
distribution. Reports AUC and the true-positive rate at low false-positive rates (1 % and 0.1 %), and, for comparison,
the global threshold attack on the same log-odds.
  python mia_lira.py --prep /root/prepared --targets central:results/central_seed0.pt:0 fl_dp:results/fl_dp_seed0.pt:1 --out results_mia2/lira.json"""
import argparse, json, os, time
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
from model import CNNLSTMAE, combined_loss
from evalutil import compute, load_prep
ap = argparse.ArgumentParser()
ap.add_argument('--prep', default='prepared'); ap.add_argument('--targets', nargs='+', required=True, help='name:weights:dp(0/1)')
ap.add_argument('--shadows', type=int, default=8); ap.add_argument('--n', type=int, default=2000); ap.add_argument('--epochs', type=int, default=10)
ap.add_argument('--batch', type=int, default=256); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
prep = load_prep(a.prep); Ntr, Nv = len(prep['train_x']), len(prep['val_x']); n = min(a.n, Ntr, Nv)
mem = np.sort(rng.choice(Ntr, n, replace=False)); non = np.sort(rng.choice(Nv, n, replace=False))
Xc = np.concatenate([np.asarray(prep['train_x'][mem]), np.asarray(prep['val_x'][non])]).astype(np.float32)
Kc = np.concatenate([np.asarray(prep['train_kg'][mem]), np.asarray(prep['val_kg'][non])]).astype(np.float32)
Yc = np.concatenate([np.asarray(prep['train_y'][mem]), np.asarray(prep['val_y'][non])]).astype(np.float32)
is_mem = np.r_[np.ones(n), np.zeros(n)]
def logodds(model):
    _, lg = compute(model, Xc, Kc, dev); return np.where(Yc == 1, lg, -lg).astype(np.float64)   # log-odds of the true label
def train_shadow(k):
    itr = np.flatnonzero(rng.random(Ntr) < 0.5); iv = np.flatnonzero(rng.random(Nv) < 0.5)
    x = np.concatenate([np.asarray(prep['train_x'][itr]), np.asarray(prep['val_x'][iv])]).astype(np.float32)
    kg = np.concatenate([np.asarray(prep['train_kg'][itr]), np.asarray(prep['val_kg'][iv])]).astype(np.float32)
    y = np.concatenate([np.asarray(prep['train_y'][itr]), np.asarray(prep['val_y'][iv])]).astype(np.float32)
    inc = np.r_[np.isin(mem, itr), np.isin(non, iv)]
    model = CNNLSTMAE(norm='group', dp=False).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    dl = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(kg), torch.from_numpy(y)), batch_size=a.batch, shuffle=True)
    for ep in range(a.epochs):
        model.train()
        for xb, kb, yb in dl:
            xb, kb, yb = xb.to(dev), kb.to(dev), yb.to(dev); opt.zero_grad(); rec, lg = model(xb, kb)
            combined_loss(rec, lg, xb, yb, 10.0).backward(); opt.step()
    print(f'shadow {k}: trained on {len(y)} windows, {int(inc.sum())} of {len(inc)} candidates in', flush=True)
    return logodds(model), inc
t0 = time.time(); S = []; IN = []
for k in range(a.shadows):
    s, inc = train_shadow(k); S.append(s); IN.append(inc)
S = np.array(S); IN = np.array(IN)
def tpr_at(y, s, f):
    fpr, tpr, _ = roc_curve(y, s); return float(np.interp(f, fpr, tpr))
def metrics(score): return dict(auc=float(roc_auc_score(is_mem, score)), tpr_at_1pct_fpr=tpr_at(is_mem, score, 0.01), tpr_at_0_1pct_fpr=tpr_at(is_mem, score, 0.001))
glob_sd = float(np.median([S[~IN[:, i], i].std() for i in range(S.shape[1]) if (~IN[:, i]).sum() >= 2]))
res = dict(n_candidates_per_class=int(n), shadows=a.shadows, shadow_epochs=a.epochs, seconds=None, targets={})
for spec in a.targets:
    name, w, dp = spec.split(':'); dp = dp == '1'
    if not os.path.exists(w): print('missing target weights', w); continue
    net = CNNLSTMAE(norm='group', dp=dp); net.load_state_dict(torch.load(w, map_location='cpu')); net.to(dev).eval()
    phi = logodds(net); score = np.empty_like(phi)
    for i in range(len(phi)):
        out = S[~IN[:, i], i]
        mu = out.mean() if len(out) else S[:, i].mean(); sd = max(out.std() if len(out) >= 2 else glob_sd, 1e-3)
        score[i] = (phi[i] - mu) / sd
    res['targets'][name] = dict(weights=w, lira_offline=metrics(score), global_threshold=metrics(phi))
    print(name, res['targets'][name], flush=True)
res['seconds'] = time.time() - t0
json.dump(res, open(a.out, 'w'), indent=2); print('saved', a.out)
