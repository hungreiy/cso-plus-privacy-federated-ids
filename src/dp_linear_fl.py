"""Private non-deep baseline B (Section 4.10.8): federated logistic regression with differential privacy.
Logistic regression on the window statistics (mean, std, max of the flows), trained with FedAvg over the same tenants
and with the same local DP-SGD as the main private model: Poisson sampling (rate = batch / tenant size), per-example
clipping to norm C, Gaussian noise with multiplier sigma, local epochs, rounds and fraction of tenants per round.
The privacy budget of every tenant is accounted with the same Renyi accountant (accountant.py).
The learning rate is chosen on the validation set from a small grid (like the main study, tuning is not privately accounted).
  python dp_linear_fl.py --prep /root/prepared --out results_private --seed 0 --sigma 1.455"""
import argparse, json, os, time, math
import numpy as np
from metrics import load_prep, eval_scores
from accountant import epsilon_from_history
ap = argparse.ArgumentParser()
ap.add_argument('--prep', default='prepared'); ap.add_argument('--out', default='results_private'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--sigma', type=float, default=1.455); ap.add_argument('--clip', type=float, default=1.0); ap.add_argument('--delta', type=float, default=1e-6)
ap.add_argument('--rounds', type=int, default=50); ap.add_argument('--local_epochs', type=int, default=5); ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--fraction', type=float, default=0.3); ap.add_argument('--lrs', default='0.05,0.2,0.5,1.0')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
prep = load_prep(a.prep); tenants = json.load(open(os.path.join(a.prep, 'tenants.json'))); T = sorted(tenants, key=int)
def feats(x, bs=20000):
    out = []
    for i in range(0, len(x), bs):
        b = np.asarray(x[i:i + bs], dtype=np.float32); out.append(np.concatenate([b.mean(axis=1), b.std(axis=1), b.max(axis=1)], axis=1))
    return np.concatenate(out)
t0 = time.time()
Xtr = feats(prep['train_x']); ytr = np.asarray(prep['train_y']).astype(np.float32); Xv = feats(prep['val_x']); Xt = feats(prep['test_x'])
mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6                      # standardise with training statistics (as the main study's scaler)
Xtr = (Xtr - mu) / sd; Xv = (Xv - mu) / sd; Xt = (Xt - mu) / sd
D = Xtr.shape[1]; sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
def train(lr, seed):
    rng = np.random.default_rng(seed); w = np.zeros(D + 1, dtype=np.float64); hist = {c: [] for c in T}
    k = max(1, int(round(a.fraction * len(T))))
    for rnd in range(a.rounds):
        chosen = rng.choice(T, size=k, replace=False); upd = []
        for c in chosen:
            idx = np.asarray(tenants[c], dtype=int); n = len(idx); q = min(1.0, a.batch / n); steps = a.local_epochs * max(1, math.ceil(n / a.batch))
            wl = w.copy(); X = Xtr[idx]; y = ytr[idx]
            for _ in range(steps):
                m = rng.random(n) < q                                   # Poisson sampling
                Xb = X[m]; yb = y[m]
                g = (sig(Xb @ wl[:-1] + wl[-1]) - yb)[:, None] * np.concatenate([Xb, np.ones((len(Xb), 1), np.float32)], axis=1)
                nrm = np.linalg.norm(g, axis=1, keepdims=True); g = g / np.maximum(1.0, nrm / a.clip)     # per-example clipping
                noisy = g.sum(0) + rng.normal(0.0, a.sigma * a.clip, size=D + 1)                       # Gaussian mechanism
                wl -= lr * noisy / (q * n)                                                               # divide by expected batch size
            hist[c].append((a.sigma, q, steps)); upd.append((wl, n))
        w = sum(u * n for u, n in upd) / sum(n for _, n in upd)        # FedAvg weighted by tenant size
    return w, hist
best = None
for lr in [float(v) for v in a.lrs.split(',')]:
    w, hist = train(lr, a.seed); v = sig(Xv @ w[:-1] + w[-1]); yv = np.asarray(prep['val_y'])
    from metrics import _auc
    auc = _auc(yv, v); print(f'lr {lr}: validation AUC {auc:.4f}', flush=True)
    if best is None or auc > best[0]: best = (auc, lr, w, hist)
_, lr, w, hist = best
v = sig(Xv @ w[:-1] + w[-1]); t = sig(Xt @ w[:-1] + w[-1])
res = eval_scores(v, prep['val_y'], t, prep['test_y'], prep['test_grp'], prep['test_type'], prep['type_names'], natt=prep.get('test_natt'))
eps = {c: epsilon_from_history(h, a.delta)[0] for c, h in hist.items() if h}
worst = max(eps, key=eps.get)
res.update(method='dp_linear_fl', seed=a.seed, lr=lr, args=vars(a), train_seconds=time.time() - t0,
           privacy=dict(delta=a.delta, n_clients=len(T), worst_client=int(worst), rounds_worst_client=len(hist[worst]), eps_numpy_accountant=float(eps[worst])))
json.dump(res, open(f'{a.out}/dp_linear_fl_seed{a.seed}.json', 'w'), indent=2)
print({k: (round(res[k], 4) if isinstance(res[k], float) else res[k]) for k in ['f1', 'precision', 'recall', 'auc', 'fpr', 'recall_seen', 'recall_unseen']}, 'lr', lr, 'eps', round(eps[worst], 2))
