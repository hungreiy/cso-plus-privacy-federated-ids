"""Baseline: centralised (non-federated, non-private) hybrid CNN-LSTM autoencoder-classifier."""
import argparse, json, os, copy, time
import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from model import CNNLSTMAE, combined_loss
from evalutil import load_prep, apply_no_kg, full_eval, val_objective

ap = argparse.ArgumentParser()
ap.add_argument('--prep', default='prepared'); ap.add_argument('--out', default='results')
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=60)
ap.add_argument('--batch', type=int, default=256); ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--wd', type=float, default=1e-5); ap.add_argument('--patience', type=int, default=10)
ap.add_argument('--norm', default='group', choices=['batch', 'group', 'none'])
ap.add_argument('--rec_weight', type=float, default=10.0)
ap.add_argument('--no_kg', action='store_true', help='ablation: zero KG vectors')
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); os.makedirs(a.out, exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prep = apply_no_kg(load_prep(a.prep), a.no_kg)
xtr = torch.from_numpy(np.ascontiguousarray(prep['train_x'])); ktr = torch.from_numpy(np.ascontiguousarray(prep['train_kg']))
ytr = torch.from_numpy(np.ascontiguousarray(prep['train_y']))
loader = DataLoader(TensorDataset(xtr, ktr, ytr), batch_size=a.batch, shuffle=True)
model = CNNLSTMAE(norm=a.norm, dp=False).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
t_start = time.time(); best, best_state, bad, hist = 1e9, None, 0, []
for ep in range(a.epochs):
    model.train(); tot = 0.0; n = 0
    for xb, kb, yb in loader:
        xb, kb, yb = xb.to(dev), kb.to(dev), yb.to(dev)
        opt.zero_grad(); rec, lg = model(xb, kb); loss = combined_loss(rec, lg, xb, yb, a.rec_weight); loss.backward(); opt.step()
        tot += loss.item() * len(xb); n += len(xb)
    v = val_objective(model, prep, dev, a.rec_weight)
    hist.append(dict(epoch=ep, train=tot / n, val=v)); print(f'epoch {ep} train {tot / n:.5f} val {v:.5f}', flush=True)
    if v < best - 1e-6: best, bad, best_state = v, 0, copy.deepcopy(model.state_dict())
    else:
        bad += 1
        if bad >= a.patience: break
model.load_state_dict(best_state)
train_seconds = time.time() - t_start
res = full_eval(model, prep, dev)
res.update(method='central_nokg' if a.no_kg else 'central', seed=a.seed, args=vars(a), history=hist, train_seconds=train_seconds)
tag = res['method']
json.dump(res, open(f'{a.out}/{tag}_seed{a.seed}.json', 'w'), indent=2)
torch.save(model.state_dict(), f'{a.out}/{tag}_seed{a.seed}.pt')
print({k: round(res[k], 4) for k in ['f1', 'precision', 'recall', 'auc', 'fpr', 'recall_seen', 'recall_unseen', 'auc_unseen']})
