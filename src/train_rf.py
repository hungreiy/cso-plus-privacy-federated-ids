"""Classical reference (not deep, not private): random forest on window statistics (mean, std, max of the 15 flows),
trained on the same training windows and evaluated on the same test windows with the same threshold rule."""
import argparse, json, os, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from metrics import load_prep, eval_scores

ap = argparse.ArgumentParser()
ap.add_argument('--prep', default='prepared'); ap.add_argument('--out', default='results')
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--trees', type=int, default=200)
ap.add_argument('--max_train', type=int, default=150000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
prep = load_prep(a.prep)

def feats(x, bs=20000):
    out = []
    for i in range(0, len(x), bs):
        b = np.asarray(x[i:i + bs]); out.append(np.concatenate([b.mean(axis=1), b.std(axis=1), b.max(axis=1)], axis=1))
    return np.concatenate(out)

t0 = time.time(); rng = np.random.default_rng(a.seed)
n = len(prep['train_x']); idx = np.sort(rng.choice(n, size=min(n, a.max_train), replace=False))
Ftr = feats(prep['train_x'][idx]); ytr = np.asarray(prep['train_y'])[idx]
rf = RandomForestClassifier(n_estimators=a.trees, n_jobs=-1, random_state=a.seed).fit(Ftr, ytr)
v = rf.predict_proba(feats(prep['val_x']))[:, 1]; t = rf.predict_proba(feats(prep['test_x']))[:, 1]
res = eval_scores(v, prep['val_y'], t, prep['test_y'], prep['test_grp'], prep['test_type'], prep['type_names'], natt=prep.get('test_natt'))
res.update(method='rf', seed=a.seed, args=vars(a), latency_ms=None, model_mb=None, train_seconds=time.time() - t0)
json.dump(res, open(f'{a.out}/rf_seed{a.seed}.json', 'w'), indent=2)
print({k: (round(res[k], 4) if isinstance(res[k], float) else res[k]) for k in ['f1', 'precision', 'recall', 'auc', 'fpr', 'recall_seen', 'recall_unseen', 'auc_unseen']})
