"""Private non-deep baseline A (Section 4.10.8): local-only random forests.
Every tenant trains a random forest on its OWN training windows only (nothing is shared), with the same window
statistics (mean, std, max of the flows) and the same alert-threshold rule as the reference random forest.
Each tenant's model is evaluated on the common test set; the result per seed is the mean over the ten tenants.
  python local_rf.py --prep /root/prepared --out results_private --seed 0"""
import argparse, json, os, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from metrics import load_prep, eval_scores
ap = argparse.ArgumentParser()
ap.add_argument('--prep', default='prepared'); ap.add_argument('--out', default='results_private')
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--trees', type=int, default=200)
ap.add_argument('--label_frac', type=float, default=1.0, help='fraction of each tenant\'s windows that are labelled; same rule as fl_flower.py')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
prep = load_prep(a.prep); tenants = json.load(open(os.path.join(a.prep, 'tenants.json')))
def feats(x, bs=20000):
    out = []
    for i in range(0, len(x), bs):
        b = np.asarray(x[i:i + bs]); out.append(np.concatenate([b.mean(axis=1), b.std(axis=1), b.max(axis=1)], axis=1))
    return np.concatenate(out)
t0 = time.time(); Fv = feats(prep['val_x']); Ft = feats(prep['test_x']); ytr_all = np.asarray(prep['train_y'])
per = []
for cid in sorted(tenants, key=int):
    idx = np.asarray(tenants[cid], dtype=int)
    if a.label_frac < 1:
        k = max(1, int(round(a.label_frac * len(idx)))); idx = idx[np.random.default_rng(12345 + 1000 * a.seed + int(cid)).permutation(len(idx))[:k]]
    idx = np.sort(idx); y = ytr_all[idx]
    if len(np.unique(y)) < 2:
        per.append(dict(tenant=int(cid), n_train=int(len(idx)), skipped='only one class')); continue
    rf = RandomForestClassifier(n_estimators=a.trees, n_jobs=-1, random_state=a.seed).fit(feats(prep['train_x'][idx]), y)
    r = eval_scores(rf.predict_proba(Fv)[:, 1], prep['val_y'], rf.predict_proba(Ft)[:, 1], prep['test_y'], prep['test_grp'], prep['test_type'], prep['type_names'], natt=prep.get('test_natt'))
    per.append(dict(tenant=int(cid), n_train=int(len(idx)), **{k: r[k] for k in ('f1', 'precision', 'recall', 'auc', 'fpr', 'recall_seen', 'recall_unseen')}, f1_clear=r['clean']['f1']))
    print('tenant', cid, {k: round(per[-1][k], 4) for k in ('f1', 'f1_clear', 'auc', 'fpr')}, flush=True)
ok = [p for p in per if 'f1' in p]
summ = {k: float(np.mean([p[k] for p in ok])) for k in ('f1', 'f1_clear', 'precision', 'recall', 'auc', 'fpr', 'recall_seen', 'recall_unseen')}
summ.update(f1_min=float(min(p['f1'] for p in ok)), f1_max=float(max(p['f1'] for p in ok)), f1_clear_min=float(min(p['f1_clear'] for p in ok)), f1_clear_max=float(max(p['f1_clear'] for p in ok)))
res = dict(method='local_rf', seed=a.seed, n_tenants=len(per), n_tenants_evaluated=len(ok), per_tenant=per, train_seconds=time.time() - t0, **summ)
tag = 'local_rf' + (f'_lf{a.label_frac:g}' if a.label_frac < 1 else ''); res['method'] = tag; res['label_frac'] = a.label_frac
json.dump(res, open(f'{a.out}/{tag}_seed{a.seed}.json', 'w'), indent=2)
print('mean over tenants:', {k: round(summ[k], 4) for k in ('f1', 'f1_clear', 'auc', 'fpr')})
