"""Classical baseline 2 (not deep, not private): gradient-boosted trees (XGBoost) on the same window statistics as the
random forest (mean, std and max of the flows in the window), same training windows, same test windows, same threshold rule."""
import argparse, json, os, time
import numpy as np
from metrics import load_prep, eval_scores
from train_rf_feats import feats

ap = argparse.ArgumentParser()
ap.add_argument('--prep', default='prepared'); ap.add_argument('--out', default='results')
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--trees', type=int, default=400)
ap.add_argument('--depth', type=int, default=8); ap.add_argument('--lr', type=float, default=0.1)
ap.add_argument('--max_train', type=int, default=150000)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
from xgboost import XGBClassifier
prep = load_prep(a.prep); t0 = time.time(); rng = np.random.default_rng(a.seed)
n = len(prep['train_x']); idx = np.sort(rng.choice(n, size=min(n, a.max_train), replace=False))
Ftr = feats(prep['train_x'][idx]); ytr = np.asarray(prep['train_y'])[idx]
# early stopping on the validation set (never the test set)
clf = XGBClassifier(n_estimators=a.trees, max_depth=a.depth, learning_rate=a.lr, subsample=0.8, colsample_bytree=0.8,
                    tree_method='hist', random_state=a.seed, n_jobs=-1, eval_metric='logloss', early_stopping_rounds=30)
Fva = feats(prep['val_x']); yva = np.asarray(prep['val_y'])
clf.fit(Ftr, ytr, eval_set=[(Fva, yva)], verbose=False)
v = clf.predict_proba(Fva)[:, 1]; t = clf.predict_proba(feats(prep['test_x']))[:, 1]
res = eval_scores(v, yva, t, prep['test_y'], prep['test_grp'], prep['test_type'], prep['type_names'], natt=prep.get('test_natt'))
res.update(method='xgb', seed=a.seed, args=vars(a), best_iteration=int(clf.best_iteration), latency_ms=None, model_mb=None, train_seconds=time.time() - t0)
json.dump(res, open(f'{a.out}/xgb_seed{a.seed}.json', 'w'), indent=2)
print({k: (round(res[k], 4) if isinstance(res[k], float) else res[k]) for k in ['f1', 'precision', 'recall', 'auc', 'fpr', 'recall_seen', 'recall_unseen', 'auc_unseen']})
