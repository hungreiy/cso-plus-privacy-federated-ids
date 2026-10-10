"""Cross-dataset test: models trained on CIC-IDS2017 are applied, unchanged, to CSE-CIC-IDS2018.

* the 2018 flows are scaled with the 2017 training scaler (scaler.npz of the 2017 prep folder) and windowed the same way
* the alert threshold is the 99.5th percentile of the 2017 benign VALIDATION scores (nothing is tuned on 2018)
* 2018 attack types are grouped as 'seen family' (a similar attack was in the 2017 training data) or 'unseen family'
Models: central hybrid and private DP-FL model (weights from the 2017 results folder), and random forest / XGBoost
retrained on the same 2017 training windows with the same seed.
"""
import argparse, glob, json, os, time
import numpy as np, torch
import ids2018
from data import make_lut
from metrics import load_prep, eval_scores
from train_rf_feats import feats
from model import CNNLSTMAE
from evalutil import compute

SEEN_FAMILY = {'DoS attacks-Hulk': 'DoS Hulk', 'DDoS attacks-LOIC-HTTP': 'DDoS', 'DDOS attack-HOIC': 'DDoS',
               'DDOS attack-LOIC-UDP': 'DDoS', 'FTP-BruteForce': 'FTP-Patator', 'SSH-Bruteforce': 'SSH-Patator'}
ap = argparse.ArgumentParser()
ap.add_argument('--prep17', required=True); ap.add_argument('--results17', required=True)
ap.add_argument('--csv18', required=True); ap.add_argument('--kg', required=True)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default='results_cross')
ap.add_argument('--max_train', type=int, default=150000)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prep = load_prep(a.prep17); meta = json.load(open(f'{a.prep17}/meta.json')); W = meta['window']
sc = np.load(f'{a.prep17}/scaler.npz'); names17 = [str(x) for x in sc['names']]
if sorted(names17) != sorted(ids2018.ORDER_2017):
    raise SystemExit(f'2017 feature names differ from the expected CIC-IDS2017 names: {set(names17) ^ set(ids2018.ORDER_2017)}')
col = [ids2018.ORDER_2017.index(n) for n in names17]; pi = names17.index('Destination Port'); lut = make_lut(a.kg)

# ---- models trained on 2017 ----
models = {}
for tag, dp in (('central', False), ('fl_dp', True)):
    p = f'{a.results17}/{tag}_seed{a.seed}.pt'
    if os.path.exists(p):
        m = CNNLSTMAE(norm='group', dp=dp); m.load_state_dict(torch.load(p, map_location='cpu')); models[tag] = m.to(dev).eval()
    else: print('missing', p)
rng = np.random.default_rng(a.seed); n = len(prep['train_x']); idx = np.sort(rng.choice(n, size=min(n, a.max_train), replace=False))
Ftr, ytr = feats(prep['train_x'][idx]), np.asarray(prep['train_y'])[idx]
from sklearn.ensemble import RandomForestClassifier
trees = {'rf': RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=a.seed).fit(Ftr, ytr)}
try:
    from xgboost import XGBClassifier
    Fva = feats(prep['val_x'])
    trees['xgb'] = XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                                 random_state=a.seed, n_jobs=-1, eval_metric='logloss', early_stopping_rounds=30).fit(
                                 Ftr, ytr, eval_set=[(Fva, np.asarray(prep['val_y']))], verbose=False)
except ImportError: print('xgboost not installed: skipping xgb')
# validation scores on 2017 (threshold source)
vscore = {t: compute(m, prep['val_x'], prep['val_kg'], dev)[1] for t, m in models.items()}
Fv = feats(prep['val_x']); vscore.update({t: c.predict_proba(Fv)[:, 1] for t, c in trees.items()})

# ---- 2018 windows, file by file ----
tscore = {t: [] for t in vscore}; Y, TYP, NATT, GRP, names = [], [], [], [], {}
for f in sorted(glob.glob(os.path.join(a.csv18, '*.csv'))):
    X, lab, _, _ = ids2018.read_csv_2018(f); X = X[:, col]
    nw = len(X) // W; X = X[:nw * W].reshape(nw, W, -1); lab = lab[:nw * W].reshape(nw, W)
    ports = np.clip(X[:, :, pi].astype(np.int64), 0, 65535); kg = lut[ports].mean(axis=1).astype(np.float32)
    X = np.clip((X - sc['lo']) / sc['span'], 0, 1).astype(np.float32)
    att = lab != 'Benign'; y = att.any(axis=1).astype(np.int8); natt = att.sum(axis=1).astype(np.int8)
    typ = np.full(nw, -1, np.int32)
    for i in np.where(y == 1)[0]:
        vals, cnt = np.unique(lab[i][att[i]], return_counts=True); nm = str(vals[cnt.argmax()])
        typ[i] = names.setdefault(nm, len(names))
    Y.append(y); NATT.append(natt); TYP.append(typ)
    for t, m in models.items(): tscore[t].append(compute(m, X, kg, dev)[1])
    F = feats(X)
    for t, c in trees.items(): tscore[t].append(c.predict_proba(F)[:, 1])
    print(f'{os.path.basename(f)}: {nw} windows, {int(y.sum())} attack', flush=True); del X, F
Y, NATT, TYP = np.concatenate(Y), np.concatenate(NATT), np.concatenate(TYP)
type_names = [None] * len(names)
for k, v in names.items(): type_names[v] = k
seen = np.array([t in SEEN_FAMILY for t in type_names]); GRP = np.where(Y == 0, 0, np.where(seen[np.maximum(TYP, 0)], 1, 2)).astype(np.int8)
summary = {}
for t in vscore:
    r = eval_scores(vscore[t], prep['val_y'], np.concatenate(tscore[t]), Y, GRP, TYP, type_names, natt=NATT)
    r.update(method=t, seed=a.seed, train='CIC-IDS2017', test='CSE-CIC-IDS2018', n_test=int(len(Y)), n_test_attack=int(Y.sum()),
             seen_family_map=SEEN_FAMILY)
    json.dump(r, open(f'{a.out}/cross_{t}_seed{a.seed}.json', 'w'), indent=2)
    summary[t] = {k: round(float(r[k]), 4) for k in ('f1', 'precision', 'recall', 'auc', 'fpr', 'recall_seen', 'recall_unseen')}
print(json.dumps(summary, indent=1))
