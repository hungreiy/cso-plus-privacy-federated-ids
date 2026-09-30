"""Torch-free helpers: loading the prepared data and computing all detection metrics from scores."""
import json, os
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

PREP_KEYS = ['train_x', 'train_kg', 'train_y', 'val_x', 'val_kg', 'val_y', 'test_x', 'test_kg', 'test_y', 'test_grp', 'test_type']

def load_prep(d):
    prep = {k: np.load(f'{d}/{k}.npy', mmap_mode='r') for k in PREP_KEYS}
    prep['type_names'] = json.load(open(f'{d}/types.json'))
    p = f'{d}/test_natt.npy'
    prep['test_natt'] = np.load(p, mmap_mode='r') if os.path.exists(p) else None
    return prep

def apply_no_kg(prep, no_kg):
    """Ablation 'without knowledge graph': replace all KG vectors by zeros."""
    if not no_kg: return prep
    for k in list(prep):
        if k.endswith('_kg'): prep[k] = np.zeros(prep[k].shape, dtype=np.float32)
    return prep

def _auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float('nan')

def _m(y, pred):
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    return dict(precision=float(p), recall=float(r), f1=float(f1), accuracy=float(accuracy_score(y, pred)),
                fpr=float(((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum())))

def _ecdf(ref_sorted, s):
    return np.searchsorted(ref_sorted, s, side='right') / len(ref_sorted)

def eval_scores(v_score, yv, t_score, y, grp, typ, type_names, pct=99.5, t_rec=None, v_rec=None, natt=None):
    """Alert if score > threshold; threshold = `pct`-th percentile of the BENIGN validation scores
    (pct = 99.5 gives a false-positive rate of about 0.5 % on validation).
    grp: 0 benign, 1 attack of a type seen in training, 2 attack of a held-out (unseen) type."""
    v_score, t_score = np.asarray(v_score, dtype=np.float64), np.asarray(t_score, dtype=np.float64)
    yv, y, grp, typ = map(np.asarray, (yv, y, grp, typ))
    y = y.astype(int)
    def at(p):
        thr = float(np.percentile(v_score[yv == 0], p)); pred = (t_score > thr).astype(int)
        m = _m(y, pred); m['threshold'] = thr
        m['recall_seen'] = float(pred[grp == 1].mean()) if (grp == 1).any() else float('nan')
        m['recall_unseen'] = float(pred[grp == 2].mean()) if (grp == 2).any() else float('nan')
        return m
    main = at(pct); ops = {str(p): at(p) for p in (90, 95, 99, 99.5, 99.9)}
    ben, seen, unseen = grp == 0, grp == 1, grp == 2
    out = dict(main)
    if natt is not None:
        # windows differ a lot in how many of their 15 flows are attack flows: report recall by that number, and metrics on
        # 'clear' windows only (benign, or at least 8 attack flows), leaving out the ambiguous windows with 1-7 attack flows
        natt = np.asarray(natt).astype(int); pm = (t_score > main['threshold']).astype(int); bp = {}
        for gname, gv in (('seen', 1), ('unseen', 2)):
            for bname, (lo, hi) in {'1-3': (1, 3), '4-7': (4, 7), '8-15': (8, 15)}.items():
                sel = (grp == gv) & (natt >= lo) & (natt <= hi)
                bp[f'{gname}_{bname}'] = dict(n=int(sel.sum()), recall=float(pm[sel].mean()) if sel.any() else float('nan'))
        keep = (natt == 0) | (natt >= 8)
        c = _m(y[keep], pm[keep]); c['auc'] = _auc(y[keep], t_score[keep]); c['n'] = int(keep.sum())
        c['recall_seen'] = float(pm[keep & seen].mean()) if (keep & seen).any() else float('nan')
        c['recall_unseen'] = float(pm[keep & unseen].mean()) if (keep & unseen).any() else float('nan')
        out['by_purity'] = bp; out['clean'] = c
    out['auc'] = _auc(y, t_score)
    out['auc_seen'] = _auc(y[ben | seen], t_score[ben | seen])
    out['auc_unseen'] = _auc(y[ben | unseen], t_score[ben | unseen])
    if t_rec is not None:
        t_rec = np.asarray(t_rec, dtype=np.float64)
        out['auc_recon'] = _auc(y, t_rec); out['auc_recon_seen'] = _auc(y[ben | seen], t_rec[ben | seen])
        out['auc_recon_unseen'] = _auc(y[ben | unseen], t_rec[ben | unseen])
    if t_rec is not None and v_rec is not None:
        # two-branch alert: classifier score OR reconstruction error, each turned into a percentile of the benign validation scores
        v_rec = np.asarray(v_rec, dtype=np.float64)
        vb_c, vb_r = np.sort(v_score[yv == 0]), np.sort(v_rec[yv == 0])
        hyb = lambda sc, rc: np.maximum(_ecdf(vb_c, sc), _ecdf(vb_r, rc))
        thr_h = float(np.percentile(hyb(v_score[yv == 0], v_rec[yv == 0]), pct)); th = hyb(t_score, t_rec); pred = (th > thr_h).astype(int)
        h = _m(y, pred); h['auc'] = _auc(y, th); h['auc_unseen'] = _auc(y[ben | unseen], th[ben | unseen])
        h['recall_seen'] = float(pred[seen].mean()) if seen.any() else float('nan'); h['recall_unseen'] = float(pred[unseen].mean()) if unseen.any() else float('nan')
        out['hybrid'] = h
    per_type = {}
    for t in np.unique(typ[typ >= 0]):
        sel = typ == t
        per_type[type_names[int(t)]] = dict(n=int(sel.sum()), recall=float((t_score[sel] > main['threshold']).mean()),
                                            unseen=bool((grp[sel] == 2).all()))
    out.update(operating_points=ops, per_type=per_type, threshold_percentile=pct,
               n_test=int(len(y)), n_test_benign=int(ben.sum()), n_test_seen=int(seen.sum()), n_test_unseen=int(unseen.sum()))
    return out
