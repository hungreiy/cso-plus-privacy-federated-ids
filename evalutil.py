import time
import numpy as np, torch
from sklearn.metrics import average_precision_score, roc_auc_score
from model import N_FLOW
from metrics import load_prep, apply_no_kg, eval_scores   # re-exported

@torch.no_grad()
def compute(model, x, kg, device, bs=2048):
    """returns (reconstruction error per window, attack logit per window)"""
    model.eval(); rec, lg = [], []
    for i in range(0, len(x), bs):
        xb = torch.as_tensor(np.asarray(x[i:i + bs]), device=device)
        kb = torch.as_tensor(np.asarray(kg[i:i + bs]), device=device)
        r, l = model(xb, kb)
        rec.append(((r[:, :, :N_FLOW] - xb) ** 2).mean(dim=(1, 2)).cpu().numpy()); lg.append(l.cpu().numpy())
    return np.concatenate(rec), np.concatenate(lg)

def val_objective(model, prep, device, rec_weight=10.0):
    """validation value of the training loss (used for model selection / monitoring)"""
    rec, lg = compute(model, prep['val_x'], prep['val_kg'], device)
    y = np.asarray(prep['val_y']).astype(np.float64)
    bce = np.logaddexp(0, lg.astype(np.float64)) - y * lg
    return float(np.mean(rec_weight * (y == 0) * rec + bce))

def full_eval(model, prep, device, pct=99.5):
    v_rec, v_lg = compute(model, prep['val_x'], prep['val_kg'], device)
    t_rec, t_lg = compute(model, prep['test_x'], prep['test_kg'], device)
    res = eval_scores(v_lg, prep['val_y'], t_lg, prep['test_y'], prep['test_grp'], prep['test_type'], prep['type_names'], pct=pct, t_rec=t_rec, v_rec=v_rec, natt=prep.get('test_natt'))
    xb = torch.as_tensor(np.asarray(prep['test_x'][:200]), device=device); kb = torch.as_tensor(np.asarray(prep['test_kg'][:200]), device=device)
    model.eval()
    with torch.no_grad():
        for i in range(20): model(xb[i:i + 1], kb[i:i + 1])
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        for i in range(200): model(xb[i:i + 1], kb[i:i + 1])
        if device.type == 'cuda': torch.cuda.synchronize()
    res['latency_ms'] = (time.time() - t0) / 200 * 1000
    res['model_mb'] = sum(p.numel() for p in model.parameters()) * 4 / 1e6
    return res

def val_selection(model, prep, device, pct=99.5):
    """Metrics on the VALIDATION set only, used to choose hyperparameters (never the test set).
    val_recall_at_fpr005: threshold = 99.5th percentile of the benign validation scores, recall on the validation attacks."""
    _, lg = compute(model, prep['val_x'], prep['val_kg'], device)
    y = np.asarray(prep['val_y']).astype(int)
    thr = float(np.percentile(lg[y == 0], pct))
    return dict(val_ap=float(average_precision_score(y, lg)), val_auc=float(roc_auc_score(y, lg)),
                val_recall_at_fpr005=float((lg[y == 1] > thr).mean()), val_threshold=thr)
