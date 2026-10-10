"""Tables for the additional experiments -> results_extra.md
   cross-dataset (results_cross), IID vs non-IID tenants (results vs results_noniid), window length (results_w5, results, results_w30).
   The within-CSE-CIC-IDS2018 study has its own tables: python make_tables.py results_2018"""
import glob, json, os, sys
import numpy as np
root = sys.argv[1] if len(sys.argv) > 1 else '.'
out = []
def ms(v, nd=4):
    v = [x for x in v if x is not None]
    if not v: return '-'
    return f'{np.mean(v):.{nd}f} ± {np.std(v, ddof=1):.{nd}f}' if len(v) > 1 else f'{v[0]:.{nd}f}'
def load(d, pat): return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, d, pat)))]
def table(title, head, rows):
    if not rows: return
    out.append(f'\n### {title}\n'); out.append('| ' + ' | '.join(head) + ' |'); out.append('|' + '---|' * len(head))
    out.extend('| ' + ' | '.join(r) + ' |' for r in rows)
NM = {'central': 'Centralised hybrid', 'fl_dp': 'Proposed DP-FL with KG', 'rf': 'Random forest', 'xgb': 'XGBoost',
      'fl_nodp': 'FedAvg without DP', 'fl_dp_nokg': 'DP-FL without KG'}

# 1. cross-dataset
rows, per_type = [], {}
for t in ('rf', 'xgb', 'central', 'fl_dp'):
    r = load('results_cross', f'cross_{t}_seed*.json')
    if not r: continue
    rows.append([NM[t], str(len(r))] + [ms([x[k] for x in r]) for k in ('f1', 'auc', 'fpr', 'recall_seen', 'recall_unseen')])
    for x in r:
        for typ, v in (x.get('per_type') or {}).items(): per_type.setdefault(typ, {}).setdefault(t, []).append(v['recall'] if isinstance(v, dict) else v)
table('Cross-dataset test: trained on CIC-IDS2017, tested unchanged on CSE-CIC-IDS2018 (threshold from 2017 validation)',
      ['Method', 'n seeds', 'F1', 'AUC-ROC', 'FPR', 'Recall seen family', 'Recall unseen family'], rows)
if per_type:
    ts = [t for t in ('rf', 'xgb', 'central', 'fl_dp') if any(t in v for v in per_type.values())]
    table('Cross-dataset recall per CSE-CIC-IDS2018 attack type', ['Attack type'] + [NM[t] for t in ts],
          [[typ] + [ms(v.get(t, []), 3) for t in ts] for typ, v in sorted(per_type.items())])

# 2. IID vs non-IID tenants
rows = []
for t in ('fl_nodp', 'fl_dp', 'fl_dp_nokg'):
    for d, lab in (('results', 'IID'), ('results_noniid', 'non-IID (Dirichlet 0.5)')):
        r = load(d, f'{t}_seed*.json')
        if r: rows.append([NM[t], lab, str(len(r))] + [ms([x[k] for x in r]) for k in ('f1', 'auc')] +
                          [ms([x['clean']['f1'] for x in r if 'clean' in x]), ms([x['recall_seen'] for x in r]),
                           ms([(x.get('privacy') or {}).get('eps_opacus') for x in r], 2)])
table('Effect of non-IID tenants (different attack mixes per tenant)', ['Method', 'Tenants', 'n seeds', 'F1', 'AUC-ROC', 'F1 clear windows', 'Recall seen', 'epsilon (worst tenant)'], rows)

# 3. window length
rows = []
for t in ('rf', 'central', 'fl_dp'):
    for d, w in (('results_w5', 5), ('results', 15), ('results_w30', 30)):
        r = load(d, f'{t}_seed*.json')
        if r: rows.append([NM[t], str(w), str(len(r))] + [ms([x[k] for x in r]) for k in ('f1', 'auc', 'recall_seen', 'recall_unseen')] +
                          [ms([x['clean']['f1'] for x in r if 'clean' in x])])
table('Sensitivity to the window length (flows per window)', ['Method', 'Window', 'n seeds', 'F1', 'AUC-ROC', 'Recall seen', 'Recall unseen', 'F1 clear windows'], rows)
open(os.path.join(root, 'results_extra.md'), 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
