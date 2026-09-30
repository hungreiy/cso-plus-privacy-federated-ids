"""Collect results/*.json into the tables of Chapter 4 (mean +/- sd over seeds).  python make_tables.py results"""
import glob, json, os, sys
import numpy as np
d = sys.argv[1] if len(sys.argv) > 1 else 'results'
NAMES = {'rf': 'Random forest (non-deep, non-private, reference)', 'central': 'Centralised hybrid model (non-private)',
         'central_nokg': 'Centralised without KG (ablation)', 'fl_nodp': 'FedAvg without DP', 'fl_dp': 'Proposed: DP-FL with KG',
         'fl_dp_nokg': 'DP-FL without KG (ablation)'}
def load(prefix, tag): return [json.load(open(f)) for f in sorted(glob.glob(f'{d}/{prefix}{tag}_seed*.json'))]
def ms(vals, nd=4):
    v = [x for x in vals if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x))]
    return '-' if not v else (f'{np.mean(v):.{nd}f}' if len(v) == 1 else f'{np.mean(v):.{nd}f} ± {np.std(v, ddof=1):.{nd}f}')
lines = []
def table(title, header, rows):
    lines.append(f'\n### {title}\n'); lines.append('| ' + ' | '.join(header) + ' |'); lines.append('|' + '---|' * len(header))
    for r in rows: lines.append('| ' + ' | '.join(r) + ' |')
runs = {t: load('', t) for t in NAMES}
g = lambda r, k: [x.get(k) for x in r]
table('Table 4.3 Detection performance on the test set (alert threshold = 99.5th percentile of benign validation scores)', ['Method', 'n seeds', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'FPR'],
      [[NAMES[t], str(len(r)), *[ms(g(r, k)) for k in ('precision', 'recall', 'f1', 'auc', 'fpr')]] for t, r in runs.items() if r])
table('Table 4.4 Known (seen) versus unseen attack types', ['Method', 'Recall seen', 'Recall unseen', 'AUC seen', 'AUC unseen', 'Two-branch recall unseen', 'AUC (reconstruction only) unseen'],
      [[NAMES[t], ms(g(r, 'recall_seen')), ms(g(r, 'recall_unseen')), ms(g(r, 'auc_seen')), ms(g(r, 'auc_unseen')),
        ms([x['hybrid']['recall_unseen'] for x in r if 'hybrid' in x]), ms(g(r, 'auc_recon_unseen'))] for t, r in runs.items() if r])
table('Table 4.3b Clear windows only (benign, or at least 8 of the 15 flows are attack flows)', ['Method', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'FPR', 'Recall seen', 'Recall unseen'],
      [[NAMES[t], *[ms([x['clean'][k] for x in r if 'clean' in x]) for k in ('precision', 'recall', 'f1', 'auc', 'fpr', 'recall_seen', 'recall_unseen')]] for t, r in runs.items() if r and 'clean' in r[0]])
bk = ['seen_8-15', 'seen_4-7', 'seen_1-3', 'unseen_8-15', 'unseen_4-7', 'unseen_1-3']
table('Recall by the number of attack flows in the window (n = test windows)', ['Method'] + [b.replace('_', ', ') for b in bk],
      [[NAMES[t]] + [ms([x['by_purity'][b]['recall'] for x in r if 'by_purity' in x], 3) + ' (n=' + str(r[0]['by_purity'][b]['n']) + ')' for b in bk] for t, r in runs.items() if r and 'by_purity' in r[0]])
for t in ('central', 'fl_dp', 'rf'):
    r = runs[t]
    if r:
        types = sorted({k for x in r for k in x['per_type']})
        table(f'Per-attack-type recall: {NAMES[t]}', ['Attack type', 'Test windows', 'Unseen in training?', 'Recall'],
              [[k, str(int(np.mean([x['per_type'][k]['n'] for x in r if k in x['per_type']]))), 'yes' if r[0]['per_type'][k]['unseen'] else 'no', ms([x['per_type'][k]['recall'] for x in r if k in x['per_type']], 3)] for k in types])
ops_rows = []
for t, r in runs.items():
    if r:
        for pc in ('95', '99', '99.5', '99.9'):
            ops_rows.append([NAMES[t], pc, ms([x['operating_points'][pc]['recall'] for x in r]), ms([x['operating_points'][pc]['fpr'] for x in r], 4), ms([x['operating_points'][pc]['f1'] for x in r])])
if ops_rows: table('Operating points (threshold = percentile of benign validation scores)', ['Method', 'Percentile', 'Recall', 'FPR', 'F1'], ops_rows)
priv = []
for t in ('fl_dp', 'fl_dp_nokg'):
    r = runs[t]
    if r:
        p = [x.get('privacy', {}) for x in r]
        priv.append([NAMES[t], ms([x['args']['sigma'] for x in r], 3), ms([x['args']['fraction'] for x in r], 2), ms([q.get('rounds_worst_client') for q in p], 1), ms([q.get('eps_opacus') for q in p], 2), ms([q.get('eps_numpy_accountant') for q in p], 2)])
table('Table 4.5 Privacy budget (delta = 1e-6, worst tenant)', ['Method', 'sigma', 'Tenants per round (fraction)', 'Rounds of worst tenant', 'epsilon (Opacus)', 'epsilon (numpy check)'], priv)
eff = []
for t, r in runs.items():
    if r and t != 'rf':
        mb = g(r, 'model_mb'); fl = t.startswith('fl')
        eff.append([NAMES[t], ms(g(r, 'latency_ms'), 2), ms(mb, 3), ms([2 * m for m in mb], 3) if fl else '-', ms([(x.get('train_seconds') or float('nan')) / 3600 for x in r], 3)])
table('Table 4.7 Efficiency', ['Method', 'Latency (ms/window)', 'Model size (MB)', 'Communication per tenant per round, up+down (MB)', 'Training time (hours)'], eff)
mia = []
for t, r in ((t, load('mia_', t)) for t in NAMES):
    if r: mia.append([NAMES[t], str(len(r)), ms([x['mia_auc'] for x in r], 3), ms([x['mia_accuracy'] for x in r], 3)])
if mia: table('Membership inference (threshold attack; 0.5 = no leakage)', ['Model', 'n', 'Attack AUC', 'Attack balanced accuracy'], mia)
fid = []
for t, r in ((t, load('fid_', t)) for t in NAMES):
    if r: fid.append([NAMES[t], str(len(r)), ms([x['fidelity_ratio'] for x in r], 2), ms([x['alerts_flipped_topk_shap'] for x in r], 3), ms([x['alerts_flipped_random'] for x in r], 3), ms([x.get('spearman_importance_vs_reference', float('nan')) for x in r], 3)])
if fid: table('Table 4.6 Explanation fidelity (removal of the top-5 flow features)', ['Model', 'n', 'Fidelity ratio (SHAP/random)', 'Alerts flipped (SHAP top-5)', 'Alerts flipped (random 5)', 'Spearman vs reference'], fid)
txt = '\n'.join(lines); print(txt); open(f'{d}/tables.md', 'w').write(txt + '\n')
