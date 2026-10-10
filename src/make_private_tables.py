"""Comparison of the private model with private non-deep baselines -> private_results.md   (run from src/)"""
import json, os, numpy as np
S = range(5); out = []
def L(d, tag): return [json.load(open(f'{d}/{tag}_seed{s}.json')) for s in S if os.path.exists(f'{d}/{tag}_seed{s}.json')]
def ms(v, nd=4): return f'{np.mean(v):.{nd}f} ± {np.std(v, ddof=1):.{nd}f}' if len(v) > 1 else (f'{v[0]:.{nd}f}' if v else '-')
rows = []
G = lambda key: (lambda j: j['clean']['f1']) if key == 'f1_clear' else (lambda j: j[key])
for name, d, tag, priv in [('Random forest, pooled data (non-private reference)', 'results', 'rf', 'none'),
                           ('Random forests, local data only (one per tenant)', 'results_private', 'local_rf', 'no data shared'),
                           ('Logistic regression, DP-FL (window statistics)', 'results_private', 'dp_linear_fl', 'DP'),
                           ('Proposed: DP-FL hybrid model with KG', 'results', 'fl_dp', 'DP')]:
    J = L(d, tag)
    if not J: continue
    f = (lambda k: (lambda j: j[k])) if tag == 'local_rf' else G
    eps = ms([j['privacy'].get('eps_opacus', j['privacy'].get('eps_numpy_accountant')) for j in J], 2) if priv == 'DP' else '-'
    rows.append([name, priv, len(J)] + [ms([f(k)(j) for j in J]) for k in ('f1', 'f1_clear', 'auc', 'fpr', 'recall_seen', 'recall_unseen')] + [eps])
out.append('\n### Private non-deep baselines (CIC-IDS2017, seeds 0-4)\n')
out.append('| Method | Privacy | n seeds | F1 | F1 clear | AUC | FPR | Recall known | Recall unseen | ε (busiest tenant) |'); out.append('|' + '---|' * 10)
out.extend('| ' + ' | '.join(map(str, r)) + ' |' for r in rows)
J = L('results_private', 'local_rf')
if J:
    out.append('\n### Local-only random forests: spread over tenants (F1 on all windows, mean over seeds)\n')
    out.append('| Statistic | F1 | F1 clear |'); out.append('|---|---|---|')
    for lab, a, b in (('Worst tenant', 'f1_min', 'f1_clear_min'), ('Mean of tenants', 'f1', 'f1_clear'), ('Best tenant', 'f1_max', 'f1_clear_max')):
        out.append(f'| {lab} | {np.mean([j[a] for j in J]):.4f} | {np.mean([j[b] for j in J]):.4f} |')
J = L('results_private', 'dp_linear_fl')
if J: out.append('\nLearning rates chosen on validation: ' + ', '.join(f"seed {j['seed']}: {j['lr']}" for j in J))
open('private_results.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
