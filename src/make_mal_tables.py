"""Malicious tenants and robust aggregation -> mal_results.md   (run from src/)"""
import json, os, numpy as np
S = range(3); out = []
def L(d, tag): return [json.load(open(f'{d}/{tag}_seed{s}.json')) for s in S if os.path.exists(f'{d}/{tag}_seed{s}.json')]
def ms(v, nd=4): return f'{np.mean(v):.{nd}f} ± {np.std(v, ddof=1):.{nd}f}' if len(v) > 1 else (f'{v[0]:.{nd}f}' if v else '-')
rows = []
for m in (0, 1, 3):
    for strat in ('fedavg', 'median'):
        tag = 'fl_dp' + (f'_m{m}' if m else '') + ('_median' if strat == 'median' else '')
        d = 'results' if (m == 0 and strat == 'fedavg') else 'results_mal'
        J = L(d, tag)
        if not J: rows.append([m, 'FedAvg' if strat == 'fedavg' else 'FedMedian', 0] + ['-'] * 6); continue
        rows.append([m, 'FedAvg' if strat == 'fedavg' else 'FedMedian', len(J), ms([j['f1'] for j in J]), ms([j['clean']['f1'] for j in J]), ms([j['recall_seen'] for j in J]),
                     ms([j['auc'] for j in J]), ms([j['fpr'] for j in J]), ms([j['privacy']['eps_opacus'] for j in J], 2)])
out.append('\n### Malicious tenants (label flipping) and robust aggregation (DP-FL with KG, seeds 0-2)\n')
out.append('| Malicious tenants (of 10) | Aggregation | n seeds | F1 | F1 clear | Recall known | AUC | FPR | ε (busiest tenant) |'); out.append('|' + '---|' * 9)
out.extend('| ' + ' | '.join(map(str, r)) + ' |' for r in rows)
open('mal_results.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
