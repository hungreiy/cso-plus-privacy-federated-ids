"""Label-scarce tenants: local random forests vs the private federated model -> label_results.md   (run from src/)"""
import json, os, numpy as np
S = range(3); out = []
def L(d, tag): return [json.load(open(f'{d}/{tag}_seed{s}.json')) for s in S if os.path.exists(f'{d}/{tag}_seed{s}.json')]
def ms(v, nd=4): return f'{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}' if len(v) > 1 else (f'{v[0]:.4f}' if v else '-')
rows = []
for f in (0.005, 0.02, 0.1, 1.0):
    lf = '' if f == 1 else f'_lf{f:g}'
    R = L('results_private' if f == 1 else 'results_label', 'local_rf' + lf); D = L('results' if f == 1 else 'results_label', 'fl_dp' + lf)
    rows.append([f'{f*100:g} %', ms([j['f1'] for j in R]), ms([j['f1_min'] for j in R]), ms([j['f1_clear'] for j in R]),
                 f"{np.mean([j['n_tenants_evaluated'] for j in R]):.1f}" if R else '-', ms([j['f1'] for j in D]), ms([j['clean']['f1'] for j in D]), ms([j['recall_seen'] for j in D])])
out.append('\n### Label-scarce tenants (IID, seeds 0-2): local random forests vs proposed DP-FL with KG\n')
out.append('| Labelled windows per tenant | Local RF F1 (mean) | Local RF F1 (weakest tenant) | Local RF F1 clear | Tenants with both classes | DP-FL F1 | DP-FL F1 clear | DP-FL recall known |')
out.append('|' + '---|' * 8); out.extend('| ' + ' | '.join(r) + ' |' for r in rows)
open('label_results.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
