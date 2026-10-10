"""Architecture baselines and the stronger membership-inference attack -> extra2_results.md   (run from src/)"""
import json, os, glob, numpy as np
out = []; S = range(3)
def L(d, tag): return [json.load(open(f'{d}/{tag}_seed{s}.json')) for s in S if os.path.exists(f'{d}/{tag}_seed{s}.json')]
def ms(v, nd=4): return f'{np.mean(v):.{nd}f} ± {np.std(v, ddof=1):.{nd}f}' if len(v) > 1 else (f'{v[0]:.{nd}f}' if v else '-')
rows = []
for name, d, tag in (('Hybrid CNN-LSTM (proposed)', 'results', 'fl_dp'), ('MLP autoencoder-classifier', 'results_arch', 'fl_dp_mlp'), ('Transformer autoencoder-classifier', 'results_arch', 'fl_dp_transformer')):
    J = L(d, tag)
    rows.append([name, len(J), ms([j['f1'] for j in J]), ms([j['clean']['f1'] for j in J]) if J else '-', ms([j['recall_seen'] for j in J]), ms([j['auc'] for j in J]), ms([j['privacy']['eps_opacus'] for j in J], 2) if J else '-', ms([j.get('model_mb', np.nan) for j in J], 2) if J else '-'])
out.append('\n### Architecture baselines under the same DP-FL (seeds 0-2)\n')
out.append('| Architecture | n seeds | F1 | F1 clear | Recall known | AUC | ε (busiest tenant) | Model size (MB) |'); out.append('|' + '---|' * 8)
out.extend('| ' + ' | '.join(map(str, r)) + ' |' for r in rows)
R = [json.load(open(f)) for f in sorted(glob.glob('results_mia2/lira_seed*.json'))]
if R:
    out.append('\n### Membership inference: offline LiRA with shadow models vs global threshold attack\n')
    out.append('| Target | Attack | n target seeds | AUC | TPR at 1 % FPR | TPR at 0.1 % FPR |'); out.append('|---|---|---|---|---|---|')
    for t in ('central', 'fl_dp'):
        for att, lab in (('lira_offline', 'LiRA (offline, shadow models)'), ('global_threshold', 'Global threshold')):
            V = [r['targets'][t][att] for r in R if t in r['targets']]
            if V: out.append(f"| {'Centralised (non-private)' if t == 'central' else 'Proposed DP-FL'} | {lab} | {len(V)} | {ms([v['auc'] for v in V])} | {ms([v['tpr_at_1pct_fpr'] for v in V])} | {ms([v['tpr_at_0_1pct_fpr'] for v in V])} |")
    out.append(f"\nShadows per run: {R[0]['shadows']} (trained {R[0]['shadow_epochs']} epochs each); candidates: {R[0]['n_candidates_per_class']} members and {R[0]['n_candidates_per_class']} non-members.")
open('extra2_results.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
