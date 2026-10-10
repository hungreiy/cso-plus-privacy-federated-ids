"""Tables for the three follow-up experiments -> more_results.md   (run from src/)"""
import json, glob, os, numpy as np
from scipy.stats import wilcoxon
out = []
def L(d, tag, seeds): return [json.load(open(f'{d}/{tag}_seed{s}.json')) for s in seeds if os.path.exists(f'{d}/{tag}_seed{s}.json')]
def ms(v, nd=4): return f'{np.mean(v):.{nd}f} ± {np.std(v, ddof=1):.{nd}f}' if len(v) > 1 else (f'{v[0]:.{nd}f}' if v else '-')
def table(title, head, rows):
    out.append(f'\n### {title}\n'); out.append('| ' + ' | '.join(head) + ' |'); out.append('|' + '---|' * len(head))
    out.extend('| ' + ' | '.join(map(str, r)) + ' |' for r in rows)
F = {'F1': lambda j: j['f1'], 'F1 clear': lambda j: j['clean']['f1'], 'AUC': lambda j: j['auc'], 'Recall known': lambda j: j['recall_seen'], 'Recall unseen': lambda j: j['recall_unseen']}
# 1. random-embedding control (paired by seed)
S = range(10)
grp = [('DP-FL with CSO+ embeddings (main)', 'results', 'fl_dp'), ('DP-FL with random embeddings (control)', 'results_kgrand', 'fl_dp'), ('DP-FL without embeddings', 'results', 'fl_dp_nokg')]
D = {n: L(d, t, S) for n, d, t in grp}
table('Random-embedding control (DP-FL, CIC-IDS2017)', ['Model', 'n seeds'] + list(F), [[n, len(D[n])] + [ms([f(j) for j in D[n]]) for f in F.values()] for n, _, _ in grp])
def paired(a, b, key='F1'):
    A = {j['seed']: F[key](j) for j in D[a]}; B = {j['seed']: F[key](j) for j in D[b]}; ss = sorted(set(A) & set(B))
    if len(ss) < 6: return [a, b, key, len(ss), '-', '-', '-']
    x = np.array([A[s] for s in ss]); y = np.array([B[s] for s in ss])
    return [a, b, key, len(ss), f'{(x - y).mean():+.4f}', f'{int((x > y).sum())}/{len(ss)}', f'{wilcoxon(x, y).pvalue:.4f}']
n0, n1, n2 = [g[0] for g in grp]
table('Paired comparisons (Wilcoxon signed-rank, two-sided)', ['A', 'B', 'Metric', 'n', 'Mean A-B', 'A > B', 'p'],
      [paired(n0, n1), paired(n0, n1, 'F1 clear'), paired(n1, n2), paired(n1, n2, 'F1 clear'), paired(n0, n2)])
# 2. privacy-utility curve
rows = []
fed = L('results', 'fl_nodp', range(5))
rows.append(['no DP (FedAvg)', '-', '-', len(fed)] + [ms([f(j) for j in fed]) for f in F.values()])
for e, d in (('1', 'results_eps1'), ('2', 'results_eps2'), ('4', 'results_eps4'), ('8 (main)', 'results')):
    J = L(d, 'fl_dp', range(5))
    if not J: continue
    eps = [j['privacy']['eps_opacus'] for j in J]
    rows.append([e, f"{J[0]['args']['sigma']:.3f}", ms(eps, 2), len(J)] + [ms([f(j) for j in J]) for f in F.values()])
table('Privacy-utility curve (DP-FL with CSO+ embeddings, seeds 0-4)', ['Target ε', 'σ', 'Achieved ε (busiest tenant)', 'n seeds'] + list(F), rows)
# 3. data splits
M = [('rf', 'Random forest'), ('central', 'Centralised hybrid'), ('fl_nodp', 'FedAvg without DP'), ('fl_dp', 'Proposed DP-FL with KG'), ('fl_dp_nokg', 'DP-FL without KG')]
splits = [('0 (main)', 'results'), ('1', 'results_split1'), ('2', 'results_split2')]
rows = []; per_split = {m: [] for m, _ in M}
for m, n in M:
    r = [n]
    for sp, d in splits:
        J = L(d, m, range(3)); v = [j['f1'] for j in J]; r.append(ms(v) if v else '-')
        if v: per_split[m].append(np.mean(v))
    r.append(ms(per_split[m]) if len(per_split[m]) > 1 else '-'); rows.append(r)
table('F1-score on all test windows by data split (seeds 0-2 per split; last column = mean ± SD across splits)', ['Method'] + [f'Split {s}' for s, _ in splits] + ['Across splits'], rows)
rank = []
for k, (sp, d) in enumerate(splits):
    vals = {n: (np.mean([j['f1'] for j in L(d, m, range(3))]) if L(d, m, range(3)) else np.nan) for m, n in M}
    rank.append([f'Split {sp}', ' > '.join(n for n, _ in sorted(vals.items(), key=lambda kv: -kv[1]) if not np.isnan(vals[n]))])
table('Ranking of the methods by F1-score in each split', ['Split', 'Ranking'], rank)
open('more_results.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
