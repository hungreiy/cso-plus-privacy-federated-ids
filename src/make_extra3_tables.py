"""Tenant-level DP, shuffled-ontology control, scaling to 50/100 tenants, and model-replacement attacks -> extra3_results.md"""
import json, os, numpy as np
out = []
def L(d, tag, S): return [json.load(open(f'{d}/{tag}_seed{s}.json')) for s in S if os.path.exists(f'{d}/{tag}_seed{s}.json')]
def ms(v, nd=4): return f'{np.mean(v):.{nd}f} ± {np.std(v, ddof=1):.{nd}f}' if len(v) > 1 else (f'{v[0]:.{nd}f}' if v else '-')
def table(title, head, rows):
    out.append(f'\n### {title}\n'); out.append('| ' + ' | '.join(head) + ' |'); out.append('|' + '---|' * len(head)); out.extend('| ' + ' | '.join(map(str, r)) + ' |' for r in rows)
F1 = lambda J: ms([j['f1'] for j in J]); CL = lambda J: ms([j['clean']['f1'] for j in J]); RK = lambda J: ms([j['recall_seen'] for j in J]); AU = lambda J: ms([j['auc'] for j in J])
# 1. tenant-level DP
rows = []
for c in ('0.5', '5'):
    J = L('results_userdp', f'fl_nodp_udp{c}', range(3))
    rows.append([c, len(J), F1(J), CL(J), RK(J), AU(J), f"{J[0]['user_privacy']['eps_tenant_level']:.2f}" if J else '-'])
J = L('results', 'fl_dp', range(3)); rows.append(['record-level DP-SGD (main)', len(J), F1(J), CL(J), RK(J), AU(J), 'record-level ' + ms([j['privacy']['eps_opacus'] for j in J], 2)])
table('Tenant-level DP-FedAvg (sigma 1.74, 10 tenants, 3 per round, 50 rounds, seeds 0-2)', ['Update clip norm', 'n', 'F1', 'F1 clear', 'Recall known', 'AUC', 'ε'], rows)
# 2. shuffled ontology
from scipy.stats import wilcoxon
G = [('CSO+ embeddings (main)', 'results', 'fl_dp'), ('Shuffled CSO+ (control)', 'results_kgshuf', 'fl_dp'), ('Random vectors (control)', 'results_kgrand', 'fl_dp'), ('No embeddings', 'results', 'fl_dp_nokg')]
D = {n: L(d, t, range(10)) for n, d, t in G}
table('Shuffled-ontology control (DP-FL, seeds 0-9)', ['Model', 'n', 'F1', 'F1 clear', 'AUC', 'Recall known'], [[n, len(D[n]), F1(D[n]), CL(D[n]), AU(D[n]), RK(D[n])] for n, _, _ in G])
def paired(a, b):
    A = {j['seed']: j['f1'] for j in D[a]}; B = {j['seed']: j['f1'] for j in D[b]}; ss = sorted(set(A) & set(B))
    if len(ss) < 6: return [a, b, len(ss), '-', '-', '-']
    x = np.array([A[s] for s in ss]); y = np.array([B[s] for s in ss]); return [a, b, len(ss), f'{(x - y).mean():+.4f}', f'{int((x > y).sum())}/{len(ss)}', f'{wilcoxon(x, y).pvalue:.4f}']
table('Paired comparisons on F1 (Wilcoxon signed-rank)', ['A', 'B', 'n', 'Mean A-B', 'A > B', 'p'],
      [paired('CSO+ embeddings (main)', 'Shuffled CSO+ (control)'), paired('Shuffled CSO+ (control)', 'No embeddings'), paired('Shuffled CSO+ (control)', 'Random vectors (control)')])
# 3. scaling
rows = []
for n, d in (('10 (main)', None), ('50', 'results_t50'), ('100', 'results_t100')):
    if d is None: R = L('results_private', 'local_rf', range(3)); A = L('results', 'fl_nodp', range(3)); P = L('results', 'fl_dp', range(3))
    else: R = L(d, 'local_rf', range(3)); A = L(d, 'fl_nodp', range(3)); P = L(d, 'fl_dp', range(3))
    rows.append([n, ms([j['f1'] for j in R]), ms([j['f1_min'] for j in R]), F1(A), F1(P), CL(P), ms([j['privacy']['eps_opacus'] for j in P], 2) if P else '-'])
table('Scaling the number of tenants (IID, seeds 0-2)', ['Tenants', 'Local RF F1 (mean)', 'Local RF F1 (weakest tenant)', 'FedAvg F1', 'DP-FL F1', 'DP-FL F1 clear', 'ε (busiest tenant)'], rows)
# 4. model replacement
rows = [['0', 'FedAvg', *(lambda J: [len(J), F1(J), CL(J), RK(J)])(L('results', 'fl_dp', range(3)))],
        ['0', 'Norm bounding', *(lambda J: [len(J), F1(J), CL(J), RK(J)])(L('results_replace', 'fl_dp_normclip', range(3)))]]
for m in (1, 3):
    for st, tag in (('FedAvg', f'fl_dp_m{m}_replace'), ('FedMedian', f'fl_dp_m{m}_median_replace'), ('Norm bounding', f'fl_dp_m{m}_replace_normclip')):
        J = L('results_replace', tag, range(3)); rows.append([m, st, len(J), F1(J), CL(J), RK(J)])
table('Model-replacement attack (label flipping, no DP, update boosted 10x) vs aggregation (seeds 0-2)', ['Malicious tenants', 'Aggregation', 'n', 'F1', 'F1 clear', 'Recall known'], rows)
open('extra3_results.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
