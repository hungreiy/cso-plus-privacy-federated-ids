"""Paired Wilcoxon signed-rank tests over seeds with Holm correction.
python stats.py results --ref fl_dp --others central fl_nodp fl_dp_nokg --metric f1
NOTE: with n paired runs the smallest two-sided exact p is 2/2^n (n=5 -> 0.0625, n=8 -> 0.0078, n=10 -> 0.002)."""
import argparse, glob, json
import numpy as np
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('--ref', required=True)
ap.add_argument('--others', nargs='+', required=True); ap.add_argument('--metric', default='f1')
a = ap.parse_args()
def load(tag):
    d = {}
    for f in glob.glob(f'{a.dir}/{tag}_seed*.json'):
        r = json.load(open(f)); d[r['seed']] = r[a.metric]
    return d
ref = load(a.ref); rows = []
for o in a.others:
    b = load(o); seeds = sorted(set(ref) & set(b))
    if len(seeds) < 6: print(f'{o}: only {len(seeds)} paired seeds -> not enough for a meaningful test'); continue
    x, y = np.array([ref[s] for s in seeds]), np.array([b[s] for s in seeds])
    p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
    rows.append((o, len(seeds), x.mean(), x.std(ddof=1), y.mean(), y.std(ddof=1), p))
order = np.argsort([r[-1] for r in rows]); m = len(rows); adj = {}
run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, rows[i][-1] * (m - rank))); adj[i] = run
print(f'{"vs":16s} n   {a.ref}(mean±sd)    other(mean±sd)     p      p_Holm')
for i, r in enumerate(rows):
    print(f'{r[0]:16s} {r[1]:<3d} {r[2]:.4f}±{r[3]:.4f}   {r[4]:.4f}±{r[5]:.4f}   {r[6]:.4f}  {adj[i]:.4f}')
