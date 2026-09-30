"""Pick the noise multiplier for a target epsilon BEFORE training (no GPU needed).
Uses the worst-case tenant: the number of rounds a tenant trains is Binomial(rounds, fraction);
we take the maximum over tenants (approximated by mean + 3 sd, capped at `rounds`).
  python choose_sigma.py --prep prepared --rounds 50 --fraction 0.3 --target_eps 8.0"""
import argparse, json, math
from accountant import find_sigma, epsilon

ap = argparse.ArgumentParser()
ap.add_argument('--prep', default='prepared'); ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--local_epochs', type=int, default=5); ap.add_argument('--rounds', type=int, default=50)
ap.add_argument('--fraction', type=float, default=0.3); ap.add_argument('--target_eps', type=float, default=8.0)
ap.add_argument('--delta', type=float, default=1e-6); ap.add_argument('--margin', type=float, default=0.95,
    help='aim for margin*target so the final epsilon lands just below the target')
ap.add_argument('--quiet', action='store_true')
a = ap.parse_args()
meta = json.load(open(f'{a.prep}/meta.json')); W = meta['windows_per_tenant']
q = min(1.0, a.batch / W)
r_worst = min(a.rounds, math.ceil(a.rounds * a.fraction + 3 * math.sqrt(a.rounds * a.fraction * (1 - a.fraction))))
steps = a.local_epochs * max(1, round(W / a.batch)) * r_worst
sigma = find_sigma(q, steps, a.delta, a.target_eps * a.margin)
if a.quiet: print(f'{sigma:.3f}')
else:
    print(f'windows per tenant {W}, sampling rate {q:.4f}, worst tenant trains {r_worst}/{a.rounds} rounds, {steps} DP steps')
    print(f'sigma = {sigma:.3f}  ->  epsilon (my accountant) = {epsilon(q, sigma, steps, a.delta)[0]:.2f} at delta = {a.delta}')
