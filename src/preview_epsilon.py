"""Preview the privacy budget BEFORE running experiments (no GPU / torch needed).

Example (your Chapter 3 setting, ~180k flows per tenant, 15-flow windows, batch 256):
  python preview_epsilon.py --windows_per_tenant 12000 --batch 256 --sigma 1.1 \
      --rounds 150 --local_epochs 5 --participation 1.0 --delta 1e-6 --target_eps 6.5
"""
import argparse
from accountant import epsilon, find_sigma

ap = argparse.ArgumentParser()
ap.add_argument('--windows_per_tenant', type=int, required=True)
ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--sigma', type=float, default=1.1)
ap.add_argument('--rounds', type=int, default=150)
ap.add_argument('--local_epochs', type=int, default=5)
ap.add_argument('--participation', type=float, default=1.0,
                help='fraction of rounds in which a given tenant trains (worst case = 1.0)')
ap.add_argument('--delta', type=float, default=1e-6)
ap.add_argument('--target_eps', type=float, default=None)
a = ap.parse_args()
q = a.batch / a.windows_per_tenant
steps_per_round = a.local_epochs * max(1, round(a.windows_per_tenant / a.batch))
steps = int(steps_per_round * a.rounds * a.participation)
eps, order = epsilon(q, a.sigma, steps, a.delta)
print(f'sampling rate q = {q:.5f}, DP steps per tenant = {steps}')
print(f'sigma = {a.sigma}: epsilon = {eps:.2f} (delta = {a.delta}, best RDP order {order})')
if a.target_eps:
    s = find_sigma(q, steps, a.delta, a.target_eps)
    print(f'sigma needed for epsilon <= {a.target_eps}: {s:.2f}')
