"""Compute the final privacy budget from the per-client DP log written by fl_flower.py.
Record-level DP: every training window belongs to ONE tenant, so the guarantee is the
WORST tenant's epsilon (the tenant that trained in the most rounds).
Reports our numpy accountant and Opacus' RDPAccountant side by side."""
import argparse, json
from collections import defaultdict
from accountant import epsilon_from_history

def compute(log_path, delta):
    per = defaultdict(list)
    for line in open(log_path):
        r = json.loads(line); per[r['cid']].append((r['sigma'], r['q'], r['steps']))
    eps = {c: epsilon_from_history(h, delta)[0] for c, h in per.items()}
    worst = max(eps, key=eps.get)
    out = dict(delta=delta, n_clients=len(eps), worst_client=worst, rounds_worst_client=len(per[worst]),
               eps_numpy_accountant=eps[worst])
    try:
        from opacus.accountants import RDPAccountant
        acc = RDPAccountant(); acc.history = list(per[worst])
        out['eps_opacus'] = acc.get_epsilon(delta)
    except Exception as e:
        out['eps_opacus'] = f'not computed ({type(e).__name__}: {e})'
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('log'); ap.add_argument('--delta', type=float, default=1e-6)
    a = ap.parse_args(); print(json.dumps(compute(a.log, a.delta), indent=2))
