"""Choose the reconstruction weight and the learning rate of the PRIVATE federated model with the VALIDATION set only,
then run the final five-seed comparison at a lower noise level (epsilon close to 8 for the busiest tenant).

Why the validation set: the test set was already used to report results, so choosing hyperparameters on it would make the
tuned numbers optimistic. Validation runs use fl_flower.py --skip_test (the test set is never evaluated); the selection metric is the
validation average precision (val_ap), with recall at 0.5 % false positives as tie-breaker.

Privacy: a tenant trains at most --cap rounds (a rule that does not look at any data), and sigma is the smallest noise
multiplier for which the epsilon of `cap` rounds is at most --target_eps, so the busiest tenant's epsilon is at most the target
and is close to it whenever the cap is reached.

  python tune_dp.py --stage all            # about 7-8 h on an RTX 4000 Ada class GPU; resumable (finished runs are skipped)
  python tune_dp.py --stage 1              # only the grid (about 2 h)
  python tune_dp.py --stage 2 / --stage 3  # top-k full-length runs / final seeds
"""
import argparse, json, os, shutil, subprocess, sys, glob, itertools
from accountant import find_sigma, epsilon

ap = argparse.ArgumentParser()
ap.add_argument('--prep', default='/root/prepared'); ap.add_argument('--out', default='results_tuned')
ap.add_argument('--stage', default='all', choices=['1', '2', '3', 'all'])
ap.add_argument('--target_eps', type=float, default=8.0); ap.add_argument('--cap', type=int, default=20)
ap.add_argument('--delta', type=float, default=1e-6); ap.add_argument('--sigma', type=float, default=0.0, help='0 = compute from target_eps and cap')
ap.add_argument('--rw', type=float, nargs='+', default=[0, 1, 10, 30]); ap.add_argument('--lrs', type=float, nargs='+', default=[3e-4, 1e-3, 3e-3])
ap.add_argument('--stage1_rounds', type=int, default=20); ap.add_argument('--stage2_rounds', type=int, default=50); ap.add_argument('--topk', type=int, default=3)
ap.add_argument('--tune_seed', type=int, default=100); ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
ap.add_argument('--baseline_results', default='results', help='folder with the earlier rf/central results, copied for the tables')
ap.add_argument('--dry_run', action='store_true')
a = ap.parse_args()
meta = json.load(open(f'{a.prep}/meta.json')); W = meta['windows_per_tenant']; q = min(1.0, 256 / W); spr = 5 * max(1, round(W / 256))
SIGMA = a.sigma or round(find_sigma(q, spr * a.cap, a.delta, a.target_eps), 3)
print(f'windows per tenant {W}; cap {a.cap} rounds; sigma {SIGMA}; epsilon (own accountant) at 18/20/{a.cap} rounds: '
      + ', '.join(f'{r}: {epsilon(q, SIGMA, spr * r, a.delta)[0]:.2f}' for r in sorted({18, 20, a.cap})), flush=True)
os.makedirs(a.out, exist_ok=True)

def run_cmd(cmd):
    print('RUN', ' '.join(cmd), flush=True)
    if a.dry_run: return 0
    return subprocess.call(cmd)

def fl(folder, seed, rw, lr, rounds, dp, kg=True, skip_test=False):
    tag = ('fl_dp' if dp else 'fl_nodp') + ('' if kg else '_nokg')
    path = f'{folder}/{tag}_seed{seed}.json'
    if os.path.exists(path): return path
    if not a.dry_run: os.makedirs(folder, exist_ok=True)
    cmd = [sys.executable, 'fl_flower.py', '--prep', a.prep, '--out', folder, '--dp', 'on' if dp else 'off', '--clients', '10', '--rounds', str(rounds),
           '--fraction', '0.3', '--local_epochs', '5', '--eval_every', '5', '--seed', str(seed), '--rec_weight', str(rw), '--lr', str(lr)]
    if dp: cmd += ['--sigma', str(SIGMA), '--max_rounds_per_tenant', str(a.cap), '--delta', str(a.delta)]
    if not kg: cmd += ['--no_kg']
    if skip_test: cmd += ['--skip_test']
    if run_cmd(cmd) != 0: print('FAILED:', path, flush=True)
    return path

def score(path):
    r = json.load(open(path))['val_selection']; return (r['val_ap'], r['val_recall_at_fpr005'])

def stage1():
    rows = []
    for rw, lr in itertools.product(a.rw, a.lrs):
        p = fl(f'{a.out}/stage1/rw{rw:g}_lr{lr:g}', a.tune_seed, rw, lr, a.stage1_rounds, dp=True, skip_test=True)
        if os.path.exists(p): rows.append(dict(rec_weight=rw, lr=lr, **json.load(open(p))['val_selection']))
    rows.sort(key=lambda r: (r['val_ap'], r['val_recall_at_fpr005']), reverse=True)
    json.dump(rows, open(f'{a.out}/stage1_summary.json', 'w'), indent=2)
    print('\nSTAGE 1 (validation only, sorted):'); [print(f"  rec_weight {r['rec_weight']:>4g}  lr {r['lr']:<7g}  val_ap {r['val_ap']:.4f}  val_auc {r['val_auc']:.4f}  val_recall@0.5%FPR {r['val_recall_at_fpr005']:.4f}") for r in rows]
    return rows

def stage2(rows):
    top = rows[:a.topk]; res = []
    for r in top:
        p = fl(f"{a.out}/stage2/rw{r['rec_weight']:g}_lr{r['lr']:g}", a.tune_seed, r['rec_weight'], r['lr'], a.stage2_rounds, dp=True, skip_test=True)
        if os.path.exists(p): res.append(dict(rec_weight=r['rec_weight'], lr=r['lr'], **json.load(open(p))['val_selection'], privacy=json.load(open(p)).get('privacy')))
    res.sort(key=lambda r: (r['val_ap'], r['val_recall_at_fpr005']), reverse=True)
    json.dump(res, open(f'{a.out}/stage2_summary.json', 'w'), indent=2)
    print('\nSTAGE 2 (full length, validation only, sorted):'); [print(f"  rec_weight {r['rec_weight']:>4g}  lr {r['lr']:<7g}  val_ap {r['val_ap']:.4f}  val_recall@0.5%FPR {r['val_recall_at_fpr005']:.4f}") for r in res]
    # tie-break fixed before seeing stage 2: among configurations within 0.005 validation AP of the best, prefer one with rec_weight > 0,
    # because the validation set has no unseen attacks and cannot reward the reconstruction branch that the design keeps for them
    chosen = res[0]
    if chosen['rec_weight'] == 0:
        near = [r for r in res if r['rec_weight'] > 0 and chosen['val_ap'] - r['val_ap'] <= 0.005]
        if near: chosen = near[0]
    best = dict(rec_weight=chosen['rec_weight'], lr=chosen['lr'], sigma=SIGMA, cap=a.cap, target_eps=a.target_eps,
                selected_on='validation average precision, seed %d (tie-break: prefer rec_weight > 0 within 0.005)' % a.tune_seed)
    json.dump(best, open(f'{a.out}/best.json', 'w'), indent=2); print('BEST:', best); return best

def stage3(best):
    rw, lr = best['rec_weight'], best['lr']; out = a.out + '/final'; os.makedirs(out, exist_ok=True)
    for s in a.seeds:
        fl(out, s, rw, lr, 50, dp=True); fl(out, s, rw, lr, 50, dp=True, kg=False); fl(out, s, rw, lr, 50, dp=False)
        if not a.dry_run and os.path.exists(f'{out}/fl_dp_seed{s}.pt'):
            if not os.path.exists(f'{out}/mia_fl_dp_seed{s}.json'): run_cmd([sys.executable, 'mia.py', '--prep', a.prep, '--weights', f'{out}/fl_dp_seed{s}.pt', '--dp_model', '--rec_weight', str(rw), '--out', f'{out}/mia_fl_dp_seed{s}.json', '--seed', str(s)])
            ref = f'{a.baseline_results}/central_seed{s}.pt'
            if not os.path.exists(f'{out}/fid_fl_dp_seed{s}.json'):
                run_cmd([sys.executable, 'explain_eval.py', '--prep', a.prep, '--weights', f'{out}/fl_dp_seed{s}.pt', '--dp_model', '--out', f'{out}/fid_fl_dp_seed{s}.json', '--seed', str(s)] + (['--ref_weights', ref] if os.path.exists(ref) else []))
    for pat in ('rf_seed*.json', 'central_seed*.json', 'central_nokg_seed*.json', 'mia_central_seed*.json', 'fid_central_seed*.json'):   # unchanged baselines, for the tables
        for f in glob.glob(f'{a.baseline_results}/{pat}'):
            if not a.dry_run: shutil.copy(f, out)
    if not a.dry_run: run_cmd([sys.executable, 'make_tables.py', out])

if a.stage in ('1', 'all'): rows = stage1()
else: rows = json.load(open(f'{a.out}/stage1_summary.json')) if os.path.exists(f'{a.out}/stage1_summary.json') else []
if a.stage in ('2', 'all'): best = stage2(rows)
elif a.stage == '3': best = json.load(open(f'{a.out}/best.json'))
if a.stage in ('3', 'all'): stage3(best)
