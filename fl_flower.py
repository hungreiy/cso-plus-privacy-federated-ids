"""Federated training with Flower (simulation) + optional record-level DP-SGD via Opacus.

  --dp off : FedAvg without DP  (baseline)
  --dp on  : FedAvg + local DP-SGD (per-example clipping, Gaussian noise, Poisson sampling)

Privacy accounting: each fit() logs (sigma, sampling rate, steps) per client per round to
dp_log_*.jsonl; final_epsilon.py composes them (worst-case tenant).
Tested against: flwr[simulation]==1.11.1 (needs Python 3.11), torch>=2.4, opacus==1.5.2.
"""
import argparse, json, os, time
import numpy as np, torch
import flwr as fl
from torch.utils.data import TensorDataset, DataLoader
from model import CNNLSTMAE, combined_loss, get_params, set_params
from evalutil import load_prep, apply_no_kg, full_eval, val_objective, val_selection
from caps import apply_cap
from final_epsilon import compute as compute_eps

ap = argparse.ArgumentParser()
ap.add_argument('--prep', default='prepared'); ap.add_argument('--out', default='results')
ap.add_argument('--dp', choices=['on', 'off'], default='off')
ap.add_argument('--sigma', type=float, default=1.1); ap.add_argument('--clip', type=float, default=1.0)
ap.add_argument('--delta', type=float, default=1e-6)
ap.add_argument('--clients', type=int, default=10); ap.add_argument('--rounds', type=int, default=50)
ap.add_argument('--fraction', type=float, default=0.3, help='fraction of tenants trained per round')
ap.add_argument('--local_epochs', type=int, default=5); ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=1e-5)
ap.add_argument('--rec_weight', type=float, default=10.0)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--eval_every', type=int, default=5)
ap.add_argument('--gpus_per_client', type=float, default=0.1); ap.add_argument('--cpus_per_client', type=float, default=1)
ap.add_argument('--no_kg', action='store_true')
ap.add_argument('--skip_test', action='store_true', help='tuning mode: compute validation metrics only, never touch the test set')
ap.add_argument('--max_rounds_per_tenant', type=int, default=0, help='0 = no cap; otherwise a tenant trains at most this many rounds')
args = ap.parse_args()
DP = args.dp == 'on'
TAG = ('fl_dp' if DP else 'fl_nodp') + ('_nokg' if args.no_kg else '')
os.makedirs(args.out, exist_ok=True)
LOG = f'{args.out}/dp_log_{TAG}_seed{args.seed}.jsonl'
if DP and os.path.exists(LOG): os.remove(LOG)
norm = 'group'   # BatchNorm is invalid under DP-SGD; GroupNorm in ALL FL runs so DP is the only difference
dev_name = 'cuda' if torch.cuda.is_available() else 'cpu'

def build(): return CNNLSTMAE(norm=norm, dp=DP)

class Tenant(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = int(cid)
        tenants = json.load(open(f'{args.prep}/tenants.json'))
        self.idx = np.array(tenants[str(self.cid)])
        prep = apply_no_kg(load_prep(args.prep), args.no_kg)
        self.x = torch.from_numpy(np.ascontiguousarray(prep['train_x'][self.idx]))
        self.kg = torch.from_numpy(np.ascontiguousarray(prep['train_kg'][self.idx]))
        self.y = torch.from_numpy(np.ascontiguousarray(prep['train_y'][self.idx]))

    def get_parameters(self, config): return get_params(build())

    def fit(self, parameters, config):
        rnd = int(config.get('round', 0))
        torch.manual_seed(args.seed * 100003 + self.cid * 1009 + rnd)
        dev = torch.device(dev_name)
        net = build(); set_params(net, parameters); net.to(dev).train()
        loader = DataLoader(TensorDataset(self.x, self.kg, self.y), batch_size=args.batch, shuffle=True)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
        model = net; steps = 0; q = args.batch / len(self.x)
        if DP:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator
            errs = ModuleValidator.validate(net, strict=False)
            assert not errs, f'Model not DP-compatible: {errs}'
            model, opt, loader = PrivacyEngine().make_private(
                module=net, optimizer=opt, data_loader=loader,
                noise_multiplier=args.sigma, max_grad_norm=args.clip, poisson_sampling=True)
            steps = len(loader) * args.local_epochs; q = loader.sample_rate
        for _ in range(args.local_epochs):
            for xb, kb, yb in loader:
                if xb.shape[0] == 0: continue
                xb, kb, yb = xb.to(dev), kb.to(dev), yb.to(dev)
                opt.zero_grad(); rec, lg = model(xb, kb)
                combined_loss(rec, lg, xb, yb, args.rec_weight).backward(); opt.step()
        if DP:
            with open(LOG, 'a') as f:
                f.write(json.dumps(dict(cid=self.cid, round=rnd, sigma=args.sigma, q=q, steps=steps)) + '\n')
        return get_params(net), len(self.x), {}

def client_fn(cid): return Tenant(cid).to_client()

prep_main = apply_no_kg(load_prep(args.prep), args.no_kg)
dev0 = torch.device(dev_name)
val_curve = []

def evaluate_fn(server_round, parameters, config):
    if server_round % args.eval_every and server_round != args.rounds: return None
    m = build(); set_params(m, parameters); m.to(dev0)
    v = val_objective(m, prep_main, dev0, args.rec_weight)
    val_curve.append(dict(round=server_round, val_objective=v)); print(f'round {server_round} val_objective {v:.5f}', flush=True)
    return v, {'val_objective': v}

class KeepFedAvg(fl.server.strategy.FedAvg):
    final_params = None
    def __init__(self, *a, max_rounds=0, **k):
        super().__init__(*a, **k); self.max_rounds = max_rounds; self.counts = {}
    def configure_fit(self, server_round, parameters, client_manager):
        return apply_cap(super().configure_fit(server_round, parameters, client_manager), self.counts, self.max_rounds)
    def aggregate_fit(self, server_round, results, failures):
        p, m = super().aggregate_fit(server_round, results, failures)
        if p is not None: self.final_params = p
        return p, m

torch.manual_seed(args.seed); np.random.seed(args.seed)
T0 = time.time()
strategy = KeepFedAvg(max_rounds=args.max_rounds_per_tenant,
        fraction_fit=args.fraction, fraction_evaluate=0.0,
    min_fit_clients=max(1, int(round(args.fraction * args.clients))), min_evaluate_clients=0,
    min_available_clients=args.clients, evaluate_fn=evaluate_fn,
    on_fit_config_fn=lambda r: {'round': r},
    initial_parameters=fl.common.ndarrays_to_parameters(get_params(build())))
fl.simulation.start_simulation(
    client_fn=client_fn, num_clients=args.clients,
    config=fl.server.ServerConfig(num_rounds=args.rounds), strategy=strategy,
    client_resources={'num_cpus': args.cpus_per_client, 'num_gpus': args.gpus_per_client if dev_name == 'cuda' else 0},
    ray_init_args={'include_dashboard': False})

TRAIN_SECONDS = time.time() - T0
final = build(); set_params(final, fl.common.parameters_to_ndarrays(strategy.final_params)); final.to(dev0)
res = {} if args.skip_test else full_eval(final, prep_main, dev0)
res['val_selection'] = val_selection(final, prep_main, dev0)
res.update(method=TAG, seed=args.seed, args=vars(args), val_curve=val_curve, train_seconds=TRAIN_SECONDS)
if DP: res['privacy'] = compute_eps(LOG, args.delta)
json.dump(res, open(f'{args.out}/{TAG}_seed{args.seed}.json', 'w'), indent=2)
torch.save(final.state_dict(), f'{args.out}/{TAG}_seed{args.seed}.pt')
print(res['val_selection'] if args.skip_test else {k: round(res[k], 4) for k in ['f1', 'precision', 'recall', 'auc', 'fpr', 'recall_seen', 'recall_unseen', 'auc_unseen']}, res.get('privacy'))
