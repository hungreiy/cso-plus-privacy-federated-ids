"""Example explanations for alerts: SHAP top-5 features of the attack logit + the ontology path of the window's
most frequent destination port (from CSO+). Writes a JSON list you can quote in Chapter 4."""
import argparse, json, os
import numpy as np, torch, shap
from model import CNNLSTMAE, N_FLOW, KG_DIM, T
from evalutil import compute, load_prep
from ontology_path import path_for_port

ap = argparse.ArgumentParser()
ap.add_argument('--prep', default='prepared'); ap.add_argument('--weights', required=True)
ap.add_argument('--dp_model', action='store_true', help='weights come from the DP run (GroupNorm + DPLSTM)')
ap.add_argument('--out', default='results/shap_examples.json'); ap.add_argument('--n_alerts', type=int, default=20)
a = ap.parse_args()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prep = load_prep(a.prep)
net = CNNLSTMAE(norm='group', dp=a.dp_model); net.load_state_dict(torch.load(a.weights, map_location='cpu')); net.to(dev).eval()
names = json.load(open(f'{a.prep}/meta.json'))['features'] + [f'kg_{i}' for i in range(KG_DIM)]

class Score(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, z): return self.m.logit(z[:, :, :N_FLOW], z[:, 0, N_FLOW:]).unsqueeze(1)
def pack(x, kg): return torch.cat([torch.as_tensor(np.asarray(x)).float(), torch.as_tensor(np.asarray(kg)).float().unsqueeze(1).expand(-1, T, -1)], 2)

yv = np.asarray(prep['val_y']); _, v_lg = compute(net, prep['val_x'], prep['val_kg'], dev); thr = float(np.percentile(v_lg[yv == 0], 99.5))
_, t_lg = compute(net, prep['test_x'], prep['test_kg'], dev); y = np.asarray(prep['test_y'])
alerts = np.where((t_lg > thr) & (y == 1))[0][:a.n_alerts]
if len(alerts) == 0: raise SystemExit('no true alerts to explain')
bg = pack(np.asarray(prep['train_x'])[:200], np.asarray(prep['train_kg'])[:200]).to(dev)
with torch.backends.cudnn.flags(enabled=False):
    sv = shap.GradientExplainer(Score(net), bg).shap_values(pack(prep['test_x'][alerts], prep['test_kg'][alerts]).to(dev))
sv = np.asarray(sv[0] if isinstance(sv, list) else sv).reshape(len(alerts), T, -1)
imp = np.abs(sv).sum(axis=1)
ports = np.load(f'{a.prep}/test_port.npy'); types = prep['type_names']; ttype = np.asarray(prep['test_type'])
out = []
for k, i in enumerate(alerts):
    top = [(names[j], float(imp[k, j])) for j in np.argsort(-imp[k])[:5]]
    path = path_for_port(int(ports[i]))
    out.append(dict(window=int(i), attack_type=types[int(ttype[i])], logit=float(t_lg[i]), threshold=thr, top5=top, ontology_path=path,
                    explanation=(f'Alert ({types[int(ttype[i])]}): attack score {t_lg[i]:.2f} (threshold {thr:.2f}). Main contributing features: '
                                 + ', '.join(f'{n} ({v:.2f})' for n, v in top) + f'. Ontology context: {path}')))
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); json.dump(out, open(a.out, 'w'), indent=2); print(f'wrote {a.out}')
