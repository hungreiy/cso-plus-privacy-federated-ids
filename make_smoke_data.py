"""SMOKE-TEST DATA ONLY (synthetic numbers shaped like CIC-IDS2017). NOT for any thesis result."""
import numpy as np, pandas as pd, os, sys
out = sys.argv[1] if len(sys.argv) > 1 else 'smoke_csv'
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
cols = ['Destination Port'] + [f'feat_{i}' for i in range(77)]
types = ['DoS Hulk', 'PortScan', 'Bot']            # 'Bot' is held out (unseen) by data.py's default
for d in range(2):
    n = 9000
    X = rng.gamma(2.0, 1.0, size=(n, 78)); X[:, 0] = rng.choice([80, 443, 53, 22, 3389], size=n)
    lab = np.array(['BENIGN'] * n, dtype=object)
    for c in range(6):
        t = types[(c + d) % 3]; start = c * 1500 + int(rng.integers(0, 1000)); length = 300
        X[start:start + length, 1:] += rng.normal(3.0, 1.0, size=(length, 77)) + types.index(t)
        lab[start:start + length] = t
    df = pd.DataFrame(X, columns=cols); df['Label'] = lab
    df.to_csv(f'{out}/day{d}.csv', index=False)
ports = np.array([80, 443, 53, 22, 3389]); vec = rng.normal(size=(5, 32)).astype(np.float32)
np.savez(f'{out}/SMOKE_ONLY_kg_embeddings.npz', ports=ports, vectors=vec)
print('wrote smoke data to', out)
