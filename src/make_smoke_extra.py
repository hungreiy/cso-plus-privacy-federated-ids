"""SMOKE-TEST DATA ONLY: synthetic CSVs in the CIC-IDS2017 column layout and the CSE-CIC-IDS2018 layout
(abbreviated names, Timestamp, Protocol, extra identifier columns in one file, a repeated header line). NOT for results."""
import os, sys, numpy as np, pandas as pd
import ids2018
out = sys.argv[1] if len(sys.argv) > 1 else 'smoke_extra'; rng = np.random.default_rng(0)
os.makedirs(f'{out}/csv17', exist_ok=True); os.makedirs(f'{out}/csv18', exist_ok=True)
for d in range(2):                                              # 2017 layout, real column names
    n = 9000; X = rng.gamma(2.0, 1.0, size=(n, 78)); X[:, 0] = rng.choice([80, 443, 21, 22, 3389], size=n); lab = np.array(['BENIGN'] * n, dtype=object)
    for c in range(8):
        t = ['DoS Hulk', 'PortScan', 'FTP-Patator', 'Bot'][(c + d) % 4]; st = c * 1100 + int(rng.integers(0, 600))
        X[st:st + 300, 1:] += rng.normal(3.0, 1.0, size=(300, 77)) + c % 3; lab[st:st + 300] = t
    df = pd.DataFrame(X, columns=[' ' + c for c in ids2018.ORDER_2017]); df[' Label'] = lab; df.to_csv(f'{out}/csv17/day{d}.csv', index=False)
inv = {v: k for k, v in ids2018.MAP_2018.items()}
feat18 = [inv[c] for c in ids2018.ORDER_2017 if c not in ('Destination Port', 'Fwd Header Length.1')]
for d in range(2):                                              # 2018 layout
    n = 9000; X = rng.gamma(2.0, 1.0, size=(n, len(feat18))); lab = np.array(['Benign'] * n, dtype=object)
    for c in range(8):
        t = ['DoS attacks-Hulk', 'FTP-BruteForce', 'Bot', 'Infilteration'][(c + d) % 4]; st = c * 1100 + int(rng.integers(0, 600))
        X[st:st + 300] += 3 + c % 3; lab[st:st + 300] = t
    df = pd.DataFrame(X, columns=feat18); s = np.arange(n)
    df.insert(0, 'Timestamp', [f'{14 + d}/02/2018 {h:02d}:{m:02d}:{x:02d}' for h, m, x in zip(s // 3600 % 24, s // 60 % 60, s % 60)])
    df.insert(0, 'Protocol', 6); df.insert(0, 'Dst Port', rng.choice([80, 443, 21, 22], size=n)); df['Label'] = lab
    if d == 1:
        for i, (k, v) in enumerate((('Flow ID', 'x'), ('Src IP', '1.1.1.1'), ('Src Port', 1), ('Dst IP', '2.2.2.2'))): df.insert(i, k, v)
    df = pd.concat([df.iloc[:4000], pd.DataFrame([df.columns], columns=df.columns), df.iloc[4000:]])
    df.to_csv(f'{out}/csv18/day{d}.csv', index=False)
ports = np.array([80, 443, 53, 21, 22, 3389]); np.savez(f'{out}/SMOKE_ONLY_kg.npz', ports=ports, vectors=rng.normal(size=(6, 32)).astype(np.float32))
print('wrote', out)
