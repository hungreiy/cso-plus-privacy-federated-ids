"""Data preparation for the supervised-hybrid detector on CIC-IDS2017 (MachineLearningCVE CSVs).

Pipeline (thesis Section 3.5):
  * per file, non-overlapping windows of 15 consecutive flows; a window is an ATTACK if any of its flows is an attack
  * the windows are cut into CONTIGUOUS BLOCKS (--block windows each) and whole blocks go to train / val / test,
    so neighbouring windows of the same attack never end up on both sides of a split
  * attack types listed in --holdout ("unseen" attacks) are removed from train and val and appear only in the test set
  * Min-Max scaler fitted on the training windows only
  * every window gets a 32-d knowledge-graph vector (mean of the destination-port embeddings of its 15 flows)
  * the training windows are partitioned across the tenants (iid, or contiguous blocks)
Files written to --out_dir: {train,val,test}_{x,kg,y}.npy, test_grp.npy (0 benign, 1 seen attack, 2 unseen attack),
test_type.npy (label id of the dominant attack type, -1 benign), test_port.npy, types.json, tenants.json, meta.json
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import ids2018

DEFAULT_HOLDOUT = 'DoS GoldenEye,DoS slowloris,DoS Slowhttptest,Bot,Web Attack,Infiltration,Heartbleed'

def read_csv_file(path):
    df = pd.read_csv(path, encoding='latin1', low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    labels = df['Label'].astype(str).str.strip().to_numpy()
    feats = df.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    ok = feats.notna().all(axis=1).to_numpy()
    return feats.to_numpy(dtype=np.float32)[ok], labels[ok], list(feats.columns), int((~ok).sum())

def make_lut(kg_path):
    """65536 x dim table: destination port -> embedding (ports missing from the file get the mean vector)"""
    z = np.load(kg_path)
    default = z['vectors'].mean(axis=0).astype(np.float32)
    lut = np.tile(default, (65536, 1))
    for p, v in zip(z['ports'], z['vectors']):
        if 0 <= int(p) < 65536: lut[int(p)] = v
    return lut

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_dir', required=True); ap.add_argument('--kg', required=True, help='npz with `ports` and `vectors` (CSO+ Node2Vec)')
    ap.add_argument('--out_dir', default='prepared'); ap.add_argument('--n_tenants', type=int, default=10)
    ap.add_argument('--window', type=int, default=15); ap.add_argument('--partition', choices=['iid', 'contiguous', 'dirichlet'], default='iid')
    ap.add_argument('--alpha', type=float, default=0.5, help='dirichlet partition: smaller = more different attack mixes per tenant')
    ap.add_argument('--dataset', choices=['cic2017', 'cic2018'], default='cic2017')
    ap.add_argument('--min_tenant', type=int, default=512, help='dirichlet: stop if a tenant gets fewer windows than this')
    ap.add_argument('--max_windows', type=int, default=0, help='0 = all; otherwise keep whole blocks up to about this many windows')
    ap.add_argument('--cap_per_type', type=int, default=0, help='with --max_windows: at most this many windows from blocks dominated by one attack type')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--benign_label', default=None)
    ap.add_argument('--holdout', default=None, help='comma-separated label prefixes kept out of training ("unseen" attacks)')
    ap.add_argument('--block', type=int, default=200, help='windows per contiguous block')
    ap.add_argument('--train_frac', type=float, default=0.6); ap.add_argument('--val_frac', type=float, default=0.1)
    a = ap.parse_args()
    if a.benign_label is None: a.benign_label = 'BENIGN' if a.dataset == 'cic2017' else 'Benign'
    if a.holdout is None: a.holdout = DEFAULT_HOLDOUT if a.dataset == 'cic2017' else ids2018.HOLDOUT_2018
    reader = read_csv_file if a.dataset == 'cic2017' else ids2018.read_csv_2018
    os.makedirs(a.out_dir, exist_ok=True); rng = np.random.default_rng(a.seed); W = a.window
    files = sorted(glob.glob(os.path.join(a.csv_dir, '*.csv')))
    if not files: raise SystemExit(f'No CSV files found in {a.csv_dir}')

    Xs, Ls, names, label2id, dropped_total = [], [], None, {}, 0
    for f in files:
        X, lab, cols, dropped = reader(f)
        if names is None: names = cols
        assert cols == names, f'column mismatch in {f}'
        dropped_total += dropped
        uniq = np.unique(lab)
        for u in uniq: label2id.setdefault(str(u), len(label2id))
        ids = np.empty(len(lab), dtype=np.int16)
        for u in uniq: ids[lab == u] = label2id[str(u)]
        n = len(X) // W
        Xs.append(X[:n * W].reshape(n, W, -1)); Ls.append(ids[:n * W].reshape(n, W)); del X
        print(f'loaded {os.path.basename(f)}: {len(lab)} flows -> {n} windows', flush=True)
    X = np.concatenate(Xs); L = np.concatenate(Ls); del Xs, Ls
    n_win, nf = X.shape[0], X.shape[2]
    print(f'dropped {dropped_total} flows with NaN/inf; {n_win} windows in total', flush=True)

    id2label = {v: k for k, v in label2id.items()}; n_lab = len(id2label)
    type_names = [id2label[i] for i in range(n_lab)]                      # raw names (used for prefix matching)
    clean = lambda s: s.replace('\u00ef\u00bf\u00bd', '-').replace('\x96', '-')
    if a.benign_label not in label2id: raise SystemExit(f'benign label {a.benign_label!r} not found in {list(label2id)}')
    benign_id = label2id[a.benign_label]
    prefixes = [s.strip() for s in a.holdout.split(',') if s.strip()]
    is_hold = np.array([any(type_names[i].startswith(p) for p in prefixes) and i != benign_id for i in range(n_lab)])
    cnt = np.stack([(L == l).sum(axis=1) for l in range(n_lab)], axis=1)
    cnt[:, benign_id] = 0
    y = (cnt.sum(axis=1) > 0).astype(np.int8)
    has_unseen = (cnt[:, is_hold].sum(axis=1) > 0) if is_hold.any() else np.zeros(n_win, bool)
    grp = np.where(has_unseen, 2, np.where(y == 1, 1, 0)).astype(np.int8)
    wtype = np.full(n_win, -1, dtype=np.int16)
    att = y == 1
    wtype[att] = cnt[att].argmax(axis=1)
    if has_unseen.any(): wtype[has_unseen] = (cnt * is_hold[None, :])[has_unseen].argmax(axis=1)
    natt = cnt.sum(axis=1).astype(np.int16)          # number of attack flows in each window (0-15)
    del cnt

    block = np.arange(n_win) // a.block; nb = int(block.max()) + 1
    if a.max_windows and n_win > a.max_windows:
        # keep WHOLE blocks (so neighbouring windows stay together): cap blocks dominated by one attack type, then fill with benign blocks
        dom = np.full(nb, -1, np.int32)
        for b in range(nb):
            t = wtype[b * a.block:(b + 1) * a.block]; t = t[t >= 0]
            if len(t): dom[b] = np.bincount(t).argmax()
        keep = np.zeros(nb, bool); size = np.bincount(block, minlength=nb)
        for t in np.unique(dom[dom >= 0]):
            bs = rng.permutation(np.where(dom == t)[0]); cs = np.cumsum(size[bs])
            keep[bs if not a.cap_per_type else bs[cs <= max(a.cap_per_type, size[bs[0]])]] = True
        ben = rng.permutation(np.where(dom == -1)[0]); room = a.max_windows - int(size[keep].sum())
        if room > 0: keep[ben[np.cumsum(size[ben]) <= room]] = True
        sel = keep[block]
        X, L, y, grp, wtype, natt = X[sel], L[sel], y[sel], grp[sel], wtype[sel], natt[sel]
        n_win = len(X); old = np.unique(block[sel]); block = np.searchsorted(old, block[sel]); nb = len(old)
        print(f'block selection: kept {nb} blocks, {n_win} windows (max_windows={a.max_windows}, cap_per_type={a.cap_per_type})', flush=True)
    perm = rng.permutation(nb); n_tr, n_va = int(a.train_frac * nb), int(a.val_frac * nb)
    split_of_block = np.empty(nb, dtype=np.int8)
    split_of_block[perm[:n_tr]] = 0; split_of_block[perm[n_tr:n_tr + n_va]] = 1; split_of_block[perm[n_tr + n_va:]] = 2
    split = split_of_block[block].copy(); split[grp == 2] = 2
    tr, va, te = (np.where(split == s)[0] for s in (0, 1, 2))
    for nm, idx in (('train', tr), ('val', va), ('test', te)):
        print(f'{nm:5s}: {len(idx):7d} windows | benign {int((grp[idx]==0).sum()):7d} | seen attack {int((grp[idx]==1).sum()):6d} | unseen attack {int((grp[idx]==2).sum()):5d}', flush=True)
    assert (y[tr] == 1).any() and (y[tr] == 0).any(), 'training set needs both classes (use a smaller --block)'
    assert (y[va] == 0).any(), 'validation set needs benign windows'

    pi = names.index('Destination Port')
    lo = np.full(nf, np.inf, np.float32); hi = np.full(nf, -np.inf, np.float32)
    for i in range(0, len(tr), 20000):
        blk = X[tr[i:i + 20000]].reshape(-1, nf); lo = np.minimum(lo, blk.min(axis=0)); hi = np.maximum(hi, blk.max(axis=0))
    span = np.where(hi - lo == 0, 1.0, hi - lo).astype(np.float32)
    np.savez(f'{a.out_dir}/scaler.npz', lo=lo, span=span, names=np.array(names))
    lut = make_lut(a.kg); kgdim = lut.shape[1]

    def dump(name, idx, with_port=False):
        n = len(idx); xs = np.empty((n, W, nf), np.float32); kg = np.empty((n, kgdim), np.float32)
        ports_mode = np.empty(n, np.int32) if with_port else None
        for i in range(0, n, 20000):
            blk = X[idx[i:i + 20000]]; ports = np.clip(blk[:, :, pi].astype(np.int64), 0, 65535)
            kg[i:i + 20000] = lut[ports].mean(axis=1)
            xs[i:i + 20000] = np.clip((blk - lo) / span, 0, 1)
            if with_port:
                ports_mode[i:i + 20000] = [np.bincount(r).argmax() for r in ports]
        np.save(f'{a.out_dir}/{name}_x.npy', xs); np.save(f'{a.out_dir}/{name}_kg.npy', kg); np.save(f'{a.out_dir}/{name}_y.npy', y[idx])
        if with_port: np.save(f'{a.out_dir}/{name}_port.npy', ports_mode)
    dump('train', tr); dump('val', va); dump('test', te, with_port=True)
    np.save(f'{a.out_dir}/test_grp.npy', grp[te]); np.save(f'{a.out_dir}/test_type.npy', wtype[te]); np.save(f'{a.out_dir}/test_natt.npy', natt[te].astype(np.int8))
    type_names = [clean(t) for t in type_names]
    json.dump(type_names, open(f'{a.out_dir}/types.json', 'w'))
    if a.partition == 'dirichlet':
        # non-IID tenants: every window class (benign and each attack type) is spread over the tenants with Dirichlet(alpha)
        # proportions, so tenants see different attack mixes and different sizes
        cls = wtype[tr]; parts = [[] for _ in range(a.n_tenants)]
        for c in np.unique(cls):
            idx = rng.permutation(np.where(cls == c)[0]); p = rng.dirichlet([a.alpha] * a.n_tenants)
            for k, chunk in enumerate(np.split(idx, (np.cumsum(p)[:-1] * len(idx)).astype(int))): parts[k].extend(chunk.tolist())
        parts = [sorted(p) for p in parts]
        if min(len(p) for p in parts) < a.min_tenant: raise SystemExit(f'a tenant got {min(len(p) for p in parts)} windows (< --min_tenant {a.min_tenant}); use a larger --alpha or another --seed')
    else:
        order = rng.permutation(len(tr)) if a.partition == 'iid' else np.arange(len(tr))
        parts = [p.tolist() for p in np.array_split(order, a.n_tenants)]
    json.dump({str(i): p for i, p in enumerate(parts)}, open(f'{a.out_dir}/tenants.json', 'w'))
    tenant_mix = [{('benign' if int(c) < 0 else type_names[int(c)]): int((wtype[tr][p] == c).sum()) for c in np.unique(wtype[tr][p])} for p in map(np.array, parts)]
    per_type_test = {type_names[int(t)]: int((wtype[te] == t).sum()) for t in np.unique(wtype[te][wtype[te] >= 0])}
    meta = dict(features=names, n_features=nf, window=W, n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)),
                n_train_attack=int(y[tr].sum()), n_test_benign=int((grp[te] == 0).sum()), n_test_seen=int((grp[te] == 1).sum()),
                n_test_unseen=int((grp[te] == 2).sum()), test_windows_per_type=per_type_test,
                test_attack_windows_by_flows={'1-3': int(((natt[te] >= 1) & (natt[te] <= 3)).sum()), '4-7': int(((natt[te] >= 4) & (natt[te] <= 7)).sum()), '8-15': int((natt[te] >= 8).sum())}, holdout_prefixes=prefixes,
                block=a.block, n_tenants=a.n_tenants, partition=a.partition, alpha=a.alpha if a.partition == 'dirichlet' else None, dataset=a.dataset,
                max_windows=a.max_windows, cap_per_type=a.cap_per_type, tenant_sizes=[len(p) for p in parts], tenant_mix=tenant_mix,
                windows_per_tenant=int(min(len(p) for p in parts)),   # smallest tenant: largest sampling rate, so the privacy bound holds for all
                seed=a.seed, kg_dim=kgdim, dropped_flows=dropped_total, n_windows=int(n_win))
    json.dump(meta, open(f'{a.out_dir}/meta.json', 'w'), indent=2)
    print(f'>>> windows per tenant = {meta["windows_per_tenant"]}; unseen attack types in test: {[t for t in per_type_test if any(t.startswith(p) for p in prefixes)]}')

if __name__ == '__main__':
    main()
