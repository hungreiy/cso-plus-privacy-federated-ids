"""Shuffled-ontology control (Section 4.10.12): keeps every CSO+ vector but breaks the port-to-service meaning.
The ports that CSO+ describes individually (their vector differs from the default vector of their port range) exchange
their vectors at random, with no port keeping its own; all other ports keep their range vector. The set of vectors, and so
the geometry of the embedding space, is unchanged; only which port gets which service's vector is wrong.
  python make_shuffled_kg.py ../ontology/cso_plus_node2vec.npz ../ontology/shuffled_kg.npz"""
import sys, numpy as np
src, dst = sys.argv[1], sys.argv[2]
z = np.load(src); ports = z['ports']; V = z['vectors'].copy()
named = []
for lo, hi in ((0, 1023), (1024, 49151), (49152, 65535)):
    idx = np.flatnonzero((ports >= lo) & (ports <= hi))
    if len(idx) == 0: continue
    rows, inv, cnt = np.unique(V[idx].round(6), axis=0, return_inverse=True, return_counts=True)
    default = cnt.argmax()                                   # the range's own vector (shared by most ports in the range)
    named.extend(idx[inv.ravel() != default].tolist())
named = np.array(sorted(named)); rng = np.random.default_rng(12345)
perm = rng.permutation(len(named))
while (perm == np.arange(len(named))).any(): perm = rng.permutation(len(named))   # no port keeps its own vector
W = V.copy(); W[named] = V[named[perm]]
np.savez(dst, ports=ports, vectors=W)
print(f'shuffled vectors among {len(named)} individually described ports; examples:', {int(p): int(named[perm][i]) for i, p in enumerate(named[:5])}, '(port -> port whose vector it now has)')
