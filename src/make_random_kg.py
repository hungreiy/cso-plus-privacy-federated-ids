"""Control for the knowledge-graph embeddings (Section 4.10.5).
Writes random vectors with the same shape and the same per-dimension mean and standard deviation as the
CSO+ Node2Vec vectors, so that every port still gets its own 32-dimensional code but without ontology semantics.
  python make_random_kg.py ../ontology/cso_plus_node2vec.npz ../ontology/random_kg.npz"""
import sys, numpy as np
src, dst = sys.argv[1], sys.argv[2]
z = np.load(src); V = z['vectors'].astype(np.float32)
rng = np.random.default_rng(12345)
R = rng.standard_normal(V.shape).astype(np.float32) * V.std(0, keepdims=True) + V.mean(0, keepdims=True)
np.savez(dst, ports=z['ports'], vectors=R)
print(f'random KG written to {dst}: {R.shape}, mean {R.mean():.4f} (real {V.mean():.4f}), std {R.std():.4f} (real {V.std():.4f})')
