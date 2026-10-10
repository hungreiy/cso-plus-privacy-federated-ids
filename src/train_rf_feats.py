"""Window statistics used by the tree baselines (random forest, XGBoost): mean, std and max of the flows in each window."""
import numpy as np
def feats(x, bs=20000):
    out = []
    for i in range(0, len(x), bs):
        b = np.asarray(x[i:i + bs]); out.append(np.concatenate([b.mean(axis=1), b.std(axis=1), b.max(axis=1)], axis=1))
    return np.concatenate(out)
