"""The similarity fit used to join windows, kept apart from the model code.

`pi3_chain` needs a GPU and several gigabytes of PyTorch; this does not. Keeping
the fit here lets the join be tested on a machine with neither — and that test
is what separated "the chaining is wrong" from "the windows are wrong" when the
seams first went bad.
"""
import numpy as np


def umeyama(source, target):
    """Similarity transform taking `source` points onto `target` points.

    Returns rotation, scale and translation, in that order. Scale is part of it
    because each window carries its own: forcing them equal would bake the
    first window's scale into everything after it.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    sc, tc = source.mean(0), target.mean(0)
    s0, t0 = source - sc, target - tc
    U, D, Vt = np.linalg.svd((t0.T @ s0) / len(source))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (s0 ** 2).sum() / len(source)
    scale = float((D * np.diag(S)).sum() / var) if var > 1e-12 else 1.0
    return R, scale, tc - scale * R @ sc
