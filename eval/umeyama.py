"""The similarity fit used to join windows, kept apart from the model code.

`pi3_chain` needs a GPU and several gigabytes of PyTorch; this does not. Keeping
the fit here lets the join be tested on a machine with neither — and that test
is what separated "the chaining is wrong" from "the windows are wrong" when the
seams first went bad.
"""
import numpy as np


class ScaleNotObservable(ValueError):
    """The source points are too tightly clustered for a scale to mean anything."""


def umeyama(source, target, *, min_rms=1e-3):
    """Similarity transform taking `source` points onto `target` points.

    Returns rotation, scale and translation, in that order. Scale is part of it
    because each window carries its own: forcing them equal would bake the
    first window's scale into everything after it.

    **Raises rather than guessing when the scale is not observable.** Points
    clustered inside a millimetre pin a rotation and a translation and leave the
    scale free, and the earlier version answered `scale = 1.0` in that case —
    a plausible number, silently invented, indistinguishable downstream from a
    measured one. That is the failure shape this repo keeps meeting: not a
    wrong answer that looks wrong, but a fallback that looks like a result.

    `min_rms` is the radius, in metres, below which the caller is asking a
    question the data cannot answer.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) < 3 or len(source) != len(target):
        raise ScaleNotObservable(
            f"a similarity needs at least three matched points, got "
            f"{len(source)} and {len(target)}")
    if not (np.isfinite(source).all() and np.isfinite(target).all()):
        raise ScaleNotObservable("source or target contains non-finite points")

    sc, tc = source.mean(0), target.mean(0)
    s0, t0 = source - sc, target - tc
    var = (s0 ** 2).sum() / len(source)
    rms = float(np.sqrt(var))
    if rms < min_rms:
        raise ScaleNotObservable(
            f"source points span {rms * 1000:.3f} mm rms, below the {min_rms * 1000:.0f} mm "
            "floor: rotation and translation are still determined, but the "
            "scale is whatever the noise prefers")

    U, D, Vt = np.linalg.svd((t0.T @ s0) / len(source))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    scale = float((D * np.diag(S)).sum() / var)
    return R, scale, tc - scale * R @ sc
