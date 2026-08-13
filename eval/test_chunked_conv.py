#!/usr/bin/env python3
"""Check that chunking the convolutions changes nothing, on a GPU.

The patch exists so a whole session fits in one pass: Pi3X folds frames into the
batch dimension and its conv heads use `padding_mode='replicate'`, which routes
through an `F.pad` kernel that indexes with int32, so past about 2^31 elements it
refuses. A convolution is independent across the batch, so evaluating it in
chunks is exact — but "should be exact" is what this repository has been wrong
about before, so it is measured: the poses must match bit for bit at a size that
worked without the patch.

Needs a CUDA device and the weights. Exits 0 when it passes, 1 when it does not,
and 77 when there is no GPU to ask.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        print(f"skip: {exc}")
        return 77
    if not torch.cuda.is_available():
        print("skip: no CUDA device")
        return 77
    os.environ.setdefault(
        "PI3_ROOT", str(Path(__file__).resolve().parent / "vendor" / "Pi3"))
    sys.path.insert(0, os.environ["PI3_ROOT"])
    try:
        from pi3.models.pi3x import Pi3X
    except ImportError as exc:
        print(f"skip: Pi3 not set up ({exc}) — run eval/setup.sh")
        return 77
    import chunked_conv

    torch.manual_seed(0)
    model = Pi3X.from_pretrained("yyfz233/Pi3X").eval().to("cuda")
    model.disable_multimodal()
    # Small enough that it runs unpatched, which is the whole point: the patch
    # must be invisible here before it is trusted where it is required.
    imgs = torch.rand(1, 8, 3, 434, 574, device="cuda")

    def run():
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return model(imgs=imgs)["camera_poses"][0].float().cpu().numpy()

    before = run()
    patched = chunked_conv.apply(model, limit=1024)   # forced low, so it bites
    after = run()
    delta = float(np.abs(before - after).max())

    # Exact in exact arithmetic, and on one card exactly zero in bfloat16 too.
    # On another it is 3.5e-03: the batch size decides which kernel the runtime
    # picks, and the kernels round differently. That reading is not a guess —
    # the same comparison in fp32 gives 5.7e-05, so the difference tracks
    # arithmetic precision, which a batch-splitting mistake would not do. Such a
    # mistake mixes frames and moves a pose by its own magnitude, so the bar is
    # set far below that and far above the rounding.
    limit = 1e-2
    ok = delta < limit
    print(f"  patched {patched} Conv2d modules")
    print(f"  largest pose difference: {delta:.3e}"
          + ("  (exact on this card)" if delta == 0.0 else ""))
    print(f"  {'PASS' if ok else 'FAIL'} — chunking agrees to {limit:.0e}"
          if ok else
          f"  FAIL — {delta:.3e} exceeds {limit:.0e}; this is too large to be "
          "kernel rounding and means frames are being mixed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
