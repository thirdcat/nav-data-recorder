"""Run the decoder's per-frame convolutions in batch chunks.

Pi3X folds frames into the batch dimension (`imgs.reshape(B*N, C, H, W)`), and
the conv heads use `padding_mode='replicate'`, which makes PyTorch call `F.pad`
separately before the convolution. That pad kernel indexes with int32, so once
frames x height x width x channels passes 2^31 elements it raises
"input tensor must fit into 32-bit index math" — at 131 frames, not for want of
memory (96 frames peak at 33.8 GiB of 98).

A convolution is independent across the batch dimension, so evaluating it in
chunks of frames is exact rather than approximate. This is why the upstream
demos never hit it: they window at 30-64 frames before reaching here.
"""
import torch
import torch.nn as nn

LIMIT = 2 ** 30          # half of int32's range, comfortably inside the kernel's


def _chunked(module, limit):
    original = module.forward

    def forward(x):
        if x.dim() != 4 or x.shape[0] <= 1 or x.numel() <= limit:
            return original(x)
        per_sample = max(x[0].numel(), 1)
        size = max(1, int(limit // per_sample))
        if size >= x.shape[0]:
            return original(x)
        return torch.cat([original(x[i:i + size])
                          for i in range(0, x.shape[0], size)], dim=0)
    return forward


def apply(model, limit=LIMIT):
    """Wrap every Conv2d so a long sequence takes the same path a short one does."""
    patched = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and not getattr(module, "_chunked", False):
            module.forward = _chunked(module, limit)
            module._chunked = True
            patched += 1
    return patched
