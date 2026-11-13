"""Load target fields and wave-speed maps from NumPy archives."""
from pathlib import Path
import hashlib
import numpy as np


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_pairs(path):
    with np.load(path, allow_pickle=False) as archive:
        target = archive['target'].astype(np.float32)
        speed = archive['c'].astype(np.float32)
    if target.ndim == 2:
        target, speed = target[None], speed[None]
    if target.ndim != 3 or target.shape != speed.shape or target.shape[-1] != target.shape[-2]:
        raise ValueError('target and c must have matching (count, n, n) shapes.')
    if not np.isfinite(target).all() or not np.isfinite(speed).all() or np.min(speed) <= 0:
        raise ValueError('Inputs must be finite, with positive wave speeds.')
    if np.any(np.sum(target**2, axis=(-2, -1)) == 0):
        raise ValueError('Relative-error evaluation requires nonzero targets.')
    return target, speed
