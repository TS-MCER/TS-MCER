"""Local durable output with atomic writes and content fingerprints."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(destination, **values):
    path = Path(destination); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(tmp, path)


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
