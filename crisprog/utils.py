"""Small shared helpers."""
import time

import numpy as np


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def natural_key(s):
    return (0, int(s), "") if str(s).isdigit() else (1, 0, str(s))


def benjamini_hochberg(p):
    """BH-adjusted p-values; NaN stays NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out
