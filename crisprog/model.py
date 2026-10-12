"""OLS of program scores on perturbations, with stratified permutation p-values.

For every test unit u and program p, over the analyzed cells:

    score_p  ~  1 + perturbed_u + covariates

All units share the same covariates, so every fit is computed at once with the
Frisch-Waugh-Lovell identity (residualize the scores on the covariates once; each
coefficient is then a sparse dot product). The numbers equal separate OLS fits.

Null: each shuffle moves whole perturbation rows between cells of the same stratum
(MOI x depth bin x batch, as far as those are covariates). Every unit keeps its number
of perturbed cells and every cell its MOI, so shuffled t-values show what depth, MOI
and batch alone can produce.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from .utils import log, natural_key

BUILTIN = ("log_n_genes", "log_total_counts", "moi")
DEPTH = ("log_n_genes", "log_total_counts")


def _z(v):
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


def _dummies(lab, name):
    lab = pd.Series(lab).astype(str).to_numpy()
    levels = sorted(set(lab), key=natural_key)
    return [(lab == lv).astype(float) for lv in levels[1:]], [f"{name}={lv}" for lv in levels[1:]]


def build_design(names, builtin, obs, batch=()):
    """Intercept + covariates. Built-ins come from `builtin`, anything else from obs columns.
    Numeric covariates are standardized (only their own coefficients change); categorical
    ones and --batch columns become 0/1 dummies. Constant covariates are dropped."""
    n = len(obs)
    cols, znames = [np.ones(n)], ["intercept"]
    for nm in names:
        if nm in builtin:
            v = pd.Series(builtin[nm])
        elif nm in obs.columns:
            v = obs[nm].reset_index(drop=True)
        else:
            raise SystemExit(f"covariate '{nm}' is neither built in ({', '.join(BUILTIN)}) nor an obs column")
        if v.isna().any():
            raise SystemExit(f"covariate '{nm}' has {int(v.isna().sum())} missing values in analyzed cells")
        if pd.api.types.is_bool_dtype(v) or pd.api.types.is_numeric_dtype(v):
            v = v.astype(float).to_numpy()
            if v.std() == 0:
                log(f"covariate '{nm}' is constant in the analyzed cells and is left out")
                continue
            cols.append(_z(v))
            znames.append(nm)
        else:
            c, cn = _dummies(v, nm)
            cols += c
            znames += cn
    for b in batch:
        if b not in obs.columns:
            raise SystemExit(f"--batch column '{b}' is not in obs")
        c, cn = _dummies(obs[b].to_numpy(), b)
        cols += c
        znames += cn
    Z = np.column_stack(cols)
    if np.linalg.matrix_rank(Z) < Z.shape[1]:
        raise SystemExit("covariates are collinear (e.g. a covariate that is constant within each batch); "
                         f"design columns: {', '.join(znames)}")
    return Z, znames


def make_strata(moi=None, depth=None, batch=None, depth_bins=10, n=None):
    """Permutation stratum per cell: MOI x depth quantile bin x batch (each part optional)."""
    keys = []
    if moi is not None:
        moi = np.asarray(moi, dtype=float)
        if np.allclose(moi, np.round(moi)) or len(np.unique(moi)) <= 50:     # counts: exact values
            keys.append(np.round(moi).astype(np.int64))
        else:                                                     # continuous MOI: quantile bins
            edges = np.unique(np.quantile(moi, np.linspace(0, 1, 21)[1:-1]))
            keys.append(np.searchsorted(edges, moi, side="right"))
    if depth is not None and depth_bins > 1:
        edges = np.quantile(depth, np.linspace(0, 1, depth_bins + 1)[1:-1])
        keys.append(np.searchsorted(edges, depth, side="right").astype(np.int64))
    if batch is not None:
        keys.append(pd.factorize(pd.Series(batch).astype(str))[0].astype(np.int64))
    if not keys:
        return np.zeros(n, dtype=np.int64)
    _, codes = np.unique(np.column_stack(keys), axis=0, return_inverse=True)
    return np.asarray(codes).ravel()


def stratified_permutation(codes, rng):
    """Random pi with codes[pi] == codes: cells only swap within their stratum."""
    n = len(codes)
    grouped = np.argsort(codes, kind="stable")
    shuffled = np.lexsort((rng.random(n), codes))
    pi = np.empty(n, dtype=np.int64)
    pi[grouped] = shuffled
    return pi


class OLSEngine:
    """Every unit x program OLS  y_p ~ x_u + Z  at once.
    X: cells x units sparse 0/1;  Y: cells x programs;  Z: cells x q (with intercept)."""

    def __init__(self, X, Y, Z):
        self.n, self.q = Z.shape
        self.Z = np.ascontiguousarray(Z, dtype=np.float64)
        self.XT = sp.csr_matrix(X.T, dtype=np.float64)
        self.n1 = np.asarray(X.sum(axis=0), dtype=np.float64).ravel()
        self.G_inv = np.linalg.inv(self.Z.T @ self.Z)
        self.cov_coef = self.G_inv @ (self.Z.T @ Y)
        self.Yr = np.ascontiguousarray(Y - self.Z @ self.cov_coef)    # scores residualized on Z
        self.yty = (self.Yr ** 2).sum(axis=0)
        self.df = self.n - self.q - 1

    def stats(self, row_order=None):
        """beta, se, t (units x programs) and d = x'Mx. row_order = inverse of a row permutation
        pi: using (Yr, Z)[row_order] with the original X equals the fit with rows X[pi]."""
        Yr = self.Yr if row_order is None else self.Yr[row_order]
        Z = self.Z if row_order is None else self.Z[row_order]
        S = self.XT @ Yr
        C = self.XT @ Z
        d = self.n1 - np.einsum("eq,qr,er->e", C, self.G_inv, C)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = S / d[:, None]
            rss = self.yty[None, :] - S * beta
            se = np.sqrt(np.maximum(rss, 0.0) / self.df / d[:, None])
            t = beta / se
        return beta, se, t, d


def n1_bins(n1, n_bins):
    """Pool tests by quantiles of their perturbed-cell count."""
    edges = np.unique(np.quantile(n1, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.searchsorted(edges, n1, side="right"), len(edges) + 1


def permutation_test(eng, t, codes, n_perm=10000, pool_bins=5, seed=1):
    """Stratified-permutation p-values, accumulated shuffle by shuffle (memory O(units x programs)).

    p_own     each test against its own shuffles: (#|t_null| >= |t| + 1) / (n_perm + 1)
    p_pooled  each test against the shuffles of all tests of the same program with a similar
              number of perturbed cells (pool_bins quantile bins); far finer p-values when many
              units are tested.
    Also returns, per program, the share of shuffled tests with OLS p < 0.05 (~0.05 = calibrated).
    """
    n = eng.n
    E, P = t.shape
    thr = np.abs(t) * (1 - 1e-9)
    exceed = np.zeros((E, P), dtype=np.int64)
    null_hits = np.zeros(P, dtype=np.int64)
    crit = stats.t.ppf(0.975, eng.df)
    bins, nb = n1_bins(eng.n1, pool_bins)
    pools = []
    for g in range(nb):
        idx = np.flatnonzero(bins == g)
        for j in range(P):
            ok = idx[np.isfinite(thr[idx, j])]
            o = ok[np.argsort(thr[ok, j])]
            pools.append({"idx": idx, "j": j, "order": o, "thr": thr[o, j],
                          "cnt": np.zeros(len(o) + 1, dtype=np.int64), "N": 0})
    sizes = np.bincount(codes)
    log(f"permutation strata: {len(sizes):,}; median {np.median(sizes):.0f} cells; "
        f"{int(np.sum(sizes == 1)):,} cells are alone in their stratum and never move")
    rng = np.random.default_rng(seed)
    step = max(1, n_perm // 10)
    for b in range(n_perm):
        pi = stratified_permutation(codes, rng)
        inv = np.empty_like(pi)
        inv[pi] = np.arange(n)
        _, _, tb, _ = eng.stats(row_order=inv)
        ab = np.abs(tb)
        exceed += ab >= thr
        null_hits += (ab > crit).sum(axis=0)
        for pl in pools:
            v = ab[pl["idx"], pl["j"]]
            v = v[np.isfinite(v)]
            pl["cnt"] += np.bincount(np.searchsorted(pl["thr"], v, side="right"), minlength=len(pl["cnt"]))
            pl["N"] += len(v)
        if (b + 1) % step == 0 or b + 1 == n_perm:
            log(f"  shuffles {b + 1:,}/{n_perm:,}")
    p_own = (exceed + 1.0) / (n_perm + 1.0)
    p_own[~np.isfinite(t)] = np.nan
    p_pooled = np.full((E, P), np.nan)
    for pl in pools:
        ge = np.cumsum(pl["cnt"][::-1])[::-1]               # ge[k] = #null values >= k thresholds
        p_pooled[pl["order"], pl["j"]] = (ge[1:] + 1.0) / (pl["N"] + 1.0)
    null_frac = null_hits / float(n_perm * E)
    return p_own, p_pooled, null_frac, min(pl["N"] for pl in pools)
