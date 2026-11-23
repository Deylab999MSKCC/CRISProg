"""OLS of program scores on perturbations, with stratified permutation p-values.

For every test unit u and program p:

    score_p  ~  1 + perturbed_u + covariates

association.reference decides which cells are the 0s:

  all       every analyzed cell without unit u (OLSEngine). All units share the same cells and covariates,
            so every fit is computed at once with the Frisch-Waugh-Lovell identity (residualize the scores
            on the covariates once; each coefficient is then a sparse dot product). The numbers equal
            separate OLS fits. Null: each shuffle moves whole perturbation rows between cells of the same
            stratum (MOI x depth bin x batch, as far as those are covariates). Every unit keeps its number
            of perturbed cells and every cell its MOI.
  controls  only the cells that carry control guides and no targeting guide (RefEngine): each unit is fitted
            on its own cells + the control cells, so other perturbations never enter its comparison. Each fit
            uses sums over the control cells (shared) plus the unit's own cells, so all units are still
            computed without refitting from scratch. Null: for each unit, its labels are shuffled among its
            own cells and the control cells of the same stratum (depth bin x batch).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from .utils import log, natural_key

BUILTIN = ("log_n_genes", "log_total_counts", "moi", "n_control")
DEPTH = ("log_n_genes", "log_total_counts")


def _z(v):
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


def _dummies(lab, name):
    lab = pd.Series(lab).astype(str).to_numpy()
    levels = sorted(set(lab), key=natural_key)
    return [(lab == lv).astype(float) for lv in levels[1:]], [f"{name}={lv}" for lv in levels[1:]]


def moi_levels(moi, min_cells=50, max_levels=40):
    """MOI as categories: integer values, the sparse top levels pooled so that each level has >= min_cells cells.
    None when MOI is not a small set of integers (then it stays a numeric covariate)."""
    moi = np.asarray(moi, dtype=float)
    if not np.allclose(moi, np.round(moi)):
        return None
    m = np.round(moi).astype(int)
    vals, cnt = np.unique(m, return_counts=True)
    top = vals[-1]
    while top > vals[0] and cnt[vals >= top].sum() < min_cells:
        top -= 1
    m = np.minimum(m, top)
    return m if len(np.unique(m)) <= max_levels else None


def build_design(names, builtin, obs, batch=(), moi_categorical="auto"):
    """Intercept + covariates. Built-ins come from `builtin`, anything else from obs columns.
    Numeric covariates are standardized (only their own coefficients change); categorical
    ones and --batch columns become 0/1 dummies. Constant covariates are dropped.
    MOI becomes one dummy per guide count (top counts pooled) when moi_categorical is true, or 'auto' and MOI
    takes few integer values: each unit is then compared with cells carrying the same number of guides."""
    n = len(obs)
    cols, znames = [np.ones(n)], ["intercept"]
    for nm in names:
        if nm == "n_control" and nm in builtin:              # always categories
            lv = moi_levels(builtin[nm])
            c, cn = _dummies(lv if lv is not None else np.round(builtin[nm]), "n_control")
            cols += c
            znames += cn
            continue
        if nm == "moi" and moi_categorical not in (False, "false", "no") and nm in builtin:
            lv = moi_levels(builtin[nm])
            if lv is not None and len(np.unique(lv)) > 1:
                c, cn = _dummies(lv, "moi")
                cols += c
                znames += cn
                continue
            if moi_categorical in (True, "true", "yes"):
                raise SystemExit("covariates.moi_categorical is true but MOI is not a small set of integers")
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


class RefEngine:
    """Every unit x program OLS  y_p ~ x_u + Z  fitted on unit u's cells + the reference (control) cells only.
    X: cells x units sparse 0/1;  Y: cells x programs;  Z: cells x q (with intercept);  ref: bool per cell.

    Unit u's cells are R_u = ref | cells of u (for a control unit, its cells are part of ref, so R_u = ref and
    its 0s are the other control cells). Z'Z, Z'Y and y'y over R_u are the control-cell sums plus the sums
    over u's cells outside ref, so each unit needs only a q x q (pseudo)inverse. A unit whose 0/1 column is
    explained by the covariates within R_u (e.g. all its cells in a batch without control cells) gets NaN."""

    def __init__(self, X, Y, Z, ref):
        self.n, self.q = Z.shape
        self.Z = np.ascontiguousarray(Z, dtype=np.float64)
        self.Y = np.ascontiguousarray(Y, dtype=np.float64)
        self.ref = np.asarray(ref, dtype=bool)
        self.ref_idx = np.flatnonzero(self.ref)
        Xc = sp.csc_matrix(X, dtype=np.float64)
        Xc.eliminate_zeros()
        self.XT = sp.csr_matrix(Xc.T)
        self.n1 = np.asarray(Xc.sum(axis=0), dtype=np.float64).ravel()
        E, P, q = Xc.shape[1], self.Y.shape[1], self.q
        Zr, Yr = self.Z[self.ref], self.Y[self.ref]
        G0, ZY0, yy0 = Zr.T @ Zr, Zr.T @ Yr, (Yr ** 2).sum(axis=0)
        self.Ginv = np.empty((E, q, q))
        self.b = np.empty((E, q, P))
        self.yMy = np.empty((E, P))
        df = np.empty(E)
        self.cells, self.extra = [], []
        for u in range(E):
            I = Xc.indices[Xc.indptr[u]:Xc.indptr[u + 1]]
            ex = I[~self.ref[I]]
            self.cells.append(I)
            self.extra.append(ex)
            Ze, Ye = self.Z[ex], self.Y[ex]
            G, ZY = G0 + Ze.T @ Ze, ZY0 + Ze.T @ Ye
            w, V = np.linalg.eigh(G)
            keep = w > max(w.max(), 1.0) * 1e-10
            Gi = (V[:, keep] / w[keep]) @ V[:, keep].T            # pseudoinverse: absent batch levels -> 0 columns
            self.Ginv[u], self.b[u] = Gi, Gi @ ZY
            self.yMy[u] = yy0 + (Ye ** 2).sum(axis=0) - (ZY * self.b[u]).sum(axis=0)
            df[u] = len(self.ref_idx) + len(ex) - int(keep.sum()) - 1
        self.n_rows = len(self.ref_idx) + np.array([len(e) for e in self.extra])
        self.df = df[:, None]
        self.cov_coef = np.linalg.pinv(G0) @ ZY0                   # covariate coefficients in the control cells

    def stats(self, row_order=None, Xalt=None):
        """beta, se, t (units x programs) and d = x'Mx. Xalt: another 0/1 matrix whose unit columns stay within
        each unit's cells R_u (a shuffle); row_order is not supported (shuffles are per unit)."""
        if row_order is not None:
            raise ValueError("RefEngine shuffles labels per unit; use permutation_test or permuted_X")
        XT = self.XT if Xalt is None else sp.csr_matrix(sp.csc_matrix(Xalt, dtype=np.float64).T)
        S = XT @ self.Y
        C = XT @ self.Z
        n1 = np.asarray(XT.sum(axis=1)).ravel()
        d = n1 - np.einsum("eq,eqr,er->e", C, self.Ginv, C)
        num = S - np.einsum("eq,eqp->ep", C, self.b)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(d > 1e-9 * np.maximum(n1, 1), d, np.nan)    # x explained by the covariates: untestable
            beta = num / d[:, None]
            se = np.sqrt(np.maximum(self.yMy - beta * num, 0.0) / self.df / d[:, None])
            t = beta / se
        return beta, se, t, d

    def strata_groups(self, codes, u):
        """For unit u: [(candidate cells, k)] per stratum holding its cells: shuffled labels go to k of the
        candidates (u's cells and the control cells of that stratum)."""
        if not hasattr(self, "_ref_by"):
            rc = codes[self.ref_idx]
            o = np.argsort(rc, kind="stable")
            vals, starts = np.unique(rc[o], return_index=True)
            self._ref_by = {v: self.ref_idx[o][s:e] for v, s, e in zip(vals, starts, list(starts[1:]) + [len(o)])}
        I, ex = self.cells[u], self.extra[u]
        s_vals, k = np.unique(codes[I], return_counts=True)
        out = []
        for s, kk in zip(s_vals, k):
            c = np.concatenate([self._ref_by.get(s, np.empty(0, dtype=np.int64)), ex[codes[ex] == s]])
            out.append((c, int(kk)))
        return out

    def permuted_X(self, codes, rng):
        """One stratified shuffle per unit, as a cells x units 0/1 matrix (for the per-gene calibration)."""
        rows, cols = [], []
        for u in range(len(self.n1)):
            for c, k in self.strata_groups(codes, u):
                rows.append(c if len(c) == k else rng.choice(c, k, replace=False))
                cols.append(np.full(k, u))
        r, c = np.concatenate(rows), np.concatenate(cols)
        return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(self.n, len(self.n1)))


def make_engine(X, Y, Z, ref=None):
    """OLSEngine (reference 'all': every other cell is a 0) or RefEngine (reference 'controls': ref marks the
    control cells, the only 0s)."""
    return OLSEngine(X, Y, Z) if ref is None else RefEngine(X, Y, Z, ref)


def _pools(n1, thr, pool_bins):
    bins, nb = n1_bins(n1, pool_bins)
    pools = {}
    for g in range(nb):
        idx = np.flatnonzero(bins == g)
        for j in range(thr.shape[1]):
            ok = idx[np.isfinite(thr[idx, j])]
            o = ok[np.argsort(thr[ok, j])]
            pools[(g, j)] = {"idx": idx, "j": j, "order": o, "thr": thr[o, j],
                             "cnt": np.zeros(len(o) + 1, dtype=np.int64), "N": 0}
    return bins, pools


def _pool_p(pools, shape):
    p_pooled = np.full(shape, np.nan)
    for pl in pools.values():
        ge = np.cumsum(pl["cnt"][::-1])[::-1]               # ge[k] = #null values >= k thresholds
        p_pooled[pl["order"], pl["j"]] = (ge[1:] + 1.0) / (pl["N"] + 1.0)
    return p_pooled


def _permutation_test_ref(eng, t, codes, n_perm, pool_bins, seed, max_block=4e6):
    """permutation_test for RefEngine: unit by unit, its labels are shuffled among its own cells and the control
    cells of the same stratum; shuffles are done in blocks (all statistics from sums over the chosen cells)."""
    E, P = t.shape
    thr = np.abs(t) * (1 - 1e-9)
    exceed = np.zeros((E, P), dtype=np.int64)
    null_hits = np.zeros(P, dtype=np.int64)
    crit = stats.t.ppf(0.975, eng.df.ravel())
    bins, pools = _pools(eng.n1, thr, pool_bins)
    n_str = len(np.unique(codes[eng.ref_idx]))
    log(f"permutation strata (control cells): {n_str:,}; each unit's labels are shuffled among its own cells and the "
        f"{len(eng.ref_idx):,} control cells of the same stratum")
    rng = np.random.default_rng(seed)
    step = max(1, E // 10)
    fixed_units = 0
    for u in range(E):
        if not np.isfinite(t[u]).any():
            continue
        groups = eng.strata_groups(codes, u)
        if all(len(c) == k for c, k in groups):
            fixed_units += 1
        n1 = int(eng.n1[u])
        B = int(max(1, min(n_perm, max_block // max(1, n1 * (P + eng.q)))))
        done = 0
        while done < n_perm:
            b = min(B, n_perm - done)
            parts = []
            for c, k in groups:
                if len(c) == k:
                    parts.append(np.broadcast_to(c, (b, k)))
                else:
                    pick = np.argpartition(rng.random((b, len(c))), k - 1, axis=1)[:, :k]
                    parts.append(c[pick])
            sel = np.hstack(parts)
            xy, xz = eng.Y[sel].sum(axis=1), eng.Z[sel].sum(axis=1)
            d = n1 - np.einsum("bq,qr,br->b", xz, eng.Ginv[u], xz)
            num = xy - xz @ eng.b[u]
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.where(d > 1e-9 * n1, d, np.nan)
                beta = num / d[:, None]
                se = np.sqrt(np.maximum(eng.yMy[u] - beta * num, 0.0) / eng.df[u] / d[:, None])
                ab = np.abs(beta / se)
            exceed[u] += (ab >= thr[u]).sum(axis=0)
            null_hits += (ab > crit[u]).sum(axis=0)
            for j in range(P):
                v = ab[:, j]
                v = v[np.isfinite(v)]
                pl = pools[(bins[u], j)]
                pl["cnt"] += np.bincount(np.searchsorted(pl["thr"], v, side="right"), minlength=len(pl["cnt"]))
                pl["N"] += len(v)
            done += b
        if (u + 1) % step == 0 or u + 1 == E:
            log(f"  units shuffled {u + 1:,}/{E:,}")
    if fixed_units:
        log(f"{fixed_units} unit(s) have no control cells in their strata: their labels cannot move (p = 1)")
    p_own = (exceed + 1.0) / (n_perm + 1.0)
    p_own[~np.isfinite(t)] = np.nan
    p_pooled = _pool_p(pools, (E, P))
    null_frac = null_hits / float(n_perm * E)
    return p_own, p_pooled, null_frac, min(pl["N"] for pl in pools.values())


def permutation_test(eng, t, codes, n_perm=10000, pool_bins=5, seed=1):
    """Stratified-permutation p-values, accumulated shuffle by shuffle (memory O(units x programs)).

    p_own     each test against its own shuffles: (#|t_null| >= |t| + 1) / (n_perm + 1)
    p_pooled  each test against the shuffles of all tests of the same program with a similar
              number of perturbed cells (pool_bins quantile bins); far finer p-values when many
              units are tested.
    Also returns, per program, the share of shuffled tests with OLS p < 0.05 (~0.05 = calibrated).
    """
    if isinstance(eng, RefEngine):
        return _permutation_test_ref(eng, t, codes, n_perm, pool_bins, seed)
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
