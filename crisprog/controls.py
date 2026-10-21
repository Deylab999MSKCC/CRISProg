"""Negative-control check: random groups of control guides tested exactly like real units.

Control guides (non-targeting, or safe-targeting e.g. at silent olfactory-receptor promoters) should not move any
program. Each control unit is a random group of control guides of one type with about as many perturbed cells as
a typical tested unit; a cell is 'perturbed' for it if it carries any of those guides. They are fitted with the same covariates,
permutation strata and number of guides per unit as the real units.

What it tells you:
  - share of control units with p_perm < 0.05: ~0.05 if the test is right. The permutation null cannot see
    confounding (e.g. guides detected more often in one cell type); control units can.
  - control units at FDR < fdr: should be ~0.
  - the control 'effect floor': the 99th percentile of |effect_sd| among control units, per program. Real
    associations get `p_vs_controls` (share of control units with an effect at least as large) and
    `beyond_controls` (|effect_sd| above that floor) in results.tsv.

Outputs (association step folder, controls/):
  control_units.tsv      control unit, type, guides, n_perturbed
  control_results.tsv    control unit x program: effect_sd, t, p_perm, fdr
  control_summary.tsv    program x type: n_units, frac_p_lt_0.05, n_fdr_hits, effect quantiles
"""
import os

import numpy as np
import pandas as pd

from .model import OLSEngine, permutation_test
from .utils import benjamini_hochberg, log


def build_control_units(A, rows, target_cells, n_units, min_cells, seed):
    """Random groups of control guides of one type, each grown guide by guide until it has about as many
    perturbed cells as a typical tested unit (target_cells) -> (cells x units 0/1 CSR, units DataFrame).
    Matching on cells, not guides, keeps control units comparable when control labels differ in size
    (e.g. one label per safe-targeting gene that pools several guides)."""
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    Gc = A.Gc[rows].tocsc()
    n = Gc.shape[0]
    per = np.diff(Gc.indptr)
    cols, defs = [], []

    def grow(order):
        mask, used = np.zeros(n, bool), []
        for g in order:
            used.append(g)
            mask[Gc.indices[Gc.indptr[g]:Gc.indptr[g + 1]]] = True
            if mask.sum() >= target_cells:
                return used, mask
        return None, None

    for typ in sorted(set(A.control_type)):
        idx = np.flatnonzero((A.control_type == typ) & (per > 0))
        cells_typ = int((np.asarray(Gc[:, idx].sum(axis=1)).ravel() > 0).sum()) if len(idx) else 0
        if cells_typ < 2 * target_cells:
            log(f"control check: {typ}: {len(idx)} control guides in {cells_typ:,} cells; need at least "
                f"{2 * target_cells:,} cells for control units of ~{target_cells:,} cells - skipped")
            continue
        groups, order, k = [], list(rng.permutation(idx)), 0
        while order and len(groups) < n_units:                 # disjoint groups first
            used, mask = grow(order)
            if used is None:
                break
            groups.append((used, mask))
            order = order[len(used):]
        n_disjoint = len(groups)
        while len(groups) < n_units:                            # then overlapping random groups
            used, mask = grow(rng.permutation(idx))
            groups.append((used, mask))
        for used, mask in groups:
            k += 1
            x = mask.astype(np.float64)
            if min_cells <= x.sum() <= n - min_cells:
                cols.append(sp.csr_matrix(x[:, None]))
                defs.append((f"{typ}_ctrl{k}", typ, len(used), ";".join(A.control_guides[used]), int(x.sum())))
        if n_disjoint < n_units:
            log(f"control check: {typ}: {n_disjoint} disjoint control units, {n_units - n_disjoint} more drawn "
                "with overlapping guides")
    if not cols:
        return None, None
    return sp.hstack(cols).tocsr(), pd.DataFrame(defs, columns=["unit", "type", "n_guides", "guides", "n_perturbed"])


def run_control_check(A, rows, Y, Z, codes, prog_names, a, res, real_n1, out):
    """Fit control units, write controls/, and add p_vs_controls / beyond_controls to res (returned)."""
    mode = a["control_units"]
    if mode in (False, "false", "no", "none"):
        return res
    n_ctrl = len(A.control_guides)
    if n_ctrl == 0 or (n_ctrl < 4 and mode == "auto"):
        msg = (f"control check: {n_ctrl} distinct control guide label(s) "
               f"({', '.join(A.control_guides[:3])}); individual control guides are needed to build control "
               "units (merged labels like 'control' cannot be split)")
        if mode is True or mode == "true":
            raise SystemExit(msg)
        log(msg + " - skipped")
        return res
    target = int(a["control_unit_cells"] or max(int(a["min_perturbed"]), int(np.median(real_n1))))
    Xc, cu = build_control_units(A, rows, target, int(a["n_control_units"]), int(a["min_perturbed"]), a["seed"])
    if Xc is None:
        log("control check: no control unit could be built - skipped")
        return res
    d = os.path.join(out, "controls")
    os.makedirs(d, exist_ok=True)
    log(f"control check: {len(cu)} control units matched to the tested units' median of {target:,} perturbed "
        f"cells ({', '.join(f'{k} {v}' for k, v in cu.type.value_counts().items())}); perturbed cells per control "
        f"unit: median {cu.n_perturbed.median():.0f}; control labels per unit: median {cu.n_guides.median():.0f}")
    sep = np.asarray(A.U[rows][np.asarray(A.Gc[rows].sum(axis=1)).ravel() > 0].sum(axis=1)).ravel()
    if len(sep) and (sep == 0).all():
        log("control check: cells with control guides carry no targeting guide (controls are a separate population); "
            "control units are compared with the other control cells of the same MOI level")
    eng = OLSEngine(Xc, Y, Z)
    beta, se, t, dd = eng.stats()
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = beta / (se * np.sqrt(dd)[:, None])
    p_own, p_pool, _, _ = permutation_test(eng, t, codes, int(a["control_n_perm"]), a["pool_bins"], a["seed"] + 1)
    p = p_pool if len(cu) > 50 else p_own
    rows_out, summ = [], []
    for typ in cu.type.unique():
        m = (cu.type == typ).to_numpy()
        for j, prog in enumerate(prog_names):
            fdr = benjamini_hochberg(p[m, j])
            for k, i in enumerate(np.flatnonzero(m)):
                rows_out.append((cu.unit.iat[i], typ, prog, cu.n_perturbed.iat[i], eff[i, j], t[i, j], p[i, j], fdr[k]))
            ae = np.abs(eff[m, j])
            summ.append(dict(program=prog, type=typ, n_units=int(m.sum()),
                             frac_p_lt_005=float(np.mean(p[m, j] < 0.05)),
                             n_fdr_hits=int(np.sum(fdr < a["fdr"])),
                             effect_q95=float(np.quantile(ae, 0.95)), effect_q99=float(np.quantile(ae, 0.99)),
                             effect_max=float(ae.max())))
    cr = pd.DataFrame(rows_out, columns=["unit", "type", "program", "n_perturbed", "effect_sd", "t", "p_perm", "fdr"])
    cs = pd.DataFrame(summ)
    cu.to_csv(os.path.join(d, "control_units.tsv"), sep="\t", index=False)
    cr.to_csv(os.path.join(d, "control_results.tsv"), sep="\t", index=False, float_format="%.6g")
    cs.to_csv(os.path.join(d, "control_summary.tsv"), sep="\t", index=False, float_format="%.4g")

    # real associations against the controls of the same program (all control types together)
    ctrl_abs = {prog: np.sort(np.abs(cr.effect_sd[cr.program == prog].to_numpy())) for prog in prog_names}
    q99 = {prog: float(np.quantile(v, 0.99)) for prog, v in ctrl_abs.items()}
    ae = res["effect_sd"].abs().to_numpy()
    pv = np.full(len(res), np.nan)
    for prog, v in ctrl_abs.items():
        k = (res["program"] == prog).to_numpy() & np.isfinite(ae)
        pv[k] = (len(v) - np.searchsorted(v, ae[k], side="left") + 1.0) / (len(v) + 1.0)
    res = res.assign(p_vs_controls=pv,
                     beyond_controls=pd.array([None if not np.isfinite(x) else bool(x > q99[pr])
                                               for x, pr in zip(ae, res["program"])], dtype="boolean"))

    log("control check (expected: ~0.05 of control units with p < 0.05, no control unit at FDR < "
        f"{a['fdr']:g}):")
    print(cs[["program", "type", "n_units", "frac_p_lt_005", "n_fdr_hits", "effect_q99"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3g}"), flush=True)
    bad = cs[(cs.frac_p_lt_005 > 0.10) | (cs.n_fdr_hits > 0)]
    if len(bad):
        log("WARNING: control units look non-null in: " + ", ".join(f"{r.program} ({r.type})" for r in bad.itertuples())
            + ". Part of the program-level signal is not specific to the targets (e.g. guide detection that differs "
              "between cell states). Prefer associations with beyond_controls = True.")
    hits = res["fdr"] < a["fdr"]
    if hits.any():
        log(f"significant associations beyond the control effect floor (99th percentile of control |effect|): "
            f"{int((hits & res['beyond_controls'].fillna(False)).sum())} of {int(hits.sum())}")
    return res
