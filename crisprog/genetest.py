"""Per-gene follow-up for the significant perturbations.

For every perturbation unit with FDR < association.fdr in at least one program, the association model is refit
with each gene's expression (.X) as the outcome, on the same cells and covariates:

    expression_g ~ perturbed + covariates,   BH FDR across genes within each unit

The unit's cis targets (every gene its guides target, including genes sharing a CRE) are flagged and left
out of all counts. This shows whether a
regulator moves many genes of the program (a broad trans effect) or mostly itself, and whether its changes are
concentrated in that program:

    ratio = (share of the program's genes that changed) / (share of all genes that changed)

Outputs (gene-test step folder):
  gene_results.tsv.gz    unit, gene, beta, t, p, fdr, is_target, programs
  program_summary.tsv    one row per significant unit x program association (counts, ratio, direction agreement,
                         t of the unit's own and shared-CRE targets, units it shares a CRE with)
  overlap.tsv            per program, pairs of its regulators: overlap of changed genes (Jaccard) and of
                         perturbed cells (shared guides/CREs make two units hard to tell apart)
  heatmap_<program>.png/.pdf
  calibration.tsv        share of shuffled per-gene tests with p < 0.05 (~0.05 = well calibrated)
"""
import os
import re

import logging

import numpy as np

logging.getLogger("fontTools").setLevel(logging.ERROR)   # silence font-subsetting chatter in PDF export
import pandas as pd
from scipy import stats

from .data import read_programs, target_genes
from .model import OLSEngine, stratified_permutation
from .utils import benjamini_hochberg, log


def _fit(Xu, Y, Z):
    eng = OLSEngine(Xu, Y, Z)
    beta, se, t, _ = eng.stats()
    return beta, t, 2.0 * stats.t.sf(np.abs(t), eng.df)


def run_gene_test(cfg, A, out):
    from .steps import DIRS, programs_path
    g, a = cfg["gene_test"], cfg["association"]
    res = pd.read_csv(os.path.join(cfg["outdir"], DIRS["test"], "results.tsv"), sep="\t")
    hits = res[res.fdr < a["fdr"]]
    if hits.empty:
        log("no significant associations; nothing to follow up")
        return
    units_sig = list(dict.fromkeys(hits.unit))
    covs = pd.read_csv(os.path.join(cfg["outdir"], DIRS["test"], "cell_covariates.tsv"), sep="\t",
                       dtype={"cell": str})
    rows = A.rows_for(covs.cell)
    Z, _, codes, strata, _ = A.design(rows)
    uidx = A.units.reset_index(drop=True)
    pos = {u: i for i, u in enumerate(uidx.unit)}
    Xu = A.U[rows][:, [pos[u] for u in units_sig]]
    X = A.X[rows].tocsc()
    det = np.diff(X.indptr)
    gkeep = np.flatnonzero(det >= max(10, int(0.001 * len(rows))))
    genes = A.genes[gkeep]
    log(f"per-gene test: {len(units_sig)} units x {len(genes):,} genes (detected in >= "
        f"{max(10, int(0.001 * len(rows)))} cells), {len(rows):,} cells")

    B = np.zeros((len(units_sig), len(genes)), np.float64)          # float64: p-values can underflow float32
    T, Pv = np.zeros_like(B), np.zeros_like(B)
    ch = int(g["chunk"])
    for s in range(0, len(genes), ch):
        Y = X[:, gkeep[s:s + ch]].toarray().astype(np.float64)
        b, t, p = _fit(Xu, Y, Z)
        B[:, s:s + ch], T[:, s:s + ch], Pv[:, s:s + ch] = b, t, p
    F = np.vstack([benjamini_hochberg(Pv[i]) for i in range(len(units_sig))])

    # calibration: stratified shuffles on a random gene subset
    rng = np.random.default_rng(g["seed"])
    sub = np.sort(rng.choice(len(genes), min(1000, len(genes)), replace=False))
    Ysub = X[:, gkeep[sub]].toarray().astype(np.float64)
    null = []
    for _ in range(int(g["n_null"])):
        pi = stratified_permutation(codes, rng)
        null.append(float(np.mean(_fit(Xu[pi], Ysub, Z)[2] < 0.05)))
    cal = pd.DataFrame({"shuffle": range(len(null)), "frac_p_lt_0.05": null})
    cal.to_csv(os.path.join(out, "calibration.tsv"), sep="\t", index=False, float_format="%.4g")
    log(f"calibration: share of shuffled tests with p < 0.05 = {np.mean(null):.3f} "
        f"(range {min(null):.3f}-{max(null):.3f}; ~0.05 = well calibrated)")

    programs = read_programs(programs_path(cfg))
    gene_set = set(genes)
    progs = {p: [x for x in gl if x in gene_set] for p, gl in programs.items() if p in set(res.program)}
    member = {}
    for p, gl in progs.items():
        for x in gl:
            member.setdefault(x, []).append(p)
    lab_own = {u: [x for x in str(t).split(";") if x] for u, t in zip(uidx.unit, uidx.targets.fillna(""))}
    lab_cis = {u: set(xs) for u, xs in lab_own.items()}
    if A.target_map is not None:               # cis set: all targets of the unit's guides (shared CREs too)
        g2t = A.target_map.groupby("guide").target.apply(set).to_dict()
        for u, gl in zip(uidx.unit, uidx.guides.fillna("")):
            for gd in str(gl).split(";"):
                lab_cis[u] |= g2t.get(gd, set())
    unit_of = {x: u for u, xs in lab_own.items() for x in xs}
    partners = {u: sorted({unit_of[x] for x in lab_cis[u] - set(lab_own[u]) if x in unit_of} - {u})
                for u in lab_own}
    gs = set(map(str, A.genes))                # target labels -> measured genes ('POU5F1_P2' -> POU5F1)
    own = {u: list(dict.fromkeys(g for x in xs for g in target_genes(x, gs))) for u, xs in lab_own.items()}
    targets = {u: {g for x in xs for g in target_genes(x, gs)} for u, xs in lab_cis.items()}

    long = pd.DataFrame({
        "unit": np.repeat(units_sig, len(genes)), "gene": np.tile(genes, len(units_sig)),
        "beta": B.ravel(), "t": T.ravel(), "p": Pv.ravel(), "fdr": F.ravel()})
    long["is_target"] = [x in targets[u] for u, x in zip(long.unit, long.gene)]
    long["programs"] = long.gene.map(lambda x: ",".join(member.get(x, [])))
    long.to_csv(os.path.join(out, "gene_results.tsv.gz"), sep="\t", index=False, float_format="%.4g")

    fcut = g["fdr"]
    gi = {x: j for j, x in enumerate(genes)}
    rows_s, ov_rows = [], []
    U_cells = {u: set(np.flatnonzero(Xu[:, i].toarray().ravel())) for i, u in enumerate(units_sig)}
    for p, gl in progs.items():
        H = hits[hits.program == p]
        if H.empty:
            continue
        pg = set(gl)
        changed = {}
        for _, h in H.iterrows():
            i = units_sig.index(h.unit)
            ok = np.array([x not in targets[h.unit] for x in genes])
            inprog = np.array([x in pg for x in genes]) & ok
            sig = (F[i] < fcut) & ok
            n_all, n_prog = int(ok.sum()), int(inprog.sum())
            k_all, k_prog = int(sig.sum()), int((sig & inprog).sum())
            up = int((sig & inprog & (T[i] > 0)).sum())
            same = int((sig & inprog & (np.sign(T[i]) == np.sign(h.t))).sum())
            tt = {x: float(T[i][gi[x]]) for x in own[h.unit] + sorted(targets[h.unit] - set(own[h.unit])) if x in gi}
            rows_s.append(dict(program=p, unit=h.unit, program_effect_sd=h.effect_sd, program_fdr=h.fdr,
                               target_in_program=h.target_in_program, program_genes_tested=n_prog,
                               changed_in_program=k_prog, up=up, down=k_prog - up,
                               same_direction_as_program=same, changed_all_genes=k_all, genes_tested=n_all,
                               ratio=(k_prog / n_prog) / (k_all / n_all) if k_all and n_prog else np.nan,
                               target_t=";".join(f"{x}:{v:.1f}" for x, v in tt.items()),
                               shares_cre_with=";".join(partners[h.unit]),
                               flag=("self-driven? target is in the program and no other program gene changed"
                                     if h.target_in_program is True and k_prog == 0 else "")))
            changed[h.unit] = set(genes[sig & inprog])
        us = list(changed)
        for x in range(len(us)):
            for y in range(x + 1, len(us)):
                a_, b_ = changed[us[x]], changed[us[y]]
                ca, cb = U_cells[us[x]], U_cells[us[y]]
                ov_rows.append(dict(program=p, unit_a=us[x], unit_b=us[y],
                                    genes_jaccard=len(a_ & b_) / len(a_ | b_) if a_ | b_ else np.nan,
                                    cells_overlap=len(ca & cb) / max(1, min(len(ca), len(cb)))))
        _heatmap(p, H, units_sig, genes, F, T, targets, gl, fcut, out, Pv, own, partners)
    summ = pd.DataFrame(rows_s)
    summ.to_csv(os.path.join(out, "program_summary.tsv"), sep="\t", index=False, float_format="%.4g")
    ov = pd.DataFrame(ov_rows, columns=["program", "unit_a", "unit_b", "genes_jaccard", "cells_overlap"])
    ov.to_csv(os.path.join(out, "overlap.tsv"), sep="\t", index=False, float_format="%.3g")
    if len(summ):
        print(summ[["program", "unit", "changed_in_program", "up", "down", "ratio", "target_in_program", "flag"]]
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"), flush=True)
    shared = ov[ov.cells_overlap > 0.5]
    if len(shared):
        log("units sharing most perturbed cells (cannot be separated): "
            + "; ".join(f"{r.unit_a} & {r.unit_b} ({r.cells_overlap:.0%})" for r in shared.itertuples()))


def _heatmap(p, H, units_sig, genes, F, T, targets, prog_genes, fcut, out, Pv, own, partners, cap=5, cap_own=10):
    """targets = cis set (excluded from the heatmap); own = the unit's own targets, shown in the left column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 6, "pdf.fonttype": 42})
    regs = list(H.sort_values(["target_in_program", "fdr"], ascending=[False, True]).unit)
    gi = {x: j for j, x in enumerate(genes)}
    cols = [gi[x] for x in prog_genes if x in gi]
    M = np.zeros((len(regs), len(cols)))
    for r_, u in enumerate(regs):
        i = units_sig.index(u)
        f, t = F[i, cols], T[i, cols]
        ok = (f < fcut) & np.array([genes[c] not in targets[u] for c in cols])
        M[r_] = np.where(ok, np.sign(t) * np.minimum(-np.log10(np.maximum(f, 1e-300)), cap), 0)
    keep = (M != 0).any(axis=0)
    M = M[:, keep]
    if M.shape[1] == 0:
        return
    Aabs = np.abs(M)
    prim = Aabs.argmax(axis=0)
    sgn = np.sign(M[prim, np.arange(M.shape[1])])
    M = M[:, np.lexsort((-Aabs.max(axis=0), sgn, prim))]
    own_txt, own_v = [], []
    for u in regs:
        i = units_sig.index(u)
        tg = [x for x in own[u] if x in gi]                   # the unit's own target(s), in name order
        if not tg:
            own_txt.append("–"); own_v.append(0.0); continue
        tv = [(T[i, gi[x]], Pv[i, gi[x]]) for x in tg]
        own_txt.append("/".join(f"{t:.1f}" for t, _ in tv))
        t_b, p_b = min(tv, key=lambda z: z[1])
        own_v.append(float(np.sign(t_b) * min(-np.log10(max(float(p_b), 1e-300)), cap_own)) if p_b < 0.05 else 0.0)
    n = len(regs)
    H_in = 0.3 * n + 1.25                                     # heatmap rows + fixed room for labels/colorbar
    fig = plt.figure(figsize=(6.4, H_in))
    bottom = 1.0 / H_in
    gs = fig.add_gridspec(1, 3, width_ratios=[0.5, 5, 1.5], wspace=0.04, left=0.12, right=0.97,
                          top=1 - 0.3 / H_in, bottom=bottom)
    axo, ax, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    kw = dict(cmap="RdBu_r", shading="flat", edgecolors="none", antialiased=False)
    axo.pcolormesh(np.arange(2), np.arange(n + 1), np.array(own_v)[:, None], norm=TwoSlopeNorm(0, -cap_own, cap_own), **kw)
    for i, txt in enumerate(own_txt):
        axo.text(0.5, i + 0.5, txt, ha="center", va="center", fontsize=4.5,
                 color="white" if abs(own_v[i]) > 6 else "black")
    axo.set_xlim(0, 1); axo.set_ylim(n, 0)
    axo.set_xticks([0.5]); axo.set_xticklabels(["own target\n(t; P<0.05)"], fontsize=5)
    axo.set_yticks(np.arange(n) + 0.5)
    tip = dict(zip(H.unit, H.target_in_program))
    def label(u):
        notes = (["in program"] if tip.get(u) is True else []) + \
                ([f"shared CRE with {', '.join(partners[u])}"] if partners.get(u) else [])
        return u + (f"\n({'; '.join(notes)})" if notes else "")
    axo.set_yticklabels([label(u) for u in regs], fontsize=5.5, linespacing=0.9)
    im = ax.pcolormesh(np.arange(M.shape[1] + 1), np.arange(n + 1), M, norm=TwoSlopeNorm(0, -cap, cap), **kw)
    ax.set_xlim(0, M.shape[1]); ax.set_ylim(n, 0)
    for i in range(1, n):
        ax.axhline(i, color="white", lw=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{M.shape[1]:,} {p} genes changed at FDR < {fcut:g} by at least one regulator "
                  f"(of {len(cols):,} measured)", fontsize=5.5)
    ax.set_title(f"{p}: regulators × program genes", loc="left", fontsize=6.5, fontweight="bold")
    up = (M > 0).sum(axis=1); dn = (M < 0).sum(axis=1)
    y = np.arange(n) + 0.5
    axb.barh(y, dn, color="#2166ac", height=0.7)
    axb.barh(y, up, left=dn, color="#b2182b", height=0.7)
    for yi, d_, u_ in zip(y, dn, up):
        axb.text(d_ + u_ + max(1, (up + dn).max()) * 0.03, yi, f"{d_} down, {u_} up", va="center", fontsize=5)
    axb.set_xlim(0, max(1, (up + dn).max()) * 2.3); axb.set_ylim(n, 0)
    axb.set_xticks([]); axb.set_yticks([])
    for s_ in ("top", "right", "bottom"):
        axb.spines[s_].set_visible(False)
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + pos.width * 0.3, bottom - 0.62 / H_in, pos.width * 0.4, 0.09 / H_in])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-cap, -cap / 2, 0, cap / 2, cap])
    cb.ax.set_xticklabels([f"≤−{cap}", f"−{cap / 2:g}", "0", f"{cap / 2:g}", f"≥{cap}"], fontsize=5)
    cb.set_label("signed −log10 FDR (red up, blue down)", fontsize=5.5)
    stem = os.path.join(out, "heatmap_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", p))
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
