"""Data check (first step): is this dataset set up correctly for CRISProg, and what can be run on it?

Looks at the analysis data with the config's settings and reports PASS / WARN / FAIL per item:
  expression     .X normalized (not raw counts), counts layer, gene-name style, duplicated genes
  guides         guide calls, guides per cell (MOI), cells without guides
  controls       negative-control guides found, their types, control-only cells, whether the negative-control
                 check can run (needs individual control guides, not one merged label)
  units          units, perturbed cells per unit, guides per unit, positive controls
  targets        target names matched to measured genes (incl. after removing promoter/guide suffixes)
  knockdown      for every unit with a measured target: t of the target gene in perturbed vs other cells
                 (same covariates as the association test); positive controls separately
  batches        batch levels, units confined to one batch, obs columns that look like unused batches
  programs       measured genes per given program; counts for discovery

Outputs (0_check/):
  check_report.md     verdicts, numbers, and 'Settings explained' (what each setting does, its value, what the data
                      shows; rule-based)
  suggested_config.yaml  settings the rules recommend changing, each with its reason
  unit_qc.tsv         per unit: type, cells, guides, targets, measured target genes, knockdown t, top-batch share
  review.md           optional: an AI reading of check_report.md (check.review: claude-code | api)
"""
import os
import re

import numpy as np
import pandas as pd

from .model import OLSEngine
from .utils import log

LEVELS = {"PASS": 0, "INFO": 1, "WARN": 2, "FAIL": 3}
BATCH_HINT = re.compile(r"batch|lane|gem|library|sample|donor|plate|run|chip|channel|pool|replicate", re.I)


def _sample_rows(M, n=2000, seed=0):
    rows = np.sort(np.random.default_rng(seed).choice(M.shape[0], min(n, M.shape[0]), replace=False))
    S = M[rows]
    return S.toarray() if hasattr(S, "toarray") else np.asarray(S)


def knockdown_t(A, units_idx, min_cells=10, chunk=200):
    """t of each unit's measured target gene(s) in perturbed vs other cells (most negative over its genes)."""
    rows = np.arange(len(A.cells))
    Z, *_ = A.design(rows)
    gi = {g: i for i, g in enumerate(A.genes)}
    pairs = [(k, gi[g]) for k, u in enumerate(units_idx)
             for g in str(A.units["target_genes"].iat[u]).split(";") if g in gi]
    out = np.full(len(units_idx), np.nan)
    if not pairs:
        return out
    U = A.U[:, units_idx]
    genes = sorted({g for _, g in pairs})
    by_gene = {}
    for k, g in pairs:
        by_gene.setdefault(g, []).append(k)
    for s in range(0, len(genes), chunk):
        cols = genes[s:s + chunk]
        Y = A.X[:, cols]
        Y = Y.toarray() if hasattr(Y, "toarray") else np.asarray(Y)
        _, _, t, _ = OLSEngine(U, Y.astype(np.float64), Z).stats()
        for j, g in enumerate(cols):
            for k in by_gene[g]:
                if np.isfinite(t[k, j]) and not (out[k] <= t[k, j]):      # keep the most negative
                    out[k] = t[k, j]
    return out


def run_check(cfg, ctx, out):
    A, d, a = ctx.analysis, cfg["data"], cfg["association"]
    items, sections = [], []

    def add(level, area, msg):
        items.append((level, area, msg))

    # ---- expression ----
    Xs = _sample_rows(A.X)
    nz = Xs[Xs != 0]
    integer = bool(nz.size and np.allclose(nz, np.round(nz)))
    if integer and nz.max() > 20:
        add("FAIL", "expression", f".X looks like raw counts (integer values up to {nz.max():.0f}); CRISProg "
            "expects normalized, log-transformed expression in .X and raw counts in data.counts_layer")
    else:
        add("PASS", "expression", f".X looks normalized (max {Xs.max():.2f} on 2,000 random cells)")
    if A.counts is not None:
        Cs = _sample_rows(A.counts)
        cz = Cs[Cs != 0]
        if cz.size and not np.allclose(cz, np.round(cz)):
            add("WARN", "expression", f"layer '{d['counts_layer']}' is not integer-valued: is it raw counts?")
        else:
            add("PASS", "expression", f"raw counts in layer '{d['counts_layer']}' (median "
                f"{np.median(Cs.sum(1)):,.0f} UMI per cell)")
    else:
        add("WARN", "expression", "no raw counts layer (data.counts_layer): program discovery and the "
            "log_total_counts covariate need it; genes detected are counted from .X")
    ens = np.mean([str(g).startswith(("ENSG", "ENSMUSG")) for g in A.genes])
    if ens > 0.5:
        add("WARN", "expression", f"{ens:.0%} of gene names look like Ensembl IDs; programs, target matching and "
            "enrichment need symbols (set data.gene_col to the symbol column)")
    dup = pd.Index(A.genes).duplicated().sum()
    if dup:
        add("INFO", "expression", f"{dup:,} duplicated gene names (the first occurrence is used)")
    add("INFO", "expression", f"{len(A.cells):,} cells x {len(A.genes):,} genes after dropping cells without guides")

    # ---- guides / MOI ----
    moi = A.builtin["moi"]
    add("INFO", "guides", f"non-control guides per cell: median {np.median(moi):.0f}, mean {moi.mean():.2f}, "
        f"max {moi.max():.0f}; {np.mean(moi == 1):.0%} of cells carry exactly one"
        + ("; high MOI: each unit is compared with cells carrying other guides" if np.median(moi) >= 3 else ""))

    # ---- controls ----
    nc = len(A.control_guides)
    ctrl_only = int(np.sum(moi == 0))
    if nc == 0:
        add("WARN", "controls", "no control guides identified (data.controls / data.control_pattern). Tests still "
            "work (every unit is compared with all other cells), but the negative-control check cannot run")
    else:
        types = pd.Series(A.control_type).value_counts().to_dict()
        add("PASS", "controls", f"{nc:,} control guide labels ({', '.join(f'{k}: {v}' for k, v in types.items())}); "
            f"{ctrl_only:,} cells carry only control guides")
        n1u = np.asarray(A.U.sum(axis=0)).ravel()
        n1u = n1u[n1u >= a["min_perturbed"]]
        target = int(a["control_unit_cells"] or (np.median(n1u) if len(n1u) else a["min_perturbed"]))
        Gcc = A.Gc.tocsc()
        cells_typ = {k: int((np.asarray(Gcc[:, np.flatnonzero(A.control_type == k)].sum(axis=1)).ravel() > 0).sum())
                     for k in types}
        small = [k for k, v in cells_typ.items() if v < 2 * target]
        if nc < 4:
            add("WARN", "controls", f"control guides appear merged into {nc} label(s) ({', '.join(A.control_guides[:3])}); "
                "the negative-control check needs individual control guide calls")
        else:
            ok = [k for k in types if k not in small]
            add("PASS" if ok else "WARN", "controls", "negative-control check: " + (
                f"up to {a['n_control_units']} control units of ~{target:,} perturbed cells per type for "
                f"{', '.join(ok)}" if ok else "no control type has enough cells")
                + (f"; too few control cells (< {2 * target:,}) for: {', '.join(small)}" if small else ""))
        has_ctrl = np.asarray(A.Gc.sum(axis=1)).ravel() > 0
        if has_ctrl.any() and (np.asarray(A.U[has_ctrl].sum(axis=1)).ravel() == 0).all():
            add("INFO", "controls", "cells with control guides carry no targeting guide: the controls are a separate "
                "population (e.g. a single control guide vs several targeting guides per cell). MOI is modeled as "
                "categories so that control units are compared with control cells of the same MOI")
        if not d["control_types"] and nc >= 4:
            kinds = {"non-targeting": r"non.?target|ntc", "safe-targeting": r"safe|^NC_OR|^OR\d"}
            seen = [k for k, rx in kinds.items() if any(re.search(rx, g, re.I) for g in A.control_guides)]
            if len(seen) > 1:
                add("INFO", "controls", f"controls look like several kinds ({', '.join(seen)}); set data.control_types "
                    "to check them separately")

    # ---- units ----
    n1 = np.asarray(A.U.sum(axis=0)).ravel().astype(int)
    tested = (n1 >= a["min_perturbed"]) & (n1 <= len(A.cells) - a["min_perturbed"])
    add("INFO", "units", f"{len(A.units):,} units; {int(tested.sum()):,} with >= {a['min_perturbed']} perturbed cells; "
        f"perturbed cells per tested unit: median {np.median(n1[tested]) if tested.any() else 0:.0f}, "
        f"10th percentile {np.percentile(n1[tested], 10) if tested.any() else 0:.0f}")
    if tested.sum() < 0.5 * len(A.units):
        add("WARN", "units", f"only {tested.mean():.0%} of units have enough cells")
    ptype = A.units["unit_type"].to_numpy()
    if d["positive_controls"]:
        found = A.units.unit[ptype == "positive control"].tolist()
        add("PASS" if found else "WARN", "units", f"positive controls found as units: {', '.join(found) or 'none'}")

    # ---- target genes ----
    has_t = A.units["targets"].fillna("").str.len() > 0
    resolved = A.units["target_genes"].fillna("").str.len() > 0
    if has_t.any():
        frac = resolved[has_t].mean()
        lvl = "PASS" if frac >= 0.8 else "WARN"
        miss = A.units.unit[has_t & ~resolved].head(8).tolist()
        add(lvl, "targets", f"{int(resolved.sum()):,} of {int(has_t.sum()):,} units have a measured target gene "
            f"({frac:.0%})" + (f"; not measured e.g. {', '.join(miss)}" if miss else ""))
    else:
        add("INFO", "targets", "no target map: units are guides and no knockdown check is possible")

    # ---- knockdown ----
    tidx = np.flatnonzero(tested & resolved.to_numpy())
    kd = np.full(len(A.units), np.nan)
    if len(tidx):
        log(f"check: knockdown of the target gene for {len(tidx):,} units")
        kd[tidx] = knockdown_t(A, tidx)
        v = kd[tidx]
        test_m = ptype[tidx] == "test"
        frac_kd = np.mean(v[test_m] < -2) if test_m.any() else np.nan
        add("PASS" if frac_kd >= 0.5 else "WARN", "knockdown",
            f"target gene down (t < -2) in {frac_kd:.0%} of tested units with a measured target; median t "
            f"{np.nanmedian(v[test_m]):.1f}. Low values can also mean the target is barely expressed")
        names_t = A.units.unit.to_numpy()[tidx]
        up2, up4 = names_t[(v > 2) & test_m], names_t[(v > 4) & test_m]
        if len(up2):
            add("INFO", "knockdown", f"{len(up2)} units have their target gene up with t > 2 (about "
                f"{0.023 * int(np.sum(np.abs(v[test_m]) < 2) + len(up2)):.0f} expected by chance among units without "
                "a detectable knockdown)")
        if len(up4) >= 3:
            add("WARN", "knockdown", f"{len(up4)} units have their target gene clearly UP (t > 4), which CRISPRi "
                f"should not do; check the guide -> target map, e.g. {', '.join(up4[:8])}")
        pc = tidx[ptype[tidx] == "positive control"]
        if len(pc):
            add("PASS" if np.nanmedian(kd[pc]) < -5 else "WARN", "knockdown", "positive controls: "
                + ", ".join(f"{A.units.unit.iat[i]} t {kd[i]:.1f}" for i in pc))

    # ---- batches ----
    top_share = np.full(len(A.units), np.nan)
    if A.batch_cols:
        b = A.obs[A.batch_cols].astype(str).agg("|".join, axis=1).to_numpy()
        lev, codes = np.unique(b, return_inverse=True)
        sizes = np.bincount(codes)
        add("INFO", "batches", f"batch ({', '.join(A.batch_cols)}): {len(lev)} levels, cells per level "
            f"{sizes.min():,}-{sizes.max():,}")
        Uc = A.U.tocsc()
        for j in np.flatnonzero(tested):
            c = codes[Uc.indices[Uc.indptr[j]:Uc.indptr[j + 1]]]
            top_share[j] = np.bincount(c, minlength=len(lev)).max() / max(1, len(c))
        one = int(np.sum(top_share[tested] >= 0.95))
        if one:
            add("INFO", "batches", f"{one:,} tested units have >= 95% of their cells in one batch (guide subpools?); "
                "they are compared within that batch only, which the batch covariate handles")
    cand = [c for c in A.obs.columns if BATCH_HINT.search(str(c)) and c not in A.batch_cols
            and 1 < A.obs[c].nunique() <= 200]
    if cand:
        add("WARN" if not A.batch_cols else "INFO", "batches", f"obs columns that look like batches but are not used: "
            f"{', '.join(cand[:6])} (covariates.batch)")

    # ---- programs ----
    if cfg["programs"]["file"]:
        from .data import read_programs
        P = read_programs(cfg["programs"]["file"])
        gs = set(A.genes)
        cov = {p: (sum(g in gs for g in gl), len(gl)) for p, gl in P.items()}
        low = [p for p, (m, n) in cov.items() if m < cfg["programs"]["min_genes"]]
        add("FAIL" if len(low) == len(cov) else ("WARN" if low else "PASS"), "programs",
            "measured genes per program: " + ", ".join(f"{p} {m}/{n}" for p, (m, n) in cov.items())
            + (f"; dropped (< {cfg['programs']['min_genes']} genes): {', '.join(low)}" if low else ""))
    elif A.counts is None:
        add("FAIL", "programs", "no programs.file and no raw counts: program discovery needs data.counts_layer")
    else:
        add("INFO", "programs", f"programs will be discovered ({', '.join(cfg['programs']['discover']['methods'])})")

    # ---- write ----
    qc = A.units.assign(n_perturbed=n1, tested=tested, target_t=kd, top_batch_share=top_share)
    qc.to_csv(os.path.join(out, "unit_qc.tsv"), sep="\t", index=False, float_format="%.4g")
    worst = max((LEVELS[l] for l, _, _ in items), default=0)
    verdict = {0: "PASS", 1: "PASS", 2: "WARN", 3: "FAIL"}[worst]
    L = [f"# Data check: {cfg['name']}", "", f"**Overall: {verdict}**", "",
         f"Data: `{d['h5ad']}`", ""]
    for area in dict.fromkeys(x[1] for x in items):
        L += [f"## {area}", ""] + [f"- **{l}** {m}" for l, ar, m in items if ar == area] + [""]
    from .settings import explanations_md, suggest_and_explain, write_suggestions
    _, znames, *_ = A.design(np.arange(len(A.cells)))
    S, E = suggest_and_explain(cfg, A, znames)
    write_suggestions(S, os.path.join(out, "suggested_config.yaml"))
    L += [explanations_md(E, S)]
    L += ["## Files", "", "- `unit_qc.tsv`: per unit cells, guides, targets, knockdown t, top-batch share",
          "- `suggested_config.yaml`: the settings changes suggested above, with reasons", ""]
    md = "\n".join(L)
    with open(os.path.join(out, "check_report.md"), "w") as fh:
        fh.write(md)
    for l, ar, m in items:
        if l != "PASS":
            log(f"check [{l}] {ar}: {m}")
    for path, value, reason in S:
        log(f"check suggests {path} = {value}  ({reason})")
    log(f"data check: {verdict} ({sum(l == 'WARN' for l, _, _ in items)} warnings, "
        f"{sum(l == 'FAIL' for l, _, _ in items)} failures); see {os.path.join(out, 'check_report.md')}")
    if cfg["check"]["review"] not in (None, "none"):
        from .annotate import review_check
        review_check(cfg, out, md)
    if verdict == "FAIL" and cfg["check"]["strict"]:
        raise SystemExit("data check failed (check.strict: true); see check_report.md")
    return verdict
