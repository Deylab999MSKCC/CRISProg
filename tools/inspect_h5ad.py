#!/usr/bin/env python3
"""
Before writing a config for a new dataset: print what CRISProg needs to know about a Perturb-seq h5ad: cells/genes,
raw counts or normalized, gene-name columns (and how well the program genes match them), guide/perturbation
columns and their format, controls, batch-like columns, obsm/uns entries (guide matrices often live there).

  python tools/inspect_h5ad.py <h5ad> [<h5ad> ...] [--programs programs.tsv]

Reads the h5ad in backed mode, but large files still need a compute node (run it inside a job).
"""
import argparse
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

GUIDE_HINT = re.compile(r"guide|sgrna|grna|protospacer|target|perturb|feature_call|gene_id|condition|knock", re.I)
CTRL_HINT = re.compile(r"non.?target|ntc|safe|control|scramble|neg|olfr|aavs", re.I)


def mstats(name, M, n=2000):
    rows = np.random.default_rng(0).choice(M.shape[0], min(n, M.shape[0]), replace=False)
    S = M[np.sort(rows)]
    S = S.toarray() if sp.issparse(S) else np.asarray(S)
    nz = S[S != 0]
    integer = bool(nz.size and np.allclose(nz, np.round(nz)))
    tot = S.sum(1)
    print(f"  {name}: dtype {M.dtype}, {'sparse' if sp.issparse(M) else 'dense'}; on {len(rows)} random cells: "
          f"integer values {integer}, min {S.min():.3g}, max {S.max():.3g}, row sums {np.percentile(tot, 5):.4g}"
          f"-{np.percentile(tot, 95):.4g} (5-95%), {np.mean(S > 0):.1%} nonzero"
          + ("  -> looks like RAW COUNTS" if integer and S.max() > 20 else ""))


def describe_col(s, k=8):
    vc = s.astype(str).value_counts()
    out = f"{s.dtype}, {s.nunique():,} unique; top: " + ", ".join(f"{v!r} ({vc[v]:,})" for v in vc.index[:k])
    ctrl = [v for v in vc.index if CTRL_HINT.search(v)]
    if ctrl:
        out += f"\n      control-like values ({len(ctrl)}): " + ", ".join(repr(v) for v in ctrl[:10])
        out += f" -> {int(vc[ctrl].sum()):,} cells"
    seps = {c: s.astype(str).str.contains(re.escape(c)).mean() for c in (";", ",", "|", "+", "_", "-")}
    seps = {c: f for c, f in seps.items() if f > 0.01}
    if seps:
        out += "\n      separators present (fraction of cells): " + ", ".join(f"'{c}' {f:.0%}" for c, f in seps.items())
    return out


def inspect(path, programs):
    import anndata
    print("=" * 100, f"\n{path}")
    ad = anndata.read_h5ad(path, backed="r")
    print(f"cells x genes: {ad.n_obs:,} x {ad.n_vars:,}")
    print(f"obs_names e.g. {list(ad.obs_names[:3])}; var_names e.g. {list(ad.var_names[:5])}")
    print("\nmatrices:")
    mstats("X", ad.X)
    for l in ad.layers:
        mstats(f"layers['{l}']", ad.layers[l])
    if ad.raw is not None:
        print(f"  raw: {ad.raw.n_vars:,} genes (raw.X below)")
        mstats("raw.X", ad.raw.X)
    print("\nvar columns:")
    var = ad.var
    for c in var.columns:
        print(f"  {c}: {describe_col(var[c], 4)}")
    if programs is not None:
        pg = set(programs.gene)
        cand = {"var_names": pd.Index(ad.var_names.astype(str))}
        cand.update({c: pd.Index(var[c].astype(str)) for c in var.columns
                     if var[c].dtype == object or str(var[c].dtype) == "category"})
        print(f"\nprogram genes ({len(pg):,}) found in:")
        for k, v in cand.items():
            print(f"  {k}: {len(pg & set(v)):,} ({len(pg & set(v)) / len(pg):.0%}); duplicated names {v.duplicated().sum():,}")
        best = max(cand, key=lambda k: len(pg & set(cand[k])))
        for p, g in programs.groupby("program"):
            print(f"    {p}: {g.gene.isin(cand[best]).sum()}/{len(g)} genes measured ({best})")
    print("\nobs columns:")
    for c in ad.obs.columns:
        tag = "  <-- guide/perturbation?" if GUIDE_HINT.search(c) else ""
        print(f"  {c}{tag}: {describe_col(ad.obs[c])}")
    print("\nobsm:", {k: (type(v).__name__, getattr(v, 'shape', None)) for k, v in ad.obsm.items()} or "none")
    for k, v in ad.obsm.items():
        if hasattr(v, "columns"):
            print(f"  obsm['{k}'] columns e.g. {list(v.columns[:8])}")
    print("uns keys:", list(ad.uns.keys()) or "none")
    for k in ad.uns.keys():
        v = ad.uns[k]
        if isinstance(v, (pd.DataFrame, np.ndarray, list)):
            print(f"  uns['{k}']: {type(v).__name__} {getattr(v, 'shape', len(v))}")
            if isinstance(v, pd.DataFrame):
                print(v.head(5).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5ad", nargs="+")
    ap.add_argument("--programs", default=None, help="program TSV (program, gene) to check gene-name matching")
    a = ap.parse_args()
    P = pd.read_csv(a.programs, sep="\t") if a.programs else None
    for p in a.h5ad:
        inspect(p, P)


if __name__ == "__main__":
    main()
