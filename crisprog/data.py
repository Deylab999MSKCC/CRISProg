"""Inputs: the h5ad, gene programs, per-cell guide calls and the optional guide -> target table."""
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .utils import log


def load_h5ad(path, counts_layer=None, gene_col=None):
    """-> obs, cell names, gene names, X (cells x genes CSR, normalized), counts (CSR or None)."""
    import anndata
    ad = anndata.read_h5ad(path)
    genes = (ad.var[gene_col] if gene_col else ad.var_names).astype(str).to_numpy()
    X = sp.csr_matrix(ad.X, dtype=np.float32)
    counts = None
    if counts_layer:
        if counts_layer not in ad.layers:
            raise SystemExit(f"layer '{counts_layer}' not in {path}; layers: {', '.join(ad.layers) or 'none'}")
        counts = sp.csr_matrix(ad.layers[counts_layer])
    log(f"{path}: {ad.n_obs:,} cells x {ad.n_vars:,} genes"
        + (f"; counts from layer '{counts_layer}'" if counts_layer else "; no counts layer given"))
    if X.nnz and X.data.min() < 0:
        log("WARNING: .X has negative values (scaled data?). AUCell ranks genes within each cell, so "
            "this changes the scores; .X should be normalized (e.g. log1p(CP10k)), not scaled.")
    return ad.obs.copy(), ad.obs_names.astype(str).to_numpy(), genes, X, counts


def read_programs(path):
    """GMT (name, description, genes...) or TSV with columns `program` and `gene`.
    -> {program: [genes]} in file order."""
    progs = {}
    if path.lower().endswith(".gmt"):
        with open(path) as fh:
            for ln in fh:
                f = [x.strip() for x in ln.rstrip("\n").split("\t")]
                if len(f) >= 3 and f[0]:
                    progs.setdefault(f[0], []).extend(g for g in f[2:] if g)
    else:
        d = pd.read_csv(path, sep="\t", dtype=str)
        if not {"program", "gene"} <= set(d.columns):
            raise SystemExit(f"{path}: a TSV program file needs columns `program` and `gene` "
                             f"(found: {', '.join(d.columns)}); or use a .gmt file")
        d = d.dropna(subset=["program", "gene"])
        for p, g in zip(d["program"].str.strip(), d["gene"].str.strip()):
            progs.setdefault(p, []).append(g)
    if not progs:
        raise SystemExit(f"no programs found in {path}")
    return {p: list(dict.fromkeys(g)) for p, g in progs.items()}


def parse_guides(values, sep=";", controls=(), control_pattern=None):
    """One obs value per cell, e.g. 'GENE1-0--1;GENE2-1-112' (empty/NaN = no guide call).

    Control guides (exact labels, or matching the regex) are not tested and do not count
    towards MOI; their cells stay in the analysis as unperturbed for every tested unit.

    -> G (cells x guides CSR 0/1, non-control guides), guide names,
       has_call (cell has any guide call), n_control (control guides per cell)
    """
    ctrl = {c.strip() for c in controls if c.strip()}
    ctrl_re = re.compile(control_pattern) if control_pattern else None
    n = len(values)
    has_call = np.zeros(n, dtype=bool)
    n_control = np.zeros(n, dtype=np.int64)
    gidx, rows, cols = {}, [], []
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        has_call[i] = True
        for g in dict.fromkeys(x.strip() for x in s.split(sep) if x.strip()):
            if g in ctrl or (ctrl_re is not None and ctrl_re.search(g)):
                n_control[i] += 1
                continue
            rows.append(i)
            cols.append(gidx.setdefault(g, len(gidx)))
    guides = np.array(list(gidx), dtype=object)
    G = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, len(guides)))
    log(f"guide calls: {has_call.sum():,} of {n:,} cells have one; {len(guides):,} distinct non-control "
        f"guides; {int((n_control > 0).sum()):,} cells carry a control guide")
    if (ctrl or ctrl_re) and not n_control.any():
        log("WARNING: no control guide matched the given labels/pattern; check --controls / --control-pattern")
    return G, guides, has_call, n_control


def read_target_map(path):
    """TSV with columns `guide` and `target` (one row per pair; a guide may hit several targets)."""
    d = pd.read_csv(path, sep="\t", dtype=str)
    if not {"guide", "target"} <= set(d.columns):
        raise SystemExit(f"{path}: needs columns `guide` and `target` (found: {', '.join(d.columns)})")
    d = d.dropna(subset=["guide", "target"])
    d = d.assign(guide=d["guide"].str.strip(), target=d["target"].str.strip())
    return d[["guide", "target"]].drop_duplicates().reset_index(drop=True)


TARGET_SUFFIX = re.compile(r"(_P\d+(?:P\d+)*|-P\d+(?:P\d+)*(?:-\d+)?|_main|_[BC]|_strong|_weak|_\d+|_[+-]_.*|#.*)$")


def target_genes(label, gene_set):
    """Measured gene(s) a target label refers to.

    The label itself if it is a measured gene; otherwise each '|'-separated part with promoter / TSS / guide
    suffixes removed until it matches: 'POU5F1_P2' -> POU5F1, 'SOX4_P1P2' -> SOX4, 'AARS_main' -> AARS,
    'ZNF226_P1|ZNF234_P1' -> ZNF226, ZNF234. Labels with no measured gene give []."""
    if label in gene_set:
        return [label]
    out = []
    for part in label.split("|"):
        x = part
        while x not in gene_set:
            y = TARGET_SUFFIX.sub("", x)
            if y == x:
                break
            x = y
        if x in gene_set and x not in out:
            out.append(x)
    return out


def add_target_genes(units, genes):
    """units['target_genes']: the measured genes behind each unit's target labels (';'-joined)."""
    gs = set(map(str, genes))
    tg, renamed, missing = [], [], []
    for t in units["targets"].fillna(""):
        labels = [x for x in str(t).split(";") if x]
        g = list(dict.fromkeys(y for x in labels for y in target_genes(x, gs)))
        tg.append(";".join(g))
        renamed += [x for x in labels if x not in gs and target_genes(x, gs)]
        missing += [x for x in labels if not target_genes(x, gs)]
    if renamed:
        ex = ", ".join(f"{x} -> {'/'.join(target_genes(x, gs))}" for x in renamed[:5])
        log(f"target names matched to measured genes after removing suffixes: {len(renamed):,} (e.g. {ex})")
    if missing:
        log(f"target names with no measured gene (no own-target check for these): {len(missing):,} "
            f"(e.g. {', '.join(missing[:8])})")
    return units.assign(target_genes=tg)


def build_units(G, guides, target_map=None, merge_identical=True):
    """Test units and the cells x units 0/1 matrix.

    No target map: every non-control guide is its own unit.
    With a map: a cell is perturbed for a target if it carries ANY of the target's guides.
    Targets with exactly the same guides would give identical tests; they are merged into one
    unit named 'A/B' (turn off with merge_identical=False).

    -> X (cells x units CSR 0/1), units DataFrame (unit, targets, guides, n_guides)
    """
    if target_map is None:
        units = pd.DataFrame({"unit": guides, "targets": "", "guides": guides, "n_guides": 1})
        return sp.csr_matrix(G), units

    gpos = {g: i for i, g in enumerate(guides)}
    in_data = target_map["guide"].isin(gpos)
    n_absent = target_map.loc[~in_data, "guide"].nunique()
    if n_absent:
        log(f"target map: {n_absent:,} guides are not in any cell and are ignored")
    unmapped = sorted(set(guides) - set(target_map["guide"]))
    if unmapped:
        log(f"target map: {len(unmapped):,} guides seen in cells have no target; they count towards MOI "
            f"but are not tested (e.g. {', '.join(unmapped[:5])})")
    tm = target_map[in_data]
    tgt_guides = {}
    for g, t in zip(tm["guide"], tm["target"]):
        tgt_guides.setdefault(t, []).append(g)
    if merge_identical:
        by_set = {}
        for t, gl in tgt_guides.items():
            by_set.setdefault(frozenset(gl), []).append(t)
        defs = [("/".join(ts), ts, tgt_guides[ts[0]]) for ts in by_set.values()]
        n_merged = sum(len(ts) > 1 for ts in by_set.values())
        if n_merged:
            log(f"target map: {n_merged:,} sets of targets share identical guides and are tested once "
                f"(e.g. {next('/'.join(ts) for ts in by_set.values() if len(ts) > 1)})")
    else:
        defs = [(t, [t], gl) for t, gl in tgt_guides.items()]
    rows, cols = [], []
    for j, (_, _, gl) in enumerate(defs):
        for g in gl:
            rows.append(gpos[g])
            cols.append(j)
    M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(guides), len(defs)))
    X = sp.csr_matrix(G @ M)
    X.data[:] = 1.0
    units = pd.DataFrame({"unit": [d[0] for d in defs], "targets": [";".join(d[1]) for d in defs],
                          "guides": [";".join(d[2]) for d in defs], "n_guides": [len(d[2]) for d in defs]})
    log(f"target map: {len(tgt_guides):,} targets -> {len(units):,} test units")
    return X, units
