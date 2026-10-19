"""Program discovery: Hotspot and/or hdWGCNA gene modules. Each module is one program: a discrete gene set
(what AUCell needs). Both methods start from the same PCA of the discovery data.

  hotspot   Hotspot (DeTomaso & Yosef 2021) builds a cell kNN graph on the PCA, finds genes whose expression is
            locally autocorrelated on it, and groups the top autocorrelated genes into modules by their local
            correlations. Hotspot has no embedding of its own; it needs one, which is why PCA comes first.
  hdwgcna   hdWGCNA (Morabito et al. 2023) averages cells into metacells (nearest neighbours on the PCA), builds
            a signed co-expression network on them and cuts it into modules (R/hdwgcna.R).

Outputs (programs step folder):
  programs.tsv                program, gene, rank, method, score (Hotspot autocorrelation Z, or hdWGCNA kME);
                              genes ranked by score within each program
  programs.gmt
  programs_summary.tsv        program, method, n_genes, top genes
  hotspot_autocorrelation.tsv, hotspot_modules.tsv, hdwgcna_modules.tsv
"""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .utils import log

LABEL = {"hotspot": "Hotspot", "hdwgcna": "hdWGCNA"}
COLS = ["program", "gene", "rank", "method", "score"]
R_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "R", "hdwgcna.R")


def _load(dc):
    import anndata
    ad = anndata.read_h5ad(dc["h5ad"])
    if not dc["counts_layer"] or dc["counts_layer"] not in ad.layers:
        raise SystemExit("program discovery needs raw counts: set data.counts_layer (or programs.discover."
                         f"counts_layer); layers in {dc['h5ad']}: {', '.join(ad.layers) or 'none'}")
    genes = (ad.var[dc["gene_col"]] if dc["gene_col"] else ad.var_names).astype(str).to_numpy()
    counts = sp.csr_matrix(ad.layers[dc["counts_layer"]])
    keep_g = ~pd.Series(genes).duplicated().to_numpy()
    rng = np.random.default_rng(dc["seed"])
    n = ad.n_obs
    rows = np.sort(rng.choice(n, dc["max_cells"], replace=False)) if n > dc["max_cells"] else np.arange(n)
    if len(rows) < n:
        log(f"discovery on a random {len(rows):,} of {n:,} cells (programs.discover.max_cells)")
    counts = counts[rows][:, keep_g]
    genes = genes[keep_g]
    det = np.asarray((counts > 0).sum(axis=0)).ravel()
    ok = det >= max(10, dc["min_cells_frac"] * len(rows))
    counts, genes = counts[:, ok], genes[ok]
    tot = np.asarray(counts.sum(axis=1)).ravel()
    good = tot > 0
    counts = counts[good]
    cells = ad.obs_names.to_numpy()[rows][good].astype(str)
    gc = dc["hdwgcna"]["group_col"]
    if gc and gc not in ad.obs:
        raise SystemExit(f"programs.discover.hdwgcna.group_col '{gc}' is not a column of {dc['h5ad']} .obs")
    groups = ad.obs[gc].astype(str).to_numpy()[rows][good] if gc else None
    log(f"discovery data: {len(cells):,} cells x {len(genes):,} genes (detected in >= "
        f"{max(10, int(dc['min_cells_frac'] * len(rows)))} cells)")
    return cells, genes, counts, groups


def _pca(counts, genes, dc):
    import anndata
    import scanpy as sc
    ad = anndata.AnnData(X=counts.astype(np.float32).copy(), var=pd.DataFrame(index=genes))
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=min(dc["n_hvg"], ad.n_vars - 1))
    ad = ad[:, ad.var.highly_variable].copy()
    sc.pp.scale(ad, max_value=10)
    n_pcs = min(dc["n_pcs"], ad.n_vars - 1, ad.n_obs - 1)
    sc.pp.pca(ad, n_comps=n_pcs, random_state=dc["seed"])
    log(f"PCA: {n_pcs} components on {ad.n_vars:,} highly variable genes")
    return ad.obsm["X_pca"]


def _hotspot(counts, genes, cells, emb, dc, out):
    hc, method = dc["hotspot"], "hotspot"
    import anndata
    import hotspot
    ad = anndata.AnnData(X=sp.csc_matrix(counts.astype(np.float64)), obs=pd.DataFrame(index=cells),
                         var=pd.DataFrame(index=genes))
    ad.layers["counts"] = ad.X
    ad.obs["total_counts"] = np.asarray(counts.sum(axis=1)).ravel()
    ad.obsm["X_emb"] = np.asarray(emb, dtype=np.float64)
    hs = hotspot.Hotspot(ad, layer_key="counts", model="danb", latent_obsm_key="X_emb",
                         umi_counts_obs_key="total_counts")
    hs.create_knn_graph(weighted_graph=False, n_neighbors=int(hc["n_neighbors"]))
    auto = hs.compute_autocorrelations(jobs=int(dc["jobs"]))
    auto.to_csv(os.path.join(out, f"{method}_autocorrelation.tsv"), sep="\t")
    sig = auto[auto.FDR < hc["fdr"]].sort_values("Z", ascending=False).head(int(hc["n_genes"]))
    log(f"Hotspot: {int((auto.FDR < hc['fdr']).sum()):,} autocorrelated genes at FDR < {hc['fdr']}; "
        f"building modules from the top {len(sig):,}")
    if len(sig) < hc["min_module_genes"]:
        log("WARNING: too few autocorrelated genes for Hotspot modules")
        return pd.DataFrame(columns=COLS)
    hs.compute_local_correlations(sig.index, jobs=int(dc["jobs"]))
    modules = hs.create_modules(min_gene_threshold=int(hc["min_module_genes"]), core_only=True,
                                fdr_threshold=hc["fdr"])
    modules.rename("module").to_frame().to_csv(os.path.join(out, f"{method}_modules.tsv"), sep="\t")
    rows = []
    for m in sorted(x for x in modules.unique() if x != -1):
        g = auto.loc[modules.index[modules == m]].sort_values("Z", ascending=False)
        name = f"Hotspot-{int(m)}"
        rows += [(name, gene, i + 1, method, float(z)) for i, (gene, z) in enumerate(g.Z.items())]
    res = pd.DataFrame(rows, columns=COLS)
    log(f"Hotspot: {res.program.nunique()} modules, "
        f"{int((modules != -1).sum())} genes assigned, {int((modules == -1).sum())} unassigned")
    return res


def _hdwgcna(counts, genes, cells, groups, emb, dc, rscript, out):
    import shutil
    import subprocess
    from .aucell import write_r_matrix
    wc = dc["hdwgcna"]
    tmp = os.path.join(out, "hdwgcna_input")
    write_r_matrix(counts, genes, cells, tmp)
    pd.DataFrame(emb, index=cells).to_csv(os.path.join(tmp, "pca.tsv"), sep="\t")
    if groups is not None:
        with open(os.path.join(tmp, "groups.txt"), "w") as fh:
            fh.write("\n".join(groups) + "\n")
    res = os.path.join(out, "hdwgcna_modules.tsv")
    cmd = [rscript, R_SCRIPT, "--input", tmp, "--out", res, "--threads", str(dc["jobs"]), "--seed", str(dc["seed"]),
           "--fraction", str(wc["gene_fraction"]), "--k", str(wc["metacell_k"]),
           "--max-shared", str(wc["max_shared"]), "--min-cells", str(wc["min_cells"]),
           "--min-module-size", str(wc["min_module_size"]), "--deep-split", str(wc["deep_split"]),
           "--merge-cut-height", str(wc["merge_cut_height"])]
    if wc["soft_power"]:
        cmd += ["--soft-power", str(wc["soft_power"])]
    log("running hdWGCNA in R: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit(f"'{rscript}' not found; install R with Seurat, WGCNA and hdWGCNA (see environment.yml)")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"hdWGCNA (R) failed with exit code {e.returncode}; see the messages above")
    shutil.rmtree(tmp, ignore_errors=True)
    M = pd.read_csv(res, sep="\t")
    rows = []
    for m, g in M.groupby("module", sort=False):
        g = g.sort_values("kME", ascending=False)
        rows += [(f"hdWGCNA-{m}", gene, i + 1, "hdwgcna", float(k)) for i, (gene, k) in enumerate(zip(g.gene, g.kME))]
    return pd.DataFrame(rows, columns=COLS)


def discover_programs(cfg, out):
    dc = cfg["programs"]["discover"]
    bad = set(dc["methods"]) - set(LABEL)
    if bad or not dc["methods"]:
        raise SystemExit(f"programs.discover.methods must be a subset of {list(LABEL)} (got {dc['methods']})")
    cells, genes, counts, groups = _load(dc)
    emb = _pca(counts, genes, dc)
    allp = []
    for method in dc["methods"]:
        if method == "hotspot":
            allp.append(_hotspot(counts, genes, cells, emb, dc, out))
        else:
            allp.append(_hdwgcna(counts, genes, cells, groups, emb, dc, cfg["aucell"]["rscript"], out))
    allp = [p for p in allp if len(p)]
    if not allp:
        raise SystemExit("no programs found; try more cells or smaller minimum module sizes")
    P = pd.concat(allp, ignore_index=True)
    P.to_csv(os.path.join(out, "programs.tsv"), sep="\t", index=False, float_format="%.4g")
    with open(os.path.join(out, "programs.gmt"), "w") as fh:
        for p, g in P.groupby("program", sort=False):
            fh.write("\t".join([p, g.method.iloc[0]] + list(g.gene)) + "\n")
    summ = P.groupby("program", sort=False).agg(method=("method", "first"), n_genes=("gene", "size"),
                                                top_genes=("gene", lambda s: ", ".join(s[:15])))
    summ.to_csv(os.path.join(out, "programs_summary.tsv"), sep="\t")
    pd.Series(genes).to_csv(os.path.join(out, "background_genes.txt"), index=False, header=False)
    print(summ.to_string(), flush=True)
