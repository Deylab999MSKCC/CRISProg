"""Program discovery: Hotspot gene modules on a PCA and/or cNMF cell embedding.

For each embedding, Hotspot builds a cell kNN graph, finds genes whose expression is locally autocorrelated
on that graph, and groups the top autocorrelated genes into modules by their local correlations. Each module
is one program: a discrete gene set (what AUCell needs). cNMF is used only as an embedding (its per-cell usages);
its continuous gene weights are not used as programs.

Outputs (programs step folder):
  programs.tsv                program, gene, rank (by autocorrelation Z within the module), method, autocorr_Z
  programs.gmt
  programs_summary.tsv        program, method, n_genes, top genes
  <method>_autocorrelation.tsv, <method>_modules.tsv
  cnmf/                       cNMF run (usages, spectra, k-selection) when cnmf is used
"""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .utils import log

LABEL = {"pca": "PCA", "cnmf": "cNMF"}


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
    log(f"discovery data: {len(cells):,} cells x {len(genes):,} genes (detected in >= "
        f"{max(10, int(dc['min_cells_frac'] * len(rows)))} cells)")
    return cells, genes, counts


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


def _cnmf_worker(args):
    from cnmf import cNMF
    d, i, n = args
    cNMF(output_dir=d, name="cnmf").factorize(worker_i=i, total_workers=n)


def _cnmf(counts, genes, cells, dc, out):
    import anndata
    from cnmf import cNMF
    d = os.path.join(out, "cnmf")
    os.makedirs(d, exist_ok=True)
    fn = os.path.join(d, "counts.h5ad")
    anndata.AnnData(X=counts.astype(np.float32), obs=pd.DataFrame(index=cells),
                    var=pd.DataFrame(index=genes)).write_h5ad(fn)
    k, dt = int(dc["cnmf_k"]), float(dc["cnmf_density_threshold"])
    obj = cNMF(output_dir=d, name="cnmf")
    log(f"cNMF: K = {k}, {dc['cnmf_n_iter']} runs over {min(int(dc['jobs']), int(dc['cnmf_n_iter']))} worker(s), "
        f"{dc['n_hvg']} highly variable genes")
    obj.prepare(counts_fn=fn, components=[k], n_iter=int(dc["cnmf_n_iter"]), seed=int(dc["seed"]),
                num_highvar_genes=int(dc["n_hvg"]))
    workers = max(1, min(int(dc["jobs"]), int(dc["cnmf_n_iter"])))
    if workers == 1:
        obj.factorize(worker_i=0, total_workers=1)
    else:                                                   # cNMF splits its runs over independent workers
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(_cnmf_worker, [(d, i, workers) for i in range(workers)]))
    obj.combine()
    obj.consensus(k=k, density_threshold=dt, show_clustering=False, close_clustergram_fig=True)
    usage, scores, tpm, top = obj.load_results(K=k, density_threshold=dt)
    usage.index = usage.index.astype(str)
    usage = usage.reindex(cells).fillna(0.0)
    usage.to_csv(os.path.join(d, "usage.tsv"), sep="\t")
    return usage.to_numpy()


def _hotspot(counts, genes, cells, emb, method, dc, out):
    import anndata
    import hotspot
    ad = anndata.AnnData(X=sp.csc_matrix(counts.astype(np.float64)), obs=pd.DataFrame(index=cells),
                         var=pd.DataFrame(index=genes))
    ad.layers["counts"] = ad.X
    ad.obs["total_counts"] = np.asarray(counts.sum(axis=1)).ravel()
    ad.obsm["X_emb"] = np.asarray(emb, dtype=np.float64)
    hs = hotspot.Hotspot(ad, layer_key="counts", model="danb", latent_obsm_key="X_emb",
                         umi_counts_obs_key="total_counts")
    hs.create_knn_graph(weighted_graph=False, n_neighbors=int(dc["n_neighbors"]))
    auto = hs.compute_autocorrelations(jobs=int(dc["jobs"]))
    auto.to_csv(os.path.join(out, f"{method}_autocorrelation.tsv"), sep="\t")
    sig = auto[auto.FDR < dc["fdr"]].sort_values("Z", ascending=False).head(int(dc["n_genes"]))
    log(f"Hotspot ({LABEL[method]}): {int((auto.FDR < dc['fdr']).sum()):,} autocorrelated genes at FDR < "
        f"{dc['fdr']}; building modules from the top {len(sig):,}")
    if len(sig) < dc["min_module_genes"]:
        log(f"WARNING: too few autocorrelated genes for {LABEL[method]} modules")
        return pd.DataFrame(columns=["program", "gene", "rank", "method", "autocorr_Z"])
    hs.compute_local_correlations(sig.index, jobs=int(dc["jobs"]))
    modules = hs.create_modules(min_gene_threshold=int(dc["min_module_genes"]), core_only=True,
                                fdr_threshold=dc["fdr"])
    modules.rename("module").to_frame().to_csv(os.path.join(out, f"{method}_modules.tsv"), sep="\t")
    rows = []
    for m in sorted(x for x in modules.unique() if x != -1):
        g = auto.loc[modules.index[modules == m]].sort_values("Z", ascending=False)
        name = f"{LABEL[method]}-HS{int(m)}"
        rows += [(name, gene, i + 1, method, float(z)) for i, (gene, z) in enumerate(g.Z.items())]
    res = pd.DataFrame(rows, columns=["program", "gene", "rank", "method", "autocorr_Z"])
    log(f"Hotspot ({LABEL[method]}): {res.program.nunique()} modules, "
        f"{int((modules != -1).sum())} genes assigned, {int((modules == -1).sum())} unassigned")
    return res


def discover_programs(cfg, out):
    dc = cfg["programs"]["discover"]
    bad = set(dc["methods"]) - set(LABEL)
    if bad or not dc["methods"]:
        raise SystemExit(f"programs.discover.methods must be a subset of {list(LABEL)}")
    cells, genes, counts = _load(dc)
    allp = []
    for method in dc["methods"]:
        emb = _pca(counts, genes, dc) if method == "pca" else _cnmf(counts, genes, cells, dc, out)
        allp.append(_hotspot(counts, genes, cells, emb, method, dc, out))
    P = pd.concat(allp, ignore_index=True)
    if P.empty:
        raise SystemExit("no programs found; try more cells, a lower min_module_genes or more n_genes")
    P.to_csv(os.path.join(out, "programs.tsv"), sep="\t", index=False, float_format="%.4g")
    with open(os.path.join(out, "programs.gmt"), "w") as fh:
        for p, g in P.groupby("program", sort=False):
            fh.write("\t".join([p, g.method.iloc[0]] + list(g.gene)) + "\n")
    summ = P.groupby("program", sort=False).agg(method=("method", "first"), n_genes=("gene", "size"),
                                                top_genes=("gene", lambda s: ", ".join(s[:15])))
    summ.to_csv(os.path.join(out, "programs_summary.tsv"), sep="\t")
    pd.Series(genes).to_csv(os.path.join(out, "background_genes.txt"), index=False, header=False)
    print(summ.to_string(), flush=True)
