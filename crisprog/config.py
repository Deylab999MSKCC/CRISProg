"""Run configuration: a YAML file merged over these defaults. Relative paths are resolved against the
directory of the config file."""
import copy
import os

import yaml

DEFAULTS = {
    "outdir": "crisprog_results",
    "data": {
        "h5ad": None,                      # analysis data: .X normalized expression, raw counts layer, guide column
        "counts_layer": None,
        "gene_col": None,                  # var column with gene symbols (default var_names)
        "guide_col": None,                 # obs column, e.g. 'g1;g2;g3'
        "guide_sep": ";",
        "controls": [],                    # control guide labels (not tested, not counted in MOI)
        "control_pattern": None,           # regex for control guides
        "target_map": None,                # TSV guide<TAB>target; null = every guide is tested on its own
        "merge_identical_targets": True,
        "keep_cells_without_guides": False,
    },
    "programs": {
        "file": None,                      # .gmt or TSV (program, gene): use these and skip discovery
        "min_genes": 20,                   # drop programs with fewer genes measured in data.h5ad
        "background": None,                # gene list (one per line) used as the enrichment background for given
                                           # programs, e.g. the genes the programs were built from
        "discover": {
            "h5ad": None,                  # data to discover programs from (default data.h5ad)
            "counts_layer": None,          # default data.counts_layer
            "gene_col": None,              # default data.gene_col
            "methods": ["pca", "cnmf"],    # cell embeddings for Hotspot
            "max_cells": 20000,            # subsample for speed (Hotspot/cNMF scale with cells)
            "min_cells_frac": 0.005,       # keep genes detected in at least this fraction of cells
            "n_hvg": 2000,
            "n_pcs": 30,
            "cnmf_k": 10,
            "cnmf_n_iter": 50,
            "cnmf_density_threshold": 0.5,
            "n_neighbors": 30,
            "n_genes": 1000,               # top autocorrelated genes (FDR < fdr) used to build modules
            "min_module_genes": 20,
            "fdr": 0.05,
            "jobs": 4,
            "seed": 0,
        },
    },
    "covariates": {"use": ["log_n_genes", "moi"], "batch": [], "moi_col": None},
    "aucell": {"rscript": "Rscript", "cores": 1, "max_rank": None, "rankings_cache": None,
               "seed": None,                # tie-breaking seed (default: association.seed)
               "n_repeats": 1},             # average the AUC over this many tie-breaking seeds
    "association": {"min_perturbed": 10, "n_perm": 10000, "perm_p": "auto", "pool_bins": 5, "depth_bins": 10,
                    "fdr_scope": "program", "fdr": 0.05, "seed": 1},
    "gene_test": {"fdr": 0.05, "n_null": 20, "chunk": 2000, "seed": 1},
    "enrichment": {"backend": "enrichr",   # enrichr (web, default) | gmt (local files)
                   "libraries": ["GO_Biological_Process", "Reactome_Pathways", "KEGG", "WikiPathways_Human",
                                 "MSigDB_Hallmark"],
                   "gmt": [],              # local .gmt files: used for backend gmt and as the fallback
                   "q": 0.05, "min_overlap": 3, "max_term_size": 500},
    "annotation": {"backend": "claude-code",   # claude-code (default; packet if the CLI is missing) | api | packet
                   "context": "",          # what the experiment is: cells, perturbation, question
                   "model": None, "web_search": True, "max_turns": 60,
                   "retries": 2},              # extra attempts if the report comes back cut off or incomplete
}

PATH_KEYS = [("data", "h5ad"), ("data", "target_map"), ("programs", "file"), ("programs", "background"),
             ("programs", "discover", "h5ad"),
             ("aucell", "rankings_cache")]


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out:
            raise SystemExit(f"unknown config key '{k}' (known: {', '.join(out)})")
        out[k] = _merge(out[k], v) if isinstance(out[k], dict) and isinstance(v, dict) else v
    return out


def load(path):
    with open(path) as fh:
        user = yaml.safe_load(fh) or {}
    cfg = _merge(DEFAULTS, user)
    root = os.path.dirname(os.path.abspath(path))

    def resolve(p):
        return p if p is None or os.path.isabs(p) else os.path.normpath(os.path.join(root, p))

    for keys in PATH_KEYS:
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = resolve(d[keys[-1]])
    cfg["enrichment"]["gmt"] = [resolve(p) for p in cfg["enrichment"]["gmt"]]
    cfg["outdir"] = resolve(cfg["outdir"])
    for k in ("h5ad", "guide_col"):
        if not cfg["data"][k]:
            raise SystemExit(f"config: data.{k} is required")
    disc = cfg["programs"]["discover"]
    for k, src in (("h5ad", "h5ad"), ("counts_layer", "counts_layer"), ("gene_col", "gene_col")):
        if disc[k] is None:
            disc[k] = cfg["data"][src]
    if isinstance(cfg["data"]["controls"], str):
        cfg["data"]["controls"] = [c for c in cfg["data"]["controls"].split(",") if c.strip()]
    return cfg
