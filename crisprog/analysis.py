"""The analysis data, loaded once and shared by the scoring, association and per-gene steps."""
import re

import numpy as np

from .data import add_target_genes, build_units, load_h5ad, parse_guides, read_target_map
from .model import DEPTH, build_design, make_strata
from .utils import log


class Analysis:
    """Cells with a guide call (unless keep_cells_without_guides), their perturbation units and covariates.

    Attributes (all restricted to kept cells, in h5ad order):
      cells, genes, X (cells x genes, normalized), counts (or None), obs,
      U (cells x units 0/1 CSR), units (DataFrame), builtin (dict of built-in covariates)
    """

    def __init__(self, cfg):
        d, cv = cfg["data"], cfg["covariates"]
        obs, cells, genes, X, counts = load_h5ad(d["h5ad"], d["counts_layer"], d["gene_col"])
        if len(set(cells)) < len(cells):
            raise SystemExit("cell names (obs_names) are not unique; run adata.obs_names_make_unique() first")
        if d["guide_col"] not in obs.columns:
            raise SystemExit(f"data.guide_col '{d['guide_col']}' is not an obs column")
        G, guides, has_call, n_control, Gc, cguides = parse_guides(obs[d["guide_col"]].astype(object).to_numpy(), d["guide_sep"],
                                              d["controls"], d["control_pattern"])
        keep = np.ones(len(cells), dtype=bool) if d["keep_cells_without_guides"] else has_call
        if not d["keep_cells_without_guides"] and (~has_call).any():
            log(f"dropping {int((~has_call).sum()):,} cells without a guide call")
        tmap = read_target_map(d["target_map"]) if d["target_map"] else None
        self.target_map = tmap
        self.U, self.units = build_units(G[keep], guides, tmap, merge_identical=d["merge_identical_targets"])
        self.cells, self.genes = cells[keep], genes
        self.units = add_target_genes(self.units, genes)
        tid = {}
        if tmap is not None and "target_id" in tmap.columns:
            tid = tmap.dropna(subset=["target_id"]).drop_duplicates("target").set_index("target")["target_id"].to_dict()
        self.units["target_ids"] = [";".join(str(tid.get(t, "")) for t in str(ts).split(";") if t) if tid else ""
                                    for ts in self.units["targets"].fillna("")]
        pos = {x.strip() for x in d["positive_controls"]}
        self.units["unit_type"] = ["positive control" if pos & set(f"{t};{g}".split(";")) else "test"
                                   for t, g in zip(self.units["targets"].fillna(""), self.units["target_genes"])]
        if pos:
            log(f"positive-control units: {', '.join(self.units.unit[self.units.unit_type != 'test']) or 'none found'}")
        self.Gc, self.control_guides = Gc[keep], cguides           # negative-control guides (not tested as units)
        self.guide_names = guides                                   # non-control guide labels
        types = d["control_types"] or {}
        self.control_type = np.array([next((k for k, rx in types.items() if re.search(rx, g)), "control")
                                      for g in cguides], dtype=object)
        self.X = X[keep]
        self.counts = counts[keep] if counts is not None else None
        self.obs = obs.iloc[np.flatnonzero(keep)].reset_index(drop=True)
        self.cov_names = [] if [c.lower() for c in cv["use"]] == ["none"] else list(cv["use"])
        nctl = n_control[keep].astype(float)
        moi_k = np.diff(G[keep].indptr)
        shared = bool(((nctl > 0) & (moi_k > 0)).any())
        cc = cv["control_count"]
        if (cc in (True, "true") or (cc == "auto" and shared)) and "n_control" not in self.cov_names \
                and len(np.unique(nctl)) > 1:
            self.cov_names.append("n_control")
            log("covariate n_control (control guides per cell, as categories) added: control guides share cells with "
                "targeting guides, so cells are compared at the same total number of detected guides")
        self.batch_cols = list(cv["batch"])
        self.moi_categorical = cv["moi_categorical"]

        src = self.counts if self.counts is not None else self.X
        if self.counts is None and "log_n_genes" in self.cov_names:
            log("no counts layer: genes detected are counted from .X")
        n_genes = np.asarray((src > 0).sum(axis=1)).ravel()
        self.builtin = {"log_n_genes": np.log(np.maximum(n_genes, 1)),
                        "moi": (self.obs[cv["moi_col"]].to_numpy(dtype=float) if cv["moi_col"]
                                else np.diff(G[keep].indptr).astype(float)),
                        "n_control": nctl}
        if self.counts is not None:
            self.builtin["log_total_counts"] = np.log(np.maximum(np.asarray(self.counts.sum(axis=1)).ravel(), 1))
        elif "log_total_counts" in self.cov_names:
            raise SystemExit("covariate log_total_counts needs data.counts_layer")
        self.depth_bins = cfg["association"]["depth_bins"]

    def rows_for(self, cell_names):
        """Positions (sorted) of the kept cells that appear in cell_names."""
        wanted = set(map(str, cell_names))
        rows = np.array([i for i, c in enumerate(self.cells) if c in wanted], dtype=int)
        if len(rows) < 100:
            raise SystemExit("fewer than 100 cells overlap; do the cell names match?")
        if len(rows) < len(self.cells):
            log(f"{len(self.cells) - len(rows):,} cells have no score and are left out")
        return rows

    def design(self, rows):
        """Covariate matrix, its column names and permutation strata for the cells in rows."""
        obs = self.obs.iloc[rows].reset_index(drop=True)
        builtin = {k: v[rows] for k, v in self.builtin.items()}
        Z, znames = build_design(self.cov_names, builtin, obs, self.batch_cols, self.moi_categorical)
        depth = next((c for c in self.cov_names if c in DEPTH), None)
        batch = obs[self.batch_cols].astype(str).agg("|".join, axis=1).to_numpy() if self.batch_cols else None
        mkey = builtin["moi"] if "moi" in self.cov_names else None
        if "n_control" in self.cov_names:                  # strata keep both guide counts fixed
            from .model import moi_levels
            nc = moi_levels(builtin["n_control"])
            mkey = (np.round(mkey).astype(int) * 1000 if mkey is not None else 0) + (nc if nc is not None else 0)
        codes = make_strata(mkey,
                            builtin[depth] if depth else None, batch, self.depth_bins, len(rows))
        strata = " x ".join((["MOI"] if "moi" in self.cov_names else [])
                            + (["control guides"] if "n_control" in self.cov_names else [])
                            + ([f"{self.depth_bins} {depth} bins"] if depth else [])
                            + (["batch"] if self.batch_cols else [])) or "none (plain shuffles)"
        return Z, znames, codes, strata, builtin
