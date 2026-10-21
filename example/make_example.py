#!/usr/bin/env python3
"""
Small simulated high-MOI screen to try the pipeline (a few seconds to make).

Writes to --outdir:
  example.h5ad       6,000 cells x 2,000 genes; .X = log1p(CP10k), layers['counts'] = counts,
                     obs['guides'] = 'g1;g2;...', obs['batch'] = A/B
  programs.gmt       PROG_A, PROG_B, PROG_C (80 genes each) and PROG_SMALL (10 genes, gets dropped)
  guide_targets.tsv  guide -> target gene

Programs are co-expressed: each cell has an activity level per program (log-normal, shared by the program's
80 genes), so program discovery (Hotspot) can find them. PROG_SMALL has no shared activity.
Planted effects (everything else is null):
  G0005 knockdown  raises PROG_A activity 1.2x  (G0005 is itself a PROG_A gene -> target_in_program = True)
  G0900 knockdown  lowers PROG_B activity 0.8x
120 other targets (G1001-G1120, 2 guides each) have no trans effect; every target's own expression drops.
Also included: guides E01_g1/E01_g2 hit two genes (G1500, G1501) -> tested once as 'G1500/G1501';
control guides NTC_1..NTC_40; one guide (X_g1) without a target; 2% of cells without a guide call.
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                     "data", "example"))
    ap.add_argument("--n-cells", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import anndata
    rng = np.random.default_rng(a.seed)
    n, m = a.n_cells, 2000
    genes = np.array([f"G{i:04d}" for i in range(1, m + 1)])
    gi = {g: i for i, g in enumerate(genes)}
    progs = {"PROG_A": genes[0:80], "PROG_B": genes[80:160], "PROG_C": genes[160:240],
             "PROG_SMALL": genes[240:250]}

    targets = ["G0005", "G0900"] + [f"G{i:04d}" for i in range(1001, 1121)]
    tmap = [(f"{t}_g{k}", t) for t in targets for k in (1, 2)]
    tmap += [("E01_g1", "G1500"), ("E01_g2", "G1500"), ("E01_g1", "G1501"), ("E01_g2", "G1501")]
    controls = [f"NTC_{k}" for k in range(1, 41)]
    library = sorted({g for g, _ in tmap}) + controls + ["X_g1"]

    # guides per cell (high MOI)
    moi = 1 + rng.poisson(5, n)
    cell_guides = [list(rng.choice(library, size=k, replace=False)) for k in moi]
    pert = {t: np.array([any(g in (f"{t}_g1", f"{t}_g2") for g in cg) for cg in cell_guides]) for t in targets}
    e01 = np.array([any(g.startswith("E01_") for g in cg) for cg in cell_guides])

    # expression
    batch = rng.choice(["A", "B"], n)
    size = np.exp(rng.normal(0, 0.35, n)) * np.where(batch == "A", 1.0, 1.4)
    base = np.exp(rng.normal(-1.0, 1.3, m))
    mu = np.outer(size, base)
    act = {p: np.exp(0.45 * rng.standard_normal(n)) for p in ("PROG_A", "PROG_B", "PROG_C")}   # co-expression
    act["PROG_A"][pert["G0005"]] *= 1.2                                    # trans: PROG_A up
    act["PROG_B"][pert["G0900"]] *= 0.8                                    # trans: PROG_B down
    for p, a_ in act.items():
        mu[:, [gi[g] for g in progs[p]]] *= a_[:, None]
    for t in targets:                                                      # cis knockdown of each target
        mu[pert[t], gi[t]] *= 0.4
    mu[np.ix_(e01, [gi["G1500"], gi["G1501"]])] *= 0.4
    counts = sp.csr_matrix(rng.poisson(mu).astype(np.float32))
    tot = np.asarray(counts.sum(axis=1)).ravel()
    X = sp.csr_matrix(counts.multiply(1e4 / tot[:, None]))
    X.data = np.log1p(X.data)

    guides_col = [";".join(cg) for cg in cell_guides]
    for i in rng.choice(n, int(0.02 * n), replace=False):
        guides_col[i] = ""
    obs = pd.DataFrame({"guides": guides_col, "batch": batch}, index=[f"cell{i:05d}" for i in range(n)])
    ad = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes), layers={"counts": counts})
    os.makedirs(a.outdir, exist_ok=True)
    ad.write_h5ad(os.path.join(a.outdir, "example.h5ad"))
    with open(os.path.join(a.outdir, "programs.gmt"), "w") as fh:
        for p, gl in progs.items():
            fh.write("\t".join([p, "simulated"] + list(gl)) + "\n")
    pd.DataFrame(tmap, columns=["guide", "target"]).to_csv(os.path.join(a.outdir, "guide_targets.tsv"),
                                                           sep="\t", index=False)
    print(f"wrote example.h5ad ({n} cells x {m} genes), programs.gmt, guide_targets.tsv to {a.outdir}")


if __name__ == "__main__":
    main()
