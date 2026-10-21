"""Pipeline steps. Each takes the config and a shared Context; outputs go to results/<name>/<step folder>/."""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

from .aucell import program_recovery, read_scores, run_aucell
from .data import read_programs
from .model import OLSEngine, permutation_test
from .plots import volcano
from .utils import benjamini_hochberg, log

STEPS = ["check", "programs", "score", "test", "genes", "enrich", "annotate"]
DIRS = {"check": "0_check", "programs": "1_programs", "score": "2_scores", "test": "3_association", "genes": "4_gene_test",
        "enrich": "5_enrichment", "annotate": "6_annotation"}


def outdir(cfg, step):
    d = os.path.join(cfg["outdir"], DIRS[step])
    os.makedirs(d, exist_ok=True)
    return d


def programs_path(cfg):
    if cfg["programs"]["file"]:
        return cfg["programs"]["file"]
    p = os.path.join(cfg["outdir"], DIRS["programs"], "programs.tsv")
    if not os.path.exists(p):
        raise SystemExit("no programs: set programs.file, or run the 'programs' step first")
    return p


class Context:
    """Loads the analysis data once, on first use."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._analysis = None

    @property
    def analysis(self):
        if self._analysis is None:
            from .analysis import Analysis
            self._analysis = Analysis(self.cfg)
        return self._analysis


# ---------------------------------------------------------------------------------------------------------------
def step_check(cfg, ctx):
    from .check import run_check
    run_check(cfg, ctx, outdir(cfg, "check"))


def step_programs(cfg, ctx):
    if cfg["programs"]["file"]:
        log(f"programs given in {cfg['programs']['file']}; discovery skipped")
        progs = read_programs(cfg["programs"]["file"])
        out = outdir(cfg, "programs")
        pd.DataFrame([(p, g, i + 1, "given") for p, gl in progs.items() for i, g in enumerate(gl)],
                     columns=["program", "gene", "rank", "method"]).to_csv(
            os.path.join(out, "programs.tsv"), sep="\t", index=False)
        return
    from .discover import discover_programs
    discover_programs(cfg, outdir(cfg, "programs"))


def step_score(cfg, ctx):
    A = ctx.analysis
    out = outdir(cfg, "score")
    programs = read_programs(programs_path(cfg))
    scored, recovery = program_recovery(programs, A.genes, cfg["programs"]["min_genes"])
    recovery.to_csv(os.path.join(out, "program_genes.tsv"), sep="\t", index=False)
    au = cfg["aucell"]
    seed = au["seed"] if au["seed"] is not None else cfg["association"]["seed"]
    S = run_aucell(A.X, A.genes, A.cells, scored, out, au["rscript"], au["cores"], au["max_rank"],
                   seed, au["rankings_cache"], repeats=au["n_repeats"])
    rank_program_genes(A, S, scored, out, cfg["association"]["seed"])


def rank_program_genes(A, S, programs, out, seed, max_cells=20000):
    """Hub genes: each program gene's correlation with the program's AUCell score (random cell subset)."""
    rows = A.rows_for(S.index)
    rng = np.random.default_rng(seed)
    if len(rows) > max_cells:
        rows = np.sort(rng.choice(rows, max_cells, replace=False))
    gi = {g: i for i, g in enumerate(A.genes)}
    X = A.X[rows].tocsc()
    recs = []
    for p, gl in programs.items():
        if p not in S.columns:
            continue
        s = S.loc[A.cells[rows], p].to_numpy(dtype=float)
        s = s - s.mean()
        idx = [gi[g] for g in gl if g in gi]
        Xg = X[:, idx].toarray().astype(float)
        Xg -= Xg.mean(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (Xg * s[:, None]).sum(axis=0) / (np.sqrt((Xg ** 2).sum(axis=0)) * np.sqrt((s ** 2).sum()))
        recs += [(p, A.genes[j], float(v)) for j, v in zip(idx, r)]
    R = pd.DataFrame(recs, columns=["program", "gene", "r_with_score"]).sort_values(
        ["program", "r_with_score"], ascending=[True, False])
    R.to_csv(os.path.join(out, "program_gene_ranking.tsv"), sep="\t", index=False, float_format="%.4g")
    log("program genes ranked by correlation with the program score (program_gene_ranking.tsv)")


def step_test(cfg, ctx):
    A, a = ctx.analysis, cfg["association"]
    t0 = time.time()
    out = outdir(cfg, "test")
    S = read_scores(os.path.join(cfg["outdir"], DIRS["score"], "aucell_scores.tsv"))
    programs = read_programs(programs_path(cfg))
    prog_names = list(S.columns)
    rows = A.rows_for(S.index)
    Y = S.loc[A.cells[rows], prog_names].to_numpy(dtype=float)
    U = A.U[rows]
    Z, znames, codes, strata, builtin = A.design(rows)
    n = len(rows)
    log(f"cells analyzed: {n:,}; design: {', '.join(znames)}; permutation strata: {strata}")

    n1 = np.asarray(U.sum(axis=0)).ravel().astype(int)
    tested = (n1 >= a["min_perturbed"]) & (n1 <= n - a["min_perturbed"])
    t_idx = np.flatnonzero(tested)
    log(f"units tested: {len(t_idx):,} of {len(A.units):,} (>= {a['min_perturbed']} perturbed cells); "
        f"programs: {len(prog_names)}")
    if not len(t_idx):
        raise SystemExit("no unit has enough perturbed cells")
    eng = OLSEngine(U[:, t_idx], Y, Z)
    beta, se, t, d = eng.stats()
    with np.errstate(divide="ignore", invalid="ignore"):
        effect_sd = beta / (se * np.sqrt(d)[:, None])
    p_ols = 2.0 * stats.t.sf(np.abs(t), eng.df)
    pooled = a["perm_p"] == "pooled" or (a["perm_p"] == "auto" and len(t_idx) > 50)
    log(f"{a['n_perm']:,} stratified shuffles; p_perm = "
        + ("pooled over tests with similar perturbed-cell counts" if pooled else "each test's own shuffles"))
    p_own, p_pool, null_frac, min_pool = permutation_test(eng, t, codes, a["n_perm"], a["pool_bins"], a["seed"])
    p_perm = p_pool if pooled else p_own
    if a["fdr_scope"] == "program":
        fdr = np.column_stack([benjamini_hochberg(p_perm[:, j]) for j in range(len(prog_names))])
    else:
        fdr = benjamini_hochberg(p_perm.ravel()).reshape(p_perm.shape)

    E, P = len(A.units), len(prog_names)

    def full(x):
        o = np.full((E, P), np.nan)
        o[t_idx] = x
        return o

    prog_sets = {p: set(programs.get(p, [])) for p in prog_names}
    tl = [s.split(";") if s else [] for s in A.units["target_genes"]]   # measured genes behind the targets
    members = [[";".join(x for x in ts if x in prog_sets[p]) for p in prog_names] for ts in tl]
    res = pd.DataFrame({
        "unit": np.repeat(A.units["unit"].to_numpy(), P), "unit_type": np.repeat(A.units["unit_type"].to_numpy(), P),
        "program": np.tile(prog_names, E),
        "n_perturbed": np.repeat(n1, P), "beta": full(beta).ravel(), "se": full(se).ravel(), "t": full(t).ravel(),
        "effect_sd": full(effect_sd).ravel(), "p_ols": full(p_ols).ravel(), "p_perm": full(p_perm).ravel(),
        "fdr": full(fdr).ravel(),
        "target_in_program": pd.array([bool(m) if tl[i] else None for i in range(E) for m in members[i]],
                                       dtype="boolean"),
        "targets_in_program": [m for row in members for m in row],
        "targets": np.repeat(A.units["targets"].to_numpy(), P),
        "target_genes": np.repeat(A.units["target_genes"].to_numpy(), P), "guides": np.repeat(A.units["guides"].to_numpy(), P),
    })
    res["program"] = pd.Categorical(res["program"], prog_names)
    res = res.sort_values(["program", "p_perm", "p_ols"]).reset_index(drop=True)
    res["program"] = res["program"].astype(str)
    from .controls import run_control_check
    res = run_control_check(A, rows, Y, Z, codes, prog_names, a, res, A.units["n_guides"].to_numpy()[t_idx], out)
    res.to_csv(os.path.join(out, "results.tsv"), sep="\t", index=False, na_rep="NA", float_format="%.6g")
    A.units.assign(n_perturbed=n1, tested=tested).to_csv(os.path.join(out, "units.tsv"), sep="\t", index=False)
    pd.DataFrame({"cell": A.cells[rows], **builtin, "stratum": codes}).to_csv(
        os.path.join(out, "cell_covariates.tsv"), sep="\t", index=False, float_format="%.6g")
    cal = pd.DataFrame({
        "program": prog_names, "n_tested": len(t_idx), "null_frac_p_ols_lt_0.05": null_frac,
        "obs_frac_p_perm_lt_0.05": np.nanmean(p_perm < 0.05, axis=0),
        f"hits_fdr_lt_{a['fdr']:g}": np.nansum(fdr < a["fdr"], axis=0).astype(int),
        **{f"coef_{nm}": eng.cov_coef[k] for k, nm in enumerate(znames) if nm != "intercept"}})
    cal.to_csv(os.path.join(out, "calibration.tsv"), sep="\t", index=False, float_format="%.4g")
    volcano(res, os.path.join(out, "volcano"), a["fdr"], program_sizes={p: len(prog_sets[p]) for p in prog_names})
    with open(os.path.join(out, "run_info.json"), "w") as fh:
        json.dump({"cells_analyzed": n, "units_tested": int(len(t_idx)), "design": znames, "strata": strata,
                   "p_perm": "pooled" if pooled else "own", "minutes": round((time.time() - t0) / 60, 2)}, fh, indent=2)
    log("calibration (share of shuffled tests with OLS p < 0.05; ~0.05 = well calibrated):")
    print(cal[["program", "n_tested", "null_frac_p_ols_lt_0.05", f"hits_fdr_lt_{a['fdr']:g}"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3g}"), flush=True)
    hits = res[res["fdr"] < a["fdr"]]
    log(f"{len(hits)} unit-program associations at FDR < {a['fdr']:g}")
    if len(hits):
        cols = ["unit", "unit_type", "program", "n_perturbed", "effect_sd", "p_perm", "fdr", "target_in_program"]
        cols += [c for c in ("beyond_controls",) if c in hits.columns]
        print(hits[cols]
              .to_string(index=False, float_format=lambda x: f"{x:.3g}"), flush=True)
    return res


def step_genes(cfg, ctx):
    from .genetest import run_gene_test
    run_gene_test(cfg, ctx.analysis, outdir(cfg, "genes"))


def step_enrich(cfg, ctx):
    from .enrich import run_enrichment
    run_enrichment(cfg, outdir(cfg, "enrich"))


def step_annotate(cfg, ctx):
    from .annotate import run_annotation
    run_annotation(cfg, outdir(cfg, "annotate"))


RUNNERS = {"check": step_check, "programs": step_programs, "score": step_score, "test": step_test, "genes": step_genes,
           "enrich": step_enrich, "annotate": step_annotate}
