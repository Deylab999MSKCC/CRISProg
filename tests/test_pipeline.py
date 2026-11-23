"""Tests: the OLS engine against direct fits, the streaming permutation p-values against a
brute-force version, and the whole pipeline (with R AUCell) on the simulated example.

    pytest -q tests/
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from crisprog.model import (OLSEngine, make_strata, n1_bins, permutation_test,  # noqa: E402
                                    stratified_permutation)


def toy(seed=0, n=600, E=8, P=3):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix((rng.random((n, E)) < 0.15).astype(float))
    Z = np.column_stack([np.ones(n), rng.normal(size=n), rng.integers(0, 2, n)])
    Y = rng.normal(size=(n, P)) + 0.3 * Z[:, [1]]
    return X, Y, Z


def test_ols_matches_direct_fit():
    X, Y, Z = toy()
    beta, se, t, _ = OLSEngine(X, Y, Z).stats()
    for e in range(X.shape[1]):
        D = np.column_stack([X[:, e].toarray().ravel(), Z])
        for j in range(Y.shape[1]):
            coef, rss, *_ = np.linalg.lstsq(D, Y[:, j], rcond=None)
            s2 = rss[0] / (len(Y) - D.shape[1])
            se_d = np.sqrt(s2 * np.linalg.inv(D.T @ D)[0, 0])
            assert np.isclose(beta[e, j], coef[0]) and np.isclose(se[e, j], se_d)


def test_streaming_pvalues_match_brute_force():
    X, Y, Z = toy(seed=1, E=30)
    eng = OLSEngine(X, Y, Z)
    _, _, t, _ = eng.stats()
    codes = make_strata(moi=np.asarray(X.sum(axis=1)).ravel(), depth=Z[:, 1], depth_bins=3)
    B = 200
    p_own, p_pool, _, _ = permutation_test(eng, t, codes, n_perm=B, pool_bins=3, seed=7)
    rng = np.random.default_rng(7)
    null = []
    for _ in range(B):
        pi = stratified_permutation(codes, rng)
        assert np.array_equal(codes[pi], codes)
        inv = np.empty_like(pi)
        inv[pi] = np.arange(len(pi))
        null.append(np.abs(eng.stats(row_order=inv)[2]))
    null = np.array(null)
    thr = np.abs(t) * (1 - 1e-9)
    assert np.allclose(p_own, ((null >= thr).sum(axis=0) + 1) / (B + 1))
    bins, nb = n1_bins(eng.n1, 3)
    for g in range(nb):
        idx = np.flatnonzero(bins == g)
        for j in range(t.shape[1]):
            pool = null[:, idx, j].ravel()
            want = [((pool >= thr[i, j]).sum() + 1) / (pool.size + 1) for i in idx]
            assert np.allclose(p_pool[idx, j], want)


needs_r = pytest.mark.skipif(
    shutil.which("Rscript") is None or subprocess.run(
        ["Rscript", "-e", "library(AUCell); library(rhdf5)"], capture_output=True).returncode != 0,
    reason="Rscript with AUCell and rhdf5 not available")
needs_hdwgcna = pytest.mark.skipif(
    shutil.which("Rscript") is None or subprocess.run(
        ["Rscript", "-e", "library(hdWGCNA); library(rhdf5)"], capture_output=True).returncode != 0,
    reason="Rscript with hdWGCNA not available")


def _config(tmp_path, data, **programs):
    import yaml
    cfg = {"name": "out", "results_dir": str(tmp_path),
           "data": {"h5ad": str(data / "example.h5ad"), "counts_layer": "counts", "guide_col": "guides",
                    "control_pattern": "^NTC_", "target_map": str(data / "guide_targets.tsv")},
           "programs": programs, "covariates": {"batch": ["batch"]}, "association": {"n_perm": 500},
           "aucell": {"n_repeats": 2},
           "enrichment": {"backend": "gmt", "gmt": []}}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


@pytest.fixture(scope="module")
def example_data(tmp_path_factory):
    sys.path.insert(0, os.path.join(ROOT, "example"))
    import make_example
    data = tmp_path_factory.mktemp("data")
    make_example.main(["--outdir", str(data)])
    return data


@needs_r
def test_given_programs_end_to_end(tmp_path, example_data):
    from crisprog import cli
    cfgp = _config(tmp_path, example_data, file=str(example_data / "programs.gmt"))
    cli.main(["run", cfgp, "--to", "genes"])
    out = tmp_path / "out"
    res = pd.read_csv(out / "3_association" / "results.tsv", sep="\t")
    hit = lambda u, p: res[(res.unit == u) & (res.program == p)].iloc[0]
    a, b = hit("G0005", "PROG_A"), hit("G0900", "PROG_B")
    assert a.fdr < 0.05 and a.t > 0 and bool(a.target_in_program)
    assert b.fdr < 0.05 and b.t < 0 and not bool(b.target_in_program)
    assert "G1500/G1501" in set(res.unit)                         # identical guide sets merged
    assert "PROG_SMALL" not in set(res.program)                   # too few genes
    planted = {("G0005", "PROG_A"), ("G0900", "PROG_B")}
    others = res[[(u, p) not in planted for u, p in zip(res.unit, res.program)]]
    assert (others.fdr < 0.05).sum() <= 2                         # (almost) no false positives
    cal = pd.read_csv(out / "3_association" / "calibration.tsv", sep="\t")
    assert cal["null_frac_p_ols_lt_0.05"].between(0.02, 0.09).all()
    summ = pd.read_csv(out / "4_gene_test" / "program_summary.tsv", sep="\t")
    b2 = summ[(summ.unit == "G0900") & (summ.program == "PROG_B")].iloc[0]
    assert b2.changed_in_program >= 5 and b2.down == b2.changed_in_program   # broad, concordant trans effect
    gcal = pd.read_csv(out / "4_gene_test" / "calibration.tsv", sep="\t")
    assert gcal["frac_p_lt_0.05"].between(0.02, 0.09).all()
    assert (out / "0_check" / "check_report.md").exists()                # data check ran first
    qc = pd.read_csv(out / "0_check" / "unit_qc.tsv", sep="\t")
    assert (qc.target_t < -2).mean() > 0.9                                 # simulated knockdown is detected
    cs = pd.read_csv(out / "3_association" / "controls" / "control_summary.tsv", sep="\t")
    assert (cs.n_fdr_hits == 0).all() and (cs.frac_p_lt_005 <= 0.12).all()  # control units are null
    assert bool(a.beyond_controls) and bool(b.beyond_controls)


def _recovered(P):
    truth = {"PROG_A": range(1, 81), "PROG_B": range(81, 161), "PROG_C": range(161, 241)}
    for name, idx in truth.items():
        t = {f"G{i:04d}" for i in idx}
        best = max(len(t & set(g.gene)) / len(t | set(g.gene)) for _, g in P.groupby("program"))
        assert best > 0.8, (name, best)


@needs_r
@pytest.mark.slow
def test_discovery_recovers_planted_programs(tmp_path, example_data):
    from crisprog import cli
    cfgp = _config(tmp_path, example_data, discover={"methods": ["hotspot"], "hotspot": {"n_genes": 500}})
    cli.main(["programs", cfgp])
    first = (tmp_path / "out" / "1_programs" / "programs.tsv").read_bytes()
    _recovered(pd.read_csv(tmp_path / "out" / "1_programs" / "programs.tsv", sep="\t"))
    cli.main(["programs", cfgp])                                  # exact neighbours: a rerun gives the same modules
    assert (tmp_path / "out" / "1_programs" / "programs.tsv").read_bytes() == first


@needs_r
def test_annotate_without_target_map(tmp_path, example_data):
    import yaml
    from crisprog import cli
    cfgp = _config(tmp_path, example_data, file=str(example_data / "programs.gmt"))
    cfg = yaml.safe_load(open(cfgp))
    del cfg["data"]["target_map"]                                 # every guide is its own unit, no targets
    cfg["annotation"] = {"backend": "packet"}
    open(cfgp, "w").write(yaml.safe_dump(cfg))
    cli.main(["run", cfgp])
    out = tmp_path / "out"
    assert pd.read_csv(out / "3_association" / "results.tsv", sep="\t").targets.isna().all()
    packet = (out / "6_annotation" / "packet.md").read_text()
    assert "| G0005_g" in packet and "PROG_A" in packet           # planted guides reported without a target


@needs_hdwgcna
@pytest.mark.slow
def test_hdwgcna_recovers_planted_programs(tmp_path, example_data):
    from crisprog import cli
    cfgp = _config(tmp_path, example_data, discover={"methods": ["hdwgcna"], "hdwgcna": {"soft_power": 12}})
    cli.main(["programs", cfgp])
    P = pd.read_csv(tmp_path / "out" / "1_programs" / "programs.tsv", sep="\t")
    assert set(P.method) == {"hdwgcna"} and P.groupby("program").score.apply(lambda s: s.is_monotonic_decreasing).all()
    _recovered(P)


def test_annotation_retries_incomplete_report(tmp_path, example_data, monkeypatch):
    """A cut-off Claude Code reply is not saved as the report; the step retries and keeps the failed output."""
    from crisprog import cli
    fake = tmp_path / "bin"
    fake.mkdir()
    good = "\n\n".join(f"## {h}\n" + "word " * 100 for h in ("Overview", "Programs", "Regulators", "References"))
    (tmp_path / "good.md").write_text(good)
    script = fake / "claude"
    script.write_text(f"""#!/bin/sh
n=$(cat {tmp_path}/count 2>/dev/null || echo 0); n=$((n+1)); echo $n > {tmp_path}/count
if [ $n -eq 1 ]; then echo "I couldn't finish the report: my last reply was stopped partway through."; else cat {tmp_path}/good.md; fi
""")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    cfgp = _config(tmp_path, example_data, file=str(example_data / "programs.gmt"))
    cli.main(["annotate", cfgp])
    ann = tmp_path / "out" / "6_annotation"
    assert (ann / "report.md").read_text().strip() == good.strip()
    assert "stopped partway" in (ann / "failed_attempt_1.md").read_text()


def test_target_genes_resolves_suffixes():
    from crisprog.data import target_genes
    gs = {"POU5F1", "SOX4", "AARS", "ZNF226", "ZNF234", "NKX2-5", "GATA6-AS1"}
    assert target_genes("POU5F1_P2", gs) == ["POU5F1"]
    assert target_genes("SOX4_P1P2", gs) == ["SOX4"]
    assert target_genes("AARS_main", gs) == ["AARS"]
    assert target_genes("ZNF226_P1|ZNF234_P1", gs) == ["ZNF226", "ZNF234"]
    assert target_genes("NKX2-5", gs) == ["NKX2-5"] and target_genes("GATA6-AS1", gs) == ["GATA6-AS1"]
    assert target_genes("NOTAGENE_P1", gs) == []


@needs_r
def test_suffixed_target_names(tmp_path, example_data):
    """A unit named 'G0005_P2' is still recognised as targeting gene G0005 (in-program check, own-target t)."""
    from crisprog import cli
    tm = pd.read_csv(example_data / "guide_targets.tsv", sep="\t")
    tm["target"] = tm.target.replace({"G0005": "G0005_P2"})
    tm.to_csv(tmp_path / "targets.tsv", sep="\t", index=False)
    cfgp = _config(tmp_path, example_data, file=str(example_data / "programs.gmt"))
    import yaml
    c = yaml.safe_load(open(cfgp))
    c["data"]["target_map"] = str(tmp_path / "targets.tsv")
    c["aucell"]["n_repeats"] = 1
    yaml.safe_dump(c, open(cfgp, "w"))
    cli.main(["run", cfgp, "--to", "genes"])
    res = pd.read_csv(tmp_path / "out" / "3_association" / "results.tsv", sep="\t")
    a = res[(res.unit == "G0005_P2") & (res.program == "PROG_A")].iloc[0]
    assert a.target_genes == "G0005" and bool(a.target_in_program)
    summ = pd.read_csv(tmp_path / "out" / "4_gene_test" / "program_summary.tsv", sep="\t")
    s = summ[(summ.unit == "G0005_P2") & (summ.program == "PROG_A")].iloc[0]
    assert str(s.target_t).startswith("G0005:")


@needs_r
def test_control_check_detects_confounding(tmp_path, example_data):
    """If some control guides (NTC_1-20) are detected mostly in cells with high PROG_A activity, control units of
    those guides become 'significant' for PROG_A: the check sees confounding that the permutation null cannot.
    (A bias shared by all control guides is absorbed by the n_control covariate, see the next test.)"""
    import anndata
    from crisprog import cli
    ad = anndata.read_h5ad(example_data / "example.h5ad")
    a_genes = [f"G{i:04d}" for i in range(1, 81)]
    score = np.asarray(ad[:, a_genes].X.mean(axis=1)).ravel()
    rng = np.random.default_rng(0)
    low = score < np.median(score)
    g = ad.obs["guides"].astype(str).to_numpy()
    for i in np.flatnonzero(low & (rng.random(len(g)) < 0.8)):
        g[i] = ";".join(x for x in g[i].split(";") if not (x.startswith("NTC_") and int(x[4:]) <= 20))
    ad.obs["guides"] = g
    ad.write_h5ad(tmp_path / "confounded.h5ad")
    cfgp = _config(tmp_path, example_data, file=str(example_data / "programs.gmt"))
    import yaml
    c = yaml.safe_load(open(cfgp))
    c["data"]["h5ad"] = str(tmp_path / "confounded.h5ad")
    c["aucell"]["n_repeats"] = 1
    c["association"].update(n_perm=200, control_n_perm=200)
    c["data"]["control_types"] = {"biased": "^NTC_([1-9]|1[0-9]|20)$", "clean": "^NTC_(2[1-9]|3[0-9]|40)$"}
    yaml.safe_dump(c, open(cfgp, "w"))
    cli.main(["run", cfgp, "--from", "score", "--to", "test"])
    cs = pd.read_csv(tmp_path / "out" / "3_association" / "controls" / "control_summary.tsv", sep="\t")
    fr = {(r.program, r.type): r.frac_p_lt_005 for r in cs.itertuples()}
    assert fr[("PROG_A", "biased")] > 0.3, fr


def test_config_free_form_control_types(tmp_path):
    """data.control_types takes the user's own type names (no 'unknown key' error)."""
    import yaml
    from crisprog import config
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"data": {"h5ad": "x.h5ad", "guide_col": "guides",
                                          "control_types": {"non-targeting": "^NTC", "safe-targeting": "^OR"}}}))
    c = config.load(str(p))
    assert c["data"]["control_types"] == {"non-targeting": "^NTC", "safe-targeting": "^OR"}


@needs_r
def test_control_population_categorical_moi(tmp_path, example_data):
    """ES/DE-like design: control guides only in their own single-guide cells, which differ a little as a
    population. With MOI as categories, control units are compared within that population and stay null."""
    import anndata
    from crisprog import cli
    ad = anndata.read_h5ad(example_data / "example.h5ad")
    rng = np.random.default_rng(1)
    g = ad.obs["guides"].astype(str).to_numpy()
    ctrl = rng.random(len(g)) < 0.25                                   # a quarter of cells: one control guide only
    g[ctrl] = [f"NTC_{k}" for k in rng.integers(1, 41, ctrl.sum())]
    g[~ctrl] = [";".join(x for x in s.split(";") if not x.startswith("NTC_")) or "G1010_g1" for s in g[~ctrl]]
    ad.obs["guides"] = g
    C = sp.csr_matrix(ad.layers["counts"]).astype(np.float64)
    a_cols = [ad.var_names.get_loc(f"G{i:04d}") for i in range(1, 81)]
    D = sp.diags(np.where(ctrl, 1.3, 1.0))                               # the control population: PROG_A a bit higher
    C = C.tolil()
    C[:, a_cols] = np.round((D @ C[:, a_cols]).toarray())
    C = sp.csr_matrix(C)
    X = sp.csr_matrix(C.multiply(1e4 / np.asarray(C.sum(1))))
    X.data = np.log1p(X.data)
    ad.X, ad.layers["counts"] = X.astype(np.float32), C.astype(np.float32)
    ad.write_h5ad(tmp_path / "pop.h5ad")
    cfgp = _config(tmp_path, example_data, file=str(example_data / "programs.gmt"))
    import yaml
    c = yaml.safe_load(open(cfgp))
    c["data"]["h5ad"] = str(tmp_path / "pop.h5ad")
    c["aucell"]["n_repeats"] = 1
    c["association"].update(n_perm=200, control_n_perm=200)
    yaml.safe_dump(c, open(cfgp, "w"))
    cli.main(["run", cfgp, "--from", "score", "--to", "test"])
    cs = pd.read_csv(tmp_path / "out" / "3_association" / "controls" / "control_summary.tsv", sep="\t")
    assert (cs.frac_p_lt_005 <= 0.12).all() and (cs.n_fdr_hits == 0).all(), cs


@needs_r
def test_control_count_covariate_absorbs_detection_bias(tmp_path, example_data):
    """CM-like: all control guides detected more often in cells with high PROG_A (guide detection depends on cell
    state). Comparing cells at the same number of control guides (n_control covariate) keeps control units null."""
    import anndata
    from crisprog import cli
    ad = anndata.read_h5ad(example_data / "example.h5ad")
    score = np.asarray(ad[:, [f"G{i:04d}" for i in range(1, 81)]].X.mean(axis=1)).ravel()
    rng = np.random.default_rng(0)
    g = ad.obs["guides"].astype(str).to_numpy()
    for i in np.flatnonzero((score < np.median(score)) & (rng.random(len(g)) < 0.8)):
        g[i] = ";".join(x for x in g[i].split(";") if not x.startswith("NTC_"))
    ad.obs["guides"] = g
    ad.write_h5ad(tmp_path / "det.h5ad")
    cfgp = _config(tmp_path, example_data, file=str(example_data / "programs.gmt"))
    import yaml
    c = yaml.safe_load(open(cfgp))
    c["data"]["h5ad"] = str(tmp_path / "det.h5ad")
    c["aucell"]["n_repeats"] = 1
    c["association"].update(n_perm=200, control_n_perm=200)
    yaml.safe_dump(c, open(cfgp, "w"))
    cli.main(["run", cfgp, "--from", "score", "--to", "test"])
    cs = pd.read_csv(tmp_path / "out" / "3_association" / "controls" / "control_summary.tsv", sep="\t")
    assert (cs.frac_p_lt_005 <= 0.12).all(), cs
    info = __import__("json").load(open(tmp_path / "out" / "3_association" / "run_info.json"))
    assert any(d.startswith("n_control") for d in info["design"])


@needs_r
def test_packet_target_ids_and_pair_summary(tmp_path, example_data):
    """Target IDs from the target map reach results.tsv and the packet; the packet summarizes ALL regulator pairs."""
    from crisprog import cli
    import yaml
    tm = pd.read_csv(example_data / "guide_targets.tsv", sep="\t")
    tm["target_id"] = "ENSG_" + tm.target
    tm.to_csv(tmp_path / "targets.tsv", sep="\t", index=False)
    cfgp = _config(tmp_path, example_data, file=str(example_data / "programs.gmt"))
    c = yaml.safe_load(open(cfgp))
    c["data"]["target_map"] = str(tmp_path / "targets.tsv")
    c["aucell"]["n_repeats"] = 1
    c["association"].update(n_perm=200, control_n_perm=200)
    c["annotation"] = {"backend": "packet"}
    yaml.safe_dump(c, open(cfgp, "w"))
    cli.main(["run", cfgp, "--steps", "score,test,genes,enrich,annotate"])
    res = pd.read_csv(tmp_path / "out" / "3_association" / "results.tsv", sep="\t")
    assert (res.loc[res.unit == "G0900", "target_ids"] == "ENSG_G0900").all()
    pk = open(tmp_path / "out" / "6_annotation" / "packet.md").read()
    assert "G0900 (ENSG_G0900)" in pk
    ov = pd.read_csv(tmp_path / "out" / "4_gene_test" / "overlap.tsv", sep="\t")
    assert ("summary over ALL pairs" in pk) == (len(ov) > 0)


@needs_r
def test_check_suggests_and_explains_settings(tmp_path, example_data):
    """A bare config gets rule-based suggestions (counts layer, controls, batch) that work when applied."""
    from crisprog import cli, config
    import yaml
    cfg = {"name": "bare", "results_dir": str(tmp_path),
           "data": {"h5ad": str(example_data / "example.h5ad"), "guide_col": "guides",
                    "target_map": str(example_data / "guide_targets.tsv")},
           "programs": {"file": str(example_data / "programs.gmt")}}
    p = tmp_path / "bare.yaml"
    p.write_text(yaml.safe_dump(cfg))
    cli.main(["run", str(p), "--steps", "check"])
    out = tmp_path / "bare" / "0_check"
    sug = yaml.safe_load(open(out / "suggested_config.yaml"))
    assert sug["data"]["counts_layer"] == "counts"
    assert sug["covariates"]["batch"] == ["batch"]
    assert "Settings explained" in open(out / "check_report.md").read()
    for k1, v in sug.items():                                  # apply the suggestions: controls are now found
        cfg[k1] = {**cfg.get(k1, {}), **v}
    p.write_text(yaml.safe_dump(cfg))
    from crisprog.analysis import Analysis
    A = Analysis(config.load(str(p)))
    assert len(A.control_guides) == 40 and A.batch_cols == ["batch"]


def test_report_factcheck_and_render(tmp_path, example_data, monkeypatch):
    """The draft is checked (rule-based + a fake AI fact-check pass), corrected, logged and rendered to HTML."""
    from crisprog import cli
    fake = tmp_path / "bin"
    fake.mkdir()
    body = "\n\n".join(f"## {h}\n" + "word " * 100 for h in ("Overview", "Programs", "Regulators"))
    draft = "# Report\n\n" + body + "\n\nG0900 changed 20/20 genes.\n\n## References\n\n1. none\n"
    fixed = draft.replace("20/20 genes", "many genes") + "\n## Fact-check log\n\n- was: 20/20 -> now: many (not in packet)\n"
    (tmp_path / "draft.md").write_text(draft)
    (tmp_path / "fixed.md").write_text(fixed)
    script = fake / "claude"
    script.write_text(f"""#!/bin/sh
case "$*" in *FACTCHECK*) cat {tmp_path}/fixed.md ;; *) cat {tmp_path}/draft.md ;; esac
""")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    cfgp = _config(tmp_path, example_data, file=str(example_data / "programs.gmt"))
    import yaml
    c = yaml.safe_load(open(cfgp))
    c["aucell"]["n_repeats"] = 1
    c["association"].update(n_perm=200, control_n_perm=200)
    c["annotation"] = {"backend": "claude-code", "retries": 0}
    yaml.safe_dump(c, open(cfgp, "w"))
    cli.main(["run", cfgp, "--steps", "score,test,genes,enrich,annotate"])
    out = tmp_path / "out" / "6_annotation"
    assert "20/20" in open(out / "report_draft.md").read()
    assert "20/20" not in open(out / "report.md").read()
    fc = open(out / "factcheck.md").read()
    assert "`20/20`" in fc and "was: 20/20" in fc
    assert (out / "report.html").exists()


# ---- association.reference: controls -------------------------------------------------------------------------
def toy_low_moi(seed=0, n=900, E=6, P=3):
    """One perturbation per cell: 30% control cells, the rest split over E units; 3 batches, one of which
    has no control cells and holds all cells of the last unit (which then cannot be estimated)."""
    rng = np.random.default_rng(seed)
    lab = np.where(rng.random(n) < 0.3, -1, rng.integers(0, E - 1, n))     # -1 = control
    batch = rng.integers(0, 2, n)
    extra = rng.choice(np.flatnonzero(lab >= 0), 40, replace=False)        # unit E-1: alone in batch 2
    lab[extra], batch[extra] = E - 1, 2
    X = sp.csr_matrix((np.ones((lab >= 0).sum()), (np.flatnonzero(lab >= 0), lab[lab >= 0])), shape=(n, E))
    Z = np.column_stack([np.ones(n), rng.normal(size=n), batch == 1, batch == 2]).astype(float)
    Y = rng.normal(size=(n, P)) + 0.3 * Z[:, [1]] + 0.5 * (lab == 0)[:, None]   # unit 0 has an effect
    return X, Y, Z, lab == -1


def _subset_fit(x, Y, Z, rows):
    D = np.column_stack([x[rows], Z[rows]])
    D = D[:, np.abs(D).sum(axis=0) > 0]
    out = []
    for j in range(Y.shape[1]):
        coef, *_ = np.linalg.lstsq(D, Y[rows, j], rcond=None)
        r = Y[rows, j] - D @ coef
        rank = np.linalg.matrix_rank(D)
        cov = np.linalg.pinv(D.T @ D) * (r @ r) / (len(rows) - rank)
        out.append((coef[0], np.sqrt(cov[0, 0])))
    return np.array(out)


def test_ref_engine_matches_subset_fit():
    """reference: controls equals a separate OLS on each unit's cells + the control cells (absent batch levels
    dropped); a unit confounded with a batch that has no control cells is NaN; a control unit is fitted against
    the other control cells."""
    from crisprog.model import RefEngine
    X, Y, Z, ref = toy_low_moi()
    beta, se, t, _ = RefEngine(X, Y, Z, ref).stats()
    E = X.shape[1]
    for u in range(E - 1):
        x = X[:, u].toarray().ravel()
        rows = np.flatnonzero(ref | (x > 0))
        want = _subset_fit(x, Y, Z, rows)
        assert np.allclose(beta[u], want[:, 0]) and np.allclose(se[u], want[:, 1])
    assert np.isnan(t[E - 1]).all()
    assert (np.abs(t[0]) > 3).all() and (np.abs(t[1:E - 1]) < 4).all()
    rng = np.random.default_rng(3)
    xc = np.zeros(len(ref))
    xc[rng.choice(np.flatnonzero(ref), 60, replace=False)] = 1
    bc, sc, *_ = RefEngine(sp.csr_matrix(xc[:, None]), Y, Z, ref).stats()
    want = _subset_fit(xc, Y, Z, np.flatnonzero(ref))
    assert np.allclose(bc[0], want[:, 0]) and np.allclose(sc[0], want[:, 1])


def test_ref_permutations_stay_within_unit_and_controls():
    """Shuffled labels of a unit only land on its own cells or control cells of the same stratum; the
    block-wise null statistics equal refits on the shuffled labels; null p-values are calibrated."""
    from crisprog.model import RefEngine
    X, Y, Z, ref = toy_low_moi(seed=2, n=2000, E=12)
    Y = np.random.default_rng(5).normal(size=Y.shape) + 0.3 * Z[:, [1]]      # all null
    eng = RefEngine(X, Y, Z, ref)
    codes = make_strata(depth=Z[:, 1], batch=Z[:, 2] + 2 * Z[:, 3], depth_bins=3)
    Xp = eng.permuted_X(codes, np.random.default_rng(1))
    for u in range(X.shape[1]):
        new = Xp[:, u].toarray().ravel() > 0
        old = X[:, u].toarray().ravel() > 0
        assert new.sum() == old.sum() and (ref[new] | old[new]).all()
        assert np.array_equal(np.bincount(codes[new], minlength=codes.max() + 1),
                              np.bincount(codes[old], minlength=codes.max() + 1))
    t_alt = eng.stats(Xalt=Xp)[2]
    for u in range(X.shape[1] - 1):
        x = Xp[:, u].toarray().ravel()
        rows = np.flatnonzero(ref | (X[:, u].toarray().ravel() > 0))
        want = _subset_fit(x, Y, Z, rows)
        assert np.allclose(t_alt[u], want[:, 0] / want[:, 1])
    _, _, t, _ = eng.stats()
    p_own, p_pool, null_frac, _ = permutation_test(eng, t, codes, n_perm=300, pool_bins=2, seed=4)
    ok = np.isfinite(t)
    assert np.isnan(p_own[~ok]).all() and 0.0 < np.mean(p_own[ok] < 0.05) < 0.2
    assert 0.02 < null_frac.mean() < 0.09


@needs_r
def test_reference_controls_vs_all(tmp_path, example_data):
    """KO-village-like data: one perturbation per cell, a large unit (30% of cells) that raises PROG_A, null
    units and control cells. reference: all compares each null unit with mostly BIG cells, so they look like
    they lower PROG_A; reference: controls compares them with control cells only and keeps them null."""
    import anndata
    import yaml
    from crisprog import cli
    ad = anndata.read_h5ad(example_data / "example.h5ad")
    rng = np.random.default_rng(0)
    r = rng.random(ad.n_obs)
    g = np.where(r < 0.25, [f"NTC_{k}" for k in rng.integers(1, 41, ad.n_obs)],
                 np.where(r < 0.55, "BIG_g1", [f"NULL{k}_g1" for k in rng.integers(1, 9, ad.n_obs)]))
    ad.obs["guides"] = g
    big = g == "BIG_g1"
    C = sp.csr_matrix(ad.layers["counts"]).astype(np.float64).tolil()
    a_cols = [ad.var_names.get_loc(f"G{i:04d}") for i in range(1, 81)]
    C[:, a_cols] = np.round((sp.diags(np.where(big, 1.6, 1.0)) @ C[:, a_cols]).toarray())
    C = sp.csr_matrix(C)
    X = sp.csr_matrix(C.multiply(1e4 / np.asarray(C.sum(1))))
    X.data = np.log1p(X.data)
    ad.X, ad.layers["counts"] = X.astype(np.float32), C.astype(np.float32)
    ad.write_h5ad(tmp_path / "village.h5ad")
    out = {}
    for ref in ("all", "controls"):
        cfgp = _config(tmp_path, example_data, file=str(example_data / "programs.gmt"))
        c = yaml.safe_load(open(cfgp))
        c["name"] = f"ref_{ref}"
        c["data"].update(h5ad=str(tmp_path / "village.h5ad"), target_map=None)
        c["aucell"]["n_repeats"] = 1
        c["association"].update(n_perm=300, control_n_perm=200, reference=ref)
        yaml.safe_dump(c, open(cfgp, "w"))
        cli.main(["run", cfgp, "--steps", "check,score,test,genes"])
        res = pd.read_csv(tmp_path / f"ref_{ref}" / "3_association" / "results.tsv", sep="\t")
        info = __import__("json").load(open(tmp_path / f"ref_{ref}" / "3_association" / "run_info.json"))
        assert info["reference"] == ref
        out[ref] = res[res.program == "PROG_A"].set_index("unit")
    nulls = [u for u in out["all"].index if u.startswith("NULL")]
    for ref in ("all", "controls"):
        assert out[ref].loc["BIG_g1", "fdr"] < 0.05 and out[ref].loc["BIG_g1", "t"] > 0
    assert (out["all"].loc[nulls, "t"] < -2).mean() > 0.7                  # shifted baseline
    assert (out["controls"].loc[nulls, "fdr"] < 0.05).sum() == 0            # vs controls: null
    assert abs(out["controls"].loc[nulls, "t"].median()) < 1.5
    rep = open(tmp_path / "ref_all" / "0_check" / "check_report.md").read()
    assert "association.reference: controls" in rep                       # check suggests it for this design
    gcal = pd.read_csv(tmp_path / "ref_controls" / "4_gene_test" / "calibration.tsv", sep="\t")
    assert gcal["frac_p_lt_0.05"].between(0.01, 0.1).all()
