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


def _config(tmp_path, data, **programs):
    import yaml
    cfg = {"name": "out", "results_dir": str(tmp_path),
           "data": {"h5ad": str(data / "example.h5ad"), "counts_layer": "counts", "guide_col": "guides",
                    "controls": [f"NTC_{k}" for k in range(1, 7)], "target_map": str(data / "guide_targets.tsv")},
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


@needs_r
@pytest.mark.slow
def test_discovery_recovers_planted_programs(tmp_path, example_data):
    from crisprog import cli
    cfgp = _config(tmp_path, example_data, discover={"methods": ["pca"], "n_genes": 500})
    cli.main(["programs", cfgp])
    P = pd.read_csv(tmp_path / "out" / "1_programs" / "programs.tsv", sep="\t")
    truth = {"PROG_A": range(1, 81), "PROG_B": range(81, 161), "PROG_C": range(161, 241)}
    for name, idx in truth.items():
        t = {f"G{i:04d}" for i in idx}
        best = max(len(t & set(g.gene)) / len(t | set(g.gene)) for _, g in P.groupby("program"))
        assert best > 0.8, (name, best)


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
