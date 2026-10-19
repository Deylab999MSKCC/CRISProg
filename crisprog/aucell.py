"""AUCell program scores through the Bioconductor AUCell package (R/aucell.R)."""
import os
import shutil
import subprocess

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .utils import log

R_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "R", "aucell.R")


def program_recovery(programs, genes, min_genes):
    """How many genes of each program are measured; programs below min_genes are dropped."""
    measured = set(genes)
    rows, keep = [], {}
    for p, gl in programs.items():
        found = [g for g in gl if g in measured]
        rows.append({"program": p, "n_genes": len(gl), "n_measured": len(found),
                     "frac_measured": round(len(found) / len(gl), 4), "scored": len(found) >= min_genes})
        if len(found) >= min_genes:
            keep[p] = found
    stats = pd.DataFrame(rows)
    for r in stats.itertuples():
        if r.frac_measured < 0.8:
            log(f"WARNING: program {r.program}: only {r.n_measured}/{r.n_genes} genes measured")
        if not r.scored:
            log(f"program {r.program} dropped: {r.n_measured} measured genes (< --min-program-genes {min_genes})")
    if not keep:
        raise SystemExit("no program has enough measured genes; are program genes and the h5ad gene names "
                         "the same kind of identifier (symbols vs Ensembl)? see --gene-col")
    return keep, stats


def write_r_matrix(X, genes, cells, d):
    """Write a cells x genes matrix for the R scripts: matrix.h5 (genes x cells CSC), genes.txt, cells.txt."""
    X = sp.csr_matrix(X)
    X.sort_indices()
    os.makedirs(d, exist_ok=True)
    with h5py.File(os.path.join(d, "matrix.h5"), "w") as f:       # cells x genes CSR == genes x cells CSC
        f["data"] = X.data.astype(np.float32)
        f["indices"] = X.indices.astype(np.int32)
        f["indptr"] = X.indptr.astype(np.int32)
        f["shape"] = np.array([X.shape[1], X.shape[0]], dtype=np.int32)
    for name, vals in (("genes.txt", genes), ("cells.txt", cells)):
        with open(os.path.join(d, name), "w") as fh:
            fh.write("\n".join(map(str, vals)) + "\n")


def run_aucell(X, genes, cells, programs, workdir, rscript="Rscript", cores=1, max_rank=None, seed=1,
               rankings=None, keep_input=False, repeats=1):
    """X: cells x genes (normalized). Returns a cells x programs DataFrame indexed by cell."""
    X = sp.csr_matrix(X)
    _, first = np.unique(genes, return_index=True)
    if len(first) < len(genes):
        keep = np.sort(first)
        log(f"{len(genes) - len(keep):,} duplicated gene names; keeping the first occurrence of each")
        X, genes = X[:, keep], genes[keep]
    X = sp.csr_matrix(X)
    X.sort_indices()
    if X.nnz >= 2 ** 31:
        raise SystemExit("expression matrix has >= 2^31 nonzero values, too many for an R sparse matrix; "
                         "subset genes (e.g. drop genes detected in very few cells) first")

    tmp = os.path.join(workdir, "aucell_input")
    write_r_matrix(X, genes, cells, tmp)
    pd.DataFrame([(p, g) for p, gl in programs.items() for g in gl], columns=["program", "gene"]).to_csv(
        os.path.join(tmp, "programs.tsv"), sep="\t", index=False)

    out = os.path.join(workdir, "aucell_scores.tsv")
    cmd = [rscript, R_SCRIPT, "--input", tmp, "--out", out, "--cores", str(cores), "--seed", str(seed),
           "--repeats", str(int(repeats))]
    if max_rank:
        cmd += ["--max-rank", str(max_rank)]
    if rankings:
        cmd += ["--rankings", rankings]
    log("running AUCell in R: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit(f"'{rscript}' not found; install R with the AUCell and rhdf5 Bioconductor packages "
                         "(see environment.yml) or pass --rscript /path/to/Rscript")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"AUCell (R) failed with exit code {e.returncode}; see the messages above")
    if not keep_input:
        shutil.rmtree(tmp, ignore_errors=True)
    return read_scores(out)


def read_scores(path):
    """cells x programs TSV whose first column is the cell name."""
    with open(path) as fh:
        first = fh.readline().rstrip("\n").split("\t")[0]
    s = pd.read_csv(path, sep="\t", dtype={first: str}).set_index(first)
    s.index.name = "cell"
    return s
