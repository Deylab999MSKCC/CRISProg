"""Command line for CRISProg (see crisprog.py at the repo root for usage)."""
import argparse
import os
import shutil
import sys
import time

from . import config as config_mod
from .steps import RUNNERS, STEPS, Context
from .utils import log

USAGE = """
CRISProg: which perturbations in a single-cell CRISPR screen drive which gene programs, and what that means.

Steps (outputs in results/<name>/<n>_<step>/):
  programs   discover gene programs with Hotspot and/or hdWGCNA (both on a PCA)
             (or take a program file you give in the config)
  score      AUCell score of every program in every cell (Bioconductor AUCell, run through R)
  test       program score ~ perturbed + covariates for every perturbation x program, stratified
             permutation p-values, FDR within each program
  genes      for the significant perturbations: the same model on every gene's expression, per-program
             heatmaps, and how concentrated the changes are in each program
  enrich     pathway enrichment of the programs and of each regulator's changed program genes
             (Enrichr web API with your background; local GMT files as option/fallback)
  annotate   AI interpretation of all results with literature support (Claude Code, the Anthropic API,
             or a packet to open in any Claude session)

Usage:
  python crisprog.py run config.yaml                  # all steps
  python crisprog.py run config.yaml --from test      # resume from a step
  python crisprog.py score config.yaml                # a single step
See config.example.yaml and README.md.
"""


class _Tee:
    """Copy everything printed to the terminal into the run's log file."""

    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, "a")

    def write(self, x):
        self.stream.write(x)
        self.file.write(x)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, k):
        return getattr(self.stream, k)


def main(argv=None):
    ap = argparse.ArgumentParser(description=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["run"] + STEPS)
    ap.add_argument("config")
    ap.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0], help="run: first step")
    ap.add_argument("--to", dest="stop", choices=STEPS, default=STEPS[-1], help="run: last step")
    a = ap.parse_args(argv)
    cfg = config_mod.load(a.config)
    os.makedirs(cfg["outdir"], exist_ok=True)
    shutil.copy(a.config, os.path.join(cfg["outdir"], "config_used.yaml"))
    sys.stdout = _Tee(sys.stdout, os.path.join(cfg["outdir"], "crisprog.log"))
    log(f"CRISProg run '{cfg['name']}': results in {cfg['outdir']}")
    steps = STEPS[STEPS.index(a.start):STEPS.index(a.stop) + 1] if a.step == "run" else [a.step]
    ctx = Context(cfg)
    for s in steps:
        t0 = time.time()
        log(f"===== {s} =====")
        RUNNERS[s](cfg, ctx)
        log(f"===== {s} done in {(time.time() - t0) / 60:.1f} min =====")
    return ctx

