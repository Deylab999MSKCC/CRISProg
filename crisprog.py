#!/usr/bin/env python3
"""
CRISProg: which perturbations in a single-cell CRISPR screen drive which gene programs, and what that means.

Steps (outputs in results/<name>/<n>_<step>/):
  programs   discover gene programs with Hotspot on a PCA and/or cNMF cell embedding
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
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crisprog.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
