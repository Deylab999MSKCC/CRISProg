# CRISProg

**Which perturbations in a single-cell CRISPR screen drive which gene programs, and what that means.**

Each perturbation changes downstream (trans) genes only a little, and testing every perturbation against every gene carries a heavy multiple-testing burden. CRISProg tests perturbations against **gene programs** (co-expressed gene sets) instead, then checks, explains and annotates the hits.

| Step | What it does | Output folder |
|---|---|---|
| `check` | **Data check** before anything runs: is `.X` normalized, are raw counts there, guide format and MOI, which controls were found and whether the negative-control check can run, cells per unit, whether target names match measured genes, **knockdown of every target**, batch structure, program coverage. PASS/WARN/FAIL report; optional AI review | `0_check/` |
| `programs` | Discovers programs with **Hotspot** and/or **hdWGCNA** (both on a PCA of the cells), or takes programs you give it | `1_programs/` |
| `score` | Scores every program in every cell with **AUCell** (Bioconductor, run through R); ranks each program's genes by their correlation with its score (hub genes) | `2_scores/` |
| `test` | `program score ~ perturbed + covariates` for every perturbation × program; stratified-permutation P-values; FDR within each program; **negative-control check** (random groups of control guides tested the same way) | `3_association/` |
| `genes` | For the significant perturbations: the same model on every gene; per-program heatmaps; how broad and how concentrated each effect is; flags | `4_gene_test/` |
| `enrich` | Pathway enrichment of the programs and of each regulator's changed program genes (**Enrichr** with your background; local GMT files as option/fallback) | `5_enrichment/` |
| `annotate` | **Claude** interprets everything, with literature support, into a report | `6_annotation/` |

## Install

```bash
git clone <repo-url> && cd CRISProg
conda env create -f environment.yml
conda activate crisprog
Rscript -e 'remotes::install_github("smorabit/hdWGCNA", ref = "dev", upgrade = "never")'   # not on conda
```

You can also use an existing setup: Python ≥ 3.10 with `numpy scipy pandas anndata h5py matplotlib scanpy pyyaml requests hotspotsc anthropic`, plus R with the Bioconductor packages `AUCell` and `rhdf5`, and `Seurat`, `WGCNA` and `hdWGCNA` for hdWGCNA discovery.

## Quick start (simulated screen)

```bash
python example/make_example.py              # writes the simulated screen to data/example/
python crisprog.py run example/config.yaml  # results in results/example/
```

The simulation has three co-expressed programs (PROG_A–C) and two planted regulators: *G0005* raises PROG_A and *G0900* lowers PROG_B. Discovery recovers the three programs from both embeddings, and the association test finds both regulators. The per-gene step also flags two null targets (*G1019*, *G1105*) whose "effect" on an extra module is only their own knockdown. Program discovery takes about 1.5 min; set `programs.file: ../data/example/programs.gmt` in the config to skip it.

## Running

Everything is set in one YAML config (`config.example.yaml` lists every option with its default):

```bash
python crisprog.py run config.yaml                 # all steps
python crisprog.py run config.yaml --from test     # resume from a step
python crisprog.py run config.yaml --steps check,test,genes,enrich,annotate   # chosen steps
python crisprog.py genes config.yaml               # one step
```

**Where things go.** Each run has a `name` (default: the h5ad file name). Results go to `results/<name>/` in the repo (or `<results_dir>/<name>/` if you set `results_dir`), with one folder per step plus `config_used.yaml` and `crisprog.log`. Keeping a dataset's inputs in `data/<name>/` keeps things together; `results/` and `data/` are not tracked by git. Each step reads the previous steps' outputs from `results/<name>/`, so you can rerun any step with new settings; use a new `name` to keep variants side by side (e.g. `myscreen_custom_background`).

## Inputs

| Input | Config key | Format |
|---|---|---|
| Analysis data | `data.h5ad` | `.X` = **normalized** expression (e.g. log1p CP10k), not scaled; unique cell names |
| Raw counts | `data.counts_layer` | Layer with raw counts. Needed for program discovery; recommended for the depth covariates |
| Guides per cell | `data.guide_col` | obs column listing each cell's guides, e.g. `g1;g2;g3` (`data.guide_sep`). Empty = no guide call |
| Control guides | `data.controls`, `data.control_pattern` | Labels and/or a regex. Controls are not tested as units and don't count towards MOI; they are used for the negative-control check |
| Control types | `data.control_types` | Optional `{type: regex}` to check kinds of negative controls separately, e.g. `{non-targeting: '^NTC', safe-targeting: '^OR'}` |
| Positive controls | `data.positive_controls` | Optional list of targets that are knockdown controls (e.g. `B2M`, `CD81`); labelled in the results and reported separately |
| Guide → target | `data.target_map` | Optional TSV with columns `guide`, `target` (and optionally `target_id`, e.g. the Ensembl ID, shown in the results and the report so that ambiguous symbols are identifiable). Without it every guide is tested on its own; with it, guides are grouped by target, and targets with identical guides are tested once (`A/B`) |
| Programs | `programs.file` | Optional `.gmt` or TSV (`program`, `gene`). Without it, programs are discovered |
| Enrichment background | `programs.background` | Optional gene list (one per line) for enriching given programs, e.g. the genes the programs were built from. Default: all measured genes |
| Discovery data | `programs.discover.h5ad` | Optional separate h5ad to discover programs from (same format; e.g. a time course). Default: the analysis data |
| Context | `annotation.context` | A few sentences on the experiment for the AI step: cell type, perturbation type, the question |

## Method details

**Data check.** The first step loads the data with the config's settings and checks what can go wrong silently: raw counts in `.X` instead of normalized values (FAIL), Ensembl IDs instead of symbols, guides per cell, control guides (none, merged into one label, or individual), units with too few cells, target names that do not match a measured gene (names like `POU5F1_P2` or `ADNP2_-_77867536.23-P1P2-1` are matched after removing promoter/guide suffixes), and whether each target is actually knocked down: the t of its own gene in perturbed vs other cells, with the same covariates as the main test. Targets that go *up* point to a wrong guide → target map. It also lists obs columns that look like unused batches. `check.strict: true` stops the run on a FAIL; `check.review: claude-code` (or `api`) adds `review.md`, an AI reading of the report with suggested config changes.

**Negative-control check.** The permutation null shows the test is right if guide assignment were random given the covariates, but it cannot see confounding such as guides being detected more easily in one cell state. Negative-control guides can: CRISProg forms random groups of control guides of each type, each grown guide by guide until it has as many perturbed cells as a typical tested unit, and tests them exactly like real units. If more than ~5% of them reach p < 0.05 in a program, or any reaches FDR < 0.05, part of that program's signal is not specific to the targets (the log warns). Each real association gets `p_vs_controls` (share of control units with an effect at least as large) and `beyond_controls` (|effect| above the 99th percentile of control effects). This needs individual control guide calls; a single merged `control` label cannot be split, and the check reports that.

**Program discovery.** Both methods start from the same PCA of the discovery cells (normalize, log, highly variable genes, scale, 30 PCs), and each module they return is a program: a discrete gene set, which is what AUCell needs. Methods that give continuous gene weights (e.g. NMF/cNMF spectra) are not used, because there is no natural cutoff for membership.
- **Hotspot** (`hotspot`) builds a cell kNN graph on the PCA, finds genes whose expression is locally autocorrelated on it, and groups the top autocorrelated genes into modules by their pairwise local correlations. Hotspot has no embedding of its own, so the PCA is required. Programs are named `Hotspot-1, -2, …`; genes are ranked by autocorrelation Z.
- **hdWGCNA** (`hdwgcna`) averages nearest-neighbour cells on the PCA into metacells (within each `hdwgcna.group_col` group, e.g. sample or time point), builds a signed co-expression network on genes expressed in ≥ 5% of cells, and cuts it into modules. Programs are named by the module colour (`hdWGCNA-turquoise`, …); genes are ranked by kME, their correlation with the module eigengene. The soft power defaults to hdWGCNA's rule, the lowest power whose scale-free fit R² reaches 0.8; check `hdwgcna_modules_soft_powers.tsv`, and set `hdwgcna.soft_power` if the chosen power has a slope far from −1 (on the simulated example the rule picks 5 with slope −14 and misses a program; 12 recovers all three).

The two methods are run independently and can find the same biology twice; near-duplicate programs both enter the FDR correction, so consider keeping one of each pair.

**Discovering programs from the screen itself** is allowed but can be circular: a target's own knockdown creates structure, and the target can end up inside a module. The per-gene step catches this (see Flags). A separate discovery dataset avoids it.

**AUCell tie-breaking.** AUCell ranks genes within each cell and breaks tied expression values at random. Normalized sparse data have many ties, so scores (and borderline hits) move slightly with the random draw. CRISProg makes this reproducible (`aucell.seed`, also passed to AUCell's parallel workers) and can average the AUC over several tie-breaking seeds (`aucell.n_repeats`) to remove the dependence on a single draw, at the cost of one ranking per seed. In our tests on a large screen, t values from two seeds correlated at r ≈ 0.997, and associations near the FDR cutoff (FDR 0.03–0.08) were significant with some seeds but not others. **For final results, use `aucell.n_repeats: 5`** and treat associations that pass only with some seeds as borderline.

**Association test.** OLS of each program's AUCell score on each perturbation, with covariates (default: log genes detected, MOI; add any obs column and `covariates.batch`). MOI enters as categories (one level per number of guides, sparse top levels pooled) when it takes few integer values (`covariates.moi_categorical: auto`), so each unit is compared with cells carrying the same number of guides; this matters when control guides sit in their own single-guide cells. When control guides share cells with targeting guides, the number of control guides per cell is added the same way (`covariates.control_count: auto`), so every comparison is between cells with the same total number of detected guides: guide detection can depend on cell state (e.g. more guides detected in cells with more RNA), and without this, cells carrying an extra control guide look systematically different. All fits are computed at once (Frisch–Waugh–Lovell). P-values come from stratified permutations: cells swap perturbation labels only within the same MOI × depth bin × batch stratum. With more than 50 units, the null is pooled over tests with similar perturbed-cell counts. FDR is Benjamini–Hochberg within each program. `3_association/calibration.tsv` shows the share of shuffled tests with OLS P < 0.05, which should be about 0.05.

**Per-gene follow-up.** For each significant perturbation, the same model is fit on every gene's expression (BH across genes). The perturbation's cis targets are excluded from all counts: every gene its guides target, including genes that share a CRE with it. `program_summary.tsv` reports, for each association:
- how many program genes changed, and in which direction
- whether they moved with the program score
- `ratio` = (share of the program's genes that changed) / (share of all genes that changed)

**Flags** (in `program_summary.tsv` and the AI packet):
- `self-driven?`: the target is a program gene and no other program gene changed. The program effect may be just the target's own knockdown.
- `cells_overlap` in `overlap.tsv`: the share of perturbed cells two regulators share. High overlap (e.g. a CRE linked to two genes) means the two effects cannot be separated.

**Enrichment.** Programs are tested against `programs.background` if given, else the discovery genes, else all measured genes. Each regulator's changed program genes are tested against the program's measured genes. The Enrichr backend uses Enrichr's background-aware API with the newest version of each library family (`enrichment.libraries`). If Enrichr is unreachable and `enrichment.gmt` files are given, a local hypergeometric test is used instead. Terms are kept at q < 0.05 with at least 3 overlapping genes.

**AI annotation.** The step writes `INSTRUCTIONS.md` and `packet.md`, with all results in compact form, then runs one backend:

| `annotation.backend` | How | Cost |
|---|---|---|
| `claude-code` (default) | Claude Code CLI in headless mode (`claude -p`), with web search. If the CLI isn't installed, only the packet is written | Your Claude subscription |
| `api` | Anthropic API (Claude Opus 5.5, adaptive thinking, web search/fetch, server-side refusal fallback) | Pay per use; needs `ANTHROPIC_API_KEY` or `ant auth login` |
| `packet` | Writes the packet only; open the two files in any Claude session | – |

To set up Claude Code once: `curl -fsSL https://claude.ai/install.sh | bash`, then run `claude` and `/login`. Enrichment and annotation need internet access; on a cluster, run them on a node that has it (`--from enrich`).

The report covers:
- a label and summary for each program
- for each regulator: direction, breadth, literature and caveats
- flagged cases
- testable hypotheses
- verified references

The output is checked for completeness (all required sections, a minimum length). A cut-off or incomplete reply is never saved as the report: the step retries (`annotation.retries`, default 2), keeps failed outputs as `failed_attempt_<n>.md`, and if every attempt fails it leaves the packet for a manual run. The instructions forbid invented references and require calibrated language. Still, check the references before using them.

## Outputs (main files)

| File | Content |
|---|---|
| `1_programs/programs.tsv`, `.gmt`, `programs_summary.tsv` | Discovered programs (gene, rank, method, score) |
| `2_scores/aucell_scores.tsv`, `program_gene_ranking.tsv` | Cells × programs scores; hub genes per program |
| `0_check/check_report.md`, `unit_qc.tsv` (+ `review.md`) | Data check; per-unit cells, knockdown t, batch share |
| `3_association/results.tsv`, `volcano.pdf`, `calibration.tsv` | All unit × program tests (incl. `unit_type`, `beyond_controls`, `p_vs_controls`) |
| `3_association/controls/control_summary.tsv`, `control_results.tsv` | Negative-control check |
| `4_gene_test/program_summary.tsv`, `heatmap_<program>.pdf`, `overlap.tsv`, `gene_results.tsv.gz` | Per-gene follow-up |
| `5_enrichment/enrichment_programs.tsv`, `enrichment_regulators.tsv`, `top_terms.tsv` | Enrichment |
| `6_annotation/report.md` (+ `packet.md`, `INSTRUCTIONS.md`) | AI report |

## Validation

- `pytest -q tests/` (about 2 min; needs R with AUCell, and hdWGCNA for its test). The tests check:
  - the OLS against direct fits
  - the permutation P-values against a brute-force version
  - planted regulators found with calibrated nulls, using given programs
  - Hotspot and hdWGCNA each recovering the planted programs (gene-set Jaccard > 0.8)
  - the data check detecting the simulated knockdown; negative-control units staying null, and flagging a planted confounder (control guides detected more often in cells with high program activity)
  - target names with promoter suffixes matched to their genes

## Citation

If you use CRISProg, please cite AUCell (Aibar et al. 2017, *Nat. Methods*), Hotspot (DeTomaso & Yosef 2021, *Cell Systems*), hdWGCNA (Morabito et al. 2023, *Cell Reports Methods*), WGCNA (Langfelder & Horvath 2008, *BMC Bioinformatics*), Enrichr (Kuleshov et al. 2016, *Nucleic Acids Res.*) and the paper this method comes from (reference to be added).

## License

To be added.
