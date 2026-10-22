"""Rule-based settings advice for the data check: what each setting is, what the data shows, what CRISProg will do
(or suggests changing), and why. No AI involved: the same data always gives the same advice.

suggest_and_explain(cfg, A, design) -> (suggestions, explanations)
  suggestions    [(config path, value, reason)]: changes the rules recommend; written to suggested_config.yaml
  explanations   [(topic, setting, current value, plain-language explanation)]: the 'Settings explained' section
"""
import re

import numpy as np
import pandas as pd
import yaml

BATCH_PRIORITY = ["gem_group", "lane", "batch", "library", "sample", "channel", "plate", "run", "pool", "replicate",
                  "donor", "chip"]
BATCH_HINT = re.compile("|".join(BATCH_PRIORITY), re.I)
CONTROL_KINDS = {                                   # kind -> regex on guide names
    "non-targeting": r"non.?target|^ntc|^nt[_-]|scrambl|negative.?control",
    "safe-targeting": r"safe.?target|^safe[_-]|^(NC_)?OR\d+[A-Z]+\d+",
    "control": r"(^|[_-])control([_-]|$)",
}
KNOWN_POSITIVE = ["B2M", "CD81", "CD151", "CD55", "CD29", "ITGB1", "TFRC", "DNAJC19", "POLR1D", "AARS", "AARS1",
                  "MALAT1", "NGFRAP1", "BEX3"]


def _layers(path):
    """Layer names of the h5ad and whether each looks like raw counts (integer values, max > 20)."""
    import anndata
    ad = anndata.read_h5ad(path, backed="r")
    out = {}
    rows = np.sort(np.random.default_rng(0).choice(ad.n_obs, min(500, ad.n_obs), replace=False))
    for name in ad.layers:
        try:
            S = ad.layers[name][rows]
            S = S.toarray() if hasattr(S, "toarray") else np.asarray(S)
            nz = S[S != 0]
            out[name] = bool(nz.size and np.allclose(nz, np.round(nz)) and nz.max() > 20)
        except Exception:
            out[name] = False
    var = ad.var.copy()
    ad.file.close()
    return out, var


def suggest_and_explain(cfg, A, znames):
    d, cv, a = cfg["data"], cfg["covariates"], cfg["association"]
    S, E = [], []

    def sug(path, value, reason):
        S.append((path, value, reason))

    def exp(topic, setting, value, text):
        E.append((topic, setting, value, text))

    # ---- expression ----
    layers, var = _layers(d["h5ad"])
    counts_like = [k for k, v in layers.items() if v]
    if not d["counts_layer"] and counts_like:
        pick = "counts" if "counts" in counts_like else counts_like[0]
        sug("data.counts_layer", pick, f"layer '{pick}' holds integer counts; raw counts give the 'genes detected' "
            "covariate and are needed for program discovery")
    exp("Expression", "data.counts_layer", d["counts_layer"],
        "Layer with raw UMI counts. Used to count genes detected per cell (a covariate for sequencing depth) and for "
        "program discovery. `.X` must hold normalized, log-transformed values; AUCell and the per-gene test use `.X`. "
        + (f"Layers that look like raw counts: {', '.join(counts_like)}." if counts_like else "No layer looks like raw counts."))
    ens = np.mean([str(g).startswith(("ENSG", "ENSMUSG")) for g in A.genes])
    if ens > 0.5 and not d["gene_col"]:
        cand = [c for c in var.columns if var[c].dtype == object and
                np.mean(var[c].astype(str).str.startswith(("ENSG", "ENSMUSG"))) < 0.1 and var[c].nunique() > 0.5 * len(var)]
        pref = [c for c in cand if re.search(r"symbol|name|gene", c, re.I)] or cand
        if pref:
            sug("data.gene_col", pref[0], f"gene names are Ensembl IDs ({ens:.0%}); column '{pref[0]}' has symbols, "
                "which programs, target matching and enrichment need")
    exp("Expression", "data.gene_col", d["gene_col"],
        f"Which gene names to use (default: the h5ad's gene index). Now {ens:.0%} look like Ensembl IDs; symbols are "
        "needed to match program genes, targets and pathway databases.")

    # ---- guides and controls ----
    moi = A.builtin["moi"]
    exp("Guides", "data.guide_col / guide_sep", f"{d['guide_col']} / '{d['guide_sep']}'",
        f"Per-cell guide calls. {len(A.guide_names):,} targeting guide labels and {len(A.control_guides):,} control "
        f"labels were found; targeting guides per cell: median {np.median(moi):.0f}, max {moi.max():.0f}. Cells "
        "without any guide call are dropped.")
    names = pd.Series(list(A.guide_names) + list(A.control_guides), dtype=str)
    if len(A.control_guides) == 0:
        found = {k: int(names.str.contains(rx, case=False, regex=True).sum()) for k, rx in CONTROL_KINDS.items()}
        found = {k: v for k, v in found.items() if v >= 2}
        if found:
            pat = "(?i)" + "|".join(f"(?:{CONTROL_KINDS[k]})" for k in found)      # case-insensitive, as detected
            sug("data.control_pattern", pat, "guide names that look like negative controls: "
                + ", ".join(f"{k} {v}" for k, v in found.items()))
            if len(found) > 1:
                sug("data.control_types", {k: "(?i)" + CONTROL_KINDS[k] for k in found},
                    "several kinds of negative controls; checking them separately tells binding from non-binding effects")
    elif not d["control_types"]:
        kinds = {k: int(pd.Series(A.control_guides, dtype=str).str.contains(rx, case=False, regex=True).sum())
                 for k, rx in CONTROL_KINDS.items() if k != "control"}
        kinds = {k: v for k, v in kinds.items() if v >= 2}
        if len(kinds) > 1:
            sug("data.control_types", {k: "(?i)" + CONTROL_KINDS[k] for k in kinds},
                "the control guides are of several kinds (" + ", ".join(f"{k} {v}" for k, v in kinds.items())
                + "); checking them separately tells binding from non-binding effects")
    shared = bool(((np.asarray(A.Gc.sum(axis=1)).ravel() > 0) & (moi > 0)).any()) if len(A.control_guides) else False
    exp("Controls", "data.controls / control_pattern", d["control_pattern"] or d["controls"] or None,
        "Negative-control guides (non-targeting, or safe-targeting at silent genes). They are not tested as TFs and "
        "not counted in MOI. They are used for the negative-control check: random groups of them are tested exactly "
        "like TFs ('fake TFs') and should show nothing; if they do, the comparison is unfair. "
        + (f"Found {len(A.control_guides):,} control labels; "
           + ("they share cells with targeting guides." if shared else
              "they are only in cells without targeting guides (a separate group of cells).")
           if len(A.control_guides) else "None found, so the negative-control check cannot run.")
        + (" Only one or a few labels (merged controls) cannot be split into fake TFs." if 0 < len(A.control_guides) < 4 else ""))
    exp("Controls", "data.control_types", d["control_types"] or None,
        "Optional split of the negative controls into kinds (e.g. non-targeting vs safe-targeting), each checked "
        "separately. Safe-targeting guides bind DNA like targeting guides do, so they are the closer comparison.")

    # ---- positive controls and targets ----
    present = set()
    for t, gset in zip(A.units["targets"].fillna(""), A.units["target_genes"].fillna("")):
        present |= set(str(t).split(";")) | set(str(gset).split(";"))
    pos_found = [g for g in KNOWN_POSITIVE if g in present]
    if not d["positive_controls"] and pos_found:
        sug("data.positive_controls", pos_found, "targets often used as CRISPRi knockdown controls are in the data; "
            "labelling them keeps them out of the regulator list (confirm against the guide library)")
    exp("Targets", "data.positive_controls", d["positive_controls"] or None,
        "Knockdowns of non-TF genes included by the library to show that knockdown works (e.g. B2M, CD81, TFRC). "
        "They are tested like TFs but labelled and reported separately, not as regulators."
        + (f" Common positive-control targets present: {', '.join(pos_found)}." if pos_found else ""))
    has_t = A.units["targets"].fillna("").str.len() > 0
    resolved = A.units["target_genes"].fillna("").str.len() > 0
    exp("Targets", "data.target_map", d["target_map"],
        "Guide -> target table. Guides of the same target are tested together as one unit. "
        + (f"{int(has_t.sum()):,} units; {int(resolved.sum()):,} have a measured target gene (names with promoter or "
           "guide suffixes, e.g. 'POU5F1_P2', are matched to their gene), which allows the knockdown check."
           if d["target_map"] else "Without it, every guide is tested on its own and knockdown cannot be checked."))
    if not d["target_map"]:
        looks = names.str.match(r"^[A-Za-z0-9.-]+[_-]\d+$").mean() if len(names) else 0
        if looks > 0.5:
            sug("data.target_map", "<guide_targets.tsv>", f"{looks:.0%} of guide names look like '<gene>_<n>': write a "
                "TSV with columns guide, target (target = name without the _<n>) so each gene's guides are tested together")

    # ---- covariates ----
    if not A.batch_cols:
        cand = [c for c in A.obs.columns if BATCH_HINT.search(str(c)) and 1 < A.obs[c].nunique() <= 200
                and A.obs[c].value_counts().min() >= 0.002 * len(A.obs)]
        cand.sort(key=lambda c: next((i for i, k in enumerate(BATCH_PRIORITY) if k in str(c).lower()), 99))
        if cand:
            c = cand[0]
            sug("covariates.batch", [c], f"obs column '{c}' looks like a batch ({A.obs[c].nunique()} levels); without a "
                "batch covariate, batch differences can look like perturbation effects")
    exp("Model", "covariates.batch", list(A.batch_cols) or None,
        "Batch columns (e.g. 10x lane). Each batch gets its own baseline in the model, and p-value shuffles only swap "
        "cells within a batch.")
    moi_cols = [z for z in znames if z.startswith("moi=")]
    nctl_cols = [z for z in znames if z.startswith("n_control=")]
    exp("Model", "covariates.use", ", ".join(cv["use"]),
        "Covariates: 'log_n_genes' = genes detected per cell (sequencing depth); 'moi' = number of targeting guides "
        "per cell. The model is: program score ~ knocked down + covariates.")
    exp("Model", "covariates.moi_categorical", cv["moi_categorical"],
        ("Guide count enters as one column per value: " + ", ".join(moi_cols[:4]) + (" ..." if len(moi_cols) > 4 else "")
         + f" ({len(moi_cols)} columns; the lowest count is the reference, rare top counts are pooled so each level "
         "has >= 50 cells). Each knocked-down cell is compared with cells carrying the same number of guides, with no "
         "assumption that guide count changes the score in a straight line.")
        if moi_cols else "Guide count enters as one straight-line term (it is not a small set of whole numbers, or this "
                         "was switched off).")
    exp("Model", "covariates.control_count", cv["control_count"],
        (f"Added ({len(nctl_cols)} columns): control guides share cells with targeting guides, so the number of control "
         "guides is matched too, i.e. cells are compared at the same total number of detected guides. Guide detection "
         "can depend on cell state, and without this, cells with an extra control guide can look systematically different.")
        if nctl_cols else ("Not added: control guides are only in cells without targeting guides, which the guide-count "
                           "columns already separate." if len(A.control_guides) and not shared else
                           "Not added: no control guides share cells with targeting guides."))
    exp("Model", "association", f"min_perturbed {a['min_perturbed']}, n_perm {a['n_perm']}, FDR {a['fdr']}",
        f"Units with fewer than {a['min_perturbed']} perturbed cells are not tested. P-values come from {a['n_perm']:,} "
        "shuffles of guide labels among cells of the same guide count, depth bin and batch (pooled across similar units "
        f"when more than 50 are tested); FDR within each program; hits are FDR < {a['fdr']}.")

    # ---- programs ----
    exp("Programs", "programs.file", cfg["programs"]["file"],
        "Gene programs to test. Given here, or discovered from the data (Hotspot and hdWGCNA modules) when empty."
        if cfg["programs"]["file"] else
        "Empty: programs are discovered from the data (Hotspot and hdWGCNA modules on a PCA); needs raw counts.")
    return S, E


def write_suggestions(S, path):
    """suggested_config.yaml: only the suggested keys, each with its reason as a comment."""
    L = ["# Settings suggested by the CRISProg data check (rule-based). Copy the ones you agree with into your",
         "# config and rerun. Values in <...> are placeholders you need to fill in.", ""]
    if not S:
        L.append("# No changes suggested.")
    top = {}
    for key, value, reason in S:
        k1, k2 = key.split(".", 1)
        top.setdefault(k1, []).append((k2, value, reason))
    for k1, items in top.items():
        L.append(f"{k1}:")
        for k2, value, reason in items:
            L.append(f"  # {reason}")
            L += ["  " + x for x in yaml.safe_dump({k2: value}, sort_keys=False, default_flow_style=False,
                                                   allow_unicode=True).rstrip().splitlines()]
        L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def explanations_md(E, S):
    L = ["## Settings explained", "",
         "What each setting does, its current value, and what the data shows. Suggested changes are in "
         "`suggested_config.yaml`.", ""]
    for topic in dict.fromkeys(e[0] for e in E):
        L += [f"### {topic}", ""]
        for _, setting, value, text in (e for e in E if e[0] == topic):
            L.append(f"- **`{setting}`** = `{value}`: {text}")
        L.append("")
    if S:
        L += ["### Suggested changes", ""] + [f"- `{p}` → `{v}`: {r}" for p, v, r in S] + [""]
    return "\n".join(L)
