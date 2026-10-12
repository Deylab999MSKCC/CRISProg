"""Pathway enrichment of the programs and of each regulator's changed program genes.

Gene lists and backgrounds:
  programs    each program's genes vs the genes programs were discovered from (or, for given programs, all
              genes measured in data.h5ad)
  regulators  for each significant unit x program association: the program genes the unit changed in the
              per-gene test (FDR < gene_test.fdr, own targets excluded) vs all measured genes of that program

Backends:
  enrichr  Enrichr's background-aware API (maayanlab.cloud/speedrichr), newest version of each library family
           in enrichment.libraries; q = Enrichr's adjusted p (BH within library)
  gmt      hypergeometric test against local .gmt files (enrichment.gmt); q = BH within file and list;
           terms with 5..max_term_size genes in the background
If Enrichr cannot be reached and GMT files are given, GMT is used instead.
Kept: q < enrichment.q and at least enrichment.min_overlap genes in the overlap.

Outputs: enrichment_programs.tsv, enrichment_regulators.tsv, top_terms.tsv (top 5 per list)
"""
import os
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .utils import benjamini_hochberg, log

API = "https://maayanlab.cloud/speedrichr/api"
LIBS = "https://maayanlab.cloud/Enrichr/datasetStatistics"
COLS = ["list_type", "list", "library", "term", "overlap", "n_list", "n_background", "p", "q", "odds_ratio", "genes"]


# --------------------------------------------------------------------------------------------- gene lists
def gene_lists(cfg):
    from .steps import DIRS, programs_path
    from .data import read_programs
    progs = read_programs(programs_path(cfg))
    bgf = os.path.join(cfg["outdir"], DIRS["programs"], "background_genes.txt")
    if not cfg["programs"]["file"] and os.path.exists(bgf):
        bg = [x.strip() for x in open(bgf) if x.strip()]
    else:
        import anndata
        ad = anndata.read_h5ad(cfg["data"]["h5ad"], backed="r")
        gc = cfg["data"]["gene_col"]
        bg = list((ad.var[gc] if gc else ad.var_names).astype(str))
        ad.file.close()
    bgs = set(bg)
    lists = [("program", p, [g for g in gl if g in bgs], bg) for p, gl in progs.items()]
    gt = os.path.join(cfg["outdir"], DIRS["genes"])
    summ_f, res_f = os.path.join(gt, "program_summary.tsv"), os.path.join(gt, "gene_results.tsv.gz")
    if os.path.exists(summ_f) and os.path.exists(res_f):
        summ = pd.read_csv(summ_f, sep="\t")
        R = pd.read_csv(res_f, sep="\t", keep_default_na=False)
        fc = cfg["gene_test"]["fdr"]
        measured = set(R.gene)
        for s in summ.itertuples():
            prog_bg = [g for g in progs.get(s.program, []) if g in measured]
            r = R[(R.unit == s.unit) & ~R.is_target.astype(str).str.lower().eq("true")]
            ch = sorted(set(r.gene[pd.to_numeric(r.fdr) < fc]) & set(prog_bg))
            if len(ch) >= 5:
                lists.append(("regulator", f"{s.unit} | {s.program}", ch, prog_bg))
    else:
        log("no per-gene test results; enriching programs only")
    return lists


# --------------------------------------------------------------------------------------------- Enrichr
def _post(url, files=None, data=None, tries=3):
    import requests
    for i in range(tries):
        try:
            r = requests.post(url, files=files, data=data, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def resolve_libraries(families):
    """Newest Enrichr library for each family, e.g. 'KEGG' -> 'KEGG_2026', 'WikiPathways_Human' ->
    'WikiPathways_2024_Human'."""
    import requests
    names = [x["libraryName"] for x in requests.get(LIBS, timeout=60).json()["statistics"]]
    out = []
    for fam in families:
        toks = fam.split("_")
        species = toks[-1] if toks[-1] in ("Human", "Mouse") else None
        base = "_".join(toks[:-1]) if species else fam
        pat = re.compile(rf"^{re.escape(base)}(?:_(\d{{4}}))?(?:_(Human|Mouse))?$")
        cand = []
        for nm in names:
            m = pat.match(nm)
            if m and (m.group(2) == species if species else m.group(2) in (None, "Human")):
                cand.append((int(m.group(1) or 0), nm))
        if not cand:
            log(f"WARNING: no Enrichr library matches '{fam}'")
            continue
        out.append(max(cand)[1])
    log("Enrichr libraries: " + ", ".join(out))
    return out


def enrichr(lists, libraries):
    bg_ids, rows = {}, []
    for kind, name, genes, bg in lists:
        if len(genes) < 3:
            continue
        key = hash(tuple(sorted(bg)))
        if key not in bg_ids:
            bg_ids[key] = _post(f"{API}/addbackground", data={"background": "\n".join(bg)})["backgroundid"]
        uid = _post(f"{API}/addList", files={"list": (None, "\n".join(genes)), "description": (None, name)})["userListId"]
        for lib in libraries:
            res = _post(f"{API}/backgroundenrich", data={"userListId": uid, "backgroundid": bg_ids[key],
                                                         "backgroundType": lib}).get(lib, [])
            for r in res:
                rows.append((kind, name, lib, r[1], len(r[5]), len(genes), len(bg), r[2], r[6], r[3],
                             ",".join(sorted(r[5]))))
            time.sleep(0.2)
    return pd.DataFrame(rows, columns=COLS)


# --------------------------------------------------------------------------------------------- local GMT
def read_gmt(path):
    sets = {}
    for ln in open(path):
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 3:
            sets[f[0]] = {g for g in f[2:] if g}
    return sets


def gmt_enrich(lists, files, max_size):
    libs = {os.path.basename(f).rsplit(".gmt", 1)[0]: read_gmt(f) for f in files}
    rows = []
    for kind, name, genes, bg in lists:
        bgs, H = set(bg), set(genes)
        if len(H) < 3:
            continue
        for lib, sets in libs.items():
            part = []
            for term, S in sets.items():
                Sb = S & bgs
                k = len(H & Sb)
                if k == 0 or not (5 <= len(Sb) <= max_size):
                    continue
                p = hypergeom.sf(k - 1, len(bgs), len(Sb), len(H))
                odds = (k / max(len(H) - k, 1)) / max(len(Sb) - k, 1) * max(len(bgs) - len(Sb) - len(H) + k, 1)
                part.append([kind, name, lib, term, k, len(H), len(bgs), p, np.nan, odds, ",".join(sorted(H & Sb))])
            if part:
                q = benjamini_hochberg([r[7] for r in part])
                for r, qq in zip(part, q):
                    r[8] = qq
                rows += part
    return pd.DataFrame(rows, columns=COLS)


# --------------------------------------------------------------------------------------------- step
def run_enrichment(cfg, out):
    e = cfg["enrichment"]
    lists = gene_lists(cfg)
    log(f"enrichment: {sum(k == 'program' for k, *_ in lists)} programs, "
        f"{sum(k == 'regulator' for k, *_ in lists)} regulator gene lists")
    backend, res = e["backend"], None
    if backend == "enrichr":
        try:
            res = enrichr(lists, resolve_libraries(e["libraries"]))
        except Exception as ex:                       # e.g. no internet on a cluster node
            log(f"WARNING: Enrichr not reachable ({type(ex).__name__}: {str(ex)[:120]})"
                + ("; falling back to local GMT files" if e["gmt"] else ""))
            backend = "gmt"
    if backend == "gmt":
        missing = [f for f in e["gmt"] if not os.path.exists(f)]
        if not e["gmt"] or missing:
            log("WARNING: no usable local GMT files (enrichment.gmt" + (f"; missing: {', '.join(missing)}" if missing
                else " is empty") + "); enrichment skipped, the annotation will run without it")
            res = pd.DataFrame(columns=COLS)
        else:
            res = gmt_enrich(lists, e["gmt"], e["max_term_size"])
    res = res[(res.q < e["q"]) & (res.overlap >= e["min_overlap"])].sort_values(["list_type", "list", "p"])
    res = res.assign(backend=backend)
    for kind, fn in (("program", "enrichment_programs.tsv"), ("regulator", "enrichment_regulators.tsv")):
        res[res.list_type == kind].to_csv(os.path.join(out, fn), sep="\t", index=False, float_format="%.4g")
    top = res.groupby("list", sort=False).head(5)
    top.to_csv(os.path.join(out, "top_terms.tsv"), sep="\t", index=False, float_format="%.3g")
    log(f"{len(res):,} terms at q < {e['q']} (>= {e['min_overlap']} genes) across {res.list.nunique()} lists")
    for nm, g in top.groupby("list", sort=False):
        print(f"  {nm}: " + "; ".join(f"{t} (q={q:.2g})" for t, q in zip(g.term.head(3), g.q.head(3))), flush=True)
