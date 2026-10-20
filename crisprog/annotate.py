"""AI annotation: Claude interprets the programs, the regulators and their caveats, with literature support.

The step first writes an annotation packet (6_annotation/):
  INSTRUCTIONS.md   what to produce and the rules (evidence, citations, calibrated language)
  packet.md         all results in compact form: programs (top genes, enrichment), significant associations,
                    per-gene follow-up (spread, concentration, flags), regulator enrichment, overlaps
  packet.json       the same, machine-readable
then runs one backend (annotation.backend):
  claude-code  (default) the Claude Code CLI in headless mode (`claude -p`); uses your Claude subscription
               login. If the CLI is not installed, only the packet is written.
  api          the Anthropic API, pay per use (needs ANTHROPIC_API_KEY or an `ant auth login` profile)
  packet       stop after writing the packet; open INSTRUCTIONS.md + packet.md in any Claude session
The report is written to report.md.
"""
import json
import os
import shutil
import subprocess

import pandas as pd

from .utils import log

INSTRUCTIONS = """# CRISProg annotation instructions

You are annotating the results of a single-cell CRISPR screen analyzed with CRISProg. Gene programs
(co-expressed gene sets) were scored per cell with AUCell; each perturbation unit (a guide, or a target such
as a gene) was tested for shifting each program's score (OLS with covariates, stratified permutation p-values,
FDR within program). For significant units, every gene was then tested individually to see whether the
perturbation moves many genes of the program (a broad trans effect) or mostly its own target.

The results are in `packet.md`. The experimental context from the user is at its top.

## What to write

A Markdown report with exactly these sections:

1. **Overview** - 4-6 sentences: what was found, at what scale, and the main biological themes.
2. **Programs** - one subsection per program that has a significant regulator, then a short list of the
   remaining programs. For each: a short label (at most 5 words), and 2-3 sentences on the biology it most
   likely represents, citing the evidence (enriched terms with q values, characteristic genes). If a program
   looks technical (ribosomal, mitochondrial, stress, cell cycle, batch) or has no clear biology, say so.
3. **Regulators** - one subsection per significant unit: which programs it moves and in which direction;
   how broad the effect is (changed program genes, concentration ratio, direction agreement); what is known
   about the target gene and whether that fits (with references); and the caveats that apply to it.
4. **Caveats** - flagged cases and limits, stated plainly:
   - `flag` = "self-driven?": the target is a program gene and no other program gene changed, so the program
     effect may come from the target's own knockdown. Do not present these as trans regulators.
   - units with high `cells_overlap` share perturbed cells (shared guides or CREs) and cannot be separated.
   - few changed genes, few perturbed cells, or programs with weak or no enrichment.
   - programs with low coherence (median correlation of their genes with the program score; real programs
     are typically well above noise-like ones) may be technical or noise.
5. **Hypotheses** - 3-5 specific, testable follow-up hypotheses, each tied to the result it comes from.
6. **References** - numbered; every reference cited in the text, with title, journal, year and a PubMed or
   DOI link.

## Rules

- Use only evidence in packet.md plus literature you actually looked up. Search the web for each regulator
  and for each program's main theme; prefer PubMed and primary papers.
- Attribute every association to the exact unit it was tested for, in every section including the Overview.
  Never describe one unit's association as another's, even when the two share perturbed cells; describe such
  pairs explicitly as a pair (e.g. "unit A raises program X; unit B, which shares its cells, raises program X
  and lowers program Y"). Use the unit and program names exactly as they appear in packet.md.
- For each regulator, search not only for the gene itself but for how its known function could connect to the
  biology of the programs it moves (signaling pathways, metabolism and nutrient sensing, chromatin, cell
  cycle). Mechanistic links are often found one step away from the gene name.
- Never invent a reference, PubMed ID, DOI, gene function or number. If you find no supporting literature,
  write that none was found.
- Use calibrated language ("consistent with", "suggests"); these are associations from a knockdown screen.
  Do not claim mechanisms the data cannot show.
- Quote numbers exactly as given (FDR, q, effect sizes, gene counts).
- Gene symbols in italics. Keep it concise: aim for 800-1500 words plus references.
- Before replying, check every unit-program claim and every number in the report against packet.md.
- Reply with the report only (no preamble).
"""


def _read(path, **kw):
    return pd.read_csv(path, sep="\t", **kw) if os.path.exists(path) else pd.DataFrame()


def build_packet(cfg, out):
    from .steps import DIRS, programs_path
    from .data import read_programs
    od, a = cfg["outdir"], cfg["association"]
    P = read_programs(programs_path(cfg))
    ptab = _read(os.path.join(od, DIRS["programs"], "programs.tsv"))
    method = dict(zip(ptab.program, ptab.method)) if len(ptab) else {}
    res = _read(os.path.join(od, DIRS["test"], "results.tsv"))
    cal = _read(os.path.join(od, DIRS["test"], "calibration.tsv"))
    info = json.load(open(os.path.join(od, DIRS["test"], "run_info.json"))) if os.path.exists(
        os.path.join(od, DIRS["test"], "run_info.json")) else {}
    hits = res[res.fdr < a["fdr"]] if len(res) else res
    summ = _read(os.path.join(od, DIRS["genes"], "program_summary.tsv"))
    ov = _read(os.path.join(od, DIRS["genes"], "overlap.tsv"))
    gcal = _read(os.path.join(od, DIRS["genes"], "calibration.tsv"))
    ep = _read(os.path.join(od, DIRS["enrich"], "enrichment_programs.tsv"))
    er = _read(os.path.join(od, DIRS["enrich"], "enrichment_regulators.tsv"))
    scored = list(cal.program) if len(cal) else list(P)
    rk = _read(os.path.join(od, DIRS["score"], "program_gene_ranking.tsv"))
    if len(rk):                                  # hub genes: correlation with the program score
        ranked = {p: list(g.gene) for p, g in rk.groupby("program", sort=False)}
        rank_note = {p: "ranked by correlation with the program score" for p in ranked}
        coherence = rk.groupby("program").r_with_score.median().round(3).to_dict()
    else:
        ranked = {p: P.get(p, []) for p in scored}
        note = {"hotspot": "ranked by Hotspot autocorrelation", "hdwgcna": "ranked by hdWGCNA kME"}
        rank_note = {p: note.get(method.get(p, "given"), "file order, not ranked") for p in scored}
        coherence = {}

    def terms(df, name, n=5):
        if df.empty:
            return []
        d = df[df.list == name].sort_values("p").head(n)
        return [dict(term=r.term, library=r.library, q=float(r.q), overlap=int(r.overlap),
                     genes=str(r.genes).split(",")[:12]) for r in d.itertuples()]

    programs = []
    for p in scored:
        h = hits[hits.program == p] if len(hits) else hits
        programs.append(dict(name=p, method=method.get(p, "given"), n_genes=len(P.get(p, [])),
                             top_genes=ranked.get(p, P.get(p, []))[:30], top_genes_order=rank_note.get(p, ""),
                             coherence=coherence.get(p),
                             enrichment=terms(ep, p),
                             regulators=[dict(unit=r.unit, direction="up" if r.t > 0 else "down",
                                              effect_sd=round(float(r.effect_sd), 3), fdr=float(r.fdr))
                                         for r in h.itertuples()] if len(h) else []))
    assoc = [dict(unit=r.unit, program=r.program, n_perturbed=int(r.n_perturbed),
                  effect_sd=round(float(r.effect_sd), 3), fdr=float(r.fdr), targets=r.targets,
                  target_in_program=None if pd.isna(r.target_in_program) else bool(r.target_in_program))
             for r in hits.itertuples()] if len(hits) else []
    follow = []
    for r in summ.itertuples() if len(summ) else []:
        follow.append(dict(unit=r.unit, program=r.program, changed_in_program=int(r.changed_in_program),
                           program_genes_tested=int(r.program_genes_tested), up=int(r.up), down=int(r.down),
                           same_direction_as_program=int(r.same_direction_as_program),
                           ratio=None if pd.isna(r.ratio) else round(float(r.ratio), 2),
                           target_t=r.target_t if isinstance(r.target_t, str) else "",
                           flag=r.flag if isinstance(r.flag, str) else "",
                           enrichment=terms(er, f"{r.unit} | {r.program}")))
    overlaps = [dict(program=r.program, unit_a=r.unit_a, unit_b=r.unit_b, genes_jaccard=r.genes_jaccard,
                     cells_overlap=r.cells_overlap) for r in ov.itertuples()
                if r.cells_overlap > 0.3 or r.genes_jaccard > 0.3] if len(ov) else []    # notable pairs only
    pk = dict(context=cfg["annotation"]["context"].strip(),
              run=dict(cells_analyzed=info.get("cells_analyzed"), units_tested=info.get("units_tested"),
                       programs_scored=len(scored), association_fdr=a["fdr"], design=info.get("design"),
                       permutation_strata=info.get("strata"),
                       null_share_p_lt_0_05_programs=None if cal.empty else round(float(cal["null_frac_p_ols_lt_0.05"].mean()), 3),
                       null_share_p_lt_0_05_genes=None if gcal.empty else round(float(gcal["frac_p_lt_0.05"].mean()), 3),
                       enrichment_backend=None if ep.empty else ep.backend.iloc[0]),
              programs=programs, associations=assoc, gene_followup=follow, overlaps=overlaps)
    with open(os.path.join(out, "packet.json"), "w") as fh:
        json.dump(pk, fh, indent=1, default=str)
    md = _packet_md(pk)
    with open(os.path.join(out, "packet.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(out, "INSTRUCTIONS.md"), "w") as fh:
        fh.write(INSTRUCTIONS)
    return md


def _packet_md(pk):
    L = ["# CRISProg results packet", "", "## Experimental context", "", pk["context"] or "(none given)", "",
         "## Run", ""]
    L += [f"- {k}: {v}" for k, v in pk["run"].items()]
    L += ["", "## Programs", ""]
    for p in pk["programs"]:
        coh = f", coherence (median gene-score r) {p['coherence']}" if p.get("coherence") is not None else ""
        L.append(f"### {p['name']} ({p['method']}, {p['n_genes']} genes{coh})")
        L.append(f"Top genes ({p['top_genes_order']}): " + ", ".join(p["top_genes"]))
        if p["enrichment"]:
            L.append("Enriched terms: " + "; ".join(f"{e['term']} [{e['library']}] q={e['q']:.2g} "
                                                    f"({e['overlap']} genes: {', '.join(e['genes'])})"
                                                    for e in p["enrichment"]))
        else:
            L.append("Enriched terms: none at the cutoff")
        if p["regulators"]:
            L.append("Significant regulators: " + "; ".join(f"{r['unit']} {r['direction']} (effect "
                                                            f"{r['effect_sd']} SD, FDR {r['fdr']:.2g})"
                                                            for r in p["regulators"]))
        L.append("")
    L += ["## Significant associations", "", "| unit | program | n perturbed cells | effect (SD) | FDR | targets "
          "| target in program |", "|---|---|---|---|---|---|---|"]
    L += [f"| {x['unit']} | {x['program']} | {x['n_perturbed']} | {x['effect_sd']} | {x['fdr']:.2g} | "
          f"{x['targets']} | {x['target_in_program']} |" for x in pk["associations"]]
    L += ["", "## Per-gene follow-up (own targets excluded)", "",
          "ratio = share of program genes changed / share of all genes changed",
          "", "| unit | program | changed / tested program genes | up | down | same direction as program | ratio "
          "| target t | flag |", "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {f['unit']} | {f['program']} | {f['changed_in_program']} / {f['program_genes_tested']} | {f['up']} | "
          f"{f['down']} | {f['same_direction_as_program']} | {f['ratio']} | {f['target_t']} | {f['flag']} |"
          for f in pk["gene_followup"]]
    enr = [f for f in pk["gene_followup"] if f["enrichment"]]
    if enr:
        L += ["", "### Enrichment of each regulator's changed program genes (vs the program's genes)", ""]
        for f in enr:
            L.append(f"- {f['unit']} | {f['program']}: " + "; ".join(
                f"{e['term']} q={e['q']:.2g} ({', '.join(e['genes'])})" for e in f["enrichment"]))
    if pk["overlaps"]:
        L += ["", "## Regulator pairs that share perturbed cells or changed genes (same program)", "",
              "cells_overlap = share of the smaller unit's perturbed cells also perturbed for the other", "",
              "| program | unit a | unit b | changed-gene Jaccard | cells overlap |", "|---|---|---|---|---|"]
        L += [f"| {o['program']} | {o['unit_a']} | {o['unit_b']} | {o['genes_jaccard']} | {o['cells_overlap']} |"
              for o in pk["overlaps"]]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------- backends
REQUIRED = ("## Overview", "## Programs", "## Regulators", "## References")


def strip_preamble(text):
    """Drop anything before the report's first heading (e.g. 'Writing the report now.')."""
    lines = text.strip().splitlines()
    first = next((i for i, x in enumerate(lines) if x.startswith("#")), 0)
    return "\n".join(lines[first:]).strip()


def check_report(text):
    """Problems that make an output unusable as the report (empty list = fine)."""
    missing = [h for h in REQUIRED if h not in text]
    words = len(text.split())
    return ([f"missing sections: {', '.join(missing)}"] if missing else []) + \
           ([f"only {words} words"] if words < 300 else [])


def _claude_code(out, an):
    cmd = ["claude", "-p", "Read INSTRUCTIONS.md and packet.md in the current folder and follow the "
           "instructions. Reply with the report only.", "--output-format", "text",
           "--allowedTools", "Read,WebSearch,WebFetch" if an["web_search"] else "Read",
           "--max-turns", str(an["max_turns"])]
    if an["model"]:
        cmd += ["--model", an["model"]]
    tries = 1 + max(0, int(an["retries"]))
    for i in range(1, tries + 1):
        log(f"running Claude Code headless (claude -p), attempt {i}/{tries}; this can take several minutes")
        r = subprocess.run(cmd, cwd=out, capture_output=True, text=True, timeout=3600)
        text = strip_preamble(r.stdout)
        problems = ([f"exit code {r.returncode}: {r.stderr.strip()[:300]}"] if r.returncode != 0 else []) \
            + check_report(text)
        if not problems:
            return text
        with open(os.path.join(out, f"failed_attempt_{i}.md"), "w") as fh:
            fh.write(text + "\n")
        log(f"attempt {i} unusable ({'; '.join(problems)}); output kept in failed_attempt_{i}.md")
    raise SystemExit(f"no usable report after {tries} attempts")


def _api(out, an, packet_md):
    import anthropic
    client = anthropic.Anthropic()
    model = an["model"] or "claude-opus-5-5"
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 40},
             {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 20}] if an["web_search"] else []
    messages = [{"role": "user", "content": packet_md + "\n\nWrite the report now."}]
    log(f"calling the Anthropic API ({model}{', web search on' if tools else ''})")
    final = None
    for _ in range(8):                                    # resume server-tool loops that pause
        with client.beta.messages.stream(
                model=model, max_tokens=64000, system=INSTRUCTIONS, messages=messages, tools=tools,
                thinking={"type": "adaptive"}, output_config={"effort": "high"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default") as stream:
            final = stream.get_final_message()
        if final.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": final.content})
    if final.stop_reason == "refusal":
        raise SystemExit(f"the model declined: {getattr(final, 'stop_details', None)}")
    text, sources = [], {}
    for b in final.content:
        if b.type == "text":
            text.append(b.text)
            for c in getattr(b, "citations", None) or []:
                url = getattr(c, "url", None)
                if url:
                    sources[url] = getattr(c, "title", "") or url
    report = strip_preamble("".join(text))
    problems = check_report(report)
    if problems:
        with open(os.path.join(out, "failed_attempt_api.md"), "w") as fh:
            fh.write(report + "\n")
        raise SystemExit(f"API output unusable ({'; '.join(problems)}); kept in failed_attempt_api.md")
    if sources:
        report += "\n\n### Web sources consulted\n" + "\n".join(f"- [{t}]({u})" for u, t in sources.items())
    return report


def _has_api_credentials():
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    if shutil.which("ant"):
        r = subprocess.run(["ant", "auth", "status"], capture_output=True, text=True)
        return r.returncode == 0 and "no " not in r.stdout.lower()
    return False


def run_annotation(cfg, out):
    an = cfg["annotation"]
    md = build_packet(cfg, out)
    log(f"annotation packet: {os.path.join(out, 'packet.md')} ({len(md):,} characters)")
    backend = an["backend"]
    if backend == "claude-code" and not shutil.which("claude"):
        log("the `claude` command (Claude Code) was not found, so only the packet was written. Install Claude Code "
            "and log in once (or set annotation.backend: api with Anthropic API credentials), then rerun: "
            "python crisprog.py annotate <config>")
        return
    if backend == "packet":
        log("packet only: open INSTRUCTIONS.md and packet.md (6_annotation/) in a Claude session and ask it to "
            "follow the instructions")
        return
    if backend == "api" and not _has_api_credentials():
        raise SystemExit("annotation.backend is api but no Anthropic credentials were found "
                         "(set ANTHROPIC_API_KEY or run `ant auth login`)")
    try:
        report = _claude_code(out, an) if backend == "claude-code" else _api(out, an, md)
    except (SystemExit, Exception) as ex:          # e.g. no internet on a cluster node, not logged in
        log(f"WARNING: annotation with {backend} failed ({str(ex)[:300]}); the packet is in {out} - rerun "
            "`python crisprog.py annotate <config>` where Claude can be reached, or open the packet in a Claude session")
        return
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write(report)
    log(f"report written: {os.path.join(out, 'report.md')}")
