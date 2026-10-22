"""Fact-checking and rendering of the AI report (end of the annotation step).

1. rule-based checks (no AI; same input -> same result)
     numbers    every q value, signed effect / t value and 'x/y' gene count in the report must occur in the packet
     references every PMID / PMC / DOI must resolve on PubMed / Crossref, with the first author or title and the year
                matching the reference line (needs internet; otherwise marked 'not checked')
2. AI fact-check pass (annotation.factcheck): a second Claude call reads the draft, the packet and the rule-based
   findings, checks every claim (numbers against the packet; each citation against the paper's abstract), fixes only
   what is wrong, adds nothing new, and lists its changes
3. the rule-based checks are rerun on the corrected report
4. rendering (annotation.render): report.html, and report.pdf with Chrome/Chromium or WeasyPrint when available

Outputs (annotation step folder): report_draft.md, report.md (checked), factcheck.md, report.html, report.pdf
"""
import datetime
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request

from .utils import log

MINUS = str.maketrans({"−": "-", "–": "-", "‒": "-"})


# ------------------------------------------------------------------------------------------------- rule-based
def _decimals(text):
    return [float(x) for x in re.findall(r"-?\d+\.\d+", text.translate(MINUS))]


def check_numbers(report, packet):
    """Numbers in the report that do not occur in the packet: q values, signed decimals, x/y counts."""
    rep, pk = report.translate(MINUS), packet.translate(MINUS)
    problems = []
    for q in sorted(set(re.findall(r"q\s*=\s*([0-9.]+(?:e-?\d+)?)", rep))):
        if f"q={q}" not in pk:
            problems.append(("q value", f"q={q}"))
    pk_vals = _decimals(pk)
    for m in re.finditer(r"(?<![\w.])([+-]\d+\.(\d{2,3}))(?![\d])", rep):
        v, nd = float(m.group(1)), len(m.group(2))
        if not any(round(abs(p), nd) == round(abs(v), nd) for p in pk_vals):
            problems.append(("number", m.group(1)))
    pairs = {(int(a), int(b)) for a, b in re.findall(r"(\d[\d,]*)\s*/\s*(\d[\d,]*)", pk.replace(",", ""))}
    for a, b in re.findall(r"(?<![\w.])(\d[\d,]*)\s*/\s*(\d[\d,]*)(?![\d.])", rep):
        a, b = int(a.replace(",", "")), int(b.replace(",", ""))
        if b >= 10 and (a, b) not in pairs:
            problems.append(("count", f"{a}/{b}"))
    return list(dict.fromkeys(problems))


def _get(url, js=True):
    req = urllib.request.Request(url, headers={"User-Agent": "CRISProg fact-check"})
    for k in range(4):
        try:
            r = urllib.request.urlopen(req, timeout=20).read().decode()
            return json.loads(r) if js else r
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(1.5 * (k + 1))
                continue
            raise
    raise RuntimeError("rate limited")


def check_references(report):
    """Each line of the References section with a PMID / PMC / DOI: does it resolve, do author and year match?"""
    sec = report.split("## References", 1)[1] if "## References" in report else ""
    lines = [ln.strip() for ln in sec.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        pmid = re.search(r"(?:PMID[:\s]*|pubmed\.ncbi\.nlm\.nih\.gov/)(\d{5,9})", ln)
        pmc = re.search(r"(PMC\d{5,9})", ln)
        doi = re.search(r"(10\.\d{4,9}/[^\s)\]>,;]+)", ln)
        if not (pmid or pmc or doi):
            out.append(dict(line=ln[:90], id="", status="no identifier"))
            continue
        try:
            if pmc and not pmid:
                c = _get("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?format=json&ids=" + pmc.group(1))
                p = c["records"][0].get("pmid")
                pmid = re.match(r"(\d+)", str(p)) if p else None
            if pmid:
                pid = pmid.group(1)
                d = _get(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id={pid}")
                d = d["result"].get(pid, {})
                author = (d.get("sortfirstauthor") or "").split(" ")[0]
                year, title, ident = (d.get("pubdate") or "")[:4], d.get("title") or "", f"PMID {pid}"
            else:
                m = _get("https://api.crossref.org/works/" + doi.group(1).rstrip("."))["message"]
                author = ((m.get("author") or [{}])[0].get("family") or "")
                year = str(((m.get("issued", {}).get("date-parts") or [[""]])[0] or [""])[0])
                title, ident = ((m.get("title") or [""])[0]), "doi " + doi.group(1)
            if not (author or title):
                out.append(dict(line=ln[:90], id=ident, status="NOT FOUND"))
                continue
            words = [w for w in re.findall(r"[a-z]{4,}", title.lower())][:8]
            title_ok = bool(words) and sum(w in ln.lower() for w in words) >= 0.6 * len(words)
            author_ok = bool(author) and author.lower() in ln.lower()
            year_ok = bool(year) and year in ln
            ok = (author_ok or title_ok) and (year_ok or not re.search(r"(19|20)\d\d", ln))
            out.append(dict(line=ln[:90], id=ident, status="ok" if ok else
                            f"MISMATCH (source: {author} {year}: {title[:70]})"))
        except Exception as ex:
            out.append(dict(line=ln[:90], id=(pmid or pmc or doi).group(1), status=f"not checked ({type(ex).__name__})"))
        time.sleep(0.4)
    return out


def rule_report(numbers, refs):
    L = [f"- numbers not found in the packet: {len(numbers)}"]
    L += [f"  - {kind}: `{val}`" for kind, val in numbers]
    bad = [r for r in refs if r["status"] != "ok"]
    L.append(f"- references checked: {len(refs)}; not confirmed: {len(bad)}")
    L += [f"  - {r['id'] or '(none)'}: {r['status']} | {r['line']}" for r in bad]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------- AI pass
FACTCHECK_INSTRUCTIONS = """# CRISProg fact-check instructions

You are fact-checking an AI-written report before it is shared. Files in the current folder:
- `report_draft.md`: the report to check
- `packet.md`: the analysis results the report was written from (the only source for numbers)
- `INSTRUCTIONS.md`: the rules the report had to follow
- `factcheck_auto.md`: problems found by rule-based checks (numbers not in the packet, references that did not
  resolve or match)

Check, and fix where needed:
1. Every number, enrichment term, gene name and unit-program claim against `packet.md`. Use the packet's exact
   value; attribute each number to the unit and program it belongs to; remove a claim the packet cannot support.
   Statements about all regulator pairs must use the packet's all-pairs summary, not the notable-pairs table.
2. Every reference: look up the paper (PubMed abstract) and check that it supports the sentence citing it.
   Reword the sentence to what the paper shows, replace the citation with one you have verified, or remove it.
   Never invent a reference, PMID or DOI.
3. The rules in `INSTRUCTIONS.md`, especially how negative and positive controls are interpreted.
4. Every item in `factcheck_auto.md`.

Do not add new findings, sections or speculation, and keep the structure and wording otherwise unchanged.
Reply with the full corrected report (starting with its first heading), then a final section
`## Fact-check log` listing each change as `- was: ... -> now: ... (reason)`, or `No changes needed.`
"""
FACTCHECK_PROMPT = ("Read FACTCHECK_INSTRUCTIONS.md and follow it: check report_draft.md against packet.md and the "
                    "cited papers. Reply with the corrected report and the fact-check log only.")


def ai_factcheck(out, an, backend, required):
    """-> (corrected report, log text) or (None, reason)."""
    from .annotate import _api, _claude_code
    with open(os.path.join(out, "FACTCHECK_INSTRUCTIONS.md"), "w") as fh:
        fh.write(FACTCHECK_INSTRUCTIONS)
    req = tuple(required) + ("## Fact-check log",)
    try:
        if backend == "claude-code":
            text = _claude_code(out, an, prompt=FACTCHECK_PROMPT, required=req, min_words=300, tag="factcheck")
        else:
            files = "\n\n".join(f"=== {f} ===\n" + open(os.path.join(out, f)).read()
                                for f in ("report_draft.md", "packet.md", "factcheck_auto.md"))
            text = _api(out, an, files, system=FACTCHECK_INSTRUCTIONS, required=req, min_words=300)
    except (SystemExit, Exception) as ex:
        return None, str(ex)[:300]
    body, _, flog = text.partition("## Fact-check log")
    return body.strip() + "\n", flog.strip()


# ------------------------------------------------------------------------------------------------- rendering
CSS = """
@page { size: A4; margin: 15mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 10pt; line-height: 1.42;
       color: #1b1b1b; max-width: 900px; margin: 0 auto; padding: 8px 16px; }
h1 { font-size: 17pt; margin: 4pt 0 6pt 0; line-height: 1.25; }
h2 { font-size: 12.5pt; margin: 16pt 0 5pt 0; border-bottom: 1px solid #cfcfcf; padding-bottom: 2pt; }
h3 { font-size: 10.8pt; margin: 11pt 0 3pt 0; }
p, li { margin: 3pt 0; }
ul, ol { margin: 2pt 0 5pt 0; padding-left: 18pt; }
table { border-collapse: collapse; margin: 5pt 0 9pt 0; width: 100%; font-size: 9pt; }
td, th { border: 1px solid #d3d3d3; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #f1f3f6; }
.meta { background: #f4f6f9; border-left: 3px solid #4f74b0; padding: 6pt 9pt; margin: 4pt 0 10pt 0; font-size: 9pt; }
a { color: #2a5a9a; text-decoration: none; word-break: break-all; }
code { font-size: 9pt; }
"""


def _normalize(md):
    """Blank line before lists and 4-space nesting (python-markdown needs both)."""
    out, prev = [], ""
    for line in md.splitlines():
        m = re.match(r"^( *)([-*]|\d+\.) ", line)
        if m:
            line = " " * (len(m.group(1)) * 2) + line.lstrip(" ")
            if prev.strip() and not re.match(r"^ *([-*]|\d+\.) ", prev) and not prev.startswith("|"):
                out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


def _chrome():
    for c in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        if shutil.which(c):
            return shutil.which(c)
    mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    return mac if os.path.exists(mac) else None


def render(out, report_md, name, note):
    """report.html (+ report.pdf when Chrome/Chromium or WeasyPrint is available). Returns the files written."""
    try:
        import markdown
    except ImportError:
        log("rendering skipped: the Python package 'markdown' is not installed (pip install markdown)")
        return []
    body = markdown.markdown(_normalize(report_md), extensions=["tables", "sane_lists"])
    if not report_md.lstrip().startswith("# "):
        body = f"<h1>CRISProg report: {name}</h1>" + body
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>CRISProg report: {name}</title>"
            f"<style>{CSS}</style></head><body><div class='meta'>{note}</div>{body}</body></html>")
    hp = os.path.join(out, "report.html")
    with open(hp, "w") as fh:
        fh.write(html)
    files, pdf = [hp], os.path.join(out, "report.pdf")
    chrome = _chrome()
    try:
        if chrome:
            subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                            f"--print-to-pdf={pdf}", "file://" + os.path.abspath(hp)],
                           check=True, capture_output=True, timeout=180)
        else:
            from weasyprint import HTML
            HTML(hp).write_pdf(pdf)
        files.append(pdf)
    except Exception as ex:
        log(f"report.pdf not written ({type(ex).__name__}: install Chrome/Chromium or weasyprint); report.html is there")
    return files


# ------------------------------------------------------------------------------------------------- driver
def finish_report(cfg, out, draft, required):
    """Fact-check the draft, write report.md / factcheck.md, render. Returns the final report text."""
    an = cfg["annotation"]
    with open(os.path.join(out, "report_draft.md"), "w") as fh:
        fh.write(draft)
    packet = open(os.path.join(out, "packet.md")).read()
    n1 = check_numbers(draft, packet)
    r1 = check_references(draft)
    auto1 = rule_report(n1, r1)
    with open(os.path.join(out, "factcheck_auto.md"), "w") as fh:
        fh.write("# Rule-based checks of report_draft.md\n\n" + auto1 + "\n")
    log(f"fact-check (rule-based) of the draft: {len(n1)} numbers not in the packet; "
        f"{sum(r['status'] != 'ok' for r in r1)} of {len(r1)} references not confirmed")
    final, flog, how = draft, "", "rule-based checks only (annotation.factcheck is off)"
    if an["factcheck"] and an["backend"] in ("claude-code", "api"):
        log("fact-check (AI): checking every claim against the packet and the cited papers")
        fixed, flog = ai_factcheck(out, an, an["backend"], required)
        if fixed:
            final, how = fixed, f"rule-based checks, then an AI fact-check pass ({an['backend']})"
        else:
            how = f"rule-based checks only (the AI fact-check pass failed: {flog})"
            log(f"WARNING: AI fact-check failed ({flog}); report.md is the unchecked draft")
            flog = ""
    n2 = check_numbers(final, packet) if final is not draft else n1
    r2 = check_references(final) if final is not draft else r1
    auto2 = rule_report(n2, r2)
    L = [f"# Fact-check of the CRISProg report ({cfg['name']})", "", f"Method: {how}.", "",
         "## Rule-based checks of the draft", "", auto1, "",
         "## Changes made by the AI fact-check", "", flog or "(none)", "",
         "## Rule-based checks of the final report", "", auto2, "",
         "Numbers are compared with the results packet; references with PubMed / Crossref. A number flagged here can "
         "still be correct (e.g. a rounded or derived value); flagged items are worth a look before sharing.", ""]
    with open(os.path.join(out, "factcheck.md"), "w") as fh:
        fh.write("\n".join(L))
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write(final)
    log(f"report written: {os.path.join(out, 'report.md')}; fact-check summary: factcheck.md "
        f"(final: {len(n2)} numbers not in the packet, {sum(r['status'] != 'ok' for r in r2)} references not confirmed)")
    if an["render"]:
        note = (f"Generated by CRISProg for run <b>{cfg['name']}</b> on {datetime.date.today().isoformat()}. "
                f"Drafted by AI from the analysis results; fact-checked ({how}). Details: factcheck.md.")
        files = render(out, final, cfg["name"], note)
        if files:
            log("rendered: " + ", ".join(os.path.basename(f) for f in files))
    return final
