# Research Gap Discovery Agent — full usage reference

An autonomous, tool-using research agent that takes a topic, searches academic literature
through real APIs, builds a research landscape, proposes candidate research gaps, **tries to
disprove each one with targeted counter-searches**, ranks the survivors with a transparent
rubric, verifies citations, and writes an evidence-backed Research Gap Discovery Report.

Design rationale, data model and acceptance criteria: [`DESIGN.md`](DESIGN.md).
Evaluation strategy and reference topics: [`EVALUATION.md`](EVALUATION.md).

## Quick start

Credentials go in a `.env` file in the project folder (copy `.env.example`). Both the CLI and the GUI load it
on start; variables already exported in the shell take precedence.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,gui]"

cp .env.example .env                        # then fill in the values; .env is git-ignored and loaded automatically

research-gap-agent "Securing Model Context Protocol (MCP) in agentic AI systems" \
    --discipline "computer security" --from-year 2022 --out report.md
```

Progress is printed to stderr; the report, full run state, and reference exports (`references.bib`,
`references.ris`, `corpus.csv` for Zotero/Mendeley/EndNote) land in `runs/<run_id>/`. The report comes as `report.html` (styled, with navigation and collapsible evidence trails), `report.md` (same content for GitHub/VS Code) and `appendix.md` (full evidence matrix, complete search log, verification log, every record).

### GUI

```bash
streamlit run app.py
```

A single-page interface designed for people who do not use the command line. The sidebar is a three-step setup
(AI model key, optional database keys, previous investigations); keys are saved to `.env`. The form uses plain
labels and three depth presets (Quick ≈ 30 papers / 1 validation round, Standard ≈ 60 / 2, Thorough ≈ 120 / 3) plus
a Custom option that exposes the exact numbers under *Advanced settings*. While a run is in progress the page shows a
twelve-step progress bar with a plain-language description of the current step; the raw log sits in a collapsible
*Technical log*. The Results tab shows the recommended next step, the surviving gaps as ranked cards with status badges
(open gap, early work only, conflicting findings, partly addressed, could not verify), the rejected candidates with
their counter-evidence, the full HTML report, a filterable paper table with links, and the search log. The theme lives
in `.streamlit/config.toml`.
Interrupted runs resume from the last completed stage:

```bash
research-gap-agent --resume 20261102-091500-ab12cd
```

Useful options: `--sources openalex,semantic_scholar,crossref,arxiv,pubmed,google_scholar`, `--max-papers 60`,
`--max-full-text 20` (open-access PDFs to read; 0 disables), `--validation-rounds 2`, `--per-source-limit 25`,
`--until landscape` (stop early), `--model claude-opus-5-5`.

Check API reachability before a long run:

```bash
RGA_MAILTO=you@example.org python scripts/smoke_sources.py "prompt injection LLM agents"
```

## Choosing an LLM provider

The pipeline is model-agnostic. Set one provider's key in `.env` (or the GUI sidebar) and it is auto-detected;
`RGA_LLM_PROVIDER` / `--provider` forces a choice, `RGA_LLM_MODEL` / `--model` picks the model.

| Provider | Key | Notes |
|---|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` | native structured outputs, prompt caching, server-side refusal fallbacks; `ANTHROPIC_WORKSPACE_ID` if your key is organisation-wide |
| `openai` | `OPENAI_API_KEY` | strict JSON-schema outputs; `reasoning_effort` mapped from the agent's effort levels |
| `azure` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | model = deployment name |
| `gemini` | `GEMINI_API_KEY` | through Google's OpenAI-compatible endpoint |
| `mistral`, `groq`, `together`, `openrouter`, `deepseek`, `xai` | `<PROVIDER>_API_KEY` | OpenAI-compatible endpoints |
| `ollama`, `lmstudio` | none | local servers; set `RGA_LLM_MODEL` to a model you have pulled |
| `openai-compatible` | `RGA_LLM_API_KEY` | any other server: set `RGA_LLM_BASE_URL` |

Structured output is negotiated per server: strict JSON schema → JSON schema → JSON mode with the schema in the
prompt → plain JSON extraction, and every reply is validated against the Pydantic schema with one corrective
retry. Default model ids live in `../research_gap_agent/providers.py` and may need updating as vendors rename models.
Quality varies by model: the gap-validation and ranking stages benefit from a strong reasoning model; small local
models will produce weaker, less well-cited reports.

```bash
research-gap-agent "topic" --provider openai --model gpt-5
research-gap-agent "topic" --provider ollama --model llama3.1
research-gap-agent "topic" --provider openai-compatible --base-url http://localhost:8000/v1 --model my-model
```

## What the agent does

| Stage | Component | What happens |
|---|---|---|
| plan | Research Planner | Decomposes the topic (concepts, synonyms, adjacent areas in/out of scope), writes 6–14 search queries, discloses assumptions. |
| discover | Literature Discovery | Runs every query on every enabled source; logs source, query, date, period, counts and status; deduplicates by DOI / arXiv / PMID / OpenAlex / S2 id / normalised title. |
| screen | Literature Analysis | Title/abstract relevance screening with a recorded reason per record. |
| fulltext | Literature Discovery | Fetches open-access PDFs (arXiv, OpenAlex OA link, Unpaywall by DOI), extracts text with pypdf, caches it; paywalled papers stay abstract-only and are counted separately. |
| extract | Literature Analysis | One structured evidence row per paper (objective, method, findings, author-acknowledged limitations, datasets, metrics, assumptions, environment). |
| landscape | Literature Analysis | Themes, cross-paper evidence statements (marked *direct* or *interpretation*), conflicts, foundational and recent papers. |
| generate_gaps | Gap Generation | 3–8 candidates across the ten gap categories, only where evidence supports them, plus ideas considered and rejected. |
| validate | Gap Validation | Per gap: counter-queries with alternative terminology → searches on all sources + citation chasing (Semantic Scholar) → gap-specific screening → adversarial judgement into one of six statuses. *Addressed* gaps are rejected; *partially addressed* gaps are narrowed and re-validated. |
| rank | Research Ranking | Qualitative high/medium/low ratings on nine criteria with rationale; deterministic ordering. |
| recommend | — | State-of-the-art prose, the 14-part recommended direction for the top gap, prioritised reading list. |
| verify | Citation Verification | Strips any paper id the LLM invented, resolves DOIs at Crossref and compares titles, surfaces retraction flags (OpenAlex/Crossref), checks that cited abstracts support the evidence statements used by surviving gaps. |
| report | Coordinator | Renders the eight-section Markdown report with clickable citations and a full search log. |

## Integrity guarantees that are enforced in code (not just prompted)

* Every paper reference in every LLM output is checked against the corpus; unknown ids are removed and logged in the report's verification log.
* A candidate gap with no verifiable supporting paper is dropped.
* A gap whose validation searches all failed is forced to *cannot validate*; "limited evidence found" is never upgraded to novelty.
* Databases that were not queried are listed as *not searched* (Google Scholar, IEEE Xplore, ACM DL; PubMed unless enabled). Rate-limited or failed queries are logged per query.
* Preprints are labelled as not peer reviewed; retracted papers are flagged.
* Counts of retrieved / unique / screened / included / analysed / full-text records are reported separately.

## Sources and scholarly tools

| Tool | Used for | Access |
|---|---|---|
| OpenAlex | primary search, retraction flags, OA links | free; `OPENALEX_API_KEY` recommended (unauthenticated use shares a daily budget per network IP and is skipped once exhausted); `RGA_MAILTO` for the polite pool |
| Semantic Scholar | primary search, citation chasing during validation | free; `S2_API_KEY` strongly recommended |
| Crossref | primary search, DOI verification, retraction relations | free; `RGA_MAILTO` for the polite pool |
| arXiv | primary search, preprint PDFs | free |
| PubMed (E-utilities) | biomedical search (`--sources ... ,pubmed`) | free; `NCBI_API_KEY` optional |
| Google Scholar via SerpAPI | extra coverage (`--sources ... ,google_scholar`) | paid; `SERPAPI_KEY` |
| GitHub repository search | existing tools and implementations as prior work (`--sources ... ,github`), typed `software` | free; `GITHUB_TOKEN` optional |
| Unpaywall | open-access PDF location by DOI | free; needs `RGA_MAILTO` |
| pypdf | PDF text extraction | bundled |
| BibTeX / RIS / CSV export | Zotero, Mendeley, EndNote import | bundled |

Google Scholar scraping libraries (e.g. `scholarly`) are deliberately not used: they violate Google's terms
and get blocked. IEEE Xplore and ACM DL are not accessed (no authorised API); DBLP's API currently sits behind a
bot-challenge page and is not usable programmatically.

## Not implemented (stated honestly)

* Paywalled full texts: only open-access PDFs are read. The report states exactly which papers had full text.
* IEEE Xplore, ACM DL, DBLP search.

## Cost

Defaults to `claude-opus-5-5` for every stage. A typical run (60 papers analysed, 5 gaps, 2 validation rounds) makes roughly 40–60 model calls; the report's final section prints token usage. Lower `--max-papers` or `--validation-rounds` to reduce cost.

## Tests

```bash
python -m pytest -q
```

Covers deduplication, every source parser (mocked HTTP), rate-limit handling, traceability stripping,
validation-loop control, DOI/retraction verification, both LLM backends (stubbed clients), full-text retrieval,
reference exports, `.env` handling, the Streamlit GUI (headless harness), and a full scripted end-to-end run
with resume.

## Looking for a technical tool rather than a research question

Pass `--goal tool` (GUI: "What do you want to end up with?") and, optionally, describe what you already built with
`--existing-work @README.md`. The planner then searches for existing tools and implementations as well as papers (add the
`github` source), gap generation prefers gaps a buildable artefact would fill and says how each differs from your own work
and from the closest existing tools, validation treats maintained repositories as prior work, and the recommendation is
written as a build plan (inputs, outputs, components, differentiation, evaluation, minimum viable version).

```bash
research-gap-agent "securing Model Context Protocol (MCP)" --goal tool --existing-work @../mcp-trustcheck/README.md \
    --sources openalex,semantic_scholar,crossref,arxiv,github
```
