# Research Gap Discovery Agent — Design

This document answers the seven pre-implementation questions in the project brief:
MVP and acceptance criteria, workflow, APIs, data model, evidence storage and linking,
gap generation and validation, and the evaluation strategy.

## 1. MVP and acceptance criteria

**MVP scope.** A command-line tool that takes a topic (plus optional discipline, period,
methodology and priority) and, without further user input, produces a Markdown
*Research Gap Discovery Report* with the eight sections required by the brief.

The MVP analyses **title + abstract + bibliographic metadata**, plus the **open-access full
text** of up to `--max-full-text` papers (arXiv, OpenAlex OA link, Unpaywall by DOI; text extracted
with pypdf). Paywalled papers remain abstract-only and the report states exactly which papers had
full text. The UI is a command line plus a Streamlit single-page GUI (`app.py`).

**Acceptance criteria (all must hold for a run to be considered successful):**

| # | Criterion | How it is checked |
|---|---|---|
| A1 | The topic is decomposed into concepts, alternative terminology and at least 6 search queries, with assumptions disclosed. | `ResearchPlan` persisted in run state; shown in report §8. |
| A2 | At least two academic sources are actually queried; every query is logged with source, date, record count, period and status (ok / rate-limited / error). | `SearchRecord` list; report §8 lists them. Sources that were not queried are listed as *not searched*. |
| A3 | Retrieved records are deduplicated across sources by DOI, arXiv id, PubMed id, OpenAlex id, Semantic Scholar id and normalised title+year; provenance of every merged record is kept. | `tests/test_dedupe.py`; `Paper.source_records`. |
| A4 | Every evidence statement, candidate gap, validation result and recommendation cites only paper ids that exist in the corpus. Any id emitted by the LLM that does not exist is removed and logged. | `tests/test_traceability.py`; verification report in state. |
| A5 | Every candidate gap goes through at least one validation round of targeted counter-searches with alternative terminology, and is classified into one of six statuses (addressed / partially addressed / conflicting / preliminary / limited evidence / cannot validate). A gap whose validation searches all failed is forced to *cannot validate*. | `tests/test_validation.py`. |
| A6 | A gap classified as *addressed* is rejected and reported as rejected with its counter-evidence; a *partially addressed* gap with a narrowed revision is re-validated (bounded number of rounds). | `tests/test_validation.py`. |
| A7 | Cited DOIs are checked against Crossref; retraction flags from OpenAlex are surfaced; preprints and peer-reviewed items are labelled separately. | `tests/test_verification.py`. |
| A8 | An interrupted run can be resumed from the last completed stage without repeating API calls. | `tests/test_pipeline_e2e.py` (resume test). |
| A9 | The report contains all eight required sections, distinguishes retrieved / unique / screened / analysed counts, and states limitations. | `tests/test_pipeline_e2e.py`. |

## 2. Research workflow

The workflow is a linear pipeline with one bounded loop (validation → revision →
re-validation). A graph framework was judged unnecessary: the only branching is inside
the validation loop, which a plain coordinator handles with explicit state.

```
topic ──► 1 plan ──► 2 discover ──► 3 screen ──► 3b fulltext ──► 4 extract ──► 5 landscape
   ──► 6 generate gaps ──► 7 validate ◄──┐  (revised gaps re-enter, max N rounds)
                              └──────────┘
   ──► 8 rank ──► 9 recommend + reading list ──► 10 verify citations ──► 11 report
```

Each stage reads and writes a single `RunState` object persisted to
`runs/<run_id>/state.json` after the stage completes. Re-running with `--resume`
skips completed stages. HTTP responses from academic APIs are cached on disk keyed
by request URL so retries and resumes do not re-hit rate limits.

Stage responsibilities map onto the components named in the brief:

| Brief component | Module |
|---|---|
| Research Planner | `agents/planner.py` |
| Literature Discovery Agent | `discovery.py` + `sources/*` |
| Literature Analysis Agent | `agents/screening.py`, `agents/analysis.py` |
| Gap Generation Agent | `agents/gap_generation.py` |
| Gap Validation Agent | `agents/validation.py` |
| Research Ranking Agent | `agents/ranking.py` |
| Citation Verification Agent | `agents/verification.py` |
| Research Coordinator | `coordinator.py` |

### LLM backends

`llm.py` defines one protocol, `structured(task, system, user, output_model, effort)`. Two backends implement it:
`AnthropicLLM` (native structured outputs, caching, workspace header, refusal fallbacks) and
`OpenAICompatibleLLM` (`llm_openai.py`), which serves OpenAI, Azure, Gemini, Mistral, Groq, Together, OpenRouter,
DeepSeek, xAI, Ollama, LM Studio and any other OpenAI-compatible server through presets in `providers.py`.
The factory `build_llm()` picks the backend from `RGA_LLM_PROVIDER` or from whichever key is present.
All prompts are provider-neutral; nothing downstream depends on the vendor.

## 3. Academic APIs and access requirements

| Source | Endpoint | Auth | Rate limit handling | Status in MVP |
|---|---|---|---|---|
| OpenAlex | `GET /works?search=…&filter=from_publication_date:…` | none; `mailto` for polite pool | 429 → exponential backoff, honour `Retry-After` | implemented |
| Semantic Scholar | `GET /graph/v1/paper/search`, `GET /graph/v1/paper/{id}/citations` | optional `x-api-key` (`S2_API_KEY`) | shared unauthenticated pool is throttled aggressively; backoff then record *rate_limited* | implemented |
| Crossref | `GET /works?query.bibliographic=…`, `GET /works/{doi}` | none; `mailto` for polite pool | backoff | implemented (search + DOI verification) |
| arXiv | `GET export.arxiv.org/api/query` (Atom) | none | ≥3 s between requests | implemented |
| PubMed | E-utilities `esearch` + `efetch` | optional `NCBI_API_KEY` | ≤3 req/s | implemented, enabled with `--sources pubmed` |
| Google Scholar | SerpAPI `engine=google_scholar` | `SERPAPI_KEY` (paid) | 1 req/s | implemented, optional (`--sources ...,google_scholar`); scraping libraries deliberately not used |
| Unpaywall | `GET /v2/{doi}?email=` | email | none needed | implemented (OA PDF discovery for full text) |
| DBLP | `search/publ/api` | none | — | **not searched**: endpoint is behind a bot-challenge page |
| IEEE Xplore / ACM DL | require institutional API keys | — | — | **not searched**; stated in report |

Retraction information comes from OpenAlex `is_retracted` and Crossref `update-to`
records. Preprint vs peer-reviewed labelling uses OpenAlex `type` / source type, Crossref
`posted-content`, arXiv origin, and Semantic Scholar venue.

## 4. Data model

All models are Pydantic (`research_gap_agent/models.py`). Key entities:

* `Paper` — one deduplicated publication. Stable internal id `P0001…`. Holds
  identifiers (DOI, arXiv, PMID, OpenAlex, S2), bibliographic fields, publication type,
  retraction flag, and `source_records` (which source/query returned it).
* `SearchRecord` — one executed query on one source, with status, counts and purpose
  (`primary` or `validation:<gap id>`).
* `ResearchPlan` — decomposition of the topic and the query set, with disclosed
  assumptions and explicitly excluded adjacent areas.
* `EvidenceRow` — structured extraction for one paper (objective, methodology,
  findings, author-acknowledged limitations, datasets, metrics, environment).
* `EvidenceStatement` — a cross-paper observation, marked `direct` (stated by the
  papers) or `interpretation` (agent inference), with the supporting `paper_ids`.
* `Theme` — a cluster of papers with approaches, strengths, weaknesses, open questions.
* `CandidateGap` — id `G01…`, category A–J, established knowledge, what is missing,
  supporting evidence/paper ids, significance, potential contribution.
* `ValidationRound` / `ValidatedGap` — the counter-queries executed, new papers found,
  judged status, counter-evidence paper ids, revised gap, confidence + explanation,
  feasibility.
* `RankedGap` — qualitative ratings per rubric criterion with rationale and main risk.
* `Recommendation` — the 14 fields of report §6.
* `VerificationReport` — hallucinated-id removals, DOI checks, retraction flags, claim
  support checks.
* `RunState` — everything above plus stage checkpoints, LLM call log and limitations.

## 5. Evidence storage and linking

* Every LLM output that refers to a paper must do so by internal id (`P0012`). Prompts
  include the id with each paper and instruct the model to use only those ids.
* After every LLM call the coordinator runs `strip_unknown_paper_ids`, which removes
  any id not in the corpus and records the removal in the verification report. A
  statement left with zero supporting ids is dropped.
* Evidence statements are the join table between papers and gaps: a gap cites
  evidence statement ids *and* paper ids; the report renders both.
* The report's citation renderer resolves `[P0012]` to the paper's DOI/arXiv/OpenAlex
  URL, so every claim is one click from its source.

## 6. Gap generation and validation

**Generation.** The gap agent receives the themes, the evidence statements and the
extraction rows, and is asked to propose candidates across the ten categories *only
where evidence supports one*, returning also the ideas it considered and rejected.
It must separate what papers state from what it infers.

**Validation (adversarial).** For each candidate the validation agent:

1. Writes 3–6 counter-queries using alternative terminology, related concepts and
   different methodological framings, aimed at *finding work that already addresses
   the gap*.
2. The coordinator runs those queries on all enabled sources, and additionally pulls
   papers citing the gap's supporting papers from Semantic Scholar when available.
3. New papers are screened for relevance to the specific gap.
4. The agent judges the gap against the new evidence and returns a status, the
   counter-evidence ids, reasoning, and (optionally) a narrowed revision.
5. Coordinator rules: *addressed* → rejected; *partially addressed* with a revision →
   revised gap enters the next round (max `--validation-rounds`, default 2); if every
   counter-search failed → status forced to *cannot validate*; the agent is never told
   that "no results" means "no prior work".

## 7. Evaluation strategy

* **Unit tests** (deterministic, no network, no LLM): deduplication, source parsers,
  traceability stripping, validation loop control, DOI/retraction verification.
* **End-to-end test** with a scripted LLM and fixture sources: verifies report
  structure, counts, rejection of addressed gaps, resume.
* **Manual reference topics** (`docs/EVALUATION.md`): topics where the state of the
  literature is well known, with expected outcomes the report must reproduce, e.g.
  a candidate gap "no benchmark for prompt-injection attacks on LLM agents" must come
  back *addressed* because such benchmarks exist (AgentDojo, InjecAgent, BIPIA); a
  deduplication check that the same arXiv paper and its published version merge.
* **Live smoke test** script (`scripts/smoke_sources.py`) that runs one query per
  source and prints counts, used to confirm API access before a full run.
