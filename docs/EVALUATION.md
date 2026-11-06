# Evaluation Strategy

## 1. Automated tests (run on every change)

`python -m pytest -q` — see `tests/`. Deterministic, no network, no LLM.

| Area | Test file | What it proves |
|---|---|---|
| Source verification | `test_sources.py` | Parsers for OpenAlex, Semantic Scholar, Crossref, arXiv, PubMed produce correct ids/types/abstracts; 429s surface as *rate_limited*, not crashes; retraction notices are not ingested as papers; responses are cached. |
| Duplicate detection | `test_dedupe.py` | DOI/arXiv/title variants merge with provenance; similar titles with different DOIs do not; published versions supersede preprints. |
| Evidence traceability | `test_traceability.py`, `test_validation.py::test_gap_without_verifiable_evidence_is_dropped`, `test_pipeline_e2e.py` | Invented paper ids are stripped everywhere and logged; gaps without evidence are dropped; prose references are scrubbed. |
| Gap validation | `test_validation.py` | Addressed → rejected with counter-evidence; partially addressed → narrowed and re-validated; round cap honoured; all-searches-failed → *cannot validate*. |
| Citation verification | `test_verification.py` | DOI resolution, title mismatch flagging, retraction via Crossref relation and OpenAlex. |
| LLM layer | `test_llm.py`, `test_llm_openai.py` | Anthropic request shape and workspace header; OpenAI-compatible schema strictification, structured-output fallback chain, validation retry, provider auto-detection. |
| Resume | `test_pipeline_e2e.py` | Completed stages are skipped; dedupe index rebuilt from state. |

## 2. Reference topics for manual evaluation (require credentials and network)

Run each topic and check the listed expectations by hand. Record the run id and outcome in a table below.

### R1 — "Prompt injection attacks on tool-using LLM agents" (from 2023)
* The landscape must contain a theme on benchmarks/evaluation; the corpus should include at least two of
  the well-known agent prompt-injection benchmarks (search the report for "AgentDojo", "InjecAgent", "BIPIA").
* If the gap generator proposes "no benchmark exists for indirect prompt injection against agents",
  validation must return **addressed** and the report must list it under *rejected*.
* Preprints must be labelled; most of this corpus is arXiv.

### R2 — "Explainable AI for IoT intrusion detection" (from 2019)
* Evidence statements should surface dataset reuse (CICIDS2017, NSL-KDD, BoT-IoT, TON_IoT appear across papers);
  the claim-support check must mark the dataset-reuse statement *supported*.
* A plausible generalisability/evaluation gap about cross-dataset evaluation should appear; its status should be
  *partially addressed* or *preliminary_evidence*, not *limited_evidence_found*, because cross-dataset XAI-IDS studies exist.

### R3 — "Securing the Model Context Protocol in agentic AI systems" (from 2024)
* Small, fast-moving literature: the run must not fabricate volume. Expect few peer-reviewed items, many preprints;
  the executive summary must state the counts and flag the preprint share.
* Validation searches must include alternative terminology ("tool poisoning", "agent tool authorization",
  "function calling security") — check the search log in §8.

### R4 — Deduplication spot check
* Pick any paper in the References that appears from both arXiv and a publisher (DOI + arXiv id on one line).
  Confirm there is exactly one entry for it and that its type is not *preprint* if a published version was found.

### R5 — Negative control
* Topic: "Hand-washing and infection control in hospitals" with `--sources pubmed,openalex`.
  A mature field: the agent should either produce narrow, well-supported gaps or report that candidates were
  rejected as addressed. Any *limited_evidence_found* here is a red flag for weak counter-searching.

## 3. Rubric for reviewing a generated report

Score each 0–2 (0 = fails, 1 = partial, 2 = meets):

1. Every substantive claim in §2 and §4 carries at least one resolvable citation.
2. Counts in §1 are mutually consistent with the search log in §8.
3. Every candidate gap documents counter-queries actually executed (visible in the search log).
4. No gap is labelled "novel"; statuses are from the six-way scheme and confidence has an explanation.
5. Rejected candidates are shown with their counter-evidence.
6. The recommendation makes no performance claims.
7. Limitations list includes unsearched databases, failed queries, and the abstract-only caveat.

## 4. Log of manual evaluations

| Date | Topic | Run id | Reviewer | Score /14 | Notes |
|---|---|---|---|---|---|
| | | | | | |
