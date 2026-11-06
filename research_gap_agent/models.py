"""Structured data models for papers, evidence, gaps, validation and run state.

Models that are produced by the LLM (suffixed ``Output``) deliberately avoid dict
fields and datetimes so that their JSON schema is accepted by the structured-output
API. Internal state models are free to use richer types.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Papers and searches
# ---------------------------------------------------------------------------

PublicationType = Literal[
    "preprint", "journal-article", "conference-paper", "article", "book-chapter", "review", "software", "other", "unknown"
]  # "software" = a code repository or released tool (e.g. a GitHub repository), not a publication  # "article" = published article whose venue type (journal/conference) could not be determined


class SourceRecord(BaseModel):
    """Provenance: which source and query returned this paper."""

    source: str
    source_id: str
    query_id: str
    url: Optional[str] = None
    retrieved_at: datetime = Field(default_factory=utcnow)


class RawRecord(BaseModel):
    """A record as returned by one source, before deduplication."""

    source: str
    source_id: str
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    authors: list[str] = Field(default_factory=list)
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    openalex_id: Optional[str] = None
    s2_id: Optional[str] = None
    venue: Optional[str] = None
    publication_type: PublicationType = "unknown"
    is_retracted: Optional[bool] = None
    cited_by_count: Optional[int] = None
    oa_url: Optional[str] = None
    url: Optional[str] = None


class ScreeningDecision(BaseModel):
    include: bool
    reason: str
    stage: Literal["primary", "validation"] = "primary"
    gap_id: Optional[str] = None


class DoiCheck(BaseModel):
    doi: str
    resolved: bool
    title_match: Optional[bool] = None
    crossref_title: Optional[str] = None
    note: Optional[str] = None


class Paper(BaseModel):
    id: str
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    authors: list[str] = Field(default_factory=list)
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    openalex_id: Optional[str] = None
    s2_id: Optional[str] = None
    venue: Optional[str] = None
    publication_type: PublicationType = "unknown"
    is_retracted: Optional[bool] = None
    cited_by_count: Optional[int] = None
    oa_url: Optional[str] = None
    url: Optional[str] = None
    source_records: list[SourceRecord] = Field(default_factory=list)
    screening: Optional[ScreeningDecision] = None
    doi_check: Optional[DoiCheck] = None
    full_text_chars: int = 0
    full_text_source: Optional[str] = None  # e.g. "arxiv", "unpaywall", "oa_url"
    full_text_error: Optional[str] = None

    @property
    def has_full_text(self) -> bool:
        return self.full_text_chars > 0

    @property
    def is_preprint(self) -> bool:
        return self.publication_type == "preprint"

    @property
    def sources(self) -> list[str]:
        return sorted({r.source for r in self.source_records})

    def best_url(self) -> Optional[str]:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        if self.openalex_id:
            return f"https://openalex.org/{self.openalex_id}"
        if self.pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        return self.url or self.oa_url

    def short_cite(self) -> str:
        first = self.authors[0].split()[-1] if self.authors else "Anon."
        et_al = " et al." if len(self.authors) > 1 else ""
        year = self.year or "n.d."
        return f"{first}{et_al}, {year}"


SearchStatus = Literal["ok", "rate_limited", "error", "skipped"]


class SearchRecord(BaseModel):
    id: str
    source: str
    query: str
    purpose: str  # "primary" | "validation:<gap_id>" | "citations:<paper_id>"
    executed_at: datetime = Field(default_factory=utcnow)
    from_year: Optional[int] = None
    to_year: Optional[int] = None
    n_retrieved: int = 0
    n_new_unique: int = 0
    paper_ids: list[str] = Field(default_factory=list)
    status: SearchStatus = "ok"
    error: Optional[str] = None
    limitation: Optional[str] = None


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class SearchQuery(BaseModel):
    query: str
    rationale: str


class ResearchQuestion(BaseModel):
    question: str
    rationale: str


class ResearchPlanOutput(BaseModel):
    """LLM output of the Research Planner."""

    assumptions: list[str] = Field(description="Assumptions made because the user did not specify them")
    core_concepts: list[str]
    related_areas_in_scope: list[str]
    related_areas_excluded: list[str] = Field(description="Adjacent areas judged out of scope, each with a short reason")
    alternative_terminology: list[str] = Field(description="Synonyms, abbreviations and alternative names")
    technical_approaches: list[str]
    foundational_frameworks: list[str]
    research_questions: list[ResearchQuestion]
    search_queries: list[SearchQuery] = Field(description="Between 6 and 14 keyword queries suitable for academic search APIs")


class ResearchPlan(ResearchPlanOutput):
    topic: str
    discipline: Optional[str] = None
    from_year: Optional[int] = None
    to_year: Optional[int] = None
    methodology_preference: Optional[str] = None
    priority: Optional[str] = None


# ---------------------------------------------------------------------------
# Screening and extraction
# ---------------------------------------------------------------------------


class ScreeningItem(BaseModel):
    paper_id: str
    include: bool
    reason: str


class ScreeningOutput(BaseModel):
    decisions: list[ScreeningItem]


Environment = Literal["theoretical", "simulation", "laboratory", "real-world", "mixed", "unknown"]


class EvidenceRow(BaseModel):
    """Structured extraction for one paper (one row of the evidence matrix)."""

    paper_id: str
    research_objective: str
    methodology: str
    main_findings: str
    limitations: list[str] = Field(description="Limitations acknowledged by the authors, or 'none stated'")
    datasets_or_benchmarks: list[str]
    evaluation_metrics: list[str]
    key_assumptions: list[str]
    environment: Environment
    relevance_to_potential_gaps: str


class ExtractionOutput(BaseModel):
    rows: list[EvidenceRow]


EvidenceKind = Literal["direct", "interpretation"]


class EvidenceStatement(BaseModel):
    id: str
    statement: str
    kind: EvidenceKind = Field(description="'direct' if stated by the papers; 'interpretation' if inferred by the agent across papers")
    paper_ids: list[str]
    theme_id: Optional[str] = None


class Theme(BaseModel):
    id: str
    name: str
    description: str
    paper_ids: list[str]
    established_approaches: list[str]
    important_contributions: list[str]
    strengths: list[str]
    weaknesses: list[str]
    unresolved_questions: list[str]


class ConflictingFinding(BaseModel):
    description: str
    paper_ids_side_a: list[str]
    paper_ids_side_b: list[str]


class LandscapeOutput(BaseModel):
    themes: list[Theme]
    evidence_statements: list[EvidenceStatement] = Field(
        description="Cross-paper observations: shared assumptions, dataset reuse, evaluation-only-in-lab, excluded scenarios, incomplete metrics, repeated limitations, trends"
    )
    conflicting_findings: list[ConflictingFinding]
    established_knowledge: list[str]
    increasing_attention: list[str]
    foundational_paper_ids: list[str]
    recent_key_paper_ids: list[str]


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------


class GapCategory(str, Enum):
    knowledge = "knowledge"
    methodological = "methodological"
    empirical = "empirical"
    evaluation = "evaluation"
    generalizability = "generalizability"
    integration = "integration"
    contradiction = "contradiction"
    deployment = "deployment"
    interdisciplinary = "interdisciplinary"
    temporal = "temporal"


class CandidateGapOutput(BaseModel):
    title: str
    category: GapCategory
    what_is_established: str
    what_is_missing: str
    supporting_evidence_ids: list[str] = Field(description="Ids of evidence statements (E###) supporting this gap")
    supporting_paper_ids: list[str] = Field(description="Ids of papers (P####) whose findings or limitations suggest this gap")
    why_it_matters: str
    potential_contribution: str
    direct_vs_interpretation: str = Field(description="Which parts are directly stated by papers and which are the agent's interpretation")


class RejectedIdea(BaseModel):
    idea: str
    why_not: str


class GapGenerationOutput(BaseModel):
    gaps: list[CandidateGapOutput]
    considered_and_rejected: list[RejectedIdea]


class CandidateGap(CandidateGapOutput):
    id: str
    revision_of: Optional[str] = None
    revision_number: int = 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidationStatus(str, Enum):
    addressed = "addressed"
    partially_addressed = "partially_addressed"
    conflicting_evidence = "conflicting_evidence"
    preliminary_evidence = "preliminary_evidence"
    limited_evidence_found = "limited_evidence_found"
    cannot_validate = "cannot_validate"


class CounterQuery(BaseModel):
    query: str
    rationale: str
    strategy: Literal["alternative_terminology", "related_concept", "different_methodology", "recent_work", "venue_specific"]


class CounterQueryOutput(BaseModel):
    queries: list[CounterQuery] = Field(description="3 to 6 queries designed to FIND work that already addresses the gap")


class CounterEvidence(BaseModel):
    paper_id: str
    relationship: Literal["addresses_fully", "addresses_partially", "conflicts", "preliminary", "tangential"]
    note: str


class Feasibility(BaseModel):
    data_availability: str
    resources_and_tools: str
    technical_requirements: str
    evaluation_challenges: str
    overall: Literal["high", "medium", "low"]


Confidence = Literal["high", "medium", "low"]


class ValidationJudgementOutput(BaseModel):
    status: ValidationStatus
    reasoning: str
    counter_evidence: list[CounterEvidence]
    revised_gap: Optional[CandidateGapOutput] = Field(
        default=None,
        description="A narrowed or reframed gap if the evidence shows the original is too broad or partly solved; null otherwise",
    )
    confidence: Confidence
    confidence_explanation: str
    feasibility: Feasibility
    what_could_not_be_established: str


class ValidationRound(BaseModel):
    round_number: int
    gap_snapshot: CandidateGap
    counter_queries: list[CounterQuery]
    search_record_ids: list[str]
    n_searches_ok: int
    n_searches_failed: int
    new_paper_ids: list[str]
    relevant_paper_ids: list[str]
    judgement: ValidationJudgementOutput


class ValidatedGap(BaseModel):
    gap: CandidateGap
    original_gap: CandidateGap
    status: ValidationStatus
    rounds: list[ValidationRound]
    counter_evidence: list[CounterEvidence]
    confidence: Confidence
    confidence_explanation: str
    feasibility: Feasibility
    rejected: bool = False
    rejection_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Ranking, recommendation, reading list
# ---------------------------------------------------------------------------

RANKING_CRITERIA = [
    "strength_of_supporting_evidence",
    "evidence_problem_unresolved",
    "importance_to_field",
    "potential_contribution",
    "feasibility",
    "data_and_tools_availability",
    "potential_for_empirical_validation",
    "relevance_to_current_developments",
    "differentiation_from_existing_work",
]

Rating = Literal["high", "medium", "low"]


class CriterionRating(BaseModel):
    criterion: str
    rating: Rating
    rationale: str


class RankingItemOutput(BaseModel):
    gap_id: str
    ratings: list[CriterionRating]
    main_risk: str
    overall_rationale: str


class RankingOutput(BaseModel):
    items: list[RankingItemOutput]
    trade_offs: str


class RankedGap(BaseModel):
    gap_id: str
    rank: int
    score: int
    ratings: list[CriterionRating]
    main_risk: str
    overall_rationale: str


class RecommendationOutput(BaseModel):
    gap_id: str
    proposed_title: str
    problem_statement: str
    motivation: str
    evidence_unresolved: str
    aim: str
    research_questions: list[str]
    hypotheses: list[str]
    methodology: str
    datasets_benchmarks_resources: list[str]
    baselines: list[str]
    evaluation_metrics: list[str]
    expected_contributions: list[str]
    potential_limitations: list[str]
    novelty_validation_plan: list[str]


class ReadingItem(BaseModel):
    paper_id: str
    role: Literal["foundational", "recent_key", "directly_related", "challenges_direction", "survey"]
    why_it_matters: str


class ReadingListOutput(BaseModel):
    items: list[ReadingItem]


class StateOfArtOutput(BaseModel):
    """Prose for report section 2, one block per theme, citing [P####] ids inline."""

    class ThemeProse(BaseModel):
        theme_id: str
        prose: str

    overview: str
    themes: list[ThemeProse]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class ClaimCheckItem(BaseModel):
    evidence_id: str
    verdict: Literal["supported", "partially_supported", "not_supported", "cannot_assess"]
    note: str


class ClaimCheckOutput(BaseModel):
    items: list[ClaimCheckItem]


class RemovedReference(BaseModel):
    where: str
    paper_id: str


class VerificationReport(BaseModel):
    removed_unknown_ids: list[RemovedReference] = Field(default_factory=list)
    doi_checks: list[DoiCheck] = Field(default_factory=list)
    retracted_paper_ids: list[str] = Field(default_factory=list)
    claim_checks: list[ClaimCheckItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------


class RunOptions(BaseModel):
    topic: str
    discipline: Optional[str] = None
    from_year: Optional[int] = None
    to_year: Optional[int] = None
    methodology_preference: Optional[str] = None
    priority: Optional[str] = None
    goal: Literal["research", "tool"] = "research"  # "tool": look for gaps a buildable technical artefact would fill
    existing_work: Optional[str] = None  # what the user already built or wants to differentiate from (free text)
    sources: list[str] = Field(default_factory=lambda: ["openalex", "semantic_scholar", "crossref", "arxiv"])
    per_source_limit: int = 25
    max_papers_to_analyze: int = 60
    validation_rounds: int = 2
    validation_per_source_limit: int = 10
    max_full_text: int = 20  # open-access full texts to fetch and analyse (0 disables)
    provider: str = "anthropic"
    model: str = "claude-opus-5-5"
    screen_model: Optional[str] = None  # cheaper model used for screening; "" = main model; None = decide at run start


class LLMCallLog(BaseModel):
    task: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    at: datetime = Field(default_factory=utcnow)


class RunState(BaseModel):
    run_id: str
    options: RunOptions
    created_at: datetime = Field(default_factory=utcnow)
    stages_completed: list[str] = Field(default_factory=list)
    plan: Optional[ResearchPlan] = None
    search_records: list[SearchRecord] = Field(default_factory=list)
    papers: dict[str, Paper] = Field(default_factory=dict)
    n_records_retrieved: int = 0
    evidence_rows: list[EvidenceRow] = Field(default_factory=list)
    landscape: Optional[LandscapeOutput] = None
    candidate_gaps: list[CandidateGap] = Field(default_factory=list)
    rejected_ideas: list[RejectedIdea] = Field(default_factory=list)
    validated_gaps: list[ValidatedGap] = Field(default_factory=list)
    ranking: list[RankedGap] = Field(default_factory=list)
    ranking_trade_offs: str = ""
    recommendation: Optional[RecommendationOutput] = None
    reading_list: list[ReadingItem] = Field(default_factory=list)
    state_of_art: Optional[StateOfArtOutput] = None
    verification: VerificationReport = Field(default_factory=VerificationReport)
    llm_calls: list[LLMCallLog] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    # --- helpers -----------------------------------------------------------
    def included_papers(self) -> list[Paper]:
        """Papers included by the primary (topic-level) screening."""
        return [p for p in self.papers.values() if p.screening and p.screening.include and p.screening.stage == "primary"]

    def primary_screened(self) -> list[Paper]:
        return [p for p in self.papers.values() if p.screening and p.screening.stage == "primary"]

    def validation_screened(self) -> list[Paper]:
        return [p for p in self.papers.values() if p.screening and p.screening.stage == "validation"]

    def analyzed_paper_ids(self) -> list[str]:
        return [r.paper_id for r in self.evidence_rows]

    def full_text_analyzed_ids(self) -> list[str]:
        return [r.paper_id for r in self.evidence_rows if self.papers.get(r.paper_id) and self.papers[r.paper_id].has_full_text]

    def add_limitation(self, text: str) -> None:
        if text not in self.limitations:
            self.limitations.append(text)
