from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pytest

from research_gap_agent.config import Settings
from research_gap_agent.llm import ScriptedLLM
from research_gap_agent.models import (
    CandidateGapOutput, ClaimCheckItem, ClaimCheckOutput, ConflictingFinding, CounterEvidence, CounterQuery,
    CounterQueryOutput, CriterionRating, EvidenceRow, EvidenceStatement, ExtractionOutput, Feasibility, GapCategory,
    GapGenerationOutput, LandscapeOutput, RANKING_CRITERIA, RankingItemOutput, RankingOutput, RawRecord, ReadingItem,
    ReadingListOutput, RecommendationOutput, RejectedIdea, ResearchPlanOutput, ResearchQuestion, RunOptions,
    ScreeningItem, ScreeningOutput, SearchQuery, StateOfArtOutput, Theme, ValidationJudgementOutput, ValidationStatus,
)
from research_gap_agent.sources.base import SearchOutcome

PID = re.compile(r"\[(P\d{4})\]")
EID = re.compile(r"\[(E\d{3})\]")
GID = re.compile(r"\[(G\d{2})\]")


def ids_in(text: str) -> list[str]:
    return list(dict.fromkeys(PID.findall(text)))


class FakeSource:
    """In-memory source: maps query substrings to records; can simulate failures."""

    def __init__(self, name: str, table: dict[str, list[RawRecord]], status: str = "ok",
                 fail_queries: Optional[set[str]] = None):
        self.name = name
        self.table = table
        self.status = status
        self.fail_queries = fail_queries or set()
        self.queries: list[str] = []

    def search(self, query: str, *, from_year=None, to_year=None, limit=25) -> SearchOutcome:
        self.queries.append(query)
        if self.status != "ok" or query in self.fail_queries:
            return SearchOutcome(status=self.status if self.status != "ok" else "rate_limited", error="HTTP 429")
        recs = []
        for key, items in self.table.items():
            if key.lower() in query.lower():
                recs.extend(items)
        return SearchOutcome(records=recs[:limit])


def rec(source, sid, title, year=2023, doi=None, arxiv=None, abstract="An abstract about the topic.", ptype="journal-article",
        retracted=None, cites=5, authors=("Ada Lovelace", "Alan Turing")):
    return RawRecord(source=source, source_id=sid, title=title, year=year, doi=doi, arxiv_id=arxiv, abstract=abstract,
                     publication_type=ptype, is_retracted=retracted, cited_by_count=cites, authors=list(authors))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(cache_dir=tmp_path / "cache", runs_dir=tmp_path / "runs", mailto="test@example.org")


@pytest.fixture
def primary_records() -> list[RawRecord]:
    return [
        rec("alpha", "a1", "Prompt injection attacks on tool-using LLM agents", 2024, doi="10.1000/abc1"),
        rec("alpha", "a2", "A benchmark for indirect prompt injection", 2024, doi="10.1000/abc2"),
        rec("alpha", "a3", "Formal verification of authorization in agent protocols", 2023, arxiv="2301.00001",
            ptype="preprint"),
        rec("beta", "b1", "Prompt Injection Attacks on Tool-Using LLM Agents.", 2024, doi="10.1000/ABC1"),  # dup of a1
        rec("beta", "b2", "Securing the Model Context Protocol: threat model", 2025, doi="10.1000/abc4"),
        rec("beta", "b3", "Unrelated study of soil microbiomes", 2022, doi="10.1000/soil"),
    ]


@pytest.fixture
def counter_records() -> list[RawRecord]:
    return [
        rec("alpha", "c1", "A deployed defense that already solves tool authorization for agents", 2025, doi="10.1000/solved"),
        rec("beta", "c2", "Partial evaluation of MCP permission models in production", 2025, doi="10.1000/partial"),
    ]


@pytest.fixture
def fake_sources(primary_records, counter_records):
    table_a = {"prompt injection": primary_records[:3], "counter": counter_records[:1], "mcp": [primary_records[4]]}
    table_b = {"prompt injection": primary_records[3:], "counter": counter_records[1:]}
    return [FakeSource("alpha", table_a), FakeSource("beta", table_b)]


def make_scripted_llm(*, judge_plan: Optional[dict[str, list[str]]] = None) -> ScriptedLLM:
    """A scripted LLM that behaves plausibly given the prompt text.

    ``judge_plan`` maps gap id -> list of statuses to return on successive judge calls.
    """
    judge_plan = judge_plan or {}
    judge_calls: dict[str, int] = {}
    feas = Feasibility(data_availability="public benchmarks exist", resources_and_tools="open-source agents",
                       technical_requirements="moderate", evaluation_challenges="ground truth is hard", overall="medium")

    def plan(system, user):
        return ResearchPlanOutput(
            assumptions=["Period assumed 2022–present"], core_concepts=["prompt injection", "MCP"],
            related_areas_in_scope=["LLM security"], related_areas_excluded=["network IDS (different problem)"],
            alternative_terminology=["indirect prompt injection"], technical_approaches=["sandboxing"],
            foundational_frameworks=["least privilege"],
            research_questions=[ResearchQuestion(question="What is known?", rationale="scope")],
            search_queries=[SearchQuery(query="prompt injection agents", rationale="core"),
                            SearchQuery(query="MCP security", rationale="adjacent")],
        )

    def screen(system, user):
        return ScreeningOutput(decisions=[
            ScreeningItem(paper_id=pid, include=("soil" not in user.split(f"[{pid}]")[1].split("\n")[0].lower()),
                          reason="scripted") for pid in ids_in(user)])

    def extract(system, user):
        return ExtractionOutput(rows=[EvidenceRow(
            paper_id=pid, research_objective="obj", methodology="method", main_findings="findings",
            limitations=["single benchmark only"], datasets_or_benchmarks=["AgentBench"], evaluation_metrics=["ASR"],
            key_assumptions=["single agent"], environment="simulation", relevance_to_potential_gaps="rel") for pid in ids_in(user)])

    def landscape(system, user):
        pids = ids_in(user)
        return LandscapeOutput(
            themes=[Theme(id="T1", name="Attacks", description="d", paper_ids=pids, established_approaches=["a"],
                          important_contributions=["c"], strengths=["s"], weaknesses=["w"], unresolved_questions=["u"])],
            evidence_statements=[
                EvidenceStatement(id="E001", statement="All studies evaluate in simulation only", kind="direct", paper_ids=pids),
                EvidenceStatement(id="E002", statement="Mixed real and fake ids", kind="interpretation", paper_ids=[pids[0], "P9999"]),
                EvidenceStatement(id="E003", statement="Only fake ids", kind="interpretation", paper_ids=["P9998"]),
            ],
            conflicting_findings=[ConflictingFinding(description="c", paper_ids_side_a=[pids[0]], paper_ids_side_b=["P9997"])],
            established_knowledge=["k"], increasing_attention=["t"], foundational_paper_ids=[pids[0]],
            recent_key_paper_ids=[pids[-1], "P9996"],
        )

    def gaps(system, user):
        pids = ids_in(user)
        mk = lambda t, ev, ps: CandidateGapOutput(  # noqa: E731
            title=t, category=GapCategory.evaluation, what_is_established="est", what_is_missing="missing",
            supporting_evidence_ids=ev, supporting_paper_ids=ps, why_it_matters="matters", potential_contribution="contrib",
            direct_vs_interpretation="mixed")
        return GapGenerationOutput(gaps=[
            mk("Gap already solved elsewhere", ["E001"], pids[:2]),
            mk("Gap with only evidence-statement support", ["E001"], ["P9999"]),
            mk("Gap with nothing verifiable", ["E999"], ["P9998"]),
        ], considered_and_rejected=[RejectedIdea(idea="x", why_not="y")])

    def counter(system, user):
        return CounterQueryOutput(queries=[CounterQuery(query="counter query one", rationale="r", strategy="alternative_terminology")])

    def judge(system, user):
        gid = GID.findall(user)[0]
        n = judge_calls.get(gid, 0)
        judge_calls[gid] = n + 1
        statuses = judge_plan.get(gid, ["limited_evidence_found"])
        status = ValidationStatus(statuses[min(n, len(statuses) - 1)])
        found = [p for p in ids_in(user.split("Papers found by the counter-searches")[1].split("Papers that originally")[0])]
        ce = [CounterEvidence(paper_id=p, relationship="addresses_fully" if status == ValidationStatus.addressed else "addresses_partially",
                              note="note") for p in found] + [CounterEvidence(paper_id="P9995", relationship="tangential", note="fake")]
        revised = None
        if status == ValidationStatus.partially_addressed:
            revised = CandidateGapOutput(title="Narrowed gap", category=GapCategory.evaluation, what_is_established="e",
                                         what_is_missing="narrower", supporting_evidence_ids=["E001"],
                                         supporting_paper_ids=ids_in(user)[:1], why_it_matters="m",
                                         potential_contribution="c", direct_vs_interpretation="d")
        return ValidationJudgementOutput(status=status, reasoning=f"scripted {status.value}", counter_evidence=ce,
                                         revised_gap=revised, confidence="medium", confidence_explanation="scripted",
                                         feasibility=feas, what_could_not_be_established="full texts")

    def rank(system, user):
        gids = list(dict.fromkeys(GID.findall(user)))
        return RankingOutput(items=[RankingItemOutput(
            gap_id=g, ratings=[CriterionRating(criterion=c, rating="high" if i == 0 else "medium", rationale="r")
                               for c in RANKING_CRITERIA], main_risk="risk", overall_rationale="because")
            for i, g in enumerate(gids)], trade_offs="trade-offs text")

    def sota(system, user):
        pids = ids_in(user)
        return StateOfArtOutput(overview=f"Overview cites [{pids[0]}] and a fake [P9999].",
                                themes=[StateOfArtOutput.ThemeProse(theme_id="T1", prose=f"Theme prose [{pids[0]}].")])

    def recommend(system, user):
        gid = GID.findall(user)[0]
        return RecommendationOutput(gap_id=gid, proposed_title="Title", problem_statement="ps", motivation="mot",
                                    evidence_unresolved=f"see [{ids_in(user)[0]}]", aim="aim", research_questions=["rq"],
                                    hypotheses=[], methodology="m", datasets_benchmarks_resources=["d"], baselines=["b"],
                                    evaluation_metrics=["asr"], expected_contributions=["c"], potential_limitations=["l"],
                                    novelty_validation_plan=["search again"])

    def reading(system, user):
        pids = ids_in(user)
        return ReadingListOutput(items=[ReadingItem(paper_id=pids[0], role="foundational", why_it_matters="w"),
                                        ReadingItem(paper_id="P9999", role="survey", why_it_matters="fake")])

    def claims(system, user):
        return ClaimCheckOutput(items=[ClaimCheckItem(evidence_id=e, verdict="supported", note="ok") for e in EID.findall(user)])

    llm = ScriptedLLM()
    for model, h in [(ResearchPlanOutput, plan), (ScreeningOutput, screen), (ExtractionOutput, extract),
                     (LandscapeOutput, landscape), (GapGenerationOutput, gaps), (CounterQueryOutput, counter),
                     (ValidationJudgementOutput, judge), (RankingOutput, rank), (StateOfArtOutput, sota),
                     (RecommendationOutput, recommend), (ReadingListOutput, reading), (ClaimCheckOutput, claims)]:
        llm.on(model, h)
    return llm


@pytest.fixture
def options() -> RunOptions:
    return RunOptions(topic="Securing MCP in agentic AI systems", from_year=2022, sources=["alpha", "beta"],
                      per_source_limit=10, validation_rounds=2, validation_per_source_limit=5, max_full_text=0)
