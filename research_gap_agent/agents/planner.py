"""Research Planner: decomposes the topic and produces the search strategy."""
from __future__ import annotations

from ..llm import LLM
from ..models import ResearchPlan, ResearchPlanOutput, RunOptions
from .common import INTEGRITY_RULES, topic_block

SYSTEM = f"""You are the Research Planner of an autonomous research-gap discovery system.
Your job is to understand a research topic deeply and design a literature search strategy for academic search APIs
(OpenAlex, Semantic Scholar, Crossref, arXiv, PubMed).

Guidance:
- Identify core concepts, adjacent research areas, alternative terminology, synonyms, abbreviations, technical approaches,
  foundational theories/frameworks, and related problems studied under different names.
- Decide which adjacent areas are genuinely in scope and which are not; list excluded areas with a reason.
- Produce 6 to 14 search queries. Queries are keyword strings (2 to 6 terms each), not sentences; use quotes for
  multi-word phrases. Cover the exact topic, its alternative terminology and its most relevant adjacent areas.
  Do not restrict yourself to the user's exact wording.
- If the user left discipline, period, methodology or priority unspecified, make reasonable assumptions and list them.
- If the desired outcome is a technical tool, include queries that find existing tools, systems, benchmarks and
  open-source implementations (words like tool, scanner, framework, benchmark, open-source, implementation), since
  the search also covers code repositories.
{INTEGRITY_RULES}"""


def plan_research(llm: LLM, options: RunOptions, period: str) -> ResearchPlan:
    user = topic_block(options.topic, options.discipline, period, options.methodology_preference, options.priority,
                       options.goal, options.existing_work)
    out = llm.structured(task="plan", system=SYSTEM, user=user, output_model=ResearchPlanOutput, effort="high")
    return ResearchPlan(
        **out.model_dump(),
        topic=options.topic,
        discipline=options.discipline,
        from_year=options.from_year,
        to_year=options.to_year,
        methodology_preference=options.methodology_preference,
        priority=options.priority,
    )
