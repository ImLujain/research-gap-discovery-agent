"""Validation-loop control: rejection, revision rounds, forced cannot_validate."""
from research_gap_agent.coordinator import Coordinator
from research_gap_agent.models import ValidationStatus

from conftest import FakeSource, make_scripted_llm


def _run(options, settings, sources, llm):
    coord = Coordinator(options, llm, settings=settings, sources=sources, run_id="t")
    coord.run(until="validate")
    return coord.state


def test_addressed_gap_is_rejected_with_counterevidence(options, settings, fake_sources):
    llm = make_scripted_llm(judge_plan={"G01": ["addressed"], "G02": ["limited_evidence_found"]})
    st = _run(options, settings, fake_sources, llm)
    by = {v.gap.id: v for v in st.validated_gaps}
    assert by["G01"].rejected and by["G01"].status == ValidationStatus.addressed
    assert "Existing work already addresses" in by["G01"].rejection_reason
    # counter-evidence came from the counter search and traces to real papers; fake id P9995 was stripped
    ce_ids = {c.paper_id for c in by["G01"].counter_evidence}
    assert ce_ids and all(i in st.papers for i in ce_ids) and "P9995" not in ce_ids
    assert any(r.paper_id == "P9995" for r in st.verification.removed_unknown_ids)
    assert not by["G02"].rejected


def test_partially_addressed_gap_is_revised_and_revalidated(options, settings, fake_sources):
    llm = make_scripted_llm(judge_plan={"G01": ["partially_addressed", "preliminary_evidence"], "G02": ["limited_evidence_found"]})
    st = _run(options, settings, fake_sources, llm)
    v = next(v for v in st.validated_gaps if v.gap.id == "G01")
    assert len(v.rounds) == 2
    assert v.rounds[0].judgement.status == ValidationStatus.partially_addressed
    assert v.gap.title == "Narrowed gap" and v.gap.revision_number == 1
    assert v.original_gap.title == "Gap already solved elsewhere"
    assert v.status == ValidationStatus.preliminary_evidence and not v.rejected
    # two rounds of counter-queries were executed for G01
    assert sum(1 for s in st.search_records if s.purpose.startswith("validation:G01")) == 4  # 1 query x 2 sources x 2 rounds


def test_round_limit_stops_revision_loop(options, settings, fake_sources):
    options.validation_rounds = 1
    llm = make_scripted_llm(judge_plan={"G01": ["partially_addressed"], "G02": ["limited_evidence_found"]})
    st = _run(options, settings, fake_sources, llm)
    v = next(v for v in st.validated_gaps if v.gap.id == "G01")
    assert len(v.rounds) == 1 and v.gap.revision_number == 0
    assert any("further narrowed revision" in n for n in st.verification.notes)


def test_all_validation_searches_failing_forces_cannot_validate(options, settings, primary_records):
    a = FakeSource("alpha", {"prompt injection": primary_records[:3], "mcp": [primary_records[4]]}, fail_queries={"counter query one"})
    b = FakeSource("beta", {"prompt injection": primary_records[3:]}, fail_queries={"counter query one"})
    llm = make_scripted_llm(judge_plan={"G01": ["limited_evidence_found"], "G02": ["limited_evidence_found"]})
    st = _run(options, settings, [a, b], llm)
    for v in st.validated_gaps:
        assert v.status == ValidationStatus.cannot_validate
        assert "failed or were rate-limited" in v.rounds[-1].judgement.reasoning
    assert any("rate-limited" in lim for lim in st.limitations)
    assert all(s.status == "rate_limited" for s in st.search_records if s.purpose.startswith("validation"))


def test_gap_without_verifiable_evidence_is_dropped(options, settings, fake_sources):
    llm = make_scripted_llm()
    st = _run(options, settings, fake_sources, llm)
    titles = [g.title for g in st.candidate_gaps]
    assert "Gap with nothing verifiable" not in titles
    g2 = next(g for g in st.candidate_gaps if g.title.startswith("Gap with only evidence"))
    assert g2.supporting_paper_ids and all(p in st.papers for p in g2.supporting_paper_ids)  # back-filled via E001
    assert any("dropped" in n for n in st.verification.notes)


def test_validation_checkpoints_per_gap_and_resumes(options, settings, fake_sources):
    """A crash during gap 2 keeps gap 1's result; the resumed run validates only the remaining gaps."""
    import pytest

    from research_gap_agent.models import ValidationJudgementOutput

    llm = make_scripted_llm(judge_plan={"G01": ["limited_evidence_found"], "G02": ["limited_evidence_found"]})
    judge_handler = llm.handlers[ValidationJudgementOutput]
    calls = {"n": 0}

    def flaky_judge(system, user):
        calls["n"] += 1
        if "[G02]" in user:
            raise RuntimeError("simulated stream drop")
        return judge_handler(system, user)

    llm.on(ValidationJudgementOutput, flaky_judge)
    coord = Coordinator(options, llm, settings=settings, sources=fake_sources, run_id="ckpt")
    with pytest.raises(RuntimeError, match="stream drop"):
        coord.run(until="validate")
    saved = Coordinator.resume("ckpt", make_scripted_llm(), settings=settings, sources=fake_sources)
    assert "validate" not in saved.state.stages_completed
    assert [v.gap.id for v in saved.state.validated_gaps] == ["G01"]  # checkpoint survived the crash
    llm2 = make_scripted_llm(judge_plan={"G02": ["limited_evidence_found"]})
    st = saved.run(until="validate")
    judged = [u for t, m, u in llm2.calls if m is ValidationJudgementOutput]
    assert [v.gap.id for v in st.validated_gaps] == ["G01", "G02"]
    assert "validate" in st.stages_completed
    assert all("[G01]" not in u for u in judged)  # G01 was not re-judged on resume


def test_resume_reuses_gap_screening_decisions(options, settings, fake_sources):
    """Papers screened for a gap before a crash are not re-screened when that gap is validated again."""
    import pytest

    from research_gap_agent.models import ScreeningOutput, ValidationJudgementOutput

    llm = make_scripted_llm(judge_plan={"G01": ["limited_evidence_found"], "G02": ["limited_evidence_found"]})
    judge_handler = llm.handlers[ValidationJudgementOutput]

    def crash_on_g01(system, user):
        if "[G01]" in user:
            raise RuntimeError("drop")
        return judge_handler(system, user)

    llm.on(ValidationJudgementOutput, crash_on_g01)
    coord = Coordinator(options, llm, settings=settings, sources=fake_sources, run_id="reuse")
    with pytest.raises(RuntimeError):
        coord.run(until="validate")
    assert any(p.screening and p.screening.stage == "validation" and p.screening.gap_id == "G01"
               for p in coord.state.papers.values())
    llm2 = make_scripted_llm(judge_plan={"G01": ["limited_evidence_found"], "G02": ["limited_evidence_found"]})
    st = Coordinator.resume("reuse", llm2, settings=settings, sources=fake_sources).run(until="validate")
    screen_calls = [u for t, m, u in llm2.calls if m is ScreeningOutput]
    assert len(screen_calls) == 1  # only G02's counter-search papers needed screening; G01's were reused
    assert [v.gap.id for v in st.validated_gaps] == ["G01", "G02"]
    assert st.validated_gaps[0].counter_evidence  # reused decisions still fed the judgement


def test_stop_request_saves_state_and_is_resumable(options, settings, fake_sources):
    import pytest

    from research_gap_agent.coordinator import RunCancelled

    llm = make_scripted_llm(judge_plan={"G01": ["limited_evidence_found"], "G02": ["limited_evidence_found"]})
    flag = {"stop": False}
    judged = []
    orig = llm.handlers[__import__("research_gap_agent.models", fromlist=["ValidationJudgementOutput"]).ValidationJudgementOutput]

    def judge_then_stop(system, user):
        judged.append(user)
        flag["stop"] = True  # ask to stop after the first gap's judgement
        return orig(system, user)

    from research_gap_agent.models import ValidationJudgementOutput
    llm.on(ValidationJudgementOutput, judge_then_stop)
    coord = Coordinator(options, llm, settings=settings, sources=fake_sources, run_id="stop", should_stop=lambda: flag["stop"])
    with pytest.raises(RunCancelled):
        coord.run()
    resumed = Coordinator.resume("stop", make_scripted_llm(judge_plan={"G02": ["limited_evidence_found"]}),
                                 settings=settings, sources=fake_sources)
    assert [v.gap.id for v in resumed.state.validated_gaps] == ["G01"]  # gap 1 finished and was saved before stopping
    assert "validate" not in resumed.state.stages_completed
    st = resumed.run()
    assert st.stages_completed[-1] == "report" and [v.gap.id for v in st.validated_gaps] == ["G01", "G02"]
