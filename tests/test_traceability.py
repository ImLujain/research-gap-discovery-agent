from research_gap_agent.models import VerificationReport
from research_gap_agent.traceability import filter_ids, scrub_text, unknown_ids_in_text


def test_filter_ids_removes_unknown_and_dedupes():
    rep = VerificationReport()
    out = filter_ids(["P0001", "P9999", "P0001", " P0002 "], {"P0001", "P0002"}, "gap G01", rep)
    assert out == ["P0001", "P0002"]
    assert [r.paper_id for r in rep.removed_unknown_ids] == ["P9999"]
    assert rep.removed_unknown_ids[0].where == "gap G01"


def test_scrub_text_marks_unknown_refs():
    rep = VerificationReport()
    txt = scrub_text("Known [P0001] and unknown [P0042] here.", {"P0001"}, "sota", rep)
    assert "[P0001]" in txt and "P0042" not in txt and "[unverified reference removed]" in txt
    assert unknown_ids_in_text("P0042 P0001", {"P0001"}) == ["P0042"]
