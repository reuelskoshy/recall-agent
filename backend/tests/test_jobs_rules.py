"""Eligibility rules, India-aware location matching, and lenient extraction parsing.

Pure functions: no database, no LLM. Listings are synthetic.
"""

from datetime import date
from typing import Any

import pytest

from app.jobs.extraction import ExtractionError, build_messages, parse_extraction
from app.jobs.locations import canonical, judge_location, location_matches
from app.jobs.rules import Listing as L
from app.jobs.rules import ProfileFacts, find_sentence, judge
from app.jobs.schemas import ExtractedJob, Location, Reason

ME = ProfileFacts(
    graduation_year=2025,
    years_experience=0.5,
    allowed_locations=["India"],
    remote_ok=True,
    role_families=["engineering"],
)

TEXT = (
    "Acme is hiring a Software Engineer in Bengaluru, India. "
    "Eligibility: 2026 batch only. "
    "You need 2+ years of experience with Python. "
    "This is a 6-month internship. "
    "You will lead a team of three engineers.\n"
    "Location: Remote (US only)."
)


def job(**fields: Any) -> ExtractedJob:
    return ExtractedJob.model_validate(fields)


def listing(title: str = "Software Engineer", text: str = TEXT, location: str = "") -> L:
    return L(title=title, location_raw=location, text=text)


def reason(reasons: list[Reason], rule: str) -> Reason:
    return next(r for r in reasons if r.rule == rule)


def run(extracted: ExtractedJob, facts: ProfileFacts = ME, text: L | None = None) -> Any:
    return judge(extracted, facts, text or listing())


# ---- Evidence --------------------------------------------------------------------------


def test_find_sentence_returns_the_listings_own_sentence() -> None:
    assert find_sentence(listing(), "2026  BATCH only") == "Eligibility: 2026 batch only."
    assert find_sentence(listing(), "graduates of 2027") is None  # invented
    assert find_sentence(listing(), "") is None
    assert find_sentence(listing(), "a") is None  # too short to mean anything


def test_find_sentence_tolerates_smart_quotes_and_dashes() -> None:
    text = f"We{chr(0x2019)}re hiring 2025{chr(0x2013)}2026 graduates."  # smart quote, en dash
    assert find_sentence(listing(text=text), "We're hiring 2025-2026 graduates") == text


def test_find_sentence_searches_title_and_location_field() -> None:
    assert find_sentence(listing(title="Ops Analyst"), "Ops Analyst") == "Ops Analyst"
    lst = listing(location="Pune, Maharashtra")
    assert find_sentence(lst, "Pune") == "Pune, Maharashtra"


# ---- Reject: not a job listing ----------------------------------------------------------


def test_rejects_people_looking_for_work() -> None:
    text = listing(
        title="Backend Engineer | India (REMOTE)",
        text="I'm a backend engineer with 3 years of Go, looking for my next role.",
    )
    verdict, reasons = run(
        job(is_job_listing=False, evidence={"is_job_listing": "looking for my next role"}),
        text=text,
    )
    r = reason(reasons, "job_listing")
    assert verdict == "rejected" and r.outcome == "reject"
    assert r.evidence == "I'm a backend engineer with 3 years of Go, looking for my next role."
    unquoted, _ = run(job(is_job_listing=False), text=text)
    assert unquoted == "flagged"


@pytest.mark.parametrize(("value", "expected"), [("false", False), (0, False), (None, True)])
def test_is_job_listing_coercion(value: Any, expected: bool) -> None:
    assert job(is_job_listing=value).is_job_listing is expected


# ---- Reject: graduation year ------------------------------------------------------------


def test_rejects_a_later_graduation_year_with_the_sentence() -> None:
    verdict, reasons = run(
        job(graduation_years=[2026], evidence={"graduation_years": "2026 batch only"})
    )
    r = reason(reasons, "graduation_year")
    assert verdict == "rejected" and r.outcome == "reject"
    assert r.evidence == "Eligibility: 2026 batch only."
    assert "2026" in r.detail and "2025" in r.detail


def test_an_unverifiable_graduation_quote_only_flags() -> None:
    verdict, reasons = run(
        job(graduation_years=[2027], evidence={"graduation_years": "2027 passouts only"})
    )
    r = reason(reasons, "graduation_year")
    assert (verdict, r.outcome, r.evidence) == ("flagged", "flag", None)
    assert "not found in the listing" in r.detail


def test_missing_graduation_quote_only_flags() -> None:
    _, reasons = run(job(graduation_years=[2026]))
    assert reason(reasons, "graduation_year").outcome == "flag"


@pytest.mark.parametrize(
    ("years", "outcome"),
    [([2025], "pass"), ([2024, 2025, 2026], "pass"), ([2023, 2024], "flag"), ([], "pass")],
)
def test_graduation_year_matches_and_earlier_years(years: list[int], outcome: str) -> None:
    _, reasons = run(job(graduation_years=years))
    assert reason(reasons, "graduation_year").outcome == outcome


def test_graduation_year_not_checked_without_profile_value() -> None:
    facts = ProfileFacts(allowed_locations=["India"])
    _, reasons = run(job(graduation_years=[2030]), facts)
    r = reason(reasons, "graduation_year")
    assert r.outcome == "not_checked" and "no graduation year" in r.detail


# ---- Reject: role family ---------------------------------------------------------------


def test_rejects_non_engineering_roles() -> None:
    text = listing(title="Operations Analyst", text="Run our operations analytics in Pune.")
    verdict, reasons = run(
        job(role_family="operations", evidence={"role_family": "Operations Analyst"}), text=text
    )
    r = reason(reasons, "role_family")
    assert verdict == "rejected" and r.evidence == "Operations Analyst"


def test_engineering_role_passes_and_other_families_can_be_wanted() -> None:
    assert reason(run(job(role_family="engineering"))[1], "role_family").outcome == "pass"
    facts = ProfileFacts(role_families=["engineering", "support"], allowed_locations=["India"])
    assert reason(run(job(role_family="support"), facts)[1], "role_family").outcome == "pass"


# ---- Reject: location (India) ----------------------------------------------------------


@pytest.mark.parametrize(
    "place",
    [
        {"city": "Bangalore"},  # alias, country inferred
        {"city": "Bengaluru", "country": "India"},
        {"city": "Gurgaon"},
        {"city": "Noida", "region": "Uttar Pradesh"},
        {"city": "Trivandrum", "country": "IN"},
        {"region": "Karnataka"},
        {"city": "Some Town", "country": "Bharat"},
    ],
)
def test_indian_places_match_india(place: dict[str, str]) -> None:
    assert location_matches(Location(**place), ["India"])


@pytest.mark.parametrize(
    "place",
    [{"city": "San Francisco", "country": "USA"}, {"city": "London"}, {"city": "Berlin"}],
)
def test_places_outside_india_do_not_match(place: dict[str, str]) -> None:
    assert not location_matches(Location(**place), ["India"])


def test_city_level_and_group_preferences() -> None:
    assert location_matches(Location(city="Bangalore"), ["Bengaluru"])
    assert not location_matches(Location(city="Pune"), ["Bengaluru"])
    assert location_matches(Location(city="Gurugram"), ["Delhi NCR"])
    assert location_matches(Location(city="Mysore"), ["Karnataka"])
    assert canonical(" U.S.A. ") == "united states"


def test_rejects_a_job_outside_india_with_its_sentence() -> None:
    text = listing(text="Join our team in Austin, Texas. Hybrid, 3 days a week.")
    verdict, reasons = run(
        job(
            locations=[{"city": "Austin", "region": "Texas", "country": "United States"}],
            evidence={"locations": "Austin, Texas"},
        ),
        text=text,
    )
    r = reason(reasons, "location")
    assert verdict == "rejected" and r.evidence == "Join our team in Austin, Texas."


def test_any_allowed_location_among_several_is_enough() -> None:
    extracted = job(locations=[{"city": "Singapore"}, {"city": "Hyderabad"}])
    assert reason(run(extracted)[1], "location").outcome == "pass"


@pytest.mark.parametrize(
    ("regions", "outcome"),
    [
        ([], "match"),  # unrestricted remote
        (["India"], "match"),
        (["APAC"], "match"),
        (["Worldwide"], "match"),
        (["United States"], "outside"),
        (["US", "Canada"], "outside"),
    ],
)
def test_remote_jobs_by_region(regions: list[str], outcome: str) -> None:
    extracted = job(work_mode="remote", remote_regions=regions)
    assert judge_location(extracted, ["India"], remote_ok=True)[0] == outcome


def test_remote_us_only_is_rejected_when_the_quote_is_real() -> None:
    verdict, reasons = run(
        job(
            work_mode="remote",
            remote_regions=["United States"],
            evidence={"locations": "Remote (US only)"},
        )
    )
    assert verdict == "rejected"
    assert reason(reasons, "location").evidence == "Location: Remote (US only)."


def test_remote_not_wanted() -> None:
    facts = ProfileFacts(allowed_locations=["Bengaluru"], remote_ok=False)
    outcome, _ = judge_location(job(work_mode="remote"), facts.allowed_locations, False)
    assert outcome == "outside"
    hybrid = job(work_mode="remote", locations=[{"city": "Bengaluru"}])
    assert judge_location(hybrid, ["Bengaluru"], remote_ok=False)[0] == "match"


def test_no_location_is_a_flag_not_a_rejection() -> None:
    verdict, reasons = run(job())
    assert verdict == "flagged" and reason(reasons, "location").outcome == "flag"


def test_location_not_checked_without_preferences() -> None:
    _, reasons = run(job(locations=[{"city": "Paris"}]), ProfileFacts(graduation_year=2025))
    assert reason(reasons, "location").outcome == "not_checked"


def test_feed_location_field_is_the_fallback_evidence() -> None:
    text = listing(text="Great role.", location="Seattle, WA")
    _, reasons = run(job(locations=[{"city": "Seattle", "country": "US"}]), text=text)
    r = reason(reasons, "location")
    assert (r.outcome, r.evidence) == ("reject", "Seattle, WA")


# ---- Flags --------------------------------------------------------------------------------


def test_flags_an_experience_gap_with_its_size() -> None:
    verdict, reasons = run(
        job(
            locations=[{"city": "Bengaluru"}],
            min_years_experience=2,
            evidence={"min_years_experience": "2+ years of experience"},
        )
    )
    r = reason(reasons, "experience")
    assert verdict == "flagged" and r.outcome == "flag"
    assert "gap of 1.5" in r.detail
    assert r.evidence == "You need 2+ years of experience with Python."


def test_freshers_and_matching_experience_pass() -> None:
    assert reason(run(job(min_years_experience=0))[1], "experience").outcome == "pass"
    assert reason(run(job())[1], "experience").outcome == "pass"


def test_flags_an_internship_without_a_graduation_year() -> None:
    _, reasons = run(job(employment_type="internship", evidence={"employment_type": "internship"}))
    r = reason(reasons, "internship")
    assert r.outcome == "flag" and r.evidence == "This is a 6-month internship."
    _, reasons = run(job(employment_type="internship", graduation_years=[2025]))
    assert reason(reasons, "internship").outcome == "pass"


def test_flags_senior_duties_on_a_junior_title() -> None:
    text = listing(title="Junior Backend Engineer")
    _, reasons = run(job(seniority_signals=["lead a team of three engineers"]), text=text)
    r = reason(reasons, "seniority")
    assert r.outcome == "flag" and r.evidence == "You will lead a team of three engineers."


@pytest.mark.parametrize("title", ["SDE-1", "SDE I", "Graduate Engineer Trainee", "Engineer I"])
def test_junior_title_patterns(title: str) -> None:
    _, reasons = run(job(seniority_signals=["lead a team"]), text=listing(title=title))
    assert reason(reasons, "seniority").outcome == "flag"


def test_seniority_ignores_invented_phrases_and_senior_titles() -> None:
    junior = listing(title="Junior Engineer")
    _, reasons = run(job(seniority_signals=["own the company roadmap"]), text=junior)
    assert reason(reasons, "seniority").outcome == "pass"  # not in the listing
    _, reasons = run(job(seniority_signals=["lead a team"]), text=listing(title="Staff Engineer"))
    assert reason(reasons, "seniority").outcome == "pass"


# ---- Verdict --------------------------------------------------------------------------


def test_eligible_when_nothing_fires_and_reasons_are_ordered() -> None:
    clean = job(
        locations=[{"city": "Bengaluru", "country": "India"}],
        role_family="engineering",
        employment_type="full_time",
        min_years_experience=0,
        graduation_years=[2025],
    )
    verdict, reasons = run(clean)
    assert verdict == "eligible"
    assert {r.rule for r in reasons} == {
        "job_listing",
        "graduation_year",
        "role_family",
        "location",
        "experience",
        "internship",
        "seniority",
    }
    mixed, reasons = run(
        job(role_family="other", min_years_experience=5, evidence={"role_family": "Acme is hiring"})
    )
    assert mixed == "rejected" and reasons[0].outcome == "reject"
    assert [r.outcome for r in reasons].index("flag") < [r.outcome for r in reasons].index("pass")


# ---- Parsing and schema coercion ------------------------------------------------------


def test_parse_is_lenient_about_fences_and_prose() -> None:
    reply = 'Sure!\n```json\n{"role_family": "Engineering", "work_mode": "On-site"}\n```\nDone.'
    parsed = parse_extraction(reply)
    assert (parsed.role_family, parsed.work_mode) == ("engineering", "onsite")


@pytest.mark.parametrize("reply", ["", "no json here", "{not json}", "[1, 2]", '"text"'])
def test_parse_rejects_replies_without_an_object(reply: str) -> None:
    with pytest.raises(ExtractionError):
        parse_extraction(reply)


def test_schema_coerces_odd_values_instead_of_failing() -> None:
    parsed = job(
        locations="Bangalore, India",
        work_mode="WFH",
        role_family="Sales",
        employment_type="Full-time",
        min_years_experience="2",
        graduation_years=["2025", 2026, "2026", "batch", 1900],
        salary={"min": "12,00,000", "max": None, "currency": "₹", "period": "per annum"},
        technologies="Python",
        evidence={"salary": "12 LPA", "made_up_field": "x", "locations": ""},
    )
    assert parsed.locations == [Location(city="Bangalore", country="India")]
    assert (parsed.work_mode, parsed.role_family, parsed.employment_type) == (
        "remote",
        "other",
        "full_time",
    )
    assert parsed.min_years_experience == 2 and parsed.graduation_years == [2025, 2026]
    assert parsed.salary is not None
    assert (parsed.salary.min, parsed.salary.currency, parsed.salary.period) == (
        1_200_000,
        "INR",
        "year",
    )
    assert parsed.technologies == ["Python"]
    assert parsed.evidence == {"salary": "12 LPA"}


def test_schema_drops_impossible_values() -> None:
    parsed = job(min_years_experience=-1, salary="lots", locations=[42, {"city": "Pune"}])
    assert parsed.min_years_experience is None and parsed.salary is None
    assert parsed.locations == [Location(city="Pune")]


def test_prompt_covers_indian_conventions_and_clips_long_listings() -> None:
    messages = build_messages("SDE 1", "", "x" * 20_000, date(2026, 11, 12))
    system, user = messages[0].content, messages[1].content
    assert "LPA" in system and "batch" in system and "2027" in system
    assert len(user) < 12_200 and user.endswith("[...]")
