"""The ranking formula: pure functions, no database, no model."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.jobs.schemas import ExtractedJob
from app.jobs.scoring import (
    ScoreWeights,
    ScoringProfile,
    canonical_tech,
    score_job,
    technology,
)

NOW = datetime(2026, 11, 12, tzinfo=UTC)
W = ScoreWeights()
ME = ScoringProfile(
    skills=frozenset(canonical_tech(s) for s in ("Python", "PostgreSQL", "AWS Lambda", "React")),
    years_experience=0.2,
    allowed_locations=("India",),
    sponsorship_locations=("United States",),
)
BENGALURU = [{"city": "Bengaluru", "country": "India"}]


def job(**fields: Any) -> ExtractedJob:
    return ExtractedJob.model_validate({"locations": BENGALURU, **fields})


def score(
    extracted: ExtractedJob,
    profile: ScoringProfile = ME,
    weights: ScoreWeights = W,
    similarity: float | None = 0.70,
    age_days: float = 0,
) -> tuple[float | None, dict[str, Any]]:
    return score_job(
        extracted,
        profile,
        weights,
        similarity=similarity,
        seen_at=NOW - timedelta(days=age_days),
        now=NOW,
    )


def test_components_are_stored_with_weights_that_sum_to_one() -> None:
    total, parts = score(job(technologies=["Python", "Go"], min_years_experience=0))
    used = {k: v for k, v in parts.items() if v["value"] is not None}
    assert set(used) == {"semantic", "technology", "experience", "location", "freshness"}
    assert sum(v["weight"] for v in used.values()) == pytest.approx(1, abs=1e-3)
    assert total == pytest.approx(sum(v["contribution"] for v in used.values()), abs=1e-3)
    assert parts["salary"] == {
        "value": None,
        "weight": 0.0,
        "contribution": 0.0,
        "detail": "Your profile sets no minimum salary.",
    }


def test_missing_data_neither_helps_nor_hurts() -> None:
    # A component that cannot be measured is left out, and the rest rescale.
    with_none, _ = score(job(), similarity=None)
    perfect, _ = score(job(min_years_experience=0), similarity=0.80)
    assert with_none == pytest.approx(1.0) and perfect == pytest.approx(1.0)


def test_technology_overlap_uses_aliases_and_word_prefixes() -> None:
    value, detail = technology(job(technologies=["postgres", "AWS", "Kubernetes", "python"]), ME)
    assert value == pytest.approx((3 + 1) / (4 + 2))  # smoothed
    assert "Matched: postgresql, aws, python" in detail and "Missing: kubernetes" in detail
    assert technology(job(), ME)[0] is None


def test_experience_falls_linearly_to_zero_at_the_reject_threshold() -> None:
    def value(years: float) -> float:
        return float(score(job(min_years_experience=years))[1]["experience"]["value"])

    assert value(0) == 1.0
    assert value(1.7) == pytest.approx(0.5)
    assert value(3.2) == 0.0
    assert value(1.2) > value(2.2)  # smaller gaps rank higher


def test_location_prefers_allowed_places_over_sponsorship() -> None:
    here = score(job())[1]["location"]
    abroad = score(job(locations=[{"city": "Austin", "country": "United States"}]))[1]["location"]
    unknown = score(job(locations=[]))[1]["location"]
    assert (here["value"], abroad["value"], unknown["value"]) == (1.0, 0.5, None)


def test_salary_only_compares_annual_amounts_in_your_currency() -> None:
    me = ScoringProfile(
        skills=ME.skills,
        years_experience=0.2,
        allowed_locations=("India",),
        min_salary=1_200_000,
        salary_currency="INR",
    )

    def value(salary: dict[str, Any]) -> float | None:
        got: float | None = score(job(salary=salary), profile=me)[1]["salary"]["value"]
        return got

    assert value({"min": 1_000_000, "max": 1_500_000, "currency": "INR", "period": "year"}) == 1.0
    assert value({"min": 600_000, "currency": "INR", "period": "year"}) == pytest.approx(0.5)
    assert value({"min": 150_000, "currency": "USD", "period": "year"}) is None
    assert value({"min": 100_000, "currency": "INR", "period": "month"}) is None


def test_freshness_halves_every_half_life() -> None:
    def value(days: float) -> float:
        return float(score(job(), age_days=days)[1]["freshness"]["value"])

    assert value(0) == 1.0
    assert value(14) == pytest.approx(0.5)
    assert value(28) == pytest.approx(0.25)


def test_similarity_is_mapped_from_floor_to_ceiling() -> None:
    def value(sim: float) -> float:
        return float(score(job(), similarity=sim)[1]["semantic"]["value"])

    assert value(0.50) == 0.0
    assert value(0.675) == pytest.approx(0.5)
    assert value(0.90) == 1.0


def test_weights_are_configurable_and_zero_removes_a_component() -> None:
    only_tech = ScoreWeights(
        semantic=0, technology=1, experience=0, location=0, salary=0, freshness=0
    )
    total, parts = score(job(technologies=["Python", "Go"]), weights=only_tech)
    assert total == pytest.approx(0.5) and parts["technology"]["weight"] == 1.0  # (1+1)/(2+2)
    assert parts["semantic"]["value"] is None


def test_a_short_technology_list_cannot_get_full_marks() -> None:
    short = technology(job(technologies=["Python", "React"]), ME)[0]
    long = technology(job(technologies=["Python", "React", "PostgreSQL", "AWS", "Go"]), ME)[0]
    assert short == pytest.approx(0.75)  # 2 of 2
    assert long == pytest.approx(5 / 7)  # 4 of 5
    assert technology(job(technologies=["Rust"] * 1 + ["Scala"]), ME)[0] == pytest.approx(0.25)


def test_better_fit_ranks_higher() -> None:
    near_miss = job(technologies=["Python", "PostgreSQL"], min_years_experience=1)
    far = job(technologies=["Rust", "Kafka", "Scala"], min_years_experience=3)
    assert score(near_miss, similarity=0.75)[0] > score(far, similarity=0.60)[0]  # type: ignore[operator]
