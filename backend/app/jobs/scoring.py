"""Rank eligible and flagged jobs with a transparent weighted score in [0, 1].

score = sum(weight_i * value_i) / sum(weight_i), over the components that could
be measured. Each value is in [0, 1]:

semantic    How close the listing is to the profile in meaning: cosine
            similarity of their embeddings (bge-small), mapped linearly from
            [floor, ceiling] to [0, 1], since real pairs rarely fall below the
            floor or above the ceiling.
technology  Share of the listing's technologies that are among your skills
            (profile skills plus the technologies of your jobs and projects),
            smoothed as (matched + 1) / (listed + 2), so a listing that names
            only "SQL, Python" cannot get full marks from two words: 2 of 2
            gives 0.75, 4 of 12 gives 0.36, 0 of 16 gives 0.06.
experience  1 when you meet the minimum, falling linearly to 0 at a gap of
            MAX_EXPERIENCE_GAP years (larger gaps are rejected by the rules).
location    1 inside your allowed locations (or remote, when allowed); 0.5 in a
            sponsorship-only place, where a visa is still a hurdle.
salary      1 when the top of the range meets your minimum, else the ratio.
            Only when both amounts are annual and in the same currency.
freshness   0.5 ** (age in days / half-life), from the posting date, or the
            date the job was first seen.

A component that cannot be measured (no salary given, no technologies listed,
no minimum experience stated, the embedding failed, no profile minimum) is left
out and the others are rescaled, so missing data neither helps nor hurts.

Default weights, and why:
- semantic 0.30 and technology 0.25: fit is what the ranking is for. Two views
  of it, one fuzzy (meaning) and one exact (named tools), so neither alone
  decides.
- experience 0.20: for an early-career candidate the experience gap is the
  biggest practical barrier. Large gaps are already rejected; this orders the
  near-misses.
- freshness 0.10: early applications do better, but an older great fit should
  still beat a new poor one.
- location 0.10: the rules already removed disallowed places, so this only
  separates "here" from "with a visa".
- salary 0.05: most listings, Indian ones especially, do not state a salary.
All weights are settings (JOBS_SCORE_W_*), and each stored component keeps
its value, effective weight and contribution, so the UI can explain a rank.
"""

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.config import Settings
from app.db.models import CandidateProfile
from app.jobs.locations import judge_location, sponsorship_place
from app.jobs.rules import MAX_EXPERIENCE_GAP
from app.jobs.schemas import ExtractedJob

SCORING_VERSION = "score-2026-11-12b"  # smoothed technology overlap

COMPONENTS = ("semantic", "technology", "experience", "location", "salary", "freshness")

# Spellings of the same technology -> one canonical name (lower case).
_TECH_ALIASES: dict[str, str] = {
    "postgres": "postgresql",
    "psql": "postgresql",
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "golang": "go",
    "node": "node.js",
    "nodejs": "node.js",
    "reactjs": "react",
    "react.js": "react",
    "amazon web services": "aws",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    "ci cd": "ci/cd",
    "cicd": "ci/cd",
    "github action": "github actions",
    "rest": "rest apis",
    "rest api": "rest apis",
    "restful apis": "rest apis",
    "llm": "llms",
    "large language models": "llms",
    "genai": "generative ai",
    "gen ai": "generative ai",
    "retrieval augmented generation": "rag",
    "retrieval-augmented generation": "rag",
    "sklearn": "scikit-learn",
}
_TECH_SPACE = re.compile(r"[\s_\-]+")


def canonical_tech(name: str) -> str:
    key = _TECH_SPACE.sub(" ", name.strip().casefold())
    return _TECH_ALIASES.get(key, key)


def _tech_matches(tech: str, skills: frozenset[str]) -> bool:
    """Equal, or one is the other plus more words: "aws" ~ "aws lambda"."""
    if tech in skills:
        return True
    return any(s.startswith(tech + " ") or tech.startswith(s + " ") for s in skills)


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    semantic: float = 0.30
    technology: float = 0.25
    experience: float = 0.20
    location: float = 0.10
    salary: float = 0.05
    freshness: float = 0.10
    freshness_half_life_days: float = 14.0
    similarity_floor: float = 0.55
    similarity_ceiling: float = 0.80

    @classmethod
    def from_settings(cls, settings: Settings) -> "ScoreWeights":
        return cls(
            semantic=settings.jobs_score_w_semantic,
            technology=settings.jobs_score_w_technology,
            experience=settings.jobs_score_w_experience,
            location=settings.jobs_score_w_location,
            salary=settings.jobs_score_w_salary,
            freshness=settings.jobs_score_w_freshness,
            freshness_half_life_days=settings.jobs_score_freshness_half_life_days,
            similarity_floor=settings.jobs_score_similarity_floor,
            similarity_ceiling=settings.jobs_score_similarity_ceiling,
        )

    def weight(self, component: str) -> float:
        return float(getattr(self, component))


@dataclass(frozen=True, slots=True)
class ScoringProfile:
    """The profile values the score reads."""

    skills: frozenset[str] = frozenset()
    years_experience: float = 0.0
    allowed_locations: tuple[str, ...] = ()
    sponsorship_locations: tuple[str, ...] = ()
    remote_ok: bool = True
    min_salary: int | None = None
    salary_currency: str | None = None

    @classmethod
    def from_profile(
        cls, profile: CandidateProfile, skill_names: Iterable[str]
    ) -> "ScoringProfile":
        return cls(
            skills=frozenset(canonical_tech(s) for s in skill_names if s.strip()),
            years_experience=float(profile.years_experience or 0),
            allowed_locations=tuple(profile.allowed_locations or ()),
            sponsorship_locations=tuple(profile.sponsorship_locations or ()),
            remote_ok=bool(profile.remote_ok),
            min_salary=profile.min_salary,
            salary_currency=(profile.salary_currency or "").upper() or None,
        )


def profile_text(headline: str, summary: str, skills: Sequence[str]) -> str:
    """What the profile embedding is made from."""
    return "\n".join(p for p in (headline, summary, "Skills: " + ", ".join(skills)) if p.strip())


def job_text(title: str, job: ExtractedJob, raw_text: str, max_chars: int = 2000) -> str:
    """What a listing's embedding is made from (the model reads ~512 tokens)."""
    techs = ", ".join(job.technologies[:30])
    return "\n".join(p for p in (title, techs and f"Technologies: {techs}", raw_text) if p)[
        :max_chars
    ]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


Measured = tuple[float | None, str]  # (value in [0, 1] or None if not measurable, detail)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def semantic(similarity: float | None, w: ScoreWeights) -> Measured:
    if similarity is None:
        return None, "No embedding for this listing or the profile."
    span = w.similarity_ceiling - w.similarity_floor
    value = _clamp((similarity - w.similarity_floor) / span) if span > 0 else 0.0
    return value, f"Cosine similarity {similarity:.3f} to your profile."


def technology(job: ExtractedJob, p: ScoringProfile) -> Measured:
    techs = list(dict.fromkeys(canonical_tech(t) for t in job.technologies if t.strip()))
    if not techs:
        return None, "The listing names no technologies."
    if not p.skills:
        return None, "Your profile has no skills."
    matched = [t for t in techs if _tech_matches(t, p.skills)]
    missing = [t for t in techs if t not in matched]
    detail = f"{len(matched)} of {len(techs)} technologies are among your skills."
    if matched:
        detail += f" Matched: {', '.join(matched[:8])}."
    if missing:
        detail += f" Missing: {', '.join(missing[:8])}."
    return (len(matched) + 1) / (len(techs) + 2), detail


def experience(job: ExtractedJob, p: ScoringProfile) -> Measured:
    needed = job.min_years_experience
    if needed is None:
        return None, "No minimum experience stated."
    gap = needed - p.years_experience
    if gap <= 0:
        return 1.0, f"Asks for {needed:g}+ years; you have {p.years_experience:g}."
    value = _clamp(1 - gap / MAX_EXPERIENCE_GAP)
    return value, f"Asks for {needed:g}+ years; a gap of {gap:g}."


def location(job: ExtractedJob, p: ScoringProfile) -> Measured:
    if not p.allowed_locations:
        return None, "Your profile has no allowed locations."
    outcome, detail = judge_location(job, list(p.allowed_locations), p.remote_ok)
    if outcome == "match":
        return 1.0, detail[:1].upper() + detail[1:] + "."
    if outcome == "unknown":
        return None, "The listing gives no location."
    place = sponsorship_place(job, list(p.sponsorship_locations), p.remote_ok)
    if place is not None:
        return 0.5, f"In {place}: needs a work visa."
    return 0.0, f"Location: {detail}."


def salary(job: ExtractedJob, p: ScoringProfile) -> Measured:
    if p.min_salary is None:
        return None, "Your profile sets no minimum salary."
    s = job.salary
    top = None if s is None else (s.max if s.max is not None else s.min)
    if s is None or top is None:
        return None, "The listing gives no salary."
    if s.period != "year" or not p.salary_currency or s.currency != p.salary_currency:
        return None, "The salary is not annual or not in your currency."
    if p.min_salary == 0:
        return 1.0, "Any salary meets your minimum."
    value = _clamp(top / p.min_salary)
    return value, f"Up to {top:,.0f} {s.currency}; your minimum is {p.min_salary:,} {s.currency}."


def freshness(seen_at: datetime, now: datetime, w: ScoreWeights) -> Measured:
    age_days = max(0.0, (now - seen_at).total_seconds() / 86400)
    value = 0.5 ** (age_days / w.freshness_half_life_days)
    return value, f"Posted or first seen {age_days:.0f} day(s) ago."


def score_job(
    job: ExtractedJob,
    profile: ScoringProfile,
    weights: ScoreWeights,
    *,
    similarity: float | None,
    seen_at: datetime,
    now: datetime,
) -> tuple[float | None, dict[str, Any]]:
    """(score, components). Score is None only if nothing could be measured."""
    measured: dict[str, Measured] = {
        "semantic": semantic(similarity, weights),
        "technology": technology(job, profile),
        "experience": experience(job, profile),
        "location": location(job, profile),
        "salary": salary(job, profile),
        "freshness": freshness(seen_at, now, weights),
    }
    used = {
        name: weights.weight(name)
        for name, (value, _) in measured.items()
        if value is not None and weights.weight(name) > 0
    }
    total = sum(used.values())
    components: dict[str, Any] = {}
    score = 0.0
    for name, (value, detail) in measured.items():
        if name in used and total > 0:
            weight = used[name] / total
            contribution = weight * float(value or 0)
            score += contribution
            components[name] = {
                "value": round(float(value or 0), 4),
                "weight": round(weight, 4),
                "contribution": round(contribution, 4),
                "detail": detail,
            }
        else:
            components[name] = {"value": None, "weight": 0.0, "contribution": 0.0, "detail": detail}
    if total <= 0:
        return None, components
    return round(score, 4), components
