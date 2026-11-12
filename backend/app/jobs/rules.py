"""Eligibility rules: plain functions over (extracted listing, profile). No LLM.

Each rule returns a `Reason`: the rule name, an outcome (reject / flag / pass /
not_checked), a human explanation, and the exact sentence from the listing
that triggered it. The verdict is the worst outcome: any reject -> rejected,
else any flag -> flagged, else eligible.

**Evidence is verified, not trusted.** The LLM copies a quote for each value it
extracts. Before a rule may *reject*, the quote must actually appear in the
listing (whitespace and case aside), and the stored evidence is the listing's
own sentence around it. If the quote is missing or invented, the rule can
only *flag*: a hallucinated "2027 batch only" must never silently drop a job.

Rules run on every pipeline run (they are cheap), so a profile edit or a
change here (bump RULES_VERSION) re-judges every job without an LLM call.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from app.db.models import CandidateProfile
from app.jobs.locations import judge_location, sponsorship_place
from app.jobs.schemas import ExtractedJob, Reason, Verdict

RULES_VERSION = "rules-2026-11-12c"  # + large experience gaps reject
# A gap above this many years rejects (with a verified quote); smaller gaps flag.
MAX_EXPERIENCE_GAP = 3.0
UNVERIFIED = " The model's quote was not found in the listing, so this is a flag, not a rejection."

JUNIOR_TITLE = re.compile(
    r"\b(junior|jr|entry[\s-]?level|associate|graduate|grad|fresher|freshers|trainee|intern|"
    r"apprentice|new[\s-]grad|early[\s-]career|sde[\s-]?(?:1|i)|swe[\s-]?(?:1|i)|"
    r"(?:engineer|developer)[\s-]+(?:1|i))\b",
    re.IGNORECASE,
)
# Typographic quotes, dashes and no-break spaces -> ASCII, so a quote copied
# with "smart" punctuation still matches the listing.
_QUOTES = str.maketrans(
    {0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0x2013: "-", 0x2014: "-", 0xA0: " "}
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
MIN_QUOTE_CHARS = 3


@dataclass(frozen=True, slots=True)
class ProfileFacts:
    """The profile values the rules read."""

    graduation_year: int | None = None
    years_experience: float = 0.0
    allowed_locations: list[str] = field(default_factory=list)
    remote_ok: bool = True
    role_families: list[str] = field(default_factory=lambda: ["engineering"])
    sponsorship_locations: list[str] = field(default_factory=list)

    @classmethod
    def from_profile(cls, profile: CandidateProfile) -> "ProfileFacts":
        return cls(
            graduation_year=profile.graduation_year,
            years_experience=float(profile.years_experience or 0),
            allowed_locations=list(profile.allowed_locations or []),
            remote_ok=bool(profile.remote_ok),
            role_families=list(profile.role_families or []),
            sponsorship_locations=list(profile.sponsorship_locations or []),
        )


@dataclass(frozen=True, slots=True)
class Listing:
    """The text a quote must be found in: title, the feed's location field, the body."""

    title: str
    location_raw: str
    text: str

    @property
    def full_text(self) -> str:
        return "\n".join((self.title, self.location_raw, self.text))


def _norm(text: str) -> str:
    return " ".join(text.translate(_QUOTES).split()).casefold()


def find_sentence(listing: Listing, quote: str | None) -> str | None:
    """The listing's own sentence containing `quote`, or None if the quote is not there."""
    needle = _norm(quote or "").strip(" .\"'")
    if len(needle) < MIN_QUOTE_CHARS:
        return None
    full = listing.full_text
    if needle not in _norm(full):
        return None
    for sentence in _SENTENCE_SPLIT.split(full):
        if needle in _norm(sentence):
            return sentence.strip()[:500]
    return (quote or "").strip()[:500]  # the quote spans sentences: keep it whole


Rule = Callable[[ExtractedJob, ProfileFacts, Listing], Reason]


def _reject_or_flag(rule: str, detail: str, evidence: str | None) -> Reason:
    if evidence is None:
        return Reason(rule=rule, outcome="flag", detail=detail + UNVERIFIED)
    return Reason(rule=rule, outcome="reject", detail=detail, evidence=evidence)


def job_listing(job: ExtractedJob, facts: ProfileFacts, listing: Listing) -> Reason:
    """Feeds like HN and web search also return text that is not a job at all."""
    rule = "job_listing"
    if job.is_job_listing:
        return Reason(rule=rule, outcome="pass", detail="A job listing.")
    evidence = find_sentence(listing, job.evidence.get("is_job_listing"))
    return _reject_or_flag(
        rule, "Not a job listing (for example, someone looking for work).", evidence
    )


def graduation_year(job: ExtractedJob, facts: ProfileFacts, listing: Listing) -> Reason:
    rule = "graduation_year"
    years = job.graduation_years
    if not years:
        return Reason(rule=rule, outcome="pass", detail="No graduation year required.")
    if facts.graduation_year is None:
        return Reason(
            rule=rule, outcome="not_checked", detail="Your profile has no graduation year."
        )
    evidence = find_sentence(listing, job.evidence.get("graduation_years"))
    wanted = ", ".join(map(str, years))
    mine = facts.graduation_year
    if mine in years:
        return Reason(
            rule=rule, outcome="pass", detail=f"Open to {mine} graduates.", evidence=evidence
        )
    if min(years) > mine:
        return _reject_or_flag(
            rule, f"Requires graduation in {wanted}; you graduated in {mine}.", evidence
        )
    return Reason(
        rule=rule,
        outcome="flag",
        detail=f"Requires graduation in {wanted}; you graduated in {mine}. Check before applying.",
        evidence=evidence,
    )


def role_family(job: ExtractedJob, facts: ProfileFacts, listing: Listing) -> Reason:
    rule = "role_family"
    if not facts.role_families:
        return Reason(rule=rule, outcome="not_checked", detail="Your profile has no role types.")
    evidence = find_sentence(listing, job.evidence.get("role_family"))
    if job.role_family in facts.role_families:
        return Reason(
            rule=rule, outcome="pass", detail=f"An {job.role_family} role.", evidence=evidence
        )
    wanted = ", ".join(facts.role_families)
    return _reject_or_flag(rule, f"A {job.role_family} role; you want {wanted}.", evidence)


def location(job: ExtractedJob, facts: ProfileFacts, listing: Listing) -> Reason:
    rule = "location"
    if not facts.allowed_locations:
        return Reason(
            rule=rule, outcome="not_checked", detail="Your profile has no allowed locations."
        )
    outcome, detail = judge_location(job, facts.allowed_locations, facts.remote_ok)
    quote = job.evidence.get("locations") or job.evidence.get("work_mode")
    evidence = find_sentence(listing, quote) or (listing.location_raw.strip() or None)
    if outcome == "match":
        return Reason(
            rule=rule, outcome="pass", detail=detail.capitalize() + ".", evidence=evidence
        )
    if outcome == "unknown":
        return Reason(rule=rule, outcome="flag", detail="The listing gives no location.")
    place = sponsorship_place(job, facts.sponsorship_locations, facts.remote_ok)
    if place is not None:
        return _sponsorship(job, listing, place, evidence)
    return _reject_or_flag(rule, f"Location: {detail}.", evidence)


def _sponsorship(job: ExtractedJob, listing: Listing, place: str, located: str | None) -> Reason:
    """Outside the allowed list, but somewhere acceptable with a work visa.

    Offered -> pass (only with a verified quote). Explicitly not offered ->
    reject (only with a verified quote). Not mentioned -> flag, to check by hand.
    """
    rule = "location"
    quote = find_sentence(listing, job.evidence.get("visa_sponsorship"))
    if job.visa_sponsorship == "offered":
        if quote is None:
            return Reason(
                rule=rule,
                outcome="flag",
                detail=f"In {place}. The model says visa sponsorship is offered, but its "
                "quote was not found in the listing. Check before applying.",
                evidence=located,
            )
        return Reason(
            rule=rule,
            outcome="pass",
            detail=f"In {place}, with visa sponsorship offered.",
            evidence=quote,
        )
    if job.visa_sponsorship == "not_offered":
        return _reject_or_flag(rule, f"In {place}, and no visa sponsorship.", quote)
    return Reason(
        rule=rule,
        outcome="flag",
        detail=f"In {place}, and the listing does not mention visa sponsorship.",
        evidence=located,
    )


def experience(job: ExtractedJob, facts: ProfileFacts, listing: Listing) -> Reason:
    rule = "experience"
    needed = job.min_years_experience
    if needed is None:
        return Reason(rule=rule, outcome="pass", detail="No minimum experience stated.")
    evidence = find_sentence(listing, job.evidence.get("min_years_experience"))
    mine = facts.years_experience
    if needed <= mine:
        return Reason(
            rule=rule,
            outcome="pass",
            detail=f"Asks for {needed:g}+ years; you have {mine:g}.",
            evidence=evidence,
        )
    gap = needed - mine
    detail = f"Asks for {needed:g}+ years; you have {mine:g} (gap of {gap:g})."
    if gap > MAX_EXPERIENCE_GAP:
        return _reject_or_flag(rule, detail, evidence)
    return Reason(rule=rule, outcome="flag", detail=detail, evidence=evidence)


def internship(job: ExtractedJob, facts: ProfileFacts, listing: Listing) -> Reason:
    rule = "internship"
    if job.employment_type != "internship":
        return Reason(rule=rule, outcome="pass", detail="Not an internship.")
    evidence = find_sentence(listing, job.evidence.get("employment_type"))
    if job.graduation_years:
        return Reason(
            rule=rule,
            outcome="pass",
            detail="An internship with a stated graduation year (checked separately).",
            evidence=evidence,
        )
    return Reason(
        rule=rule,
        outcome="flag",
        detail="An internship with no graduation-year requirement: "
        "check whether graduates can apply.",
        evidence=evidence,
    )


def seniority(job: ExtractedJob, facts: ProfileFacts, listing: Listing) -> Reason:
    rule = "seniority"
    if not JUNIOR_TITLE.search(listing.title):
        return Reason(rule=rule, outcome="pass", detail="The title is not a junior title.")
    found = [(s, find_sentence(listing, s)) for s in job.seniority_signals]
    verified = [(s, sentence) for s, sentence in found if sentence]
    if not verified:
        return Reason(rule=rule, outcome="pass", detail="A junior title with no senior duties.")
    phrases = "; ".join(f'"{s}"' for s, _ in verified[:3])
    return Reason(
        rule=rule,
        outcome="flag",
        detail=f"A junior title, but the listing also says {phrases}.",
        evidence=verified[0][1],
    )


RULES: tuple[Rule, ...] = (
    job_listing,
    graduation_year,
    role_family,
    location,
    experience,
    internship,
    seniority,
)


def judge(job: ExtractedJob, facts: ProfileFacts, listing: Listing) -> tuple[Verdict, list[Reason]]:
    """Run every rule. Rejections first, then flags, then the rest (stable order)."""
    reasons = [rule(job, facts, listing) for rule in RULES]
    order = {"reject": 0, "flag": 1, "pass": 2, "not_checked": 3}
    reasons.sort(key=lambda r: order[r.outcome])
    outcomes = {r.outcome for r in reasons}
    verdict: Verdict = (
        "rejected" if "reject" in outcomes else "flagged" if "flag" in outcomes else "eligible"
    )
    return verdict, reasons
