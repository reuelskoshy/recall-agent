"""A cheap, deterministic filter that runs before LLM extraction.

Large boards list hundreds of jobs, and the per-run extraction cap would be
spent on sales roles in Dublin. This filter reads only the title and the feed's
raw location field, and it only skips a listing the rules would reject anyway:

- a title that is clearly in a role family the profile does not want (and has
  no engineering word in it), or
- a raw location that names only countries outside the allowed and sponsorship
  places (and no remote mention, when remote is acceptable).

When in doubt it keeps the listing and lets the LLM and the rules decide.
Skipped listings are not stored: they are re-checked on every run, so a
profile edit (a new country, a new role family) brings them back at once.

Senior titles are not skipped (the experience rule only flags them); they are
extracted after everything else, so the cap is spent on likely fits first.
"""

import re
from dataclasses import dataclass

from app.jobs.locations import raw_location_outside
from app.jobs.rules import ProfileFacts

# Any of these keeps the title, whatever else it says ("Sales Engineer", "Data Analyst").
_ENGINEERING = re.compile(
    r"\b(engineer\w*|developer|programmer|sde|swe|software|data|ml|machine learning|ai|"
    r"llm|devops|sre|reliability|qa|test\w*|automation|security|backend|back-end|frontend|"
    r"front-end|full[\s-]?stack|platform|infrastructure|cloud|scientist|research\w*|"
    r"architect|technical|tech)\b",
    re.IGNORECASE,
)
# Role family -> title words that clearly belong to it.
_FAMILY_TITLES: dict[str, re.Pattern[str]] = {
    "other": re.compile(
        r"\b(sales|account executive|account manager|business development|bdr|sdr|"
        r"recruit\w*|talent|sourcer|marketing|brand|growth marketer|content|copywriter|"
        r"communications|public relations|legal|counsel|paralegal|attorney|lawyer|"
        r"finance|financial|accountant|accounting|payroll|tax|treasury|audit\w*|"
        r"human resources|hr|people partner|people operations|designer|design|"
        r"executive assistant|office manager|receptionist|administrative|"
        r"partnerships|procurement|merchandis\w*)\b",
        re.IGNORECASE,
    ),
    "operations": re.compile(
        r"\b(operations|collections?|accounts payable|accounts receivable|accounts|"
        r"program manager|project manager|product manager|key account|"
        r"personal assistant|real estate|workplace|facilities|supply chain|logistics)\b",
        re.IGNORECASE,
    ),
    "support": re.compile(
        r"\b(customer support|customer success|customer service|support specialist|"
        r"support associate|help ?desk)\b",
        re.IGNORECASE,
    ),
}
SENIOR_TITLE = re.compile(
    r"\b(senior|sr|staff|principal|lead|manager|director|head|vp|vice president|"
    r"distinguished|fellow|chief)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Skip:
    rule: str  # "role_family" or "location"
    detail: str  # safe to count by rule; never logged with the title


def prefilter(title: str, location_raw: str, facts: ProfileFacts) -> Skip | None:
    """A Skip if the listing would certainly be rejected, else None (extract it)."""
    if facts.role_families and not _ENGINEERING.search(title):
        for family, pattern in _FAMILY_TITLES.items():
            if family not in facts.role_families and pattern.search(title):
                return Skip("role_family", f"a {family} title")
    if facts.allowed_locations:
        wanted = [*facts.allowed_locations, *facts.sponsorship_locations]
        outside = raw_location_outside(location_raw, wanted, facts.remote_ok)
        if outside is not None:
            return Skip("location", f"located in {outside}")
    return None


def priority(title: str) -> int:
    """Lower is extracted first: 0 for most titles, 1 for senior ones."""
    return 1 if SENIOR_TITLE.search(title) else 0
