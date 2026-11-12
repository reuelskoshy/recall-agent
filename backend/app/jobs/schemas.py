"""What the LLM extracts from a listing, and what the rules say about it.

The extraction schema is deliberately forgiving on input (a single string
where a list is expected, "Full-time" for full_time, an unknown label) and
strict on output: every value lands in a known type or a safe default, so one
odd field never discards the whole extraction.

Fields a rule can act on carry an `evidence` quote copied from the listing.
The rules check that the quote really is in the listing (see rules.py): an
unverifiable quote can raise a flag but never reject a job.
"""

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

WorkMode = Literal["onsite", "hybrid", "remote", "unknown"]
RoleFamily = Literal["engineering", "operations", "support", "other"]
EmploymentType = Literal[
    "full_time", "internship", "apprenticeship", "contract", "part_time", "unknown"
]
SalaryPeriod = Literal["year", "month", "hour", "unknown"]
Outcome = Literal["reject", "flag", "pass", "not_checked"]
Verdict = Literal["eligible", "flagged", "rejected"]

EXTRACTOR_VERSION = "extract-2026-11-12"
EVIDENCE_FIELDS = (
    "is_job_listing",
    "locations",
    "work_mode",
    "role_family",
    "employment_type",
    "min_years_experience",
    "graduation_years",
    "degree_field",
    "salary",
)


def _label(value: Any) -> str:
    """'Full-Time ' -> 'full_time'."""
    return re.sub(r"[\s\-]+", "_", str(value or "").strip().casefold())


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


class Location(BaseModel):
    city: str = ""
    region: str = ""
    """State or province, e.g. Karnataka."""
    country: str = ""

    @field_validator("city", "region", "country", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return str(value or "").strip()[:100]


class Salary(BaseModel):
    """Annual amounts in whole currency units (12 LPA -> 1200000, INR, year)."""

    min: float | None = None
    max: float | None = None
    currency: str = ""
    period: SalaryPeriod = "unknown"

    @field_validator("period", mode="before")
    @classmethod
    def _period(cls, value: Any) -> str:
        label = _label(value)
        aliases = {"annual": "year", "yearly": "year", "per_annum": "year", "monthly": "month"}
        label = aliases.get(label, label)
        return label if label in ("year", "month", "hour") else "unknown"

    @field_validator("min", "max", mode="before")
    @classmethod
    def _amount(cls, value: Any) -> float | None:
        try:
            amount = float(str(value).replace(",", "")) if value not in (None, "") else None
        except ValueError:
            return None
        return amount if amount is not None and amount >= 0 else None

    @field_validator("currency", mode="before")
    @classmethod
    def _currency(cls, value: Any) -> str:
        text = str(value or "").strip().upper()
        return {"₹": "INR", "RS": "INR", "RS.": "INR", "$": "USD", "€": "EUR", "£": "GBP"}.get(
            text, text[:3]
        )


class ExtractedJob(BaseModel):
    """Structured facts about one listing."""

    is_job_listing: bool = True
    """False for text that is not an employer advertising a role: a person looking
    for work (common in HN threads), a news article, a list of companies."""
    locations: list[Location] = Field(default_factory=list)
    work_mode: WorkMode = "unknown"
    remote_regions: list[str] = Field(default_factory=list)
    """Where a remote job may be done from; empty means anywhere (or not remote)."""
    role_family: RoleFamily = "other"
    employment_type: EmploymentType = "unknown"
    min_years_experience: float | None = None
    graduation_years: list[int] = Field(default_factory=list)
    """Graduation years (batches) the listing requires; empty when it sets none."""
    degree_field: str = ""
    salary: Salary | None = None
    technologies: list[str] = Field(default_factory=list)
    seniority_signals: list[str] = Field(default_factory=list)
    """Phrases copied from the listing that imply seniority, e.g. "lead a team"."""
    evidence: dict[str, str] = Field(default_factory=dict)
    """field name -> the listing text the value came from, copied exactly."""

    @field_validator("is_job_listing", mode="before")
    @classmethod
    def _is_job(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().casefold() not in ("false", "no", "0")
        return value is not False and value != 0

    @field_validator("locations", mode="before")
    @classmethod
    def _locations(cls, value: Any) -> list[Any]:
        items = []
        for item in _as_list(value)[:20]:
            if isinstance(item, str):
                parts = [p.strip() for p in item.split(",")]
                item = {"city": parts[0], "country": parts[-1] if len(parts) > 1 else ""}
            if isinstance(item, dict):
                items.append(item)
        return items

    @field_validator("work_mode", mode="before")
    @classmethod
    def _work_mode(cls, value: Any) -> str:
        label = _label(value)
        aliases = {"on_site": "onsite", "office": "onsite", "in_office": "onsite", "wfh": "remote"}
        label = aliases.get(label, label)
        return label if label in ("onsite", "hybrid", "remote") else "unknown"

    @field_validator("role_family", mode="before")
    @classmethod
    def _role_family(cls, value: Any) -> str:
        label = _label(value)
        return label if label in ("engineering", "operations", "support") else "other"

    @field_validator("employment_type", mode="before")
    @classmethod
    def _employment_type(cls, value: Any) -> str:
        label = _label(value)
        aliases = {
            "fulltime": "full_time",
            "permanent": "full_time",
            "intern": "internship",
            "apprentice": "apprenticeship",
            "contractor": "contract",
            "parttime": "part_time",
        }
        label = aliases.get(label, label)
        allowed = ("full_time", "internship", "apprenticeship", "contract", "part_time")
        return label if label in allowed else "unknown"

    @field_validator("min_years_experience", mode="before")
    @classmethod
    def _years(cls, value: Any) -> float | None:
        try:
            years = float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None
        return years if years is not None and 0 <= years <= 50 else None

    @field_validator("graduation_years", mode="before")
    @classmethod
    def _grad_years(cls, value: Any) -> list[int]:
        years: list[int] = []
        for item in _as_list(value):
            try:
                year = int(str(item).strip()[:4])
            except ValueError:
                continue
            if 1980 <= year <= 2100 and year not in years:
                years.append(year)
        return sorted(years)

    @field_validator("remote_regions", "technologies", "seniority_signals", mode="before")
    @classmethod
    def _strings(cls, value: Any) -> list[str]:
        return [str(v).strip()[:200] for v in _as_list(value)[:40] if str(v).strip()]

    @field_validator("degree_field", mode="before")
    @classmethod
    def _degree(cls, value: Any) -> str:
        return str(value or "").strip()[:200]

    @field_validator("salary", mode="before")
    @classmethod
    def _salary(cls, value: Any) -> Any:
        return value if isinstance(value, dict) else None

    @field_validator("evidence", mode="before")
    @classmethod
    def _evidence(cls, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {str(k): str(v)[:500] for k, v in value.items() if k in EVIDENCE_FIELDS and v}


class Reason(BaseModel):
    """One rule's outcome, stored on the analysis so every verdict can be audited."""

    rule: str
    outcome: Outcome
    detail: str
    evidence: str | None = None
    """The exact sentence from the listing that triggered the rule, when one was found."""
