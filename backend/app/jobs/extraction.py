"""LLM extraction: one listing -> ExtractedJob.

The model only *reads*: it turns prose into fields and copies the sentence
each value came from. Every judgment (reject, flag) happens afterwards in
plain code (rules.py), where it can be tested and audited.

Parsing is lenient like memory extraction (code fences and surrounding prose
are ignored), and the schema coerces odd values to safe defaults. Output with
no JSON object at all is an `ExtractionError`.

The listing is public text; the prompt contains nothing from the profile.
"""

import asyncio
import json
import re
from datetime import date
from typing import Any

from pydantic import ValidationError

from app.jobs.schemas import ExtractedJob
from app.llm.base import ChatMessage, LLMProvider, TextDelta

MAX_LISTING_CHARS = 12_000

EXTRACTION_PROMPT = """You read job listings and extract facts into JSON. Do not judge
the job or the candidate: only report what the listing says.

Return ONLY this JSON object, no other text:
{
 "is_job_listing": true,
 "locations": [{"city": "", "region": "", "country": ""}],
 "work_mode": "onsite|hybrid|remote|unknown",
 "remote_regions": [],
 "role_family": "engineering|operations|support|other",
 "employment_type": "full_time|internship|apprenticeship|contract|part_time|unknown",
 "min_years_experience": null,
 "graduation_years": [],
 "degree_field": "",
 "salary": null,
 "technologies": [],
 "seniority_signals": [],
 "evidence": {}
}

Rules:
- is_job_listing: false when the text is NOT an employer advertising a role:
  a person describing themselves to find work ("I'm a backend engineer looking
  for..."), a news article, a list of companies, a search results page. Then
  quote the giveaway phrase in evidence.is_job_listing and leave the rest empty.
- locations: every place the job can be based, with the country written out in
  full ("Bangalore" -> {"city": "Bengaluru", "region": "Karnataka", "country":
  "India"}). Empty list if none is given.
- remote_regions: for remote jobs, where the candidate may live ("Remote, India
  only" -> ["India"]; "Remote (US/Canada)" -> ["United States", "Canada"]).
  Empty if unrestricted or not remote.
- role_family: engineering = software, data, ML/AI, QA, DevOps, SRE, security
  engineering. operations = ops, analyst, program/project management. support =
  customer or technical support. other = sales, marketing, design, HR, etc.
- min_years_experience: the lowest number asked for ("2-4 years" -> 2;
  "freshers"/"0-1 years" -> 0). null if not stated.
- graduation_years: the batches/years of graduation the listing requires, each
  year listed ("2025/2026 batch" or "2025 and 2026 passouts" -> [2025, 2026];
  "2024 onwards" -> every year from 2024 to {year_plus_one}). Empty if none.
- salary: annual amounts in whole units, or null. Indian formats: "12 LPA",
  "12 lakhs per annum" or "CTC 12L" -> 1200000 INR; "1.2 Cr" -> 12000000.
  {"min": 1200000, "max": 1800000, "currency": "INR", "period": "year"}.
- technologies: languages, frameworks, databases, cloud and tools named.
- seniority_signals: short phrases copied word for word that imply seniority,
  e.g. "lead a team of engineers", "own the architecture", "mentor juniors".
- evidence: for each of locations, work_mode, role_family, employment_type,
  min_years_experience, graduation_years, degree_field, salary that you filled
  in, the shortest exact text from the listing that supports it, copied
  character for character (the title counts as listing text). Omit a key when
  the listing says nothing about it. Never paraphrase."""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ExtractionError(Exception):
    """The model's reply could not be turned into an ExtractedJob."""


def build_messages(title: str, location_raw: str, text: str, today: date) -> list[ChatMessage]:
    clipped = text if len(text) <= MAX_LISTING_CHARS else text[:MAX_LISTING_CHARS] + "\n[...]"
    prompt = EXTRACTION_PROMPT.replace("{year_plus_one}", str(today.year + 1))
    return [
        ChatMessage(role="system", content=prompt),
        ChatMessage(
            role="user",
            content=f"Title: {title}\nLocation field: {location_raw or '(none)'}\n\n{clipped}",
        ),
    ]


def parse_extraction(reply: str) -> ExtractedJob:
    """Lenient parse; raises ExtractionError when there is no usable JSON object."""
    cleaned = _FENCE.sub("", reply.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ExtractionError("no JSON object in the reply")
    try:
        payload: Any = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ExtractionError("the reply is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ExtractionError("the reply is not a JSON object")
    try:
        return ExtractedJob.model_validate(payload)
    except ValidationError as exc:  # the validators coerce most things; this is a backstop
        raise ExtractionError(
            f"the reply does not match the schema ({exc.error_count()} errors)"
        ) from exc


async def extract(
    provider: LLMProvider,
    title: str,
    location_raw: str,
    text: str,
    *,
    timeout_s: float,
    today: date | None = None,
) -> ExtractedJob:
    """One LLM call under a deadline. Raises LLMError, TimeoutError or ExtractionError."""
    messages = build_messages(title, location_raw, text, today or date.today())
    async with asyncio.timeout(timeout_s):
        parts = [
            event.text
            async for event in provider.stream_chat(messages)
            if isinstance(event, TextDelta)
        ]
    return parse_extraction("".join(parts))
