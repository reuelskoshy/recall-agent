"""Does a job's location fall inside the candidate's allowed list?

Allowed entries can be countries ("India"), states ("Karnataka"), cities
("Bengaluru") or groups ("Delhi NCR"). Matching is done on canonical names, so
"Bangalore", "Bengaluru" and "BLR" are the same place. Cities in the table also
know their country, so a listing that says only "Pune" still counts as India.

The LLM normalises each location to city/region/country (it knows far more
places than this table). The table exists for the matching itself: aliases,
groups and cities whose country the model left out. It is focused on India,
which is where the candidate is looking; other countries match by name.
"""

import re
from collections.abc import Iterable
from typing import Literal

from app.jobs.schemas import ExtractedJob, Location

LocationOutcome = Literal["match", "outside", "unknown"]

INDIA = "india"

# canonical -> aliases (all lower case)
_ALIASES: dict[str, tuple[str, ...]] = {
    INDIA: ("in", "ind", "bharat", "republic of india"),
    "bengaluru": ("bangalore", "blr", "bengaluru urban"),
    "mumbai": ("bombay", "navi mumbai", "thane"),
    "chennai": ("madras",),
    "kolkata": ("calcutta",),
    "gurugram": ("gurgaon",),
    "delhi": ("new delhi", "ncr delhi"),
    "kochi": ("cochin", "ernakulam"),
    "thiruvananthapuram": ("trivandrum", "tvm"),
    "mysuru": ("mysore",),
    "puducherry": ("pondicherry",),
    "vadodara": ("baroda",),
    "visakhapatnam": ("vizag",),
    "prayagraj": ("allahabad",),
    "hyderabad": ("secunderabad", "cyberabad"),
    "pune": ("poona",),
    "delhi ncr": ("ncr", "national capital region", "delhi/ncr", "delhi-ncr"),
    "united states": ("us", "usa", "united states of america", "america"),
    "united kingdom": ("uk", "great britain", "britain", "england"),
    "united arab emirates": ("uae",),
}

# Indian cities (canonical) -> state. Used to infer the country and to match states.
_INDIAN_CITIES: dict[str, str] = {
    "bengaluru": "karnataka",
    "mysuru": "karnataka",
    "mangaluru": "karnataka",
    "mumbai": "maharashtra",
    "pune": "maharashtra",
    "nagpur": "maharashtra",
    "hyderabad": "telangana",
    "chennai": "tamil nadu",
    "coimbatore": "tamil nadu",
    "kolkata": "west bengal",
    "delhi": "delhi",
    "gurugram": "haryana",
    "faridabad": "haryana",
    "noida": "uttar pradesh",
    "greater noida": "uttar pradesh",
    "ghaziabad": "uttar pradesh",
    "lucknow": "uttar pradesh",
    "prayagraj": "uttar pradesh",
    "ahmedabad": "gujarat",
    "gandhinagar": "gujarat",
    "vadodara": "gujarat",
    "surat": "gujarat",
    "jaipur": "rajasthan",
    "chandigarh": "chandigarh",
    "mohali": "punjab",
    "kochi": "kerala",
    "thiruvananthapuram": "kerala",
    "kozhikode": "kerala",
    "indore": "madhya pradesh",
    "bhopal": "madhya pradesh",
    "bhubaneswar": "odisha",
    "visakhapatnam": "andhra pradesh",
    "vijayawada": "andhra pradesh",
    "goa": "goa",
    "puducherry": "puducherry",
}
_INDIAN_STATES = frozenset(_INDIAN_CITIES.values()) | {
    "assam",
    "bihar",
    "jharkhand",
    "uttarakhand",
    "himachal pradesh",
    "chhattisgarh",
    "jammu and kashmir",
}

# Groups that contain several cities.
_GROUPS: dict[str, frozenset[str]] = {
    "delhi ncr": frozenset(
        {"delhi", "gurugram", "noida", "greater noida", "ghaziabad", "faridabad"}
    ),
}

# Remote-work regions that include each country (lower case).
_REGIONS_CONTAINING: dict[str, frozenset[str]] = {
    INDIA: frozenset(
        {"apac", "asia", "asia pacific", "asia-pacific", "south asia", "indian subcontinent"}
    ),
}
_ANYWHERE = frozenset(
    {"anywhere", "worldwide", "global", "remote", "fully remote", "work from anywhere"}
)

_CANONICAL = {alias: name for name, aliases in _ALIASES.items() for alias in aliases}

# Countries the pre-extraction filter recognises in a feed's raw location field.
# Anything not listed here is "unknown", and unknown places are never skipped.
_COUNTRIES = frozenset(
    {
        INDIA,
        "united states",
        "united kingdom",
        "united arab emirates",
        "canada",
        "mexico",
        "brazil",
        "argentina",
        "colombia",
        "chile",
        "ireland",
        "germany",
        "france",
        "netherlands",
        "belgium",
        "spain",
        "portugal",
        "italy",
        "switzerland",
        "austria",
        "sweden",
        "norway",
        "denmark",
        "finland",
        "poland",
        "czech republic",
        "romania",
        "israel",
        "saudi arabia",
        "qatar",
        "singapore",
        "malaysia",
        "indonesia",
        "philippines",
        "vietnam",
        "thailand",
        "japan",
        "south korea",
        "china",
        "hong kong",
        "taiwan",
        "australia",
        "new zealand",
        "south africa",
        "nigeria",
        "kenya",
        "egypt",
    }
)
_RAW_SPLIT = re.compile(r"[,;/|()\-\u2013\u2014]+")  # also en and em dashes
_REMOTE_WORD = re.compile(r"\b(remote|anywhere|worldwide|distributed|wfh)\b", re.IGNORECASE)


def canonical(name: str) -> str:
    """'  Bangalore ' -> 'bengaluru'; 'USA' -> 'united states'; unknown names lower-cased."""
    key = " ".join(name.replace(".", "").split()).casefold()
    return _CANONICAL.get(key, key)


def country_of(location: Location) -> str:
    """Canonical country, inferred from a known city or state when the model left it out."""
    if location.country:
        return canonical(location.country)
    city, region = canonical(location.city), canonical(location.region)
    if city in _INDIAN_CITIES or city in _GROUPS or region in _INDIAN_STATES:
        return INDIA
    return ""


def _places(location: Location) -> set[str]:
    """Every name this location answers to: city, its groups, state, country."""
    city = canonical(location.city)
    names = {city, canonical(location.region), country_of(location)}
    if city in _INDIAN_CITIES:
        names.add(_INDIAN_CITIES[city])
    names |= {group for group, members in _GROUPS.items() if city in members}
    if city in _GROUPS:  # "Delhi NCR" itself
        names.add(city)
    names.discard("")
    return names


def location_matches(location: Location, allowed: Iterable[str]) -> bool:
    """True if the location is inside any allowed place."""
    places = _places(location)
    return any(canonical(a) in places for a in allowed if a.strip())


def allowed_countries(allowed: Iterable[str]) -> set[str]:
    """The countries the allowed places are in ("Bengaluru" -> india)."""
    countries: set[str] = set()
    for entry in allowed:
        name = canonical(entry)
        if not name:
            continue
        if name in _INDIAN_CITIES or name in _GROUPS or name in _INDIAN_STATES:
            countries.add(INDIA)
        else:
            countries.add(name)
    return countries


def remote_region_allows(regions: list[str], allowed: Iterable[str]) -> bool:
    """A remote job's regions include somewhere allowed (no regions = anywhere)."""
    if not regions:
        return True
    countries = allowed_countries(allowed)
    allowed_names = {canonical(a) for a in allowed} | countries
    for region in regions:
        name = canonical(region)
        if name in _ANYWHERE or name in allowed_names:
            return True
        if any(name in _REGIONS_CONTAINING.get(c, frozenset()) for c in countries):
            return True
    return False


def judge_location(
    job: ExtractedJob, allowed: list[str], remote_ok: bool
) -> tuple[LocationOutcome, str]:
    """('match' | 'outside' | 'unknown', explanation)."""
    if job.work_mode == "remote" and remote_ok:
        if remote_region_allows(job.remote_regions, allowed):
            where = ", ".join(job.remote_regions) or "anywhere"
            return "match", f"remote ({where})"
        if not any(location_matches(loc, allowed) for loc in job.locations):
            return "outside", f"remote, but only from {', '.join(job.remote_regions)}"
    if not job.locations:
        if job.work_mode == "remote" and not remote_ok:
            return "outside", "remote only, and remote work is not in your preferences"
        return "unknown", "the listing gives no location"
    matching = [loc for loc in job.locations if location_matches(loc, allowed)]
    if matching:
        return "match", f"{describe(matching[0])} is in your allowed locations"
    return "outside", f"{'; '.join(describe(loc) for loc in job.locations[:5])} not allowed"


def sponsorship_place(job: ExtractedJob, places: list[str], remote_ok: bool) -> str | None:
    """The first of the job's locations (or remote regions) inside `places`, if any.

    Used for places the candidate accepts only with visa sponsorship.
    """
    if not places:
        return None
    for loc in job.locations:
        if location_matches(loc, places):
            return describe(loc)
    if job.work_mode == "remote" and remote_ok:
        wanted = {canonical(p) for p in places}
        for region in job.remote_regions:
            if canonical(region) in wanted:
                return f"remote ({region})"
    return None


def raw_location_outside(raw: str, wanted: Iterable[str], remote_ok: bool) -> str | None:
    """For the pre-extraction filter: the known countries a feed's raw location names,
    when it names at least one and nothing inside `wanted`.

    Returns None ("keep") whenever there is doubt: an empty field, a remote
    mention (when remote is acceptable), a place in a wanted country, or no
    recognisable country at all. "London, UK" -> "united kingdom"; "London" -> None.
    """
    if not raw.strip():
        return None
    if remote_ok and _REMOTE_WORD.search(raw):
        return None
    wanted = list(wanted)
    countries = allowed_countries(wanted)
    wanted_names = {canonical(w) for w in wanted if w.strip()} | countries
    found: set[str] = set()
    for part in _RAW_SPLIT.split(raw):
        name = canonical(part)
        if not name:
            continue
        if name in wanted_names:
            return None
        if name in _INDIAN_CITIES or name in _GROUPS or name in _INDIAN_STATES:
            if INDIA in countries:
                return None
            found.add(INDIA)
        elif name in _COUNTRIES:
            found.add(name)
    if not found or found & countries:
        return None
    return ", ".join(sorted(found))


def describe(location: Location) -> str:
    return ", ".join(p for p in (location.city, location.region, location.country) if p) or "?"
