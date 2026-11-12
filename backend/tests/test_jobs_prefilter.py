"""The pre-extraction filter: skips only what the rules would certainly reject."""

import pytest

from app.jobs.locations import raw_location_outside
from app.jobs.prefilter import prefilter, priority
from app.jobs.rules import ProfileFacts

FACTS = ProfileFacts(
    allowed_locations=["India"],
    sponsorship_locations=["United States"],
    role_families=["engineering"],
)


@pytest.mark.parametrize(
    "title",
    [
        "Account Executive, Enterprise",
        "Senior Recruiter",
        "Product Designer",
        "Legal Counsel",
        "Customer Success Manager",
        "HR Business Partner",
        "Product Manager II",  # operations, per the extraction prompt
    ],
)
def test_non_engineering_titles_are_skipped(title: str) -> None:
    skip = prefilter(title, "Bengaluru, India", FACTS)
    assert skip is not None and skip.rule == "role_family"


@pytest.mark.parametrize(
    "title",
    [
        "Software Engineer",
        "Sales Engineer",  # an engineering word wins: the LLM decides
        "Data Analyst",
        "Design Engineer",
        "Security Operations",
        "Solutions Consultant",  # matches no list: kept
        "SDE-1",
    ],
)
def test_engineering_or_unclear_titles_are_kept(title: str) -> None:
    assert prefilter(title, "Bengaluru, India", FACTS) is None


def test_a_wanted_role_family_is_not_skipped() -> None:
    facts = ProfileFacts(allowed_locations=["India"], role_families=["engineering", "support"])
    assert prefilter("Customer Support Specialist", "Pune", facts) is None


def test_no_role_families_means_no_title_filter() -> None:
    facts = ProfileFacts(allowed_locations=["India"], role_families=[])
    assert prefilter("Account Executive", "Pune", facts) is None


@pytest.mark.parametrize(
    "raw",
    [
        "London, UK",
        "Dublin, Ireland",
        "Singapore",
        "Berlin, Germany; Amsterdam, Netherlands",
        "Toronto - Canada",
    ],
)
def test_locations_only_in_unwanted_countries_are_skipped(raw: str) -> None:
    skip = prefilter("Software Engineer", raw, FACTS)
    assert skip is not None and skip.rule == "location"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "Bengaluru",
        "Bangalore, Karnataka",
        "IN-Bangalore",
        "London, UK; Hyderabad, India",
        "Gurgaon / Remote",
        "Remote - Europe",  # remote: the LLM reads the region restriction
        "San Francisco, CA",  # no recognisable country: kept
        "New York, NY, United States",  # a sponsorship place
        "London",  # a city with no country: kept
        "Delhi-NCR",
    ],
)
def test_wanted_or_unclear_locations_are_kept(raw: str) -> None:
    assert prefilter("Software Engineer", raw, FACTS) is None


def test_remote_is_not_a_way_in_when_remote_is_unwanted() -> None:
    facts = ProfileFacts(allowed_locations=["India"], remote_ok=False)
    assert raw_location_outside("Remote - Germany", ["India"], remote_ok=False) == "germany"
    assert prefilter("Software Engineer", "Remote - Germany", facts) is not None


def test_no_allowed_locations_means_no_location_filter() -> None:
    facts = ProfileFacts(allowed_locations=[])
    assert prefilter("Software Engineer", "London, UK", facts) is None


def test_a_city_in_the_allowed_list_keeps_its_country() -> None:
    facts = ProfileFacts(allowed_locations=["Bengaluru"])
    assert prefilter("Software Engineer", "Pune, India", facts) is None
    assert prefilter("Software Engineer", "Austin, United States", facts) is not None


def test_senior_titles_go_last() -> None:
    assert priority("Software Engineer") == 0
    assert priority("SDE-1") == 0
    assert priority("Senior Software Engineer") == 1
    assert priority("Staff Engineer, Platform") == 1
    assert priority("Engineering Manager") == 1


def test_raw_locations_split_on_dashes_but_not_digits() -> None:
    en_dash, em_dash = chr(0x2013), chr(0x2014)
    assert raw_location_outside(f"Dublin{en_dash}Ireland", ["India"], True) == "ireland"
    assert raw_location_outside(f"Berlin {em_dash} Germany", ["India"], True) == "germany"
    assert raw_location_outside("Pune 411001, India", ["India"], True) is None
