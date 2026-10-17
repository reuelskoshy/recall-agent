"""The Acme sample dataset: a small, fictional company database for query_database.

It is deterministic (seeded RNG, fixed dates), so the same questions always have
the same answers, which the Phase 6 eval relies on. Acme is also the company in
`samples/acme-employee-handbook.md`, so one question can need both the handbook
(RAG) and the data (SQL), e.g. leave carry-over rules vs. leave balances.

The table definitions are the single source of truth: the seed script creates
them from this metadata and the query tool describes them to the model from it.
Column comments are part of that description, so they say what values mean.
"""

import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
)

SCHEMA = "acme"
DATASET_VERSION = "acme-v1"
"""Bump when the data or tables change; the seed script then rebuilds the schema."""

metadata = MetaData(schema=SCHEMA)

departments = Table(
    "departments",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(60), nullable=False, unique=True),
    Column("location", String(60), nullable=False, comment="Office city"),
    Column("annual_budget", Numeric(12, 2), nullable=False, comment="EUR per year"),
    comment="Company departments",
)

employees = Table(
    "employees",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("first_name", String(40), nullable=False),
    Column("last_name", String(40), nullable=False),
    Column("email", String(120), nullable=False, unique=True),
    Column("department_id", ForeignKey("departments.id"), nullable=False),
    Column("manager_id", ForeignKey("employees.id"), comment="NULL for department heads"),
    Column("title", String(80), nullable=False),
    Column("hire_date", Date, nullable=False),
    Column("salary", Numeric(10, 2), nullable=False, comment="Gross EUR per year"),
    Column("country", String(40), nullable=False),
    Column("is_remote", Boolean, nullable=False, comment="Works remotely most of the week"),
    comment="Current employees",
)

leave_balances = Table(
    "leave_balances",
    metadata,
    Column("employee_id", ForeignKey("employees.id"), primary_key=True),
    Column("year", Integer, primary_key=True),
    Column("entitled_days", Integer, nullable=False, comment="24 per year (handbook 2.1)"),
    Column("carried_over_days", Integer, nullable=False, comment="From last year, max 10"),
    Column("taken_days", Integer, nullable=False, comment="Days of annual leave used so far"),
    comment="Annual leave per employee per year; remaining = entitled + carried_over - taken",
)

customers = Table(
    "customers",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(80), nullable=False, unique=True),
    Column("industry", String(40), nullable=False),
    Column("country", String(40), nullable=False),
    Column("segment", String(20), nullable=False, comment="'enterprise' | 'mid-market' | 'smb'"),
    Column("signup_date", Date, nullable=False),
    comment="Companies that buy from Acme",
)

products = Table(
    "products",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("sku", String(20), nullable=False, unique=True),
    Column("name", String(80), nullable=False),
    Column("category", String(40), nullable=False),
    Column("unit_price", Numeric(10, 2), nullable=False, comment="List price in EUR"),
    comment="Products Acme sells",
)

orders = Table(
    "orders",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("customer_id", ForeignKey("customers.id"), nullable=False),
    Column("sales_rep_id", ForeignKey("employees.id"), nullable=False, comment="Employee"),
    Column("order_date", Date, nullable=False),
    Column(
        "status",
        String(20),
        nullable=False,
        comment="'completed' | 'shipped' | 'pending' | 'cancelled'; revenue excludes cancelled",
    ),
    comment="Customer orders, 2025-01-01 to 2026-09-30",
)

order_items = Table(
    "order_items",
    metadata,
    Column("order_id", ForeignKey("orders.id"), primary_key=True),
    Column("product_id", ForeignKey("products.id"), primary_key=True),
    Column("quantity", Integer, nullable=False),
    Column("unit_price", Numeric(10, 2), nullable=False, comment="Price charged, EUR"),
    Column(
        "discount_pct",
        Numeric(4, 1),
        nullable=False,
        comment="0-30; line total = quantity * unit_price * (1 - discount_pct / 100)",
    ),
    comment="Order lines",
)


@dataclass(frozen=True, slots=True)
class Dataset:
    """A queryable data source: tables (for creation and description) plus rows."""

    name: str
    version: str
    metadata: MetaData
    rows: dict[str, list[dict[str, Any]]]
    """Table name -> rows, in insertion (foreign-key) order."""


# ---- Deterministic data generation ------------------------------------------------

_DEPARTMENTS = [
    ("Engineering", "Berlin", 4_200_000),
    ("Sales", "Amsterdam", 2_600_000),
    ("Marketing", "Amsterdam", 1_300_000),
    ("Customer Success", "Lisbon", 900_000),
    ("Finance", "Berlin", 700_000),
    ("People", "Lisbon", 500_000),
]
_TITLES = {
    "Engineering": ["Software Engineer", "Senior Software Engineer", "Staff Engineer"],
    "Sales": ["Account Executive", "Senior Account Executive", "Sales Development Rep"],
    "Marketing": ["Marketing Manager", "Content Strategist", "Growth Analyst"],
    "Customer Success": ["Customer Success Manager", "Support Engineer"],
    "Finance": ["Financial Analyst", "Accountant"],
    "People": ["People Partner", "Recruiter"],
}
_HEAD_TITLES = {
    "Engineering": "VP Engineering",
    "Sales": "VP Sales",
    "Marketing": "Head of Marketing",
    "Customer Success": "Head of Customer Success",
    "Finance": "CFO",
    "People": "Head of People",
}
_HEADCOUNT = {
    "Engineering": 14,
    "Sales": 9,
    "Marketing": 5,
    "Customer Success": 5,
    "Finance": 3,
    "People": 3,
}
_FIRST = [
    "Ava",
    "Liam",
    "Noah",
    "Mia",
    "Lena",
    "Jonas",
    "Sofia",
    "Lucas",
    "Emma",
    "Mateo",
    "Ines",
    "Tiago",
    "Clara",
    "Finn",
    "Zara",
    "Omar",
    "Priya",
    "Ravi",
    "Hana",
    "Kenji",
    "Elif",
    "Can",
    "Nora",
    "Felix",
    "Maya",
    "Leo",
    "Amara",
    "Yusuf",
    "Julia",
    "Pedro",
]
_LAST = [
    "Becker",
    "Jansen",
    "Silva",
    "Costa",
    "Novak",
    "Fischer",
    "Visser",
    "Moreau",
    "Rossi",
    "Kowalski",
    "Okafor",
    "Nakamura",
    "Haddad",
    "Larsen",
    "Murphy",
    "Santos",
    "Weber",
    "de Vries",
    "Ferreira",
    "Kaya",
    "Lindqvist",
    "Patel",
    "Dubois",
    "Meyer",
]
_COUNTRY_BY_CITY = {"Berlin": "Germany", "Amsterdam": "Netherlands", "Lisbon": "Portugal"}
_BASE_SALARY = {
    "Engineering": 72_000,
    "Sales": 58_000,
    "Marketing": 56_000,
    "Customer Success": 48_000,
    "Finance": 60_000,
    "People": 52_000,
}
_CUSTOMER_PREFIX = [
    "Bluepeak",
    "Cobalt",
    "Driftwood",
    "Emberline",
    "Foxglove",
    "Granite",
    "Harbor",
    "Ironleaf",
    "Juniper",
    "Kestrel",
    "Lumen",
    "Meridian",
    "Nimbus",
    "Orchid",
    "Pinecrest",
]
_CUSTOMER_SUFFIX = ["Analytics", "Logistics", "Health", "Retail", "Energy", "Labs"]
_INDUSTRY_BY_SUFFIX = {
    "Analytics": "Software",
    "Logistics": "Logistics",
    "Health": "Healthcare",
    "Retail": "Retail",
    "Energy": "Energy",
    "Labs": "Biotech",
}
_CUSTOMER_COUNTRIES = [
    "Germany",
    "Netherlands",
    "France",
    "Spain",
    "Portugal",
    "Belgium",
    "Austria",
    "Ireland",
    "Denmark",
    "Italy",
]
_PRODUCTS = [
    ("REC-CORE", "Recall Core licence", "Software", 1200),
    ("REC-PRO", "Recall Pro licence", "Software", 2900),
    ("REC-ENT", "Recall Enterprise licence", "Software", 7500),
    ("ADD-SSO", "Single sign-on add-on", "Add-on", 600),
    ("ADD-AUDIT", "Audit log add-on", "Add-on", 450),
    ("ADD-API", "API access add-on", "Add-on", 800),
    ("SUP-STD", "Standard support plan", "Support", 300),
    ("SUP-PREM", "Premium support plan", "Support", 1100),
    ("SVC-ONB", "Onboarding workshop", "Services", 2000),
    ("SVC-MIG", "Data migration service", "Services", 4500),
    ("SVC-TRAIN", "Team training day", "Services", 1500),
    ("HW-HUB", "Edge sync hub", "Hardware", 950),
]
_ORDER_STATUS = ["completed"] * 14 + ["shipped"] * 3 + ["pending"] * 2 + ["cancelled"] * 2
_DATA_START = date(2025, 1, 1)
_DATA_END = date(2026, 9, 30)


def _day_between(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def generate(seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    """Build every row from a fixed seed. Same seed -> byte-identical dataset."""
    rng = random.Random(seed)
    rows: dict[str, list[dict[str, Any]]] = {}

    rows["departments"] = [
        {"id": i, "name": name, "location": city, "annual_budget": Decimal(budget)}
        for i, (name, city, budget) in enumerate(_DEPARTMENTS, 1)
    ]

    names = rng.sample([(f, last) for f in _FIRST for last in _LAST], sum(_HEADCOUNT.values()))
    staff: list[dict[str, Any]] = []
    for dept_id, (dept, city, _budget) in enumerate(_DEPARTMENTS, 1):
        head_id: int | None = None
        for n in range(_HEADCOUNT[dept]):
            first, last = names[len(staff)]
            emp_id = len(staff) + 1
            is_head = n == 0
            title = _HEAD_TITLES[dept] if is_head else rng.choice(_TITLES[dept])
            seniority = 1.8 if is_head else 1.3 if "Senior" in title or "Staff" in title else 1.0
            remote = rng.random() < 0.3
            staff.append(
                {
                    "id": emp_id,
                    "first_name": first,
                    "last_name": last,
                    "email": f"{first}.{last.replace(' ', '')}@acme.example".lower(),
                    "department_id": dept_id,
                    "manager_id": None if is_head else head_id,
                    "title": title,
                    "hire_date": _day_between(rng, date(2016, 1, 1), date(2026, 6, 30)),
                    "salary": Decimal(
                        round(_BASE_SALARY[dept] * seniority * rng.uniform(0.9, 1.15), -2)
                    ),
                    "country": rng.choice(list(_COUNTRY_BY_CITY.values()))
                    if remote
                    else _COUNTRY_BY_CITY[city],
                    "is_remote": remote,
                }
            )
            if is_head:
                head_id = emp_id
    rows["employees"] = staff

    balances: list[dict[str, Any]] = []
    for emp in staff:
        for year in (2025, 2026):
            if emp["hire_date"].year > year:
                continue
            carried = 0 if year == 2025 else rng.randint(0, 10)
            # 2026 is still in progress (data ends in September), so less is taken.
            taken = rng.randint(8, 24) if year == 2025 else rng.randint(2, 20)
            balances.append(
                {
                    "employee_id": emp["id"],
                    "year": year,
                    "entitled_days": 24,
                    "carried_over_days": carried,
                    "taken_days": taken,
                }
            )
    rows["leave_balances"] = balances

    company_names = rng.sample([(p, s) for p in _CUSTOMER_PREFIX for s in _CUSTOMER_SUFFIX], 30)
    rows["customers"] = [
        {
            "id": i,
            "name": f"{prefix} {suffix}",
            "industry": _INDUSTRY_BY_SUFFIX[suffix],
            "country": rng.choice(_CUSTOMER_COUNTRIES),
            "segment": rng.choices(["enterprise", "mid-market", "smb"], [2, 3, 5])[0],
            "signup_date": _day_between(rng, date(2023, 1, 1), date(2025, 12, 31)),
        }
        for i, (prefix, suffix) in enumerate(company_names, 1)
    ]

    rows["products"] = [
        {"id": i, "sku": sku, "name": name, "category": cat, "unit_price": Decimal(price)}
        for i, (sku, name, cat, price) in enumerate(_PRODUCTS, 1)
    ]

    reps = [e["id"] for e in staff if e["department_id"] == 2]  # Sales
    orders_rows: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for order_id in range(1, 301):
        customer = rng.choice(rows["customers"])
        order_date = _day_between(rng, max(_DATA_START, customer["signup_date"]), _DATA_END)
        orders_rows.append(
            {
                "id": order_id,
                "customer_id": customer["id"],
                "sales_rep_id": rng.choice(reps),
                "order_date": order_date,
                "status": rng.choice(_ORDER_STATUS),
            }
        )
        for product in rng.sample(rows["products"], rng.randint(1, 4)):
            items.append(
                {
                    "order_id": order_id,
                    "product_id": product["id"],
                    "quantity": rng.choice([1, 1, 2, 3, 5, 10]),
                    "unit_price": product["unit_price"],
                    "discount_pct": Decimal(rng.choice([0, 0, 0, 5, 10, 15, 20, 30])),
                }
            )
    rows["orders"] = sorted(orders_rows, key=lambda o: o["id"])
    rows["order_items"] = items
    return rows


ACME = Dataset(name=SCHEMA, version=DATASET_VERSION, metadata=metadata, rows=generate())
