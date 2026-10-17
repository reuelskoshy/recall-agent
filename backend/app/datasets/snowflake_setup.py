"""Generate everything needed to run query_database on Snowflake.

    python -m app.datasets.snowflake_setup

writes two files under `secrets/` (git-ignored):

- `snowflake_rsa_key.p8`: a new RSA private key for the app's service user
  (reused if it already exists). It never leaves this machine.
- `snowflake_setup.sql`: one script to run in a Snowsight worksheet (Run All)
  as a user with ACCOUNTADMIN, e.g. the trial's first user. It creates the
  warehouse, database, schema, the Acme tables and rows, the read-only role,
  and the service user with the matching public key.

Why a SQL file instead of loading over a connection: the app then never holds
a credential that can create or write anything. The privileged steps run once,
by a human, in Snowflake's own UI (with their MFA).

Least privilege inside the script: objects are created by SYSADMIN, roles,
users, and grants by SECURITYADMIN; the reader role gets USAGE + SELECT only.
"""

import argparse
import base64
from collections.abc import Iterable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from sqlalchemy import Boolean, Column, Date, Integer, Numeric, String, Table

from app.core.config import Settings, get_settings
from app.datasets.acme import ACME, Dataset
from app.sql.snowflake import REPO_ROOT, resolve_key_path

INSERT_BATCH = 500


def ensure_key_pair(path: Path) -> str:
    """Create an unencrypted PKCS#8 RSA key at `path` if missing; return the public key body.

    The body is the base64 DER of the public key, the format Snowflake's
    `RSA_PUBLIC_KEY` expects (PEM without header, footer, or newlines).
    """
    if path.exists():
        private_key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        path.write_bytes(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        path.chmod(0o600)
    der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return base64.b64encode(der).decode()


def quote(value: str) -> str:
    """A Snowflake string literal."""
    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"


def literal(value: Any) -> str:
    """Render one Python value as a Snowflake SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | Decimal):
        return str(value)
    if isinstance(value, date):
        return quote(value.isoformat())
    return quote(str(value))


def column_type(column: Column[Any]) -> str:
    """Map the dataset's SQLAlchemy types onto Snowflake types."""
    t = column.type
    if isinstance(t, Boolean):
        return "BOOLEAN"
    if isinstance(t, Numeric):
        return f"NUMBER({t.precision},{t.scale})"
    if isinstance(t, Integer):
        return "NUMBER(38,0)"
    if isinstance(t, String):
        return f"VARCHAR({t.length})" if t.length else "VARCHAR"
    if isinstance(t, Date):
        return "DATE"
    raise ValueError(f"No Snowflake type for {column.table.name}.{column.name}: {t!r}")


def create_table(table: Table) -> str:
    """CREATE TABLE with comments and (informational, unenforced) keys."""
    lines = []
    for column in table.columns:
        line = f"    {column.name} {column_type(column)}"
        if not column.nullable:
            line += " NOT NULL"
        if column.comment:
            line += f" COMMENT {quote(column.comment)}"
        lines.append(line)
    pk = [c.name for c in table.primary_key.columns]
    if pk:
        lines.append(f"    PRIMARY KEY ({', '.join(pk)})")
    for fk in table.foreign_keys:
        lines.append(
            f"    FOREIGN KEY ({fk.parent.name}) "
            f"REFERENCES {fk.column.table.name} ({fk.column.name})"
        )
    comment = f"\nCOMMENT = {quote(table.comment)}" if table.comment else ""
    return f"CREATE TABLE {table.name} (\n" + ",\n".join(lines) + f"\n){comment};"


def insert_rows(table: Table, rows: list[dict[str, Any]]) -> Iterable[str]:
    """Multi-row INSERTs, in batches."""
    names = [c.name for c in table.columns]
    for start in range(0, len(rows), INSERT_BATCH):
        values = ",\n".join(
            "(" + ", ".join(literal(row[n]) for n in names) + ")"
            for row in rows[start : start + INSERT_BATCH]
        )
        yield f"INSERT INTO {table.name} ({', '.join(names)}) VALUES\n{values};"


def build_setup_sql(dataset: Dataset, settings: Settings, public_key: str) -> str:
    """The whole setup script for one dataset."""
    wh = settings.snowflake_warehouse
    db = settings.snowflake_database
    schema = f"{db}.{settings.snowflake_schema}"
    role = settings.snowflake_role
    user = settings.snowflake_user
    rows = sum(len(r) for r in dataset.rows.values())

    parts = [
        f"""-- Recall: Snowflake setup for query_database (dataset {dataset.version}, {rows} rows).
-- Run in a Snowsight worksheet as a user with ACCOUNTADMIN: select all, then Run.
-- Safe to re-run: the schema and its data are rebuilt, the user and role are updated.

-- 1. Compute and storage (owned by SYSADMIN) ------------------------------------
USE ROLE SYSADMIN;
CREATE WAREHOUSE IF NOT EXISTS {wh}
  WAREHOUSE_SIZE = XSMALL
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Recall agent: read-only demo queries';
-- Hard limits for anything run on this warehouse.
ALTER WAREHOUSE {wh} SET
  STATEMENT_TIMEOUT_IN_SECONDS = 30
  STATEMENT_QUEUED_TIMEOUT_IN_SECONDS = 30;
CREATE DATABASE IF NOT EXISTS {db};
CREATE OR REPLACE SCHEMA {schema} COMMENT = {quote(dataset.version)};
USE WAREHOUSE {wh};
USE SCHEMA {schema};

-- 2. Tables and data -------------------------------------------------------------"""
    ]
    for table in dataset.metadata.sorted_tables:  # foreign-key order
        parts.append(create_table(table))
        parts.extend(insert_rows(table, dataset.rows.get(table.name, [])))

    parts.append(
        f"""-- 3. Read-only role: USAGE to reach the data, SELECT to read it, nothing else --
USE ROLE SECURITYADMIN;
CREATE ROLE IF NOT EXISTS {role} COMMENT = 'Recall agent: SELECT on {schema} only';
GRANT USAGE ON WAREHOUSE {wh} TO ROLE {role};
GRANT USAGE ON DATABASE {db} TO ROLE {role};
GRANT USAGE ON SCHEMA {schema} TO ROLE {role};
GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO ROLE {role};

-- 4. Service user: key-pair login only (no password, no interactive sign-in) ---
CREATE USER IF NOT EXISTS {user}
  TYPE = SERVICE
  DEFAULT_ROLE = {role}
  DEFAULT_WAREHOUSE = {wh}
  DEFAULT_NAMESPACE = {schema}
  COMMENT = 'Recall agent (query_database)';
ALTER USER {user} SET
  RSA_PUBLIC_KEY = '{public_key}'
  STATEMENT_TIMEOUT_IN_SECONDS = 10;
GRANT ROLE {role} TO USER {user};

-- 5. Check: the role should list only USAGE and SELECT privileges --------------
SHOW GRANTS TO ROLE {role};

-- To remove everything later (as ACCOUNTADMIN):
--   DROP USER IF EXISTS {user}; DROP ROLE IF EXISTS {role};
--   DROP DATABASE IF EXISTS {db}; DROP WAREHOUSE IF EXISTS {wh};
"""
    )
    return "\n\n".join(parts)


def main() -> None:
    """Write the key (if missing) and the setup script."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "secrets" / "snowflake_setup.sql"), help="SQL file"
    )
    args = parser.parse_args()
    settings = get_settings()

    key_path = resolve_key_path(settings.snowflake_private_key_path)
    existed = key_path.exists()
    public_key = ensure_key_pair(key_path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_setup_sql(ACME, settings, public_key), encoding="utf-8")

    print(f"Private key: {key_path} ({'existing' if existed else 'new'}; keep it secret)")
    print(f"Setup SQL:   {out}")
    print("Next: open Snowsight > Projects > Worksheets > + > 'Create from SQL file',")
    print("pick the setup SQL, select all, and Run. Then set SQL_BACKEND=snowflake.")


if __name__ == "__main__":
    main()
