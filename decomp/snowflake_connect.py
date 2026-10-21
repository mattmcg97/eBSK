"""Snowpark session from the same .env used in the sis repo.

Credentials are read from environment variables (or .env) so nothing
sensitive is committed. Defaults to SSO (SNOWFLAKE_AUTHENTICATOR=externalbrowser);
set SNOWFLAKE_AUTHENTICATOR=snowflake and SNOWFLAKE_PASSWORD for password auth.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from snowflake.snowpark import Session

# .env at the repo root, wherever the script is launched from
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

REQUIRED_VARS = [
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
]


def get_session() -> Session:
    missing = [var for var in REQUIRED_VARS if not os.environ.get(var)]
    if missing:
        sys.exit(f"Missing required environment variables: {', '.join(missing)}")

    authenticator = os.environ.get("SNOWFLAKE_AUTHENTICATOR", "externalbrowser")

    configs = dict(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        authenticator=authenticator,
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
        database=os.environ.get("SNOWFLAKE_DATABASE"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA"),
        role=os.environ.get("SNOWFLAKE_ROLE"),
    )

    if authenticator == "snowflake":
        password = os.environ.get("SNOWFLAKE_PASSWORD")
        if not password:
            sys.exit("SNOWFLAKE_PASSWORD is required when SNOWFLAKE_AUTHENTICATOR=snowflake")
        configs["password"] = password

    return Session.builder.configs({k: v for k, v in configs.items() if v}).create()
