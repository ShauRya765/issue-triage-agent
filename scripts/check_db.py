"""Verify DATABASE_URL points at a usable checkpoint store.

Run this before the API. It connects, runs PostgresSaver.setup() (which
creates the checkpoint tables if they don't exist), and reports what it
found -- so a bad Supabase URI fails here with a clear message instead of
inside a half-started run.

    ./venv/bin/python scripts/check_db.py
"""

import os
import sys
from pathlib import Path

# Running this as "python scripts/check_db.py" puts scripts/ on sys.path, not
# the repo root, so "import app" fails. Add the root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv


def main() -> int:
    load_dotenv()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL is not set (copy .env.example to .env)")
        return 1

    # Imported after the env check so the failure above doesn't need psycopg.
    from langgraph.checkpoint.postgres import PostgresSaver

    from app.graph import _connect

    host = database_url.split("@")[-1].split("/")[0]
    print(f"connecting to {host} ...")

    try:
        conn = _connect(database_url)
    except Exception as exc:
        print(f"connection failed: {exc}")
        print(
            "\nIf this is Supabase: check the password is filled in, and prefer the\n"
            "session pooler URI (pooler.supabase.com:5432) -- the direct\n"
            "db.<ref>.supabase.co host is IPv6-only and unreachable on many networks."
        )
        return 1

    with conn:
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()

        with conn.cursor() as cur:
            cur.execute("select version()")
            version = cur.fetchone()["version"]
            cur.execute(
                """
                select table_name from information_schema.tables
                where table_schema = 'public' and table_name like 'checkpoint%'
                order by table_name
                """
            )
            tables = [row["table_name"] for row in cur.fetchall()]
            cur.execute("select count(distinct thread_id) as n from checkpoints")
            threads = cur.fetchone()["n"]

    print(f"ok: {version.split(',')[0]}")
    print(f"checkpoint tables: {', '.join(tables) or 'none'}")
    print(f"threads currently checkpointed: {threads}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
