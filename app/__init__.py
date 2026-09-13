"""Load .env before any submodule reads the environment.

app/policy.py resolves COMPONENTS and LABEL_PREFIX at import time, so this
has to happen here -- the package is imported before any of its submodules --
rather than in main.py. Real environment variables always win: load_dotenv()
does not override what's already set, so container/CI config beats a stray
local .env file.
"""

from dotenv import load_dotenv

load_dotenv()
