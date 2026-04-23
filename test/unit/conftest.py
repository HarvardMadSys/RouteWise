"""Unit-test level bootstrap to stub optional dependencies at import time.

Some packages (e.g., serving.__init__ importing serving.config) depend on
third-party modules optional in CI. We stub them here to avoid import-time
failures while focusing on pure-unit tests that don't need their behavior.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

# Stub python-dotenv if missing
if "dotenv" not in sys.modules:  # pragma: no cover - import-time shim
    sys.modules["dotenv"] = SimpleNamespace(load_dotenv=lambda *a, **k: None)

# Stub aiohttp if missing to satisfy imports in serving.http
if "aiohttp" not in sys.modules:  # pragma: no cover - import-time shim

    class _DummySession:  # minimal placeholder
        def __init__(self, *a, **k):
            self.closed = False

        async def close(self):
            self.closed = True

        # Methods used by tests are patched, so we keep placeholders only.

    sys.modules["aiohttp"] = SimpleNamespace(
        ClientError=Exception,
        ClientTimeout=lambda total=None: None,
        ClientSession=_DummySession,
    )
