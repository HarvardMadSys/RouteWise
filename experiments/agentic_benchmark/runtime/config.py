"""Connection settings for the live hybridInference gateway."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://freeinference.org"
# Default to the plain minimax-m2.5 (fixed routing) as the baseline while the
# harness is brought up. minimax-fast is the same model under RouteWise routing,
# swapped in later for the routewise-vs-baseline comparison.
DEFAULT_MODEL = "minimax-m2.5"

# The public key is stored under a few possible names across local .env files.
_PROD_API_KEY_ENV_CANDIDATES = (
    "FREEINFERENCE_API_KEY",
    "FreeInference_API_KEY",
)
_STAGING_API_KEY_ENV_CANDIDATES = (
    "FREEINFERENCE_STAGE_API_KEY",
    "FreeInference_Stage_API_Key",
)
# Admin key (for the admin-only X-Route-Pin baseline). ADMIN_TOKEN is a guess;
# verify it is accepted as a chat-API admin Bearer key before relying on it.
_ADMIN_KEY_ENV_CANDIDATES = (
    "FREEINFERENCE_ADMIN_API_KEY",
    "ADMIN_TOKEN",
)


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _api_key_from_env(base_url: str) -> str | None:
    """Return the gateway key matching the configured deployment when possible."""
    if "staging." in base_url:
        return _first_env(_STAGING_API_KEY_ENV_CANDIDATES + _PROD_API_KEY_ENV_CANDIDATES)
    return _first_env(_PROD_API_KEY_ENV_CANDIDATES + _STAGING_API_KEY_ENV_CANDIDATES)


@dataclass(frozen=True)
class GatewayConfig:
    """Where the gateway lives and which keys to authenticate with.

    ``base_url`` is the gateway root without a trailing ``/v1``; the client
    appends the concrete endpoint paths itself.
    """

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    admin_api_key: str | None = None
    timeout_sec: float = 240.0

    @classmethod
    def from_env(cls, **overrides: object) -> GatewayConfig:
        """Build a config from the environment (load a .env beforehand if needed)."""
        # Resolve the effective base_url BEFORE picking the api key, so a
        # base_url override (e.g. the staging A/B driver) selects the matching
        # deployment's key rather than the FREEINFERENCE_BASE_URL default's.
        raw_base = str(
            overrides.pop("base_url", None)
            or os.environ.get("FREEINFERENCE_BASE_URL", DEFAULT_BASE_URL)
        )
        base_url = raw_base.rstrip("/").removesuffix("/v1")
        values: dict[str, object] = {
            "base_url": base_url,
            "model": os.environ.get("AGENTIC_MODEL", DEFAULT_MODEL),
            "api_key": _api_key_from_env(base_url),
            "admin_api_key": _first_env(_ADMIN_KEY_ENV_CANDIDATES),
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

    def require_api_key(self) -> str:
        """Return the public API key or fail with a clear message."""
        if not self.api_key:
            raise RuntimeError(
                "No FreeInference API key found. Set FREEINFERENCE_API_KEY "
                "(or FreeInference_API_KEY). For staging, set "
                "FREEINFERENCE_STAGE_API_KEY or FreeInference_Stage_API_Key."
            )
        return self.api_key
