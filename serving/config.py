"""Configuration helpers for provider endpoints and API keys.

Loads environment variables (via dotenv) and returns a simple mapping used by
the serving components. Keep this file lightweight and free of heavy logic.
"""

import os

from dotenv import load_dotenv


def get_config() -> dict[str, dict[str, str | None]]:
    """Load environment variables and return a simple config map.

    Returns:
      A nested mapping of provider names to configuration values.
    """
    load_dotenv()

    return {
        "local": {"base_url": os.getenv("LOCAL_BASE_URL", "http://localhost:8001")},
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        },
        "llama": {
            "api_key": os.getenv("LLAMA_API_KEY"),
            "base_url": os.getenv("LLAMA_BASE_URL", "https://api.llama.com/compat/v1/"),
        },
        "openrouter": {
            "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            # Optional attribution headers for OpenRouter rankings
            "http_referer": os.getenv("OPENROUTER_HTTP_REFERER", None),
            "x_title": os.getenv("OPENROUTER_X_TITLE", None),
        },
    }


def get_db_config() -> dict[str, str | int]:
    """Load PostgreSQL database configuration from environment.

    Returns:
        A mapping of asyncpg connection parameters.

    Raises:
        ValueError: If required database credentials are not set.
    """
    load_dotenv()

    # Required fields with no defaults for security
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME")

    if not db_user:
        raise ValueError("DB_USER must be set in environment or .env file")
    if not db_password:
        raise ValueError("DB_PASSWORD must be set in environment or .env file")
    if not db_name:
        raise ValueError("DB_NAME must be set in environment or .env file")

    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": db_name,
        "user": db_user,
        "password": db_password,
    }
