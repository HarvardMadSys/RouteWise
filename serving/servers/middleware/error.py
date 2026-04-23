"""Global exception handlers that produce OpenRouter-style error bodies."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from serving.schemas import ErrorDetail, ErrorResponse
from serving.utils.errors import categorize_exception
from serving.utils.logging import get_logger

logger = get_logger(__name__)


def _build_error_response(
    message: str,
    *,
    code: int | None = None,
    typ: str = "server_error",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standardized error response payload.

    Args:
        message: Human-readable error message.
        code: Optional HTTP status code associated with the error.
        typ: Error type identifier.
        extra: Optional additional fields to include under ``error``.

    Returns:
        Dict[str, Any]: Serialized error object conforming to OpenRouter style.
    """
    detail = ErrorDetail(type=typ, message=message, code=code, **(extra or {}))
    return ErrorResponse(error=detail).model_dump()


def install_error_handlers(app: FastAPI) -> None:
    """Install global exception handlers that return OpenRouter-like errors."""

    @app.middleware("http")
    async def _fallback_error_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Ensure JSON error bodies for uncaught exceptions.

        Some exception paths in ASGI stacks can bypass exception handlers in
        certain test transports or debug configurations. This lightweight
        middleware guarantees we still emit a structured JSON error.
        """
        try:
            return await call_next(request)
        except HTTPException:
            # Let the specific HTTP handler below shape the payload/code.
            raise
        except Exception as exc:
            err_type = categorize_exception(exc)
            logger.error(
                "unhandled_exception",
                extra={
                    "error_type": err_type,
                    "path": request.url.path,
                    "method": request.method,
                },
                exc_info=exc,
            )
            content = _build_error_response(str(exc), code=500, typ=err_type)
            return JSONResponse(status_code=500, content=content)

    @app.exception_handler(HTTPException)
    async def http_exc_handler(request: Request, exc: HTTPException) -> JSONResponse:
        # If detail already shaped like our error, forward as-is
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(
                status_code=exc.status_code, content=exc.detail, headers=exc.headers
            )
        err_type = categorize_exception(exc)
        logger.error(
            "http_error",
            extra={
                "error_type": err_type,
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method,
            },
            exc_info=exc,
        )
        content = _build_error_response(str(exc.detail), code=exc.status_code, typ=err_type)
        return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)

    @app.exception_handler(Exception)
    async def any_exc_handler(request: Request, exc: Exception) -> JSONResponse:
        err_type = categorize_exception(exc)
        logger.error(
            "unhandled_exception",
            extra={
                "error_type": err_type,
                "path": request.url.path,
                "method": request.method,
            },
            exc_info=exc,
        )
        content = _build_error_response(str(exc), code=500, typ=err_type)
        return JSONResponse(status_code=500, content=content)
