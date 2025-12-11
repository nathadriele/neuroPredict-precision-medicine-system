"""
Exceções customizadas e handlers para API.
"""

from datetime import datetime
from typing import Any, Dict

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from loguru import logger


class NeuroPredictException(Exception):
    """Exceção base do NeuroPredict."""
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Dict[str, Any] = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ModelNotFoundError(NeuroPredictException):
    """Modelo não encontrado."""
    
    def __init__(self, model_id: str) -> None:
        super().__init__(
            message=f"Modelo {model_id} não encontrado",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"model_id": model_id},
        )


class ModelLoadError(NeuroPredictException):
    """Erro ao carregar modelo."""
    
    def __init__(self, error: str) -> None:
        super().__init__(
            message=f"Erro ao carregar modelo: {error}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"error": error},
        )


class InvalidInputError(NeuroPredictException):
    """Entrada inválida."""
    
    def __init__(self, field: str, message: str) -> None:
        super().__init__(
            message=f"Entrada inválida no campo '{field}': {message}",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"field": field, "validation_error": message},
        )


class PredictionError(NeuroPredictException):
    """Erro durante predição."""
    
    def __init__(self, error: str) -> None:
        super().__init__(
            message=f"Erro na predição: {error}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error": error},
        )


class RAGError(NeuroPredictException):
    """Erro no sistema RAG."""
    
    def __init__(self, error: str) -> None:
        super().__init__(
            message=f"Erro no sistema RAG: {error}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error": error},
        )


class DatabaseError(NeuroPredictException):
    """Erro de banco de dados."""
    
    def __init__(self, error: str) -> None:
        super().__init__(
            message=f"Erro de banco de dados: {error}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"error": error},
        )


class RateLimitExceeded(NeuroPredictException):
    """Rate limit excedido."""
    
    def __init__(self, limit: int, window: str) -> None:
        super().__init__(
            message=f"Rate limit excedido: {limit} requests por {window}",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"limit": limit, "window": window},
        )


async def neuropredict_exception_handler(
    request: Request,
    exc: NeuroPredictException,
) -> JSONResponse:
    logger.error(
        f"NeuroPredictException: {exc.message}",
        extra={
            "status_code": exc.status_code,
            "details": exc.details,
            "path": request.url.path,
        },
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path),
        },
    )


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    logger.warning(
        f"HTTPException: {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "path": request.url.path,
        },
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path),
        },
    )


async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception(
        "Exceção não tratada",
        extra={
            "exception_type": type(exc).__name__,
            "path": request.url.path,
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Erro interno do servidor",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path),
            # "details": str(exc),
        },
    )


async def validation_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    from fastapi.exceptions import RequestValidationError
    
    if isinstance(exc, RequestValidationError):
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            })
        
        logger.warning(
            "Erro de validação",
            extra={
                "errors": errors,
                "path": request.url.path,
            },
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Erro de validação",
                "validation_errors": errors,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path),
            },
        )
    
    return await general_exception_handler(request, exc)


def register_exception_handlers(app) -> None:
    from fastapi.exceptions import RequestValidationError
    
    app.add_exception_handler(NeuroPredictException, neuropredict_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)