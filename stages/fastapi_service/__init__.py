"""
Stage 4: FastAPI Service & HTTP Interface

Public API for the HTTP service layer.

Entry point:
    from stages.fastapi_service import create_app
    app = create_app()

    # Run with: uvicorn stages.fastapi_service:app --host 0.0.0.0 --port 8000
"""

from .main import create_app, app

__all__ = [
    "create_app",
    "app",
]
