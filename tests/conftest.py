"""Pytest fixtures for the data-service tests.

Uses SQLite in-memory instead of Postgres so tests run without any external
infra. The FastAPI dependency override rewires the session factory."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Prepend the service root so ``app.*`` imports resolve during test runs.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Point the service at a throwaway data dir *before* app.config is imported.
_TMP = Path(tempfile.mkdtemp(prefix="ads_test_"))
os.environ.setdefault("DATA_ROOT", str(_TMP))
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.db import Base, get_session  # noqa: E402
from app.main import app  # noqa: E402
from app import models  # noqa: F401, E402  ensure Dataset is registered


@pytest.fixture(scope="function")
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture(scope="function")
def client(db_session):
    def _override():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_session] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
