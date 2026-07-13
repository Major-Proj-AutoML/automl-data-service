"""FastAPI entry point for the automl-data-service."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_session
from app.schemas import (
    DatasetPreview,
    DatasetResponse,
    HealthResponse,
    OpenMLImportRequest,
    TaskTypeStr,
)
from app.service import (
    delete_dataset,
    get_dataset,
    import_openml,
    known_openml_ids,
    list_datasets,
    preview_dataset,
    upload_csv,
)

logging.basicConfig(level=settings.log_level)

app = FastAPI(
    title="automl-data-service",
    version="0.1.0",
    description="Dataset registry for the AutoML microservices. Uploads/fetches datasets, "
    "stores metadata in Postgres, persists CSVs to disk.",
)


@app.get("/health", response_model=HealthResponse)
def health(session: Session = Depends(get_session)) -> HealthResponse:
    try:
        session.execute(text("SELECT 1"))
        return HealthResponse(status="ok", database="connected")
    except Exception:
        return HealthResponse(status="degraded", database="unreachable")


@app.post("/datasets", response_model=DatasetResponse, status_code=201)
def create_dataset_from_csv(
    file: UploadFile = File(..., description="CSV file with header row."),
    target_col: str = Form(..., description="Name of the target column."),
    task_type: Optional[TaskTypeStr] = Form(
        None, description="Auto-detected if omitted."
    ),
    seed: int = Form(42),
    test_size: float = Form(0.2, gt=0.0, lt=1.0),
    session: Session = Depends(get_session),
) -> DatasetResponse:
    ds = upload_csv(
        session, file, target_col=target_col, task_type=task_type,
        seed=seed, test_size=test_size,
    )
    return DatasetResponse.model_validate(ds)


@app.post("/datasets/openml", response_model=DatasetResponse, status_code=201)
def create_dataset_from_openml(
    body: OpenMLImportRequest,
    session: Session = Depends(get_session),
) -> DatasetResponse:
    ds = import_openml(
        session, openml_id=body.openml_id, seed=body.seed, test_size=body.test_size,
    )
    return DatasetResponse.model_validate(ds)


@app.get("/datasets", response_model=list[DatasetResponse])
def list_all(session: Session = Depends(get_session)) -> list[DatasetResponse]:
    return [DatasetResponse.model_validate(d) for d in list_datasets(session)]


@app.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_one(dataset_id: int, session: Session = Depends(get_session)) -> DatasetResponse:
    return DatasetResponse.model_validate(get_dataset(session, dataset_id))


@app.get("/datasets/{dataset_id}/preview", response_model=DatasetPreview)
def preview(
    dataset_id: int,
    n_rows: int = Query(10, ge=1, le=1000),
    split: str = Query("train", pattern="^(train|test)$"),
    session: Session = Depends(get_session),
) -> DatasetPreview:
    return DatasetPreview.model_validate(preview_dataset(session, dataset_id, n_rows, split))


@app.delete("/datasets/{dataset_id}", status_code=204)
def delete_one(
    dataset_id: int,
    wipe_files: bool = Query(False, description="Also delete CSVs from disk."),
    session: Session = Depends(get_session),
) -> None:
    delete_dataset(session, dataset_id, wipe_files=wipe_files)


@app.get("/datasets/openml/catalog")
def openml_catalog() -> dict[int, str]:
    """OpenML datasets pre-registered in automl-reusables (28 CC18 classification + Ames Housing regression)."""
    return known_openml_ids()
