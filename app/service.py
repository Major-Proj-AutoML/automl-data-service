# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 The VESIT AutoML Project contributors

"""Business logic for the data-service.

Thin wrapper over ``src.experiments.datasets`` from automl-reusables. Registers
dataset metadata in Postgres and persists CSVs under ``settings.data_root``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.experiments.datasets import (
    SELECTED_CLASSIFICATION,
    SELECTED_REGRESSION,
    _detect_task_type,
    _drop_id_columns,
    load_custom_dataset,
    load_dataset,
)

from app.config import settings
from app.models import Dataset


_STABLE_ROOT = Path("/opt/automl-reusables/data/experiments/stable")


def _data_root() -> Path:
    root = Path(settings.data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _relocate_to_stable(dataset: Dataset) -> None:
    """Move a dataset's CSVs into stable/<id>/ so paths survive host moves."""
    dest_dir = _STABLE_ROOT / str(dataset.id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for attr, fname in (("train_path", "train.csv"), ("test_path", "test.csv")):
        src = Path(getattr(dataset, attr))
        dst = dest_dir / fname
        if src.resolve() == dst.resolve():
            continue
        if not src.exists():
            continue
        shutil.copyfile(src, dst)
        setattr(dataset, attr, str(dst))


def upload_csv(
    session: Session,
    upload: UploadFile,
    target_col: str,
    task_type: str | None,
    seed: int = 42,
    test_size: float = 0.2,
) -> Dataset:
    """Persist an uploaded CSV, split it, and register in Postgres."""
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Missing filename in upload")
    stem = Path(upload.filename).stem or "dataset"

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".csv", mode="wb"
    ) as tmp:
        shutil.copyfileobj(upload.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        info = load_custom_dataset(
            str(tmp_path),
            target_col=target_col,
            task_type=task_type,
            seed=seed,
            test_size=test_size,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    dataset = Dataset(
        name=stem,
        source="custom",
        openml_id=None,
        target_col=info["target_col"],
        task_type=info["task_type"].value,
        train_path=info["train_path"],
        test_path=info["test_path"],
        n_rows=int(info["df_train"].shape[0] + info["df_test"].shape[0]),
        n_cols=int(info["df_train"].shape[1]),
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    _relocate_to_stable(dataset)
    session.commit()
    session.refresh(dataset)
    return dataset


def import_openml(
    session: Session, openml_id: int, seed: int = 42, test_size: float = 0.2
) -> Dataset:
    existing = session.scalar(
        select(Dataset).where(Dataset.openml_id == openml_id, Dataset.source == "openml")
    )
    if existing is not None:
        return existing

    try:
        info = load_dataset(openml_id, seed=seed, test_size=test_size)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch OpenML dataset {openml_id}: {exc}"
        )

    dataset = Dataset(
        name=info["dataset_name"],
        source="openml",
        openml_id=openml_id,
        target_col=info["target_col"],
        task_type=info["task_type"].value,
        train_path=info["train_path"],
        test_path=info["test_path"],
        n_rows=int(info["df_train"].shape[0] + info["df_test"].shape[0]),
        n_cols=int(info["df_train"].shape[1]),
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    _relocate_to_stable(dataset)
    session.commit()
    session.refresh(dataset)
    return dataset


def list_datasets(session: Session) -> list[Dataset]:
    return list(session.scalars(select(Dataset).order_by(Dataset.created_at.desc())))


def get_dataset(session: Session, dataset_id: int) -> Dataset:
    ds = session.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return ds


def delete_dataset(session: Session, dataset_id: int, wipe_files: bool = False) -> None:
    ds = get_dataset(session, dataset_id)
    if wipe_files:
        for p in (ds.train_path, ds.test_path):
            path = Path(p)
            if path.exists():
                path.unlink(missing_ok=True)
        parent = Path(ds.train_path).parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    session.delete(ds)
    session.commit()


def preview_dataset(
    session: Session, dataset_id: int, n_rows: int = 10, split: str = "train"
) -> dict:
    ds = get_dataset(session, dataset_id)
    path = ds.train_path if split == "train" else ds.test_path
    if not Path(path).exists():
        raise HTTPException(
            status_code=410,
            detail=f"CSV file no longer exists on disk: {path}",
        )
    df = pd.read_csv(path)
    head = df.head(n_rows)
    return {
        "dataset_id": ds.id,
        "columns": list(df.columns),
        "rows": head.to_dict(orient="records"),
        "n_rows_total": int(len(df)),
    }


def known_openml_ids() -> dict[int, str]:
    return {**SELECTED_CLASSIFICATION, **SELECTED_REGRESSION}
