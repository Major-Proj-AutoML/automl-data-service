# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 The VESIT AutoML Project contributors

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


TaskTypeStr = Literal[
    "binary_classification",
    "multiclass_classification",
    "regression",
]


class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    source: Literal["openml", "custom"]
    openml_id: Optional[int] = None
    target_col: str
    task_type: TaskTypeStr
    train_path: str
    test_path: str
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    created_at: datetime


class OpenMLImportRequest(BaseModel):
    openml_id: int = Field(..., gt=0)
    seed: int = 42
    test_size: float = Field(0.2, gt=0.0, lt=1.0)


class DatasetPreview(BaseModel):
    dataset_id: int
    columns: list[str]
    rows: list[dict]
    n_rows_total: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["connected", "unreachable"]
