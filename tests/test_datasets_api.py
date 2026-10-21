# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 The VESIT AutoML Project contributors

from __future__ import annotations

import io


def _fake_csv() -> tuple[str, bytes, str]:
    rows = ["a,b,label"] + [f"{i%3},{i*0.1},{i%2}" for i in range(60)]
    return ("data.csv", "\n".join(rows).encode("utf-8"), "text/csv")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"] == "connected"


def test_upload_csv_then_list(client):
    files = {"file": _fake_csv()}
    data = {"target_col": "label"}
    r = client.post("/datasets", files=files, data=data)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["source"] == "custom"
    assert body["target_col"] == "label"
    assert body["task_type"] == "binary_classification"
    assert body["n_rows"] == 60
    ds_id = body["id"]

    r = client.get("/datasets")
    assert r.status_code == 200
    listing = r.json()
    assert len(listing) == 1
    assert listing[0]["id"] == ds_id

    r = client.get(f"/datasets/{ds_id}")
    assert r.status_code == 200
    assert r.json()["id"] == ds_id


def test_upload_missing_target_returns_400(client):
    files = {"file": _fake_csv()}
    data = {"target_col": "does_not_exist"}
    r = client.post("/datasets", files=files, data=data)
    assert r.status_code == 400
    assert "not found" in r.json()["detail"]


def test_preview_dataset(client):
    files = {"file": _fake_csv()}
    r = client.post("/datasets", files=files, data={"target_col": "label"})
    ds_id = r.json()["id"]

    r = client.get(f"/datasets/{ds_id}/preview?n_rows=3")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"] == ds_id
    assert body["columns"] == ["a", "b", "label"]
    assert len(body["rows"]) == 3


def test_delete_dataset(client):
    files = {"file": _fake_csv()}
    r = client.post("/datasets", files=files, data={"target_col": "label"})
    ds_id = r.json()["id"]

    r = client.delete(f"/datasets/{ds_id}?wipe_files=true")
    assert r.status_code == 204

    r = client.get(f"/datasets/{ds_id}")
    assert r.status_code == 404


def test_openml_catalog_lists_known_ids(client):
    r = client.get("/datasets/openml/catalog")
    assert r.status_code == 200
    catalog = r.json()
    assert "37" in catalog
    assert catalog["37"] == "diabetes"
    assert "41211" in catalog
    assert catalog["41211"] == "ames_housing"
