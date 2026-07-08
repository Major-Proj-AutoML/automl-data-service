# automl-data-service

Dataset registry microservice for the AutoML pipeline. Port **8001**.

## Responsibilities

- Accept custom CSV uploads → validate → split → cache CSV files → register in Postgres.
- Import OpenML datasets by ID (uses the 28 CC18 catalogue from `automl-reusables`).
- List / fetch / delete registered datasets.
- Serve dataset previews (first N rows).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + DB reachability |
| POST | `/datasets` | Upload CSV (multipart) with `target_col`, optional `task_type` |
| POST | `/datasets/openml` | Import an OpenML dataset by id |
| GET | `/datasets` | List all registered datasets |
| GET | `/datasets/{id}` | Fetch a dataset's metadata |
| GET | `/datasets/{id}/preview` | First N rows of train or test split |
| DELETE | `/datasets/{id}` | Unregister (optionally `?wipe_files=true`) |
| GET | `/datasets/openml/catalog` | The 28 OpenML CC18 ids known to the pipeline |

Interactive docs at `http://localhost:8001/docs`.

## Local run (without Docker)

Prerequisites:
- Postgres running (start `automl-infra`: `cd ../automl-infra && docker compose up -d`)
- `automl-reusables` installed as an editable dependency

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # or: source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -e ../automl-reusables
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload --port 8001
```

## Docker

Built and orchestrated from `automl-infra` (added there in a later step).

## Tests

Uses an in-memory SQLite so no Postgres required to test:

```bash
pytest -v
```
