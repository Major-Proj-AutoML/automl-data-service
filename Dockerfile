FROM python:3.11-slim

WORKDIR /srv

# System deps needed by numpy/scipy/scikit-learn wheels + psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Install the shared library first so it caches independently of the service code.
# The compose file bind-mounts ../automl-reusables to /opt/automl-reusables.
COPY pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip setuptools

# Install service (the automl-reusables dep is satisfied at runtime via editable install)
COPY app ./app
RUN pip install --no-cache-dir --no-deps -e . && \
    pip install --no-cache-dir \
        fastapi "uvicorn[standard]" sqlalchemy psycopg2-binary python-multipart pydantic-settings \
        pandas numpy scikit-learn scipy pydantic requests openml tabulate

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
