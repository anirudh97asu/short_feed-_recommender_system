# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile
# ──────────────────────────────────────────────────────────────────────────────
# The JSON catalog is BAKED INTO the image at build time via COPY.
# This means the offline pipeline must be run BEFORE docker build:
#
#   Step 1:  python -m src.offline.run_all
#            → produces src/backend/data/products_with_embeddings.json
#
#   Step 2:  docker compose up --build
#            → COPY src/backend/ picks up the JSON automatically
#
# To update the catalog: re-run the offline pipeline, then rebuild.
# For catalog versioning: use -v to override the baked-in file at runtime.
#
# Ports:
#   7860 — Gradio   (HF Spaces requires this to be the public-facing port)
#   8000 — FastAPI  (internal; called by Gradio on localhost)
#   5432 — Postgres (internal only)
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql \
    postgresql-client \
    supervisor \
    libpq-dev \
    gcc \
    gosu \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
COPY src/backend/requirements.txt  /tmp/backend_req.txt
COPY src/frontend/requirements.txt /tmp/frontend_req.txt

RUN pip install --no-cache-dir \
    -r /tmp/backend_req.txt \
    -r /tmp/frontend_req.txt

# ── Application code + data ────────────────────────────────────────────────────
# COPY src/backend/ includes data/products_with_embeddings.json if it exists.
# Run the offline pipeline first to generate it:
#   python -m src.offline.run_all
WORKDIR /app
COPY src/backend/  /app/backend/
COPY src/frontend/ /app/frontend/

# ── Process manager config ─────────────────────────────────────────────────────
COPY src/supervisord.conf /etc/supervisor/supervisord.conf

# ── Entrypoint ─────────────────────────────────────────────────────────────────
COPY src/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Persistent storage directories ────────────────────────────────────────────
# /data — HF Spaces persistent volume for Postgres data
RUN mkdir -p /data/pgdata /var/run/postgresql \
    && chown -R postgres:postgres /var/run/postgresql \
    && chmod 777 /data

EXPOSE 7860 8000

ENTRYPOINT ["/entrypoint.sh"]