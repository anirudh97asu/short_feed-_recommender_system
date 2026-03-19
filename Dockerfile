FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql \
    postgresql-client \
    supervisor \
    libpq-dev \
    gcc \
    gosu \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY src/backend/requirements.txt  /tmp/backend_req.txt
COPY src/frontend/requirements.txt /tmp/frontend_req.txt

RUN pip install --no-cache-dir \
    -r /tmp/backend_req.txt \
    -r /tmp/frontend_req.txt

WORKDIR /app
COPY src/backend/  /app/backend/
COPY src/frontend/ /app/frontend/

COPY src/supervisord.conf /etc/supervisor/supervisord.conf

COPY src/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Pre-create and own pgdata at build time.
# entrypoint.sh also runs chown at runtime to handle HF Spaces
# remounting /data as root on every cold start.
RUN mkdir -p /data/pgdata /var/run/postgresql \
    && chown -R postgres:postgres /var/run/postgresql \
    && chown -R postgres:postgres /data/pgdata \
    && chmod 700 /data/pgdata \
    && chmod 777 /data

EXPOSE 7860 8000

ENTRYPOINT ["/entrypoint.sh"]