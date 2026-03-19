#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# entrypoint.sh
# ──────────────────────────────────────────────────────────────────────────────
# Writes the correct supervisord config at runtime based on environment,
# then hands off to supervisord.
#
#   LOCAL (docker-compose):
#     DATABASE_URL points at external postgres service.
#     → Write a config with FastAPI + Gradio ONLY (no internal postgres)
#
#   HF SPACES (single container):
#     No external DATABASE_URL.
#     → Initialise postgres, write config with all three processes.
# ──────────────────────────────────────────────────────────────────────────────

set -e

PGDATA="/data/pgdata"
PGUSER="fashion"
PGDB="fashion_rec"
PGPASSWORD="${POSTGRES_PASSWORD:-fashion_secret}"
RUNTIME_CONF="/tmp/supervisord-runtime.conf"

# ── Shared supervisord header ──────────────────────────────────────────────────
write_header() {
cat > "$RUNTIME_CONF" << 'EOF'
[supervisord]
nodaemon=true
logfile=/tmp/supervisord.log
pidfile=/tmp/supervisord.pid
loglevel=info

[unix_http_server]
file=/tmp/supervisor.sock

[supervisorctl]
serverurl=unix:///tmp/supervisor.sock

[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface
EOF
}

# ── Postgres program block ─────────────────────────────────────────────────────
write_postgres() {
cat >> "$RUNTIME_CONF" << 'EOF'

[program:postgres]
command=/bin/bash -c "exec gosu postgres /usr/lib/postgresql/$(ls /usr/lib/postgresql/)/bin/postgres -D /data/pgdata -c listen_addresses=localhost -c log_destination=stderr"
priority=100
autostart=true
autorestart=true
startretries=3
startsecs=5
user=root
stdout_logfile=/tmp/postgres.log
stderr_logfile=/tmp/postgres.log
EOF
}

# ── FastAPI program block ──────────────────────────────────────────────────────
write_fastapi() {
cat >> "$RUNTIME_CONF" << EOF

[program:fastapi]
command=uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
directory=/app/backend
priority=200
autostart=true
autorestart=true
startretries=5
startsecs=10
environment=DATABASE_URL="$DATABASE_URL"
stdout_logfile=/tmp/fastapi.log
stderr_logfile=/tmp/fastapi.log
EOF
}

# ── Gradio program block ───────────────────────────────────────────────────────
write_gradio() {
cat >> "$RUNTIME_CONF" << EOF

[program:gradio]
command=python app.py
directory=/app/frontend
priority=300
autostart=true
autorestart=true
startretries=5
startsecs=20
environment=BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
stdout_logfile=/tmp/gradio.log
stderr_logfile=/tmp/gradio.log
EOF
}

# ── Detect environment and build config ───────────────────────────────────────

if [ -n "$DATABASE_URL" ] && [[ "$DATABASE_URL" != *"localhost"* ]]; then
    # ── docker-compose mode: external Postgres ────────────────────────────────
    echo "[entrypoint] External DATABASE_URL detected."
    echo "[entrypoint] Writing config: FastAPI + Gradio only (no internal Postgres)."

    write_header
    write_fastapi
    write_gradio

else
    # ── HF Spaces mode: internal Postgres ────────────────────────────────────
    echo "[entrypoint] No external DATABASE_URL — setting up internal Postgres."

    if [ ! -f "$PGDATA/PG_VERSION" ]; then
        echo "[entrypoint] First boot — initialising Postgres at $PGDATA ..."

        gosu postgres /usr/lib/postgresql/*/bin/initdb \
            --pgdata="$PGDATA" \
            --auth=md5 \
            --username=postgres \
            --encoding=UTF8

        gosu postgres /usr/lib/postgresql/*/bin/pg_ctl \
            -D "$PGDATA" -o "-c listen_addresses=''" -w start

        gosu postgres psql --username=postgres <<-EOSQL
            CREATE USER ${PGUSER} WITH PASSWORD '${PGPASSWORD}';
            CREATE DATABASE ${PGDB} OWNER ${PGUSER};
            GRANT ALL PRIVILEGES ON DATABASE ${PGDB} TO ${PGUSER};
EOSQL

        gosu postgres /usr/lib/postgresql/*/bin/pg_ctl \
            -D "$PGDATA" -w stop

        echo "[entrypoint] Postgres initialised successfully."
    else
        echo "[entrypoint] Postgres data directory already exists — skipping init."
    fi

    export DATABASE_URL="postgresql://${PGUSER}:${PGPASSWORD}@localhost:5432/${PGDB}"

    echo "[entrypoint] Writing config: Postgres + FastAPI + Gradio."

    write_header
    write_postgres
    write_fastapi
    write_gradio
fi

echo "[entrypoint] Starting supervisord with $RUNTIME_CONF ..."
exec /usr/bin/supervisord -c "$RUNTIME_CONF"