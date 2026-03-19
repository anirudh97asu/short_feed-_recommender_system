---
title: Short Feed Recommendation
emoji: 🐨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
# 🛍️ Amazon Fashion Feed Recommendation System

A short-form product discovery feed — TikTok-style personalised recommendations
for fashion, built with CLIP embeddings, hierarchical clustering, and an
explore-exploit strategy driven by explicit like and implicit skip signals.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Step 1 — Offline Pipeline (Local Machine)](#3-step-1--offline-pipeline-local-machine)
4. [Step 2 — Docker Build](#4-step-2--docker-build)
5. [Step 3 — Run Locally with Docker Compose](#5-step-3--run-locally-with-docker-compose)
6. [Step 4 — Test the System](#6-step-4--test-the-system)
7. [Step 5 — Deploy to HuggingFace Spaces](#7-step-5--deploy-to-huggingface-spaces)
8. [Catalog Versioning](#8-catalog-versioning)
9. [How the Recommendation Engine Works](#9-how-the-recommendation-engine-works)
10. [API Reference](#10-api-reference)
11. [Architecture](#11-architecture)
12. [Environment Variables](#12-environment-variables)

---

## 1. Prerequisites

### For the offline pipeline
- Python 3.10+
- A GPU is recommended for CLIP inference (~2 min with GPU, ~30 min on CPU)
- The dataset on your local machine:
  ```
  data/16k_apperal_data_preprocessed   ← pickle file (no extension)
  data/16k_images/<ASIN>.jpeg           ← image folder
  ```

### For Docker
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- [Docker Compose](https://docs.docker.com/compose/install/) (included with Docker Desktop)
- The `products_with_embeddings.json` file produced by the offline pipeline

### Verify everything is working
```bash
python --version        # 3.10 or higher
docker --version        # 24.x or higher
docker compose version  # v2.x or higher
```

---

## 2. Project Structure

```
fashion_rec/
├── Dockerfile                        # single image: Python + PostgreSQL + Supervisor
├── docker-compose.yml                # local dev: external postgres + app
├── supervisord.conf                  # process order: postgres → fastapi → gradio
├── entrypoint.sh                     # first-boot Postgres initialisation
├── .env.example                      # copy to .env for local config overrides
│
├── backend/
│   ├── main.py                       # FastAPI app + all endpoints
│   ├── database.py                   # PostgreSQL: auth, sessions, interactions
│   ├── recommender.py                # pure recommendation logic
│   ├── requirements.txt
│   └── data/
│       └── products_with_embeddings.json   ← produced by offline pipeline
│
├── frontend/
│   ├── app.py                        # Gradio UI
│   └── requirements.txt
│
└── offline/                          # run locally to generate embeddings
    ├── module_01_data_loader.py      # load pickle, clean, match images
    ├── module_02_clip_embeddings.py  # CLIP encode image + text → 512-dim
    ├── module_03_clustering.py       # agglomerative clustering + dendrogram
    ├── module_04_export.py           # merge metadata + embeddings → JSON
    ├── run_all.py                    # runs all 4 modules in sequence
    └── requirements.txt
```

---

## 3. Step 1 — Offline Pipeline (Local Machine)

The offline pipeline generates `products_with_embeddings.json` — the single
artifact that the Docker container needs. Run this once per catalog version.

### 3.1 — Set up a virtual environment

```bash
cd fashion_rec

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows

# Install offline dependencies
pip install -r offline/requirements.txt
```

### 3.2 — Place your data files

```bash
# Create the data directory
mkdir -p data/16k_images

# Your folder should look like this:
# data/
#   16k_apperal_data_preprocessed       ← pickle file (no extension)
#   16k_images/
#     B004GSI2OS.jpeg
#     B012YX2ZPI.jpeg
#     ...
```

### 3.3 — Run modules one by one

Run each module independently so you can inspect intermediate outputs
before committing to the next (heavier) step.

```bash
# Module 1 — Load pickle, clean data, match images
# Input:  data/16k_apperal_data_preprocessed  +  data/16k_images/
# Output: data/cleaned_products.csv
python -m offline.module_01_data_loader
```

Expected output:
```
[load_pickle] Loaded 16,047 rows, 7 columns
[clean] Dropped 12 rows → 16,035 remain
[match_images] 15,980 images found | 55 will use CDN URL
── Data Summary ──────────────────────────────────────
  Total products   : 16,035
  Unique categories: 3
  Unique brands    : 1,203
  Price range      : $0.00 – $299.99
  Images available : 15,980 / 16,035
──────────────────────────────────────────────────────
[run] Saved cleaned data → data/cleaned_products.csv
```

```bash
# Module 2 — CLIP embeddings (~2 min with GPU, ~30 min on CPU)
# Input:  data/cleaned_products.csv
# Output: data/embeddings.npy  +  data/embedding_index.json
python -m offline.module_02_clip_embeddings
```

Expected output:
```
[load_clip] Loading openai/clip-vit-base-patch32 on cuda ...
[load_clip] Model ready.
Embedding batches: 100%|████████| 251/251 [02:03<00:00]
[run] Saved embeddings → data/embeddings.npy  shape=(16035, 512)
[run] Saved ASIN index → data/embedding_index.json
```

```bash
# Module 3 — Hierarchical clustering (~3 min on CPU)
# Input:  data/embeddings.npy
# Output: data/clustered_products.csv  +  data/cluster_summary.json
#         data/dendrogram.png          ← inspect this before proceeding
python -m offline.module_03_clustering
```

> **Important — read the dendrogram before continuing.**
> Open `data/dendrogram.png`. Look for long vertical lines — these mark
> natural cluster boundaries. Set `DISTANCE_THRESHOLD` at the top of
> `offline/module_03_clustering.py` to the distance where the biggest
> jumps occur, then re-run this step.
>
> If the default produces a reasonable cluster count (50–200), continue.

Expected output:
```
[cluster] Mode: distance_threshold = None  (cluster count decided by data)
[cluster] Input shape : (16035, 512)
[cluster] Distance matrix : ~1024 MB on CPU RAM
[cluster] Done.  Clusters found: 87
[evaluate] Silhouette score (n=2000): 0.312
[evaluate] Cluster sizes — min: 42  max: 318  mean: 184  std: 61
[run] Saved → data/clustered_products.csv
[run] Saved → data/cluster_summary.json
```

```bash
# Module 4 — Export final JSON
# Input:  data/clustered_products.csv  +  data/embeddings.npy
# Output: backend/data/products_with_embeddings.json  (~32 MB)
python -m offline.module_04_export
```

Expected output:
```
[run] Products: 16,035  |  Embeddings: (16035, 512)
── Export Summary ────────────────────────────────────
  Records written : 16,035
  File size       : 31.4 MB
  Output path     : backend/data/products_with_embeddings.json
  Sample ASIN     : B004GSI2OS
  Embedding length: 512
  Cluster ID      : 12
──────────────────────────────────────────────────────
[run] ✓ products_with_embeddings.json is ready.
```

### 3.4 — Alternative: run all modules at once

```bash
python -m offline.run_all
```

### 3.5 — Verify the output

```bash
ls -lh backend/data/products_with_embeddings.json
# Should show: ~32 MB

# Deactivate the virtual environment when done
deactivate
```

---

## 4. Step 2 — Docker Build

```bash
# Build the image (~3–5 min on first build, ~30s after caching)
docker build -t fashion-rec:latest .

# Verify the image was created
docker images | grep fashion-rec
# fashion-rec   latest   abc123def456   2 minutes ago   1.2GB
```

> **What is in the image:**
> Python 3.11, PostgreSQL 15, Supervisor, all Python dependencies,
> and the application code (`backend/` and `frontend/`).
>
> **What is NOT in the image:**
> `products_with_embeddings.json` (mounted as a volume at runtime) and
> PostgreSQL data (stored in a named Docker volume).

---

## 5. Step 3 — Run Locally with Docker Compose

Docker Compose starts two services: a dedicated Postgres container and the
application container (FastAPI + Gradio via Supervisor).

```bash
# Start everything (builds image if not already built)
docker compose up --build

# Or if the image is already built
docker compose up

# Run in the background
docker compose up -d
```

You should see logs from all three processes starting in order:

```
postgres_1  | database system is ready to accept connections
app_1       | [startup] Loading products_with_embeddings.json ...
app_1       | [startup] 16,035 products | 87 clusters
app_1       | [db] Schema initialised (users, sessions, interactions)
app_1       | INFO:     Application startup complete.
app_1       | Running on local URL: http://0.0.0.0:7860
```

### Available URLs

| Service | URL | Description |
|---|---|---|
| Gradio UI | http://localhost:7860 | Main user interface |
| FastAPI | http://localhost:8000 | Backend API |
| Swagger docs | http://localhost:8000/docs | Interactive API documentation |
| ReDoc | http://localhost:8000/redoc | Alternative API docs |
| Postgres | localhost:5432 | Database (inspect with psql or TablePlus) |

### View logs

```bash
# All services
docker compose logs -f

# One service at a time
docker compose logs -f app
docker compose logs -f postgres

# Individual process logs inside the app container
docker compose exec app tail -f /tmp/fastapi.log
docker compose exec app tail -f /tmp/gradio.log
docker compose exec app tail -f /tmp/postgres.log
```

### Stop

```bash
docker compose down        # stop containers, keep DB data
docker compose down -v     # stop containers AND wipe all DB data
docker compose restart     # restart without rebuilding
```

---

## 6. Step 4 — Test the System

### 6.1 — Health check

```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "ok",
  "products": 16035,
  "clusters": 87,
  "db": true
}
```

If `db` is `false`, Postgres hasn't finished starting yet. Wait 10 seconds and retry.

### 6.2 — Create a new user

```bash
curl -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username": "alice"}'
```

Expected:
```json
{
  "user_id": "a1b2c3d4-...",
  "username": "alice",
  "session_id": "e5f6g7h8-...",
  "is_new": true
}
```

Save `user_id` and `session_id` — you'll need them for all subsequent calls.

### 6.3 — Login as returning user

```bash
curl -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username": "alice"}'
```

Expected: same `user_id`, a new `session_id`, and `"is_new": false`.

### 6.4 — Get cold-start recommendations

```bash
curl "http://localhost:8000/recommend?user_id=<USER_ID>&session_id=<SESSION_ID>&n=4"
```

Expected: 4 products from different clusters (maximum diversity):
```json
[
  {
    "asin": "B004GSI2OS",
    "title": "featherlite ladies long sleeve stain resistant shirt",
    "brand": "FeatherLite",
    "price": 26.26,
    "image_url": "https://images-na.ssl-images-amazon.com/...",
    "cluster_id": 12
  },
  ...
]
```

### 6.5 — Log an explicit like

```bash
curl -X POST http://localhost:8000/interact \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<SESSION_ID>",
    "user_id": "<USER_ID>",
    "asin": "B004GSI2OS",
    "action": "liked",
    "view_duration_seconds": 8.4
  }'
```

Expected: `{"status": "ok"}`

### 6.6 — Log an implicit skip

Items viewed for more than 15 seconds without a like are automatically
logged as implicit skips by the frontend. You can also log them manually:

```bash
curl -X POST http://localhost:8000/interact \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<SESSION_ID>",
    "user_id": "<USER_ID>",
    "asin": "B012YX2ZPI",
    "action": "skipped",
    "view_duration_seconds": 22.1
  }'
```

### 6.7 — Get personalised recommendations

```bash
curl "http://localhost:8000/recommend?user_id=<USER_ID>&session_id=<SESSION_ID>&n=4"
```

Results should now reflect the like and skip signals from above.

### 6.8 — Logout

```bash
curl -X POST http://localhost:8000/logout \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<SESSION_ID>"}'
```

Expected: `{"status": "session closed"}`

### 6.9 — Verify history persists across sessions

```bash
# Login again as the same user
curl -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username": "alice"}'

# Get recommendations with the new session_id
curl "http://localhost:8000/recommend?user_id=<USER_ID>&session_id=<NEW_SESSION_ID>&n=4"
# Should still reflect the like and skip from the previous session
```

### 6.10 — Use Swagger UI (easier than curl)

Open http://localhost:8000/docs — execute all API calls interactively
without writing curl commands.

### 6.11 — Inspect the database

```bash
# Connect to Postgres inside the container
docker compose exec postgres psql -U fashion -d fashion_rec

# Useful queries
SELECT * FROM users;
SELECT * FROM sessions ORDER BY started_at DESC;
SELECT * FROM interactions ORDER BY created_at DESC LIMIT 20;

# View duration breakdown — are implicit skips working?
SELECT action,
       COUNT(*)                              AS count,
       ROUND(AVG(view_duration_seconds), 1) AS avg_duration_secs,
       ROUND(MIN(view_duration_seconds), 1) AS min_duration_secs,
       ROUND(MAX(view_duration_seconds), 1) AS max_duration_secs
FROM   interactions
WHERE  view_duration_seconds IS NOT NULL
GROUP  BY action;

# Interaction breakdown per user
SELECT u.username, i.action, COUNT(*)
FROM   interactions i
JOIN   users u ON u.user_id = i.user_id
GROUP  BY u.username, i.action
ORDER  BY u.username, i.action;

\q   # exit psql
```

---

## 7. Step 5 — Deploy to HuggingFace Spaces

### 7.1 — Prepare the repo

```bash
git add backend/data/products_with_embeddings.json
git commit -m "add product embeddings v1"
git push
```

> If the file exceeds 50 MB, use [Git LFS](https://git-lfs.com):
> ```bash
> git lfs install
> git lfs track "*.json"
> git add .gitattributes
> git add backend/data/products_with_embeddings.json
> git commit -m "add embeddings via LFS"
> git push
> ```

### 7.2 — Create a HuggingFace Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose a name (e.g. `fashion-feed`)
3. Select **Docker** as the Space SDK
4. Set visibility to Public or Private
5. Click **Create Space**

### 7.3 — Push your repo to the Space

```bash
git remote add space https://huggingface.co/spaces/<your-username>/fashion-feed
git push space main
```

HF Spaces builds the Dockerfile automatically. No further config needed.

### 7.4 — Monitor the build

Go to your Space URL → **Logs** tab. You should see:

```
Building Docker image...
[entrypoint] No external DATABASE_URL — setting up internal Postgres ...
[entrypoint] First boot — initialising Postgres data directory ...
[entrypoint] Postgres initialised successfully.
[startup] Loading products_with_embeddings.json ...
[startup] 16,035 products | 87 clusters
Running on local URL: http://0.0.0.0:7860
```

### 7.5 — Optional secrets

In Space Settings → **Secrets**:

```
POSTGRES_PASSWORD    your_strong_production_password
```

---

## 8. Catalog Versioning

The JSON catalog is mounted as a Docker volume at runtime — not baked into
the image. Swap the catalog without rebuilding:

```bash
# Build the image once
docker build -t fashion-rec:latest .

# Autumn catalog
docker run -d -p 7860:7860 -p 8000:8000 \
  -v $(pwd)/data/v1/products_with_embeddings.json:/app/backend/data/products_with_embeddings.json \
  --name fashion-v1 \
  fashion-rec:latest

# Winter catalog — same image, different data, no rebuild
docker run -d -p 7861:7860 -p 8001:8000 \
  -v $(pwd)/data/v2/products_with_embeddings.json:/app/backend/data/products_with_embeddings.json \
  --name fashion-v2 \
  fashion-rec:latest
```

With docker-compose, update the volume source path and run `docker compose up`.

---

## 9. How the Recommendation Engine Works

### Interaction signals

| Signal | How triggered | Weight |
|---|---|---|
| `liked` | User clicks 👍 Like | `+1.0 × decay` |
| `skipped` (implicit) | Item viewed > 15 seconds without a like | `−0.3 × decay` |
| `shown` | Item displayed in feed | not used in embedding |

The 15-second threshold is applied by the frontend automatically — on every
Like or Refresh, it checks how long each currently visible product has been
on screen and logs any product viewed over 15 seconds (without a like) as
an implicit skip, including the actual `view_duration_seconds`.

### User Embedding (signed + recency-weighted)

```
weight = sign × exp(−λ × days_since_interaction)

  liked    → sign = +1.0   strong pull toward this style
  skipped  → sign = −0.3   weak push away (implicit skips are noisier than likes)

  λ = 0.1:
    like today        → weight = +1.000
    like 7 days ago   → weight = +0.497
    like 30 days ago  → weight = +0.050
    skip today        → weight = −0.300
```

### Explore-Exploit

```
α = 0.3   for users with < 5 likes  (explore more — still learning taste)
α = 0.1   for users with ≥ 5 likes  (exploit more — good signal established)

Per recommendation slot:
  random() > α  →  EXPLOIT: highest cosine similarity to user embedding
  random() ≤ α  →  EXPLORE: random product from a never-liked cluster
```

### Full session lifecycle

```
Login
  └── returning user → load all liked/skipped history across all sessions
  └── new user       → cold start: 1 random product per cluster

During session
  ├── items appear   → logged as 'shown' (no duration yet)
  ├── user likes     → logged as 'liked'  with view_duration_seconds
  │                    feed refreshes immediately
  │                    implicit skips logged for other visible items > 15s
  └── user refreshes → implicit skips logged for visible items > 15s
                        feed refreshes

Logout
  └── implicit skips logged for any remaining visible items > 15s
  └── session.ended_at set to NOW()
  └── full history preserved for next login
```

---

## 10. API Reference

| Method | Endpoint | Body / Params | Response |
|---|---|---|---|
| `GET` | `/health` | — | `{status, products, clusters, db}` |
| `POST` | `/login` | `{username}` | `{user_id, username, session_id, is_new}` |
| `POST` | `/logout` | `{session_id}` | `{status}` |
| `POST` | `/interact` | `{session_id, user_id, asin, action, view_duration_seconds?}` | `{status}` |
| `GET` | `/recommend` | `?user_id&session_id&n=12` | `[{asin, title, brand, price, image_url, cluster_id}]` |
| `GET` | `/docs` | — | Swagger UI |

Valid `action` values: `shown` · `liked` · `skipped`

`view_duration_seconds` is optional — pass `null` for `shown` events,
and the actual elapsed seconds for `liked` and `skipped` events.

---

## 11. Architecture

```
LOCAL (docker-compose)                   HF SPACES (single container)
┌──────────────────────────┐             ┌───────────────────────────────────┐
│  postgres (service)      │             │  [supervisor — PID 1]             │
│  image: postgres:15-alpine│            │    ├── postgres  :5432 (internal) │
│  healthcheck: pg_isready │             │    ├── fastapi   :8000 (internal) │
│  volume: postgres_data   │             │    └── gradio    :7860 (public)   │
└─────────────┬────────────┘             │                                   │
              │ DATABASE_URL             │  /data/pgdata  ← persistent vol   │
┌─────────────▼────────────┐             │  /app/backend/data ← JSON mount   │
│  app (service)           │             └───────────────────────────────────┘
│  [supervisor]            │
│    ├── fastapi   :8000   │        entrypoint.sh detects environment:
│    └── gradio    :7860   │          external DATABASE_URL
│  JSON: volume mount :ro  │            → skip initdb, start supervisor
└──────────────────────────┘          no external DATABASE_URL
                                        → run initdb, create user/db,
                                          start supervisor
```

---

## 12. Environment Variables

| Variable | Default | Description | Where to set |
|---|---|---|---|
| `DATABASE_URL` | auto-set by entrypoint | Full Postgres connection string | `docker-compose.yml` or HF Secrets |
| `POSTGRES_PASSWORD` | `fashion_secret` | Internal Postgres password | `.env` or HF Secrets |
| `BACKEND_URL` | `http://localhost:8000` | FastAPI URL used by Gradio | HF Secrets (only if splitting services) |
