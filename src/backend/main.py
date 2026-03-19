"""
MODULE 7 — FastAPI Application
================================
Purpose : HTTP layer — wires auth, sessions, interactions, and
          the recommender together into a clean REST API.

Auth flow:
  POST /login   → lookup username → return existing user OR create new one
  POST /logout  → close the session, persist interaction history

Interaction flow:
  POST /interact  → log 'shown', 'liked', or 'skipped'
  GET  /recommend → fetch history, build embedding, return N products

Data:
  products_with_embeddings.json is MOUNTED as a Docker volume at runtime.
  This means you can update the catalog by mounting a new file and
  restarting the container — no image rebuild required.
  Different image versions simply default to different mounted files.
"""

import json
import os
import numpy as np

from contextlib         import asynccontextmanager
from fastapi            import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic           import BaseModel

import database
import recommender as rec


# ── Constants ──────────────────────────────────────────────────────────────────

# Mounted at runtime via Docker volume — see docker-compose.yml
DATA_FILE = os.path.join(
    os.path.dirname(__file__), "data", "products_with_embeddings.json"
)

# In-memory stores — loaded once at startup, read-only during serving
PRODUCTS    : dict                  = {}   # { asin: product_dict }
EMBEDDINGS  : dict                  = {}   # { asin: np.ndarray(512,) }
CLUSTERS    : dict                  = {}   # { cluster_id_str: [asin, ...] }
RECOMMENDER : rec.Recommender | None = None  # single instance, reused per request


# ── Startup ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the JSON catalog into RAM once at startup."""
    global PRODUCTS, EMBEDDINGS, CLUSTERS

    print("[startup] Loading products_with_embeddings.json ...")
    with open(DATA_FILE) as f:
        raw = json.load(f)

    for p in raw:
        asin             = p["asin"]
        PRODUCTS[asin]   = p
        EMBEDDINGS[asin] = np.array(p["embedding"], dtype=np.float32)
        cid              = str(p["cluster_id"])
        CLUSTERS.setdefault(cid, []).append(asin)

    print(f"[startup] {len(PRODUCTS):,} products | {len(CLUSTERS)} clusters")

    # Build the global embedding matrix ONCE — shared across all requests
    rec.init_matrix(EMBEDDINGS)

    # Instantiate the recommender — stateless, reused for every request
    global RECOMMENDER
    RECOMMENDER = rec.Recommender(PRODUCTS, EMBEDDINGS, CLUSTERS)

    database.init_db()
    yield
    print("[shutdown] Done.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Fashion Feed Recommender",
    description = "Username-based auth, session tracking, liked/skipped signals.",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str

class LoginResponse(BaseModel):
    user_id   : str
    username  : str
    session_id: str
    is_new    : bool          # True = just registered, False = returning user

class LogoutRequest(BaseModel):
    session_id: str

class InteractRequest(BaseModel):
    session_id            : str
    user_id               : str
    asin                  : str
    action                : str          # 'shown' | 'liked' | 'skipped'
    view_duration_seconds : float | None = None   # None for 'shown' events

class ProductOut(BaseModel):
    asin      : str
    title     : str
    brand     : str
    price     : float
    image_url : str
    cluster_id: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status"  : "ok",
        "products": len(PRODUCTS),
        "clusters": len(CLUSTERS),
        "db"      : database.health_check(),
    }


@app.post("/login", response_model=LoginResponse, tags=["Auth"])
def login(body: LoginRequest):
    """
    Single endpoint for both login and registration.

    Logic:
      - Look up the username in the DB
      - If found    → returning user, open a new session
      - If not found→ register new user, open a new session

    The frontend shows a welcome-back message vs a first-time message
    based on the `is_new` field in the response.

    Username is normalised to lowercase + stripped of whitespace
    so 'Alice', 'alice', and ' Alice ' all resolve to the same account.
    """
    if not body.username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty.")

    existing = database.get_user_by_username(body.username)

    if existing:
        user   = existing
        is_new = False
    else:
        user   = database.create_user(body.username)
        is_new = True

    session_id = database.create_session(str(user["user_id"]))

    return {
        "user_id"   : str(user["user_id"]),
        "username"  : user["username"],
        "session_id": session_id,
        "is_new"    : is_new,
    }


@app.post("/logout", tags=["Auth"])
def logout(body: LogoutRequest):
    """
    Close the current session. Interaction history is persisted in the DB
    and will automatically inform the user's embedding on their next login.
    """
    database.close_session(body.session_id)
    return {"status": "session closed"}


@app.post("/interact", tags=["Interactions"])
def interact(body: InteractRequest):
    """
    Log a single user interaction.

    action must be one of:
      'shown'   — item was displayed to the user (neutral)
      'liked'   — user clicked the like button   (strong positive)
      'skipped' — user clicked skip              (weak negative)

    Call this:
      - With 'shown'   for every item displayed in the feed
      - With 'liked'   when the user presses Like
      - With 'skipped' when the user presses Skip
    """
    valid_actions = {"shown", "liked", "skipped"}
    if body.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"action must be one of {valid_actions}"
        )
    if body.asin not in PRODUCTS:
        raise HTTPException(status_code=404, detail=f"ASIN '{body.asin}' not found")

    database.log_interaction(
        session_id            = body.session_id,
        user_id               = body.user_id,
        asin                  = body.asin,
        action                = body.action,
        view_duration_seconds = body.view_duration_seconds,
    )
    return {"status": "ok"}


@app.get("/recommend", response_model=list[ProductOut], tags=["Recommendations"])
def recommend(user_id: str, session_id: str):
    """
    Core recommendation endpoint.

    Returns TOP_K (5) ranked products via explore-exploit.
    The frontend queues these 5 and displays them one at a time.

    New users (no history) receive a diverse cold-start selection.
    Returning users benefit immediately from all past session history.
    """
    interactions = database.get_user_interactions(user_id)
    return RECOMMENDER.recommend(interactions)