"""
MODULE 8 — Gradio Frontend
============================
Purpose : User-facing UI with login, a product feed, and
          implicit/explicit interaction signals.

Interaction signals:
  Explicit:
    👍 Like    → log action='liked'  with view duration
                 then refresh feed

  Implicit:
    ⏱ View > 15s without liking
               → automatically logged as action='skipped'
                 with the actual view duration
               → triggered on Like AND on Refresh

  Neutral:
    Shown      → logged when each product first appears
                 (no duration yet — recorded as NULL)

View duration tracking:
  state["shown_at"] stores { asin: timestamp_float } for every product
  currently visible in the feed. When the user clicks Like or Refresh,
  we compute (now - shown_at) for each visible product:
    - If the product was just liked         → log as 'liked'
    - If view_duration > IMPLICIT_SKIP_SECS → log as 'skipped'
    - Otherwise                             → no further action

Screens:
  1. Login  — username input
  2. Feed   — 4×3 product grid + Like button + Refresh + Logout
"""

import os
import time
import requests
import gradio as gr


# ── Config ─────────────────────────────────────────────────────────────────────

BACKEND             = os.environ.get("BACKEND_URL", "http://localhost:8000")
N                   = 12      # products per page
IMPLICIT_SKIP_SECS  = 15.0    # view longer than this without liking → implicit skip


# ── API helpers ────────────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs) -> dict | list | None:
    """Thin wrapper around requests — returns parsed JSON or None on error."""
    try:
        r = requests.request(method, f"{BACKEND}{path}", timeout=10, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[api] {method} {path} failed: {e}")
        return None


def api_login(username: str) -> dict | None:
    return api("POST", "/login", json={"username": username})

def api_logout(session_id: str) -> None:
    api("POST", "/logout", json={"session_id": session_id})

def api_interact(session_id: str, user_id: str, asin: str,
                 action: str, duration: float | None = None) -> None:
    """Log a single interaction. duration is None for 'shown' events."""
    api("POST", "/interact", json={
        "session_id"           : session_id,
        "user_id"              : user_id,
        "asin"                 : asin,
        "action"               : action,
        "view_duration_seconds": round(duration, 2) if duration is not None else None,
    })

def api_recommend(user_id: str, session_id: str) -> list[dict]:
    result = api("GET", "/recommend", params={
        "user_id"   : user_id,
        "session_id": session_id,
        "n"         : N,
    })
    return result if isinstance(result, list) else []


# ── State ──────────────────────────────────────────────────────────────────────

def fresh_state() -> dict:
    return {
        "user_id"   : None,
        "username"  : None,
        "session_id": None,
        "products"  : [],          # list[dict] currently shown in feed
        "shown_at"  : {},          # { asin: time.time() } — when each product appeared
    }


# ── Core timing logic ──────────────────────────────────────────────────────────

def process_implicit_skips(
    state     : dict,
    liked_asin: str | None = None,
) -> None:
    """
    Called before every feed refresh (Like or Refresh button).

    For each product currently visible in the feed:
      - If it was just liked            → log as 'liked' with duration
      - If viewed > IMPLICIT_SKIP_SECS  → log as 'skipped' with duration
      - Otherwise                       → do nothing (too brief to be a signal)

    liked_asin: the ASIN the user clicked Like on, or None (Refresh case).
    """
    now        = time.time()
    session_id = state["session_id"]
    user_id    = state["user_id"]
    shown_at   = state["shown_at"]

    for product in state["products"]:
        asin     = product["asin"]
        shown_ts = shown_at.get(asin)

        if shown_ts is None:
            continue    # safety guard — should always exist

        duration = now - shown_ts

        if asin == liked_asin:
            # Explicit like — always log regardless of duration
            api_interact(session_id, user_id, asin, "liked", duration)

        elif duration >= IMPLICIT_SKIP_SECS:
            # Viewed long enough without liking → implicit skip
            api_interact(session_id, user_id, asin, "skipped", duration)

        # else: viewed too briefly — neutral, no log


def show_new_feed(state: dict, products: list[dict]) -> dict:
    """
    Update state with a fresh set of products and record shown_at
    for every product. Also logs 'shown' for each to the backend.
    """
    now        = time.time()
    session_id = state["session_id"]
    user_id    = state["user_id"]

    state["products"] = products
    state["shown_at"] = {p["asin"]: now for p in products}

    # Log each product as 'shown' (duration unknown → None)
    for p in products:
        api_interact(session_id, user_id, p["asin"], "shown", None)

    return state


# ── Feed renderer ──────────────────────────────────────────────────────────────

def render_gallery(products: list[dict]) -> list[tuple]:
    """Convert product dicts → (image_url, caption) for gr.Gallery."""
    return [
        (
            p["image_url"],
            f"{p['title'][:45]}...\n${p['price']:.2f} · {p['brand']}"
        )
        for p in products
    ]


# ── Event handlers ─────────────────────────────────────────────────────────────

def on_login(username: str, state: dict):
    """Login or register — single endpoint handles both."""
    if not username.strip():
        return state, gr.update(), gr.update(), gr.update(), \
               "⚠️ Please enter a username."

    result = api_login(username)
    if not result:
        return state, gr.update(), gr.update(), gr.update(), \
               "⚠️ Could not reach the backend. Is it running?"

    state["user_id"]    = result["user_id"]
    state["username"]   = result["username"]
    state["session_id"] = result["session_id"]

    # Fetch initial feed and record shown timestamps
    products = api_recommend(result["user_id"], result["session_id"])
    state    = show_new_feed(state, products)

    greeting = (
        f"👋 Welcome back, **{result['username']}**! Your personalised feed is ready."
        if not result["is_new"] else
        f"🎉 Welcome, **{result['username']}**! "
        f"Like items to personalise your feed. "
        f"Items viewed for over {IMPLICIT_SKIP_SECS:.0f}s without a like "
        f"are automatically used as negative signals."
    )

    return (
        state,
        gr.update(visible=False),   # hide login screen
        gr.update(visible=True),    # show feed screen
        render_gallery(products),
        greeting,
    )


def on_like(idx: int, state: dict):
    """
    User liked the product at gallery position idx.

    1. Process implicit skips for all other visible products
       (those viewed > 15s log as skipped, the liked one logs as liked)
    2. Fetch fresh recommendations
    3. Record new shown_at timestamps
    """
    products = state.get("products", [])
    if not products or idx >= len(products):
        return state, gr.update(), "⚠️ Could not identify product."

    liked_asin = products[idx]["asin"]

    # Step 1 — log liked + any implicit skips from current feed
    process_implicit_skips(state, liked_asin=liked_asin)

    # Step 2 — fetch fresh feed
    new_products = api_recommend(state["user_id"], state["session_id"])

    # Step 3 — record new shown timestamps
    state = show_new_feed(state, new_products)

    return (
        state,
        render_gallery(new_products),
        f"❤️ Liked! Items viewed over {IMPLICIT_SKIP_SECS:.0f}s "
        f"were recorded as implicit skips. Feed refreshed.",
    )


def on_refresh(state: dict):
    """
    User manually refreshed the feed.

    1. Process implicit skips for all currently visible products
       (no liked_asin — everything is evaluated for implicit skip only)
    2. Fetch fresh recommendations
    3. Record new shown_at timestamps
    """
    # Step 1 — check all visible products for implicit skips
    process_implicit_skips(state, liked_asin=None)

    # Step 2 — fetch fresh feed
    new_products = api_recommend(state["user_id"], state["session_id"])

    # Step 3 — record new shown timestamps
    state = show_new_feed(state, new_products)

    return (
        state,
        render_gallery(new_products),
        "🔄 Feed refreshed. Implicit skips recorded for items viewed "
        f"over {IMPLICIT_SKIP_SECS:.0f}s.",
    )


def on_logout(state: dict):
    """Close session and return to login screen."""
    # Process any remaining implicit skips before closing
    process_implicit_skips(state, liked_asin=None)

    if state.get("session_id"):
        api_logout(state["session_id"])

    username  = state.get("username", "")
    new_state = fresh_state()

    return (
        new_state,
        gr.update(visible=True),    # show login screen
        gr.update(visible=False),   # hide feed screen
        [],
        f"👋 Goodbye, **{username}**! "
        "Your preferences have been saved for next time.",
    )


# ── Layout ─────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Fashion Discovery", theme=gr.themes.Soft()) as app:

    state = gr.State(fresh_state())

    gr.Markdown("# 🛍️ Fashion Discovery Feed")
    status = gr.Markdown("Enter your username to begin.")

    # ── Screen 1: Login ───────────────────────────────────────────────────────
    with gr.Column(visible=True) as login_screen:
        gr.Markdown("### Login or create an account")
        gr.Markdown(
            "Enter any username. Existing accounts are loaded automatically. "
            "New usernames create a fresh account."
        )
        username_box = gr.Textbox(
            placeholder = "e.g. alice",
            label       = "Username",
            max_lines   = 1,
        )
        login_btn = gr.Button("Continue →", variant="primary")

    # ── Screen 2: Feed ────────────────────────────────────────────────────────
    with gr.Column(visible=False) as feed_screen:

        gallery = gr.Gallery(
            label      = "Your personalised feed",
            columns    = 4,
            rows       = 3,
            height     = 560,
            object_fit = "contain",
        )

        gr.Markdown(
            f"Select a product position (0–{N-1}) and click **Like** to signal interest. "
            f"Products you view for more than **{IMPLICIT_SKIP_SECS:.0f} seconds** "
            f"without liking are automatically recorded as implicit skips."
        )

        with gr.Row():
            idx_slider  = gr.Slider(
                minimum = 0,
                maximum = N - 1,
                step    = 1,
                value   = 0,
                label   = f"Product position (0 = top-left, {N-1} = bottom-right)",
            )

        with gr.Row():
            like_btn    = gr.Button("👍 Like",    variant="primary")
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            logout_btn  = gr.Button("Log out",    variant="stop")

    # ── Wire events ───────────────────────────────────────────────────────────

    login_btn.click(
        fn      = on_login,
        inputs  = [username_box, state],
        outputs = [state, login_screen, feed_screen, gallery, status],
    )

    like_btn.click(
        fn      = on_like,
        inputs  = [idx_slider, state],
        outputs = [state, gallery, status],
    )

    refresh_btn.click(
        fn      = on_refresh,
        inputs  = [state],
        outputs = [state, gallery, status],
    )

    logout_btn.click(
        fn      = on_logout,
        inputs  = [state],
        outputs = [state, login_screen, feed_screen, gallery, status],
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
