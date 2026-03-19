"""
MODULE 6 — Recommender Engine
================================
Purpose : Class-based recommendation logic.

Design decisions:
  1. MATRIX is built at MODULE LEVEL — the moment this file is imported,
     the 16k × 512 numpy matrix is constructed once and held in memory.
     No function or method ever rebuilds it. Every cosine similarity call
     is a single matrix multiply against this shared object.

  2. Recommender class holds no mutable state — it receives the data
     stores (products, embeddings, clusters) at construction and exposes
     pure methods. Safe to instantiate once and reuse across all requests.

  3. Top 5 recommendations returned — the feed shows one product at a
     time, so 5 is the right queue depth: enough to fill one session
     page without over-fetching.

User embedding model:
  liked   → weight = +1.0 × exp(−λ × days_ago)
  skipped → weight = −0.3 × exp(−λ × days_ago)

  Asymmetry: likes are a stronger signal than skips.
  Negative weights push the user embedding away from skipped styles.

Explore-exploit:
  α = 0.3  for users with < 5 likes  (explore more)
  α = 0.1  for users with ≥ 5 likes  (exploit more)
"""

import random
import numpy as np

from datetime import datetime


# ── Tunable constants ──────────────────────────────────────────────────────────

DECAY_LAMBDA = 0.1   # recency decay — higher = forget old signals faster
SKIP_WEIGHT  = 0.3   # negative signal strength relative to like (+1.0)
ALPHA_LOW    = 0.3   # explore rate: users with < LIKE_THRESH likes
ALPHA_HIGH   = 0.1   # explore rate: users with >= LIKE_THRESH likes
LIKE_THRESH  = 5     # like count that flips explore rate from high → low
TOP_K        = 5     # number of recommendations to return per call


# ── Module-level embedding matrix ─────────────────────────────────────────────
# Built ONCE when this module is first imported.
# Shared across all Recommender instances and all requests.
# Shape: (N_products, 512)
#
# Why here and not inside the class:
#   If it lived inside a method, numpy would allocate a new 32MB array
#   on every recommendation call. At the module level it is allocated
#   exactly once and stays pinned in memory for the lifetime of the process.

_MATRIX    : np.ndarray | None = None   # populated by init_matrix()
_ALL_ASINS : list[str]         = []


def init_matrix(embeddings: dict) -> None:
    """
    Build the global embedding matrix from the embeddings dict.
    Must be called once at application startup (main.py lifespan),
    before any Recommender instance is used.

    Args:
        embeddings: { asin: np.ndarray(512,) }
    """
    global _MATRIX, _ALL_ASINS
    _ALL_ASINS = list(embeddings.keys())
    _MATRIX    = np.array(
        [embeddings[a] for a in _ALL_ASINS],
        dtype=np.float32
    )   # shape (N, 512)
    print(f"[recommender] Matrix initialised — shape: {_MATRIX.shape}")


# ── Recommender class ──────────────────────────────────────────────────────────

class Recommender:
    """
    Stateless recommendation engine.
    Instantiate once at startup; all methods are pure (no side effects).

    Args:
        products   : { asin: product_dict }
        embeddings : { asin: np.ndarray(512,) }
        clusters   : { cluster_id_str: [asin, ...] }
    """

    def __init__(self, products: dict, embeddings: dict, clusters: dict):
        self.products   = products
        self.embeddings = embeddings
        self.clusters   = clusters

    # ── Public API ─────────────────────────────────────────────────────────────

    def recommend(
        self,
        interactions : list[dict],   # from database.get_user_interactions()
        shown_asins  : set[str] = None,   # already shown this session
    ) -> list[dict]:
        """
        Main entry point. Returns TOP_K ranked product dicts.

        Flow:
          no interactions  → cold start  (diverse, one per cluster)
          has interactions → build signed user embedding
                           → explore or exploit per slot
        """
        shown_asins = shown_asins or set()

        # No history — cold start
        if not interactions:
            return self._cold_start()

        # Build user embedding from interaction history
        user_emb = self._build_user_embedding(interactions)
        if user_emb is None:
            return self._cold_start()

        return self._explore_exploit(user_emb, interactions, shown_asins)

    # ── Private methods ────────────────────────────────────────────────────────

    def _build_user_embedding(
        self,
        interactions: list[dict],
    ) -> np.ndarray | None:
        """
        Signed, recency-decayed weighted sum of interacted embeddings.

        liked   → +1.0 × exp(−λ × days_ago)
        skipped → −0.3 × exp(−λ × days_ago)

        Returns None if no valid embeddings found (triggers cold start).
        """
        now     = datetime.utcnow()
        weights = []
        vecs    = []

        for row in interactions:
            asin = row["asin"]
            if asin not in self.embeddings:
                continue

            days_ago = (now - row["created_at"]).total_seconds() / 86400
            decay    = np.exp(-DECAY_LAMBDA * days_ago)
            sign     = +1.0 if row["action"] == "liked" else -SKIP_WEIGHT

            weights.append(sign * decay)
            vecs.append(self.embeddings[asin])

        if not vecs:
            return None

        weights  = np.array(weights, dtype=np.float32)   # signed, shape (N,)
        vecs     = np.array(vecs,    dtype=np.float32)   # shape (N, 512)
        user_emb = (weights[:, None] * vecs).sum(axis=0) # weighted sum

        norm = np.linalg.norm(user_emb)
        if norm < 1e-8:
            return None   # signals cancelled out — fall back to cold start

        return user_emb / norm   # normalise to unit vector

    def _cosine_similarity(
        self,
        query_emb    : np.ndarray,
        exclude_asins: set[str],
        top_k        : int,
    ) -> list[str]:
        """
        Dot product of query against the global MATRIX (pre-built at import).
        Equivalent to cosine similarity since all vectors are L2-normalised.

        Returns top_k ASINs sorted by descending similarity score,
        excluding any ASIN in exclude_asins.
        """
        if _MATRIX is None:
            raise RuntimeError("Call init_matrix() before using Recommender.")

        scores = _MATRIX @ query_emb        # shape (N,) — single BLAS call
        ranked = np.argsort(scores)[::-1]

        result = []
        for idx in ranked:
            asin = _ALL_ASINS[idx]
            if asin not in exclude_asins:
                result.append(asin)
            if len(result) == top_k:
                break
        return result

    def _cold_start(self) -> list[dict]:
        """
        No interaction history available.
        Pick 1 random product from each of TOP_K different clusters.
        Guarantees visual diversity on the first screen.
        """
        cluster_ids = list(self.clusters.keys())
        random.shuffle(cluster_ids)

        picks = []
        for cid in cluster_ids:
            candidates = self.clusters[cid]
            if candidates:
                picks.append(random.choice(candidates))
            if len(picks) == TOP_K:
                break

        return [self._to_dict(a) for a in picks if a in self.products]

    def _explore_exploit(
        self,
        user_emb    : np.ndarray,
        interactions: list[dict],
        shown_asins : set[str],
    ) -> list[dict]:
        """
        Fill TOP_K recommendation slots via ε-greedy explore-exploit.

        EXPLOIT slot: next most similar product to user_emb (cosine sim)
        EXPLORE slot: random product from a cluster the user has never liked

        α switches from 0.3 → 0.1 once the user has LIKE_THRESH likes.
        """
        liked_asins = {
            r["asin"] for r in interactions if r["action"] == "liked"
        }
        liked_cluster_ids = {
            str(self.products[a]["cluster_id"])
            for a in liked_asins if a in self.products
        }

        alpha        = ALPHA_LOW if len(liked_asins) < LIKE_THRESH else ALPHA_HIGH
        exploit_pool = self._cosine_similarity(
            user_emb,
            exclude_asins = shown_asins,
            top_k         = TOP_K * 3,   # fetch extra buffer to survive explore picks
        )
        exploit_idx  = 0
        result       = []

        for _ in range(TOP_K):
            if random.random() <= alpha:
                # EXPLORE — random product from an unseen cluster
                asin = self._explore(liked_cluster_ids)
            else:
                # EXPLOIT — next best by cosine similarity
                if exploit_idx < len(exploit_pool):
                    asin = exploit_pool[exploit_idx]
                    exploit_idx += 1
                else:
                    asin = self._explore(liked_cluster_ids)

            if asin and asin not in result:
                result.append(asin)

        return [self._to_dict(a) for a in result if a in self.products]

    def _explore(self, liked_cluster_ids: set) -> str:
        """
        Pick a random product from a cluster the user has never liked.
        Falls back to any cluster if all have been liked.
        """
        all_ids    = set(self.clusters.keys())
        unseen_ids = all_ids - liked_cluster_ids
        target     = random.choice(
            list(unseen_ids) if unseen_ids else list(all_ids)
        )
        return random.choice(self.clusters[target])

    def _to_dict(self, asin: str) -> dict:
        """Return product metadata without the embedding."""
        p = self.products[asin]
        return {
            "asin"      : asin,
            "title"     : p["title"],
            "brand"     : p["brand"],
            "price"     : p["price"],
            "image_url" : p["image_url"],
            "cluster_id": p["cluster_id"],
        }