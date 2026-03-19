"""
MODULE 4 — Export Final Products JSON
========================================
Purpose : Merge product metadata + CLIP embeddings + cluster IDs into
          one self-contained JSON file that the FastAPI backend will
          load into memory at startup.

Why a flat JSON and not a database :
  - 16k products × 512 floats ≈ 32 MB — fits comfortably in RAM
  - The FastAPI server loads it ONCE at startup; all subsequent
    recommendation calls are pure in-memory numpy operations (< 5 ms)
  - No network round-trip to a vector DB on every request
  - Works on any hosting platform with zero extra infrastructure
  - Committed to the GitHub repo → the backend always has it available

Input   : data/clustered_products.csv  (from Module 3)
          data/embeddings.npy           (from Module 2)
          data/embedding_index.json     (ASIN list, from Module 2)

Output  : backend/data/products_with_embeddings.json
          (also printed: file size and a sample record)

Run on  : Google Colab or local machine (no GPU needed)
"""

import json
import os
import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────

CLUSTERED_CSV    = "datasets/clustered_products.csv"
EMBEDDINGS_PATH  = "offline_calculations/embeddings/embeddings.npy"
INDEX_PATH       = "offline_calculations/embeddings/embedding_index.json"
OUTPUT_PATH      = "offline_calculations/products_with_embeddings.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

def merge(df: pd.DataFrame,
          embeddings: np.ndarray,
          asin_list: list[str]) -> list[dict]:
    """
    Combine each product row with its 512-dim embedding into one dict.

    The embedding is stored as a plain Python list of floats so it can
    be serialised to JSON. The backend will cast it back to np.array
    on load.
    """
    # Build a quick lookup: asin → row index in embeddings array
    asin_to_idx = {asin: i for i, asin in enumerate(asin_list)}

    records = []
    missing = 0

    for _, row in df.iterrows():
        asin = row["asin"]

        if asin not in asin_to_idx:
            missing += 1
            continue

        emb = embeddings[asin_to_idx[asin]]

        records.append({
            # ── Identifiers ──────────────────────────────────────────────
            "asin"        : asin,
            # ── Display metadata (used by Gradio frontend) ────────────────
            "title"       : str(row.get("title",        "")),
            "brand"       : str(row.get("brand",        "")),
            "color"       : str(row.get("color",        "")),
            "category"    : str(row.get("product_type_name", "")),
            "price"       : float(row.get("formatted_price", 0.0)),
            "image_url"   : str(row.get("medium_image_url",  "")),
            # ── Recommendation signals ────────────────────────────────────
            "cluster_id"  : int(row.get("cluster_id", -1)),
            "embedding"   : emb.tolist(),   # list[float], len=512
        })

    if missing:
        print(f"[merge] Warning: {missing} products had no embedding — skipped")

    return records


def save(records: list[dict], path: str) -> None:
    """Write records to JSON, creating the output directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f)


def report(records: list[dict], path: str) -> None:
    """Print a quick sanity-check summary."""
    size_mb = os.path.getsize(path) / (1024 * 1024)
    sample  = records[0]

    print("\n── Export Summary ────────────────────────────────────")
    print(f"  Records written : {len(records):,}")
    print(f"  File size       : {size_mb:.1f} MB")
    print(f"  Output path     : {path}")
    print(f"\n  Sample record keys : {list(sample.keys())}")
    print(f"  Sample ASIN        : {sample['asin']}")
    print(f"  Sample title       : {sample['title'][:60]}")
    print(f"  Embedding length   : {len(sample['embedding'])}")
    print(f"  Cluster ID         : {sample['cluster_id']}")
    print("──────────────────────────────────────────────────────\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> list[dict]:
    """Load all inputs, merge, save, and report."""

    print("[run] Loading inputs ...")
    df         = pd.read_csv(CLUSTERED_CSV)
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    with open(INDEX_PATH) as f:
        asin_list = json.load(f)

    print(f"[run] Products: {len(df):,}  |  Embeddings: {embeddings.shape}")

    records = merge(df, embeddings, asin_list)
    save(records, OUTPUT_PATH)
    report(records, OUTPUT_PATH)

    print(f"[run] ✓ products_with_embeddings.json is ready.")
    print(f"[run]   Commit it to your GitHub repo and deploy.")

    return records


if __name__ == "__main__":
    run()
