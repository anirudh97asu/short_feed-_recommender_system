"""
run_all.py
===========
Runs all 4 offline modules in sequence on your local machine.

Project structure expected:
  apparel_feed_recommendation_system/
  └── src/
      ├── offline/
      │   ├── run_all.py          ← this file
      │   ├── module_01_data_loader.py
      │   ├── module_02_clip_embeddings.py
      │   ├── module_03_clustering.py
      │   └── module_04_export.py
      └── backend/
          └── data/               ← output JSON lands here

Usage (from project root):
  python -m src.offline.run_all

Or run individual modules:
  python -m src.offline.module_01_data_loader
  python -m src.offline.module_02_clip_embeddings
  python -m src.offline.module_03_clustering
  python -m src.offline.module_04_export
"""

import os
import sys

# ── Configure paths ────────────────────────────────────────────────────────────
# All paths are relative to the project root.
# Change DATA_ROOT if your raw data lives somewhere else.

# Project root = two levels up from this file (src/offline/run_all.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT    = os.path.join(PROJECT_ROOT, "src", "offline", "data")

PICKLE_NAME  = "16k_apperal_data_preprocessed"   # no extension
IMAGES_DIR   = "16k_images"

# Output JSON goes directly into backend/data so Docker can mount it
OUTPUT_JSON  = os.path.join(PROJECT_ROOT, "src", "backend", "data",
                            "products_with_embeddings.json")

# ── Make sure the data directories exist ──────────────────────────────────────
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

# ── Import modules ─────────────────────────────────────────────────────────────
# Add src/offline/ to path so the modules can import each other cleanly
sys.path.insert(0, os.path.dirname(__file__))

import module_01_data_loader    as m1
import module_02_clip_embeddings as m2
import module_03_clustering     as m3
import module_04_export         as m4

# ── Patch paths in each module ─────────────────────────────────────────────────

m1.PICKLE_PATH     = os.path.join(DATA_ROOT, PICKLE_NAME)
m1.IMAGES_DIR      = os.path.join(DATA_ROOT, IMAGES_DIR)
m1.OUTPUT_CSV_PATH = os.path.join(DATA_ROOT, "cleaned_products.csv")

m2.CLEANED_CSV     = m1.OUTPUT_CSV_PATH
m2.EMBEDDINGS_PATH = os.path.join(DATA_ROOT, "embeddings.npy")
m2.INDEX_PATH      = os.path.join(DATA_ROOT, "embedding_index.json")

m3.EMBEDDINGS_PATH      = m2.EMBEDDINGS_PATH
m3.INDEX_PATH           = m2.INDEX_PATH
m3.CLEANED_CSV          = m1.OUTPUT_CSV_PATH
m3.CLUSTERED_CSV        = os.path.join(DATA_ROOT, "clustered_products.csv")
m3.CLUSTER_SUMMARY_PATH = os.path.join(DATA_ROOT, "cluster_summary.json")

m4.CLUSTERED_CSV   = m3.CLUSTERED_CSV
m4.EMBEDDINGS_PATH = m2.EMBEDDINGS_PATH
m4.INDEX_PATH      = m2.INDEX_PATH
m4.OUTPUT_PATH     = OUTPUT_JSON


# ── Pipeline ───────────────────────────────────────────────────────────────────

def main():
    print("\nProject root :", PROJECT_ROOT)
    print("Data root    :", DATA_ROOT)
    print("Output JSON  :", OUTPUT_JSON)
    print()

    # ── Step 1 ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1/4 — Data Loading & Cleaning")
    print("=" * 60)
    print(f"  Pickle : {m1.PICKLE_PATH}")
    print(f"  Images : {m1.IMAGES_DIR}")

    if not os.path.exists(m1.PICKLE_PATH):
        print(f"\n  ❌ Pickle file not found: {m1.PICKLE_PATH}")
        print(f"  Place your dataset at that path and re-run.")
        sys.exit(1)

    m1.run()

    # ── Step 2 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2/4 — CLIP Embeddings  (~2 min with GPU, ~30 min CPU)")
    print("=" * 60)
    m2.run()

    # ── Step 3 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3/4 — Hierarchical Clustering")
    print("=" * 60)
    print("  After this step, open data/dendrogram.png.")
    print("  If cluster count looks wrong, set DISTANCE_THRESHOLD in")
    print("  module_03_clustering.py and re-run just this step.")
    m3.run()

    # ── Step 4 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4/4 — Export Final JSON")
    print("=" * 60)
    m4.run()

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✓ Pipeline complete!")
    print(f"  Output : {OUTPUT_JSON}")
    print()
    print("  Next steps:")
    print("  1. docker compose up --build")
    print("  2. Open http://localhost:7860")
    print("=" * 60)


if __name__ == "__main__":
    main()