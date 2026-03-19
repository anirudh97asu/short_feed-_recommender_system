"""
COLAB RUNNER — run_all.py
===========================
Purpose : Single script that runs all 4 offline modules in sequence.
          Paste this into a Google Colab cell (or run as a .py script).

Before running:
  1. Upload your CSV and images folder to Google Drive
  2. Mount Drive (cell below)
  3. Update DATA_ROOT to point at your folder
  4. Set DATABASE_URL if you want to verify DB connectivity

Cell 0 — Mount Drive + install deps
─────────────────────────────────────
  from google.colab import drive
  drive.mount('/content/drive')
  !pip install -r offline/requirements.txt -q

Cell 1 — Run this file
─────────────────────────────────────
  !python offline/run_all.py
"""

import os
import sys

# ── Point these at your actual paths ──────────────────────────────────────────
DATA_ROOT   = "/home/anirudh97/Data/apparel_feed_recommendation_system/datasets"   # ← change this
PICKLE_NAME = "pickels/16k_apperal_data_preprocessed"             # no extension
IMAGES_DIR  = "16k_images"

# Override module-level constants before importing each module
os.environ["DATA_ROOT"] = DATA_ROOT

# Add offline/ to path so modules can import each other
sys.path.insert(0, os.path.dirname(__file__))

# Patch data paths dynamically
import module_01_data_loader   as m1
import module_02_clip_embeddings as m2
import module_03_clustering    as m3
import module_04_export        as m4

m1.PICKLE_PATH     = os.path.join(DATA_ROOT, PICKLE_NAME)
m1.IMAGES_DIR      = os.path.join(DATA_ROOT, IMAGES_DIR)
m1.OUTPUT_CSV_PATH = os.path.join(DATA_ROOT, "cleaned_products.csv")

m2.CLEANED_CSV      = m1.OUTPUT_CSV_PATH
m2.EMBEDDINGS_PATH  = os.path.join(DATA_ROOT, "embeddings.npy")
m2.INDEX_PATH       = os.path.join(DATA_ROOT, "embedding_index.json")

m3.EMBEDDINGS_PATH  = m2.EMBEDDINGS_PATH
m3.INDEX_PATH       = m2.INDEX_PATH
m3.CLEANED_CSV      = m1.OUTPUT_CSV_PATH
m3.CLUSTERED_CSV    = os.path.join(DATA_ROOT, "clustered_products.csv")
m3.CLUSTER_SUMMARY_PATH = os.path.join(DATA_ROOT, "cluster_summary.json")

m4.CLUSTERED_CSV    = m3.CLUSTERED_CSV
m4.EMBEDDINGS_PATH  = m2.EMBEDDINGS_PATH
m4.INDEX_PATH       = m2.INDEX_PATH
m4.OUTPUT_PATH      = os.path.join(DATA_ROOT, "..", "src/backend/data/products_with_embeddings.json")


# ── Run pipeline ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 1/4 — Data Loading & Cleaning")
    print("=" * 60)
    m1.run()

    print("\n" + "=" * 60)
    print("STEP 2/4 — CLIP Embeddings  (needs GPU, ~2 min)")
    print("=" * 60)
    m2.run()

    print("\n" + "=" * 60)
    print("STEP 3/4 — Clustering")
    print("=" * 60)
    m3.run()

    print("\n" + "=" * 60)
    print("STEP 4/4 — Export Final JSON")
    print("=" * 60)
    m4.run()

    print("\n" + "=" * 60)
    print("✓ Pipeline complete!")
    print("  Commit backend/data/products_with_embeddings.json to GitHub.")
    print("  Then deploy backend/ to Render and frontend/ to HF Spaces.")
    print("=" * 60)


if __name__ == "__main__":
    main()
