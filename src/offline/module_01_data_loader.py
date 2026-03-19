"""
MODULE 1 — Data Loader & Inspector
====================================
Purpose : Load the raw CSV, match each row with its image file,
          clean up the data, and produce a single clean DataFrame
          that every downstream module will consume.

Input   : data/16k_apparel_data_preprocessed.csv
          data/16k_images/<ASIN>.jpeg

Output  : data/cleaned_products.csv   (saved to disk)
          Returns a pandas DataFrame  (used in-memory by next modules)

Run on  : Google Colab (no GPU needed for this step)
"""

import os
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────

PICKLE_PATH     = "datasets/pickels/16k_apperal_data_preprocessed"   # no file extension
IMAGES_DIR      = "datasets/16k_images"
OUTPUT_CSV_PATH = "datasets/cleaned_products.csv"

# Columns we actually need — drop everything else
REQUIRED_COLUMNS = ["asin", "brand", "color", "product_type_name",
                    "title", "formatted_price", "medium_image_url"]


# ── Step 1 : Load CSV ──────────────────────────────────────────────────────────

def load_pickle(path: str) -> pd.DataFrame:
    """
    Load the raw pickle file and keep only the columns we need.

    Why pickle and not CSV:
      The dataset is distributed as a pandas pickle — it preserves
      dtypes exactly and loads faster than CSV for large DataFrames.
      pd.read_pickle() handles it with no extra arguments.

    Note: no file extension in the path — this matches the original
    dataset filename 16k_apperal_data_preprocessed (no .pkl/.pickle).
    """
    df = pd.read_pickle(path)

    # Keep only required columns that exist in this pickle
    available = [c for c in REQUIRED_COLUMNS if c in df.columns]
    df = df[available].copy()

    print(f"[load_pickle] Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


# ── Step 2 : Clean ─────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that would break downstream modules:
      - Missing ASIN          → can't identify the product
      - Missing title         → can't build text embedding
      - Duplicate ASIN        → keep the first occurrence
    """
    before = len(df)

    df = df.dropna(subset=["asin", "title"])   # must have ID + text
    df = df.drop_duplicates(subset="asin")      # one row per product

    # Normalise price: strip "$" and cast to float; fill missing with 0.0
    if "formatted_price" in df.columns:
        df["formatted_price"] = (
            df["formatted_price"]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)  # remove $ and commas
            .replace("", "0")
            .astype(float)
        )

    # Lowercase category for consistency
    if "product_type_name" in df.columns:
        df["product_type_name"] = df["product_type_name"].str.lower().str.strip()

    print(f"[clean] Dropped {before - len(df):,} rows → {len(df):,} remain")
    return df.reset_index(drop=True)


# ── Step 3 : Match images ──────────────────────────────────────────────────────

def match_images(df: pd.DataFrame, images_dir: str) -> pd.DataFrame:
    """
    For each ASIN, check whether a local image file exists.
    Adds column:
      image_path  — absolute path to the JPEG (or None if missing)

    Products without a local image are kept but flagged — we can still
    use the Amazon CDN URL (medium_image_url) for the frontend display.
    """
    def find_image(asin: str) -> str | None:
        # Images may be stored as ASIN.jpeg or ASIN.jpg
        for ext in ("jpeg", "jpg"):
            path = os.path.join(images_dir, f"{asin}.{ext}")
            if os.path.exists(path):
                return path
        return None

    df["image_path"] = df["asin"].apply(find_image)

    found    = df["image_path"].notna().sum()
    missing  = df["image_path"].isna().sum()
    print(f"[match_images] {found:,} images found | {missing:,} will use CDN URL")
    return df


# ── Step 4 : Summarise ─────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame) -> None:
    """Print a quick data profile so we can sanity-check before embeddings."""
    print("\n── Data Summary ──────────────────────────────────────")
    print(f"  Total products   : {len(df):,}")
    print(f"  Unique categories: {df['product_type_name'].nunique()}")
    print(f"  Unique brands    : {df['brand'].nunique()}")
    print(f"  Price range      : ${df['formatted_price'].min():.2f}"
          f" – ${df['formatted_price'].max():.2f}")
    print(f"  Avg title length : {df['title'].str.len().mean():.0f} chars")
    print(f"  Images available : {df['image_path'].notna().sum():,} / {len(df):,}")
    print("──────────────────────────────────────────────────────\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    """
    Execute all steps in order and return the cleaned DataFrame.
    Also saves to disk so other modules can reload independently.
    """
    df = load_pickle(PICKLE_PATH)
    df = clean(df)
    df = match_images(df, IMAGES_DIR)
    summarise(df)

    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"[run] Saved cleaned data → {OUTPUT_CSV_PATH}")
    return df


if __name__ == "__main__":
    run()
