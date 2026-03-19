"""
MODULE 2 — CLIP Embedding Pipeline
=====================================
Purpose : Generate a single 512-dim embedding per product by fusing
          its image and title through OpenAI's CLIP model.

Why CLIP : It encodes BOTH images and text into the same vector space,
           so image_embedding and text_embedding are directly comparable.
           Averaging them gives a richer signal than either alone.

Input   : data/cleaned_products.csv  (output of Module 1)
Output  : data/embeddings.npy        — float32 array, shape (N, 512)
          data/embedding_index.json  — maps row index → asin

Run on  : Google Colab with T4 GPU (~2 min for 16k products)
"""

import json
import numpy as np
import pandas as pd
import torch
import requests

from io import BytesIO
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


# ── Constants ──────────────────────────────────────────────────────────────────

CLEANED_CSV       = "datasets/cleaned_products.csv"
EMBEDDINGS_PATH   = "offline_calculations/embeddings/embeddings.npy"
INDEX_PATH        = "offline_calculations/embeddings/embedding_index.json"

CLIP_MODEL_NAME   = "openai/clip-vit-base-patch32"   # 512-dim output
BATCH_SIZE        = 128    # adjust down if GPU runs out of memory
IMAGE_SIZE        = (224, 224)


# ── Model loader ───────────────────────────────────────────────────────────────

def load_clip(device: str):
    """Load CLIP model and processor once, reuse for all batches."""
    print(f"[load_clip] Loading {CLIP_MODEL_NAME} on {device} ...")
    model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()   # inference mode — disables dropout
    print("[load_clip] Model ready.\n")
    return model, processor


# ── Image helpers ──────────────────────────────────────────────────────────────

def load_image(row: pd.Series) -> Image.Image | None:
    """
    Try to load image in this order:
      1. Local file (image_path column)  — fastest, no network
      2. Amazon CDN URL (medium_image_url) — fallback if local missing
      3. Return None if both fail       — will use text embedding only
    """
    # Try local file first
    if pd.notna(row.get("image_path")):
        try:
            return Image.open(row["image_path"]).convert("RGB")
        except Exception:
            pass

    # Fall back to CDN URL
    if pd.notna(row.get("medium_image_url")):
        try:
            response = requests.get(row["medium_image_url"], timeout=5)
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            pass

    return None   # image unavailable — text-only embedding will be used


# ── Embedding functions ────────────────────────────────────────────────────────

@torch.no_grad()
def embed_images(images: list[Image.Image], model, processor, device: str) -> np.ndarray:
    """
    Encode a batch of PIL images → normalised float32 array (batch, 512).
    @torch.no_grad() disables gradient tracking → faster, less memory.
    """
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    feats  = model.get_image_features(**inputs)
    feats  = feats / feats.norm(dim=-1, keepdim=True)   # L2 normalise
    return feats.cpu().numpy().astype(np.float32)


@torch.no_grad()
def embed_texts(titles: list[str], model, processor, device: str) -> np.ndarray:
    """
    Encode a batch of product titles → normalised float32 array (batch, 512).
    """
    inputs = processor(text=titles, return_tensors="pt",
                       padding=True, truncation=True, max_length=77).to(device)
    feats  = model.get_text_features(**inputs)
    feats  = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


# ── Core pipeline ──────────────────────────────────────────────────────────────

def generate_embeddings(df: pd.DataFrame, model, processor, device: str) -> np.ndarray:
    """
    For every product:
      - Encode image  (if available)
      - Encode title  (always available)
      - Final = mean(image_emb, text_emb)   or text_emb alone if no image

    Returns float32 array of shape (len(df), 512).
    """
    all_embeddings = np.zeros((len(df), 512), dtype=np.float32)

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Embedding batches"):
        batch = df.iloc[start : start + BATCH_SIZE]

        # ── Text embeddings (always computed) ────────────────────────────────
        titles      = batch["title"].fillna("").tolist()
        text_embs   = embed_texts(titles, model, processor, device)

        # ── Image embeddings (best-effort) ────────────────────────────────────
        images      = [load_image(row) for _, row in batch.iterrows()]
        has_image   = [img is not None for img in images]

        if any(has_image):
            # Only run image encoder on rows that actually have an image
            valid_images  = [img for img in images if img is not None]
            valid_indices = [i for i, ok in enumerate(has_image) if ok]
            image_embs    = embed_images(valid_images, model, processor, device)

            # Fuse: average image + text for rows with an image
            fused = text_embs.copy()
            for local_i, global_i in enumerate(valid_indices):
                fused[global_i] = (image_embs[local_i] + text_embs[global_i]) / 2.0
                # Re-normalise the fused vector
                norm = np.linalg.norm(fused[global_i])
                if norm > 0:
                    fused[global_i] /= norm
        else:
            fused = text_embs   # no images in this batch → text only

        all_embeddings[start : start + len(batch)] = fused

    return all_embeddings


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> tuple[np.ndarray, list[str]]:
    """
    Load data, generate all embeddings, save to disk.
    Returns (embeddings array, list of ASINs in row order).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run] Using device: {device}\n")

    df = pd.read_csv(CLEANED_CSV)
    print(f"[run] Loaded {len(df):,} products\n")

    model, processor = load_clip(device)
    embeddings       = generate_embeddings(df, model, processor, device)

    # Save embeddings array
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"\n[run] Saved embeddings → {EMBEDDINGS_PATH}  shape={embeddings.shape}")

    # Save ASIN index so we can map row number ↔ ASIN later
    asin_list = df["asin"].tolist()
    with open(INDEX_PATH, "w") as f:
        json.dump(asin_list, f)
    print(f"[run] Saved ASIN index → {INDEX_PATH}")

    return embeddings, asin_list


if __name__ == "__main__":
    run()
