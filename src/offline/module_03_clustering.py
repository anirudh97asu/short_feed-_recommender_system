"""
MODULE 3 — Product Clustering (Hierarchical / Agglomerative)
==============================================================
Purpose : Group the 16k products into N clusters using Agglomerative
          (Hierarchical) clustering directly on CLIP embeddings.

Why Hierarchical over K-Means :
  - Does not assume spherical clusters (fashion styles aren't spherical)
  - Builds a full dendrogram — the complete merge history of all products
  - Can determine the number of clusters AUTOMATICALLY by finding where
    merge distances jump significantly (distance_threshold mode)
  - More natural groupings: "floral prints" stays together rather
    than being split by K-Means' Voronoi boundaries

How cluster count is decided — two modes:
  Mode A │ distance_threshold (DEFAULT — recommended)
         │   You set a distance value; the algorithm cuts the tree
         │   wherever merges exceed that distance.
         │   Number of clusters is an OUTPUT — naturally determined
         │   by the structure of the data itself.
         │
  Mode B │ n_clusters (manual override)
         │   You specify exactly how many clusters you want.
         │   Algorithm cuts the tree to produce that count.
         │   Number of clusters is an INPUT — you decide.

  → Use Mode A first. Read the dendrogram to pick a good threshold.
    Switch to Mode B only if you need a specific cluster count.

Memory reality check :
  16,000 × 16,000 × 4 bytes ≈ 1 GB distance matrix on CPU RAM.
  Fits in Colab's standard 12 GB runtime — no reduction needed.
  Note: sklearn runs on CPU. GPU VRAM is not relevant here.

Input   : data/embeddings.npy        (from Module 2)
          data/embedding_index.json  (ASIN list, from Module 2)
          data/cleaned_products.csv  (product metadata, from Module 1)

Output  : data/clustered_products.csv
          data/cluster_summary.json
          data/dendrogram.png         ← use this to pick distance_threshold
          data/cluster_distribution.png

Run on  : Google Colab standard runtime (CPU, ~12 GB RAM)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster         import AgglomerativeClustering
from sklearn.metrics         import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from collections             import defaultdict


# ── Constants ──────────────────────────────────────────────────────────────────

EMBEDDINGS_PATH      = "offline_calculations/embeddings/embeddings.npy"
INDEX_PATH           = "offline_calculations/embeddings/embedding_index.json"
CLEANED_CSV          = "datasets/cleaned_products.csv"
CLUSTERED_CSV        = "datasets/clustered_products.csv"
CLUSTER_SUMMARY_PATH = "datasets/cluster_summary.json"
DENDROGRAM_PATH      = "datasets/dendrogram.png"

# ── Clustering mode — pick ONE ─────────────────────────────────────────────────
#
# RECOMMENDED: Start with distance_threshold = None (auto mode).
#   1. Run the module once → inspect the dendrogram (data/dendrogram.png)
#   2. Find the distance level where merges show a big jump
#   3. Set DISTANCE_THRESHOLD to that value → re-run
#
# OVERRIDE: Set N_CLUSTERS to a specific number instead.
#   Set DISTANCE_THRESHOLD = None and N_CLUSTERS = 100 (or your choice).
#
DISTANCE_THRESHOLD   = None    # e.g. 12.5 after reading the dendrogram
N_CLUSTERS           = 100     # only used when DISTANCE_THRESHOLD is None


# ── Clustering ─────────────────────────────────────────────────────────────────

def agglomerative_cluster(embeddings: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Fit Agglomerative (hierarchical) clustering on CLIP embeddings.

    Mode A — distance_threshold (automatic):
      AgglomerativeClustering builds the FULL dendrogram when
      distance_threshold is set and n_clusters is None.
      It then cuts wherever merge distance exceeds the threshold.
      The number of resulting clusters is determined by the data.

    Mode B — n_clusters (manual):
      Cuts the dendrogram to produce exactly n_clusters groups.

    Linkage = 'ward':
      Minimises within-cluster variance at each merge step.
      Produces compact clusters — best for visual style grouping.

    Returns (labels array, n_clusters_found).
    """
    using_threshold = DISTANCE_THRESHOLD is not None

    if using_threshold:
        print(f"[cluster] Mode: distance_threshold = {DISTANCE_THRESHOLD}  "
              f"(cluster count will be decided by the data)")
        model = AgglomerativeClustering(
            n_clusters         = None,
            distance_threshold = DISTANCE_THRESHOLD,
            metric             = "euclidean",
            linkage            = "ward",
            compute_full_tree  = True,   # required for distance_threshold mode
        )
    else:
        print(f"[cluster] Mode: n_clusters = {N_CLUSTERS}  "
              f"(manually specified)")
        model = AgglomerativeClustering(
            n_clusters = N_CLUSTERS,
            metric     = "euclidean",
            linkage    = "ward",
        )

    print(f"[cluster] Input shape : {embeddings.shape}")
    print(f"[cluster] Distance matrix : "
          f"~{embeddings.shape[0]**2 * 4 / 1024**2:.0f} MB on CPU RAM\n")

    labels          = model.fit_predict(embeddings)
    n_clusters_found = len(set(labels))

    print(f"[cluster] Done.  Clusters found: {n_clusters_found}")
    if using_threshold:
        print(f"[cluster] Tip: if {n_clusters_found} clusters is too many/few, "
              f"adjust DISTANCE_THRESHOLD up/down and re-run.\n")

    return labels, n_clusters_found


# ── Dendrogram ─────────────────────────────────────────────────────────────────

def plot_dendrogram(embeddings: np.ndarray, sample_size: int = 500) -> None:
    """
    Plot a dendrogram on a random sample of products.

    Full 16k dendrogram is unreadable — 500 items shows the hierarchical
    structure clearly and computes in seconds.

    How to read it:
      - Long vertical lines = natural cluster boundaries
      - The horizontal cut level tells you how many clusters are "natural"
      - If your k=100 cuts through many long lines, it's a good k
    """
    print(f"[dendrogram] Sampling {sample_size} points ...")
    idx    = np.random.choice(len(embeddings), sample_size, replace=False)
    sample = embeddings[idx]

    Z = linkage(sample, method="ward")

    plt.figure(figsize=(16, 6))
    dendrogram(
        Z,
        truncate_mode  = "lastp",   # show only the last p merges
        p              = 30,        # top 30 merges
        leaf_rotation  = 90,
        leaf_font_size = 8,
        show_contracted= True,
    )
    plt.title(f"Agglomerative Dendrogram  (sample n={sample_size}, top 30 merges)")
    plt.xlabel("Product index / cluster size")
    plt.ylabel("Ward distance")
    plt.tight_layout()
    plt.savefig(DENDROGRAM_PATH, dpi=120)
    plt.show()
    print(f"[dendrogram] Saved → {DENDROGRAM_PATH}\n")


# ── Quality Evaluation ─────────────────────────────────────────────────────────

def evaluate(embeddings: np.ndarray, labels: np.ndarray) -> None:
    """
    Silhouette score on a random sample — full dataset is slow.
    > 0.25 = acceptable   |   > 0.40 = good
    """
    sample_n = min(2000, len(embeddings))
    idx      = np.random.choice(len(embeddings), sample_n, replace=False)
    score    = silhouette_score(embeddings[idx], labels[idx], metric="euclidean")

    unique, counts = np.unique(labels, return_counts=True)
    print(f"[evaluate] Silhouette score (n={sample_n})  : {score:.4f}")
    print(f"[evaluate] Cluster sizes — "
          f"min: {counts.min()}  max: {counts.max()}  "
          f"mean: {counts.mean():.0f}  std: {counts.std():.0f}")

    plt.figure(figsize=(14, 4))
    plt.bar(unique, counts, color="steelblue", width=0.8)
    plt.xlabel("Cluster ID")
    plt.ylabel("Product count")
    plt.title(f"Products per cluster  (k={len(unique)}, total={len(labels):,})")
    plt.tight_layout()
    plt.savefig("datasets/cluster_distribution.png", dpi=100)
    plt.show()
    print("[evaluate] Saved → datasets/cluster_distribution.png\n")


# ── Build outputs ──────────────────────────────────────────────────────────────

def attach_cluster_ids(
    df        : pd.DataFrame,
    asin_list : list,
    labels    : np.ndarray
) -> pd.DataFrame:
    """Attach cluster_id to each product row by matching ASIN to label index."""
    label_map        = {asin: int(lbl) for asin, lbl in zip(asin_list, labels)}
    df["cluster_id"] = df["asin"].map(label_map)
    return df


def build_cluster_summary(df: pd.DataFrame) -> dict:
    """Build { cluster_id_str: [asin, ...] } for cold-start and explore."""
    summary = defaultdict(list)
    for _, row in df.iterrows():
        summary[str(int(row["cluster_id"]))].append(row["asin"])

    sizes = [len(v) for v in summary.values()]
    print(f"[summary] {len(summary)} clusters | "
          f"min={min(sizes)}  max={max(sizes)}  avg={sum(sizes)//len(sizes)}")
    return dict(summary)


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    """Load → cluster → evaluate → visualise → save."""

    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    with open(INDEX_PATH) as f:
        asin_list = json.load(f)
    df = pd.read_csv(CLEANED_CSV)

    print(f"[run] Embeddings : {embeddings.shape}  |  Products : {len(df):,}\n")

    # ── Step 1: Plot dendrogram FIRST so you can pick a good threshold ────────
    # On first run, read dendrogram.png and set DISTANCE_THRESHOLD at the top
    # of this file, then re-run to get naturally-sized clusters.
    plot_dendrogram(embeddings)

    # ── Step 2: Cluster ───────────────────────────────────────────────────────
    labels, n_found = agglomerative_cluster(embeddings)
    print(f"[run] Final cluster count : {n_found}\n")

    # ── Step 3: Evaluate quality ──────────────────────────────────────────────
    evaluate(embeddings, labels)

    # ── Step 4: Attach IDs and save ───────────────────────────────────────────
    df = attach_cluster_ids(df, asin_list, labels)
    df.to_csv(CLUSTERED_CSV, index=False)
    print(f"[run] Saved → {CLUSTERED_CSV}")

    summary = build_cluster_summary(df)
    with open(CLUSTER_SUMMARY_PATH, "w") as f:
        json.dump(summary, f)
    print(f"[run] Saved → {CLUSTER_SUMMARY_PATH}")

    return df, labels


if __name__ == "__main__":
    run()
