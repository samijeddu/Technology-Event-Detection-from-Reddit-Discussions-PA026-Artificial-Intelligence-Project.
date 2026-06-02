from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import time

import hdbscan
import numpy as np
import pandas as pd


# ==================================================
# CONFIG
# ==================================================

CONFIG_PATH = Path(__file__).resolve().parents[1] / "00_config" / "config.py"
_CONFIG_SPEC = importlib.util.spec_from_file_location("pipeline_config", CONFIG_PATH)
if _CONFIG_SPEC is None or _CONFIG_SPEC.loader is None:
    raise ImportError(f"Cannot load PipelineConfig from {CONFIG_PATH}")
_CONFIG_MODULE = importlib.util.module_from_spec(_CONFIG_SPEC)
sys.modules[_CONFIG_SPEC.name] = _CONFIG_MODULE
_CONFIG_SPEC.loader.exec_module(_CONFIG_MODULE)
PipelineConfig = _CONFIG_MODULE.PipelineConfig

DEFAULT_CONFIG = PipelineConfig(
    start_month="2025-11",
    end_month="2026-01",
    output_run_name="nov_2025_jan_2026",
)

MIN_CLUSTER_SIZE = 100
MIN_SAMPLES = 15
METRIC = "euclidean"
CLUSTER_SELECTION_METHOD = "eom"
PREDICTION_DATA = True


def run_hdbscan(cfg: PipelineConfig) -> None:
    """Cluster UMAP embeddings with HDBSCAN for the configured run."""
    if not cfg.umap_embeddings_path.exists():
        raise FileNotFoundError(
            f"UMAP embeddings file not found: {cfg.umap_embeddings_path}"
        )

    if not cfg.metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {cfg.metadata_path}")

    umap_embeddings = np.load(cfg.umap_embeddings_path)
    metadata = pd.read_parquet(cfg.metadata_path)

    if umap_embeddings.ndim != 2:
        raise ValueError(
            f"UMAP embeddings must be a 2D array, got shape {umap_embeddings.shape}"
        )

    if umap_embeddings.shape[0] != metadata.shape[0]:
        raise ValueError(
            "Mismatch: UMAP rows do not match metadata rows "
            f"({umap_embeddings.shape[0]} != {metadata.shape[0]})"
        )

    print("UMAP embeddings shape:", umap_embeddings.shape)
    print("Metadata shape:", metadata.shape)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric=METRIC,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        prediction_data=PREDICTION_DATA,
    )

    start = time.time()
    cluster_labels = clusterer.fit_predict(umap_embeddings)
    elapsed = time.time() - start

    metadata = metadata.copy()
    metadata["cluster_id"] = cluster_labels
    metadata["cluster_probability"] = clusterer.probabilities_

    n_total = len(metadata)
    n_noise = int((metadata["cluster_id"] == -1).sum())
    n_clusters = metadata.loc[
        metadata["cluster_id"] != -1,
        "cluster_id",
    ].nunique()
    noise_ratio = n_noise / n_total if n_total else 0.0

    cfg.run_processed_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(cfg.posts_with_clusters_path, index=False)
    np.save(cfg.cluster_labels_path, cluster_labels)

    print("\nHDBSCAN finished")
    print("Time:", round(elapsed / 60, 2), "minutes")
    print("Number of clusters:", n_clusters)
    print("Noise points:", n_noise)
    print("Noise ratio:", round(noise_ratio, 4))
    print("\nSaved posts with clusters:")
    print(cfg.posts_with_clusters_path)
    print("Saved cluster labels:")
    print(cfg.cluster_labels_path)


def main(cfg: PipelineConfig | None = None) -> None:
    run_hdbscan(cfg or DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
