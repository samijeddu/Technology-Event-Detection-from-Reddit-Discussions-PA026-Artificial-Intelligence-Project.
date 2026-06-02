from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import umap


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

N_NEIGHBORS = 10
N_COMPONENTS = 15
MIN_DIST = 0.03
METRIC = "cosine"
RANDOM_STATE = 42


def run_umap(cfg: PipelineConfig) -> None:
    """Reduce sentence embeddings with UMAP for the configured run."""
    if not cfg.embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {cfg.embeddings_path}")

    embeddings = np.load(cfg.embeddings_path, mmap_mode="r")

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be a 2D array, got shape {embeddings.shape}"
        )

    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError(f"Embeddings array is empty or invalid: {embeddings.shape}")

    print("Input embeddings shape:", embeddings.shape)

    umap_model = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        n_components=N_COMPONENTS,
        min_dist=MIN_DIST,
        metric=METRIC,
        random_state=RANDOM_STATE,
    )

    umap_embeddings = umap_model.fit_transform(embeddings)
    umap_embeddings = np.asarray(umap_embeddings, dtype=np.float32)

    cfg.run_processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.umap_embeddings_path, umap_embeddings)

    print("Output UMAP shape:", umap_embeddings.shape)
    print("Saved UMAP embeddings:")
    print(cfg.umap_embeddings_path)


def main(cfg: PipelineConfig | None = None) -> None:
    run_umap(cfg or DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
