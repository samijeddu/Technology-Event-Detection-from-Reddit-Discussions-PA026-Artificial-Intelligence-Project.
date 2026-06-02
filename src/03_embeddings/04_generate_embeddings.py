from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


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

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128
TEXT_COLUMN = "text_for_embedding"


def generate_embeddings(cfg: PipelineConfig, sample_size: int | None = None) -> None:
    """
    Generate sentence embeddings from the cleaned dataset for one configured run.

    The cleaned dataframe is preserved as-is for metadata except for the
    embedding text column, which is excluded to keep the metadata artifact light.
    """
    if not cfg.cleaned_parquet_path.exists():
        raise FileNotFoundError(
            f"Cleaned parquet not found: {cfg.cleaned_parquet_path}"
        )

    print("Loading cleaned dataset:")
    print(cfg.cleaned_parquet_path)

    df = pd.read_parquet(cfg.cleaned_parquet_path)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Missing required column '{TEXT_COLUMN}' in {cfg.cleaned_parquet_path}"
        )

    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be a positive integer or None.")
        df = df.head(sample_size).copy()
        print(f"Smoke test mode: using first {sample_size} rows")
    else:
        print("Full run mode: using the complete dataset.")

    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()

    print("Rows:", len(df))
    print("Loading embedding model:", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    cfg.run_processed_dir.mkdir(parents=True, exist_ok=True)

    np.save(cfg.embeddings_path, embeddings)

    metadata = df.drop(columns=[TEXT_COLUMN])
    metadata.to_parquet(cfg.metadata_path, index=False)

    print("\nSaved embeddings:")
    print(cfg.embeddings_path)
    print("Embeddings shape:", embeddings.shape)

    print("\nSaved metadata:")
    print(cfg.metadata_path)
    print("Metadata shape:", metadata.shape)


def main(cfg: PipelineConfig | None = None) -> None:
    generate_embeddings(cfg or DEFAULT_CONFIG, sample_size=None)


if __name__ == "__main__":
    main()
