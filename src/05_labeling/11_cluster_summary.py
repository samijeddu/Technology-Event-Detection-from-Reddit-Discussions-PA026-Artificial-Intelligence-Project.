from __future__ import annotations

from collections import Counter
from pathlib import Path
import importlib.util
import re
import sys

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

TOP_KEYWORDS = 12
SAMPLE_TITLES = 8

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "you", "your", "are",
    "was", "were", "have", "has", "had", "but", "not", "can", "could", "would",
    "should", "will", "just", "like", "about", "into", "than", "then", "they",
    "them", "their", "what", "when", "where", "why", "how", "all", "any",
    "new", "use", "using", "get", "got", "one", "out", "more", "some",
    "really", "does", "did", "been", "because", "after", "before",
    "chatgpt", "openai", "model",
    "https", "http", "com", "www", "removed", "deleted", "now", "there",
    "here", "want", "people", "time", "who", "only", "make", "need",
}


def select_text_column(df: pd.DataFrame) -> str:
    for column in ["text_for_embedding", "full_text", "title"]:
        if column in df.columns:
            return column
    raise ValueError("No text column found. Expected text_for_embedding, full_text or title.")


def get_top_keywords(texts: pd.Series, top_k: int = TOP_KEYWORDS) -> str:
    text = " ".join(texts.fillna("").astype(str).tolist()).lower()
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", text)
    words = [
        word
        for word in words
        if (
            word not in STOPWORDS
            and len(word) >= 3
            and not word.isnumeric()
            and "http" not in word
            and "www" not in word
            and ".com" not in word
        )
    ]
    counts = Counter(words)
    return ", ".join([f"{word}:{count}" for word, count in counts.most_common(top_k)])


def get_sample_titles(group: pd.DataFrame, n: int = SAMPLE_TITLES) -> str:
    if "cluster_probability" in group.columns:
        group = group.sort_values("cluster_probability", ascending=False)

    titles = (
        group["title"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )
    titles = [title for title in titles if title]
    return " || ".join(titles[:n])


def build_cluster_summary(cfg: PipelineConfig) -> None:
    """Build one row of summary statistics for each non-noise cluster."""
    if not cfg.posts_with_clusters_path.exists():
        raise FileNotFoundError(
            f"Posts with clusters file not found: {cfg.posts_with_clusters_path}"
        )

    df = pd.read_parquet(cfg.posts_with_clusters_path)

    required_cols = {"cluster_id", "date", "subreddit", "title"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    text_column = select_text_column(df)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.floor("D")

    valid_df = df[df["cluster_id"] != -1].copy()
    summary_rows = []

    for cluster_id, group in valid_df.groupby("cluster_id"):
        n_posts = len(group)

        top_subreddits = (
            group["subreddit"]
            .value_counts()
            .head(5)
            .to_dict()
        )

        daily_counts = group.groupby("date").size().sort_values(ascending=False)
        peak_date = daily_counts.index[0] if len(daily_counts) > 0 else None
        peak_count = int(daily_counts.iloc[0]) if len(daily_counts) > 0 else 0
        peak_ratio = peak_count / n_posts if n_posts > 0 else 0.0

        summary_rows.append({
            "cluster_id": cluster_id,
            "n_posts": n_posts,
            "date_start": group["date"].min(),
            "date_end": group["date"].max(),
            "peak_date": peak_date,
            "peak_count": peak_count,
            "peak_ratio": peak_ratio,
            "top_subreddits": str(top_subreddits),
            "top_keywords": get_top_keywords(group[text_column]),
            "sample_titles": get_sample_titles(group),
        })

    cluster_summary = pd.DataFrame(summary_rows)

    if not cluster_summary.empty:
        cluster_summary = (
            cluster_summary
            .sort_values("n_posts", ascending=False)
            .reset_index(drop=True)
        )

    cfg.run_processed_dir.mkdir(parents=True, exist_ok=True)
    cluster_summary.to_csv(cfg.cluster_summary_path, index=False)

    print("Clusters processed:", len(cluster_summary))
    print("Saved cluster summary:")
    print(cfg.cluster_summary_path)
    print("\nTop 10 clusters by n_posts:")
    print(cluster_summary.head(10).to_string(index=False))


def main(cfg: PipelineConfig | None = None) -> None:
    build_cluster_summary(cfg or DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
