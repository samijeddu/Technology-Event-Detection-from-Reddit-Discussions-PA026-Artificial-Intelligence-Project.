from __future__ import annotations

from pathlib import Path
import importlib.util
from itertools import combinations
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


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

TOP_N_TERMS = 5
TOP_K_DEDUP = 30
SCORE_METHOD = "S4_full_metadata"
DEDUP_JACCARD_THRESHOLD = 0.30
DEDUP_PEAK_DATE_WINDOW_DAYS = 2


def select_text_column(df: pd.DataFrame) -> str:
    for column in ["text_for_embedding", "full_text", "title"]:
        if column in df.columns:
            return column
    raise ValueError("No text column found. Expected text_for_embedding, full_text or title.")


def labeled_summary_path(cfg: PipelineConfig) -> Path:
    return cfg.run_processed_dir / "reddit_tech_cluster_summary_labeled.csv"


def event_candidates_path(cfg: PipelineConfig) -> Path:
    return cfg.run_outputs_dir / "temporal_analysis" / "event_candidates_ranked.csv"


def event_candidates_deduplicated_path(cfg: PipelineConfig) -> Path:
    return (
        cfg.run_outputs_dir
        / "temporal_analysis"
        / "event_candidates_ranked_deduplicated.csv"
    )


def event_candidates_top30_deduplicated_path(cfg: PipelineConfig) -> Path:
    return (
        cfg.run_outputs_dir
        / "temporal_analysis"
        / "event_candidates_top30_deduplicated.csv"
    )


def add_s4_features(
    posts: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> pd.DataFrame:
    required_cols = {"cluster_id"}
    missing_cols = required_cols - set(posts.columns)
    if missing_cols:
        raise ValueError(f"Missing required post columns: {sorted(missing_cols)}")

    df = posts.copy()
    if "date" not in df.columns:
        if "created_dt" not in df.columns:
            raise ValueError("Missing date column. Expected date or created_dt.")
        df["date"] = pd.to_datetime(df["created_dt"], utc=True, errors="coerce")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.floor("D")
    valid_df = df[df["cluster_id"] != -1].copy()

    if valid_df.empty:
        raise ValueError("No non-noise posts available for S4 feature calculation.")

    aggregations = {"n_posts_recomputed": ("cluster_id", "size")}
    if "score" in valid_df.columns:
        aggregations["avg_score"] = ("score", "mean")
    if "num_comments" in valid_df.columns:
        aggregations["avg_comments"] = ("num_comments", "mean")

    base_stats = valid_df.groupby("cluster_id").agg(**aggregations).reset_index()
    if "avg_score" not in base_stats.columns:
        base_stats["avg_score"] = 0.0
    if "avg_comments" not in base_stats.columns:
        base_stats["avg_comments"] = 0.0

    daily_counts = (
        valid_df
        .groupby(["cluster_id", "date"])
        .size()
        .reset_index(name="daily_posts")
    )
    daily_stats = (
        daily_counts
        .groupby("cluster_id")
        .agg(
            mean_daily_posts=("daily_posts", "mean"),
            std_daily_posts=("daily_posts", "std"),
            active_days=("date", "count"),
        )
        .reset_index()
    )

    recomputed_cols = [
        "avg_score",
        "avg_comments",
        "engagement",
        "burst_share",
        "peak_ratio",
        "peak_prominence",
        "mean_daily_posts",
        "std_daily_posts",
        "active_days",
        "S1_baseline",
        "S4_full_metadata",
        "event_candidate_score",
        "score_method",
    ]
    summary_base = cluster_summary.drop(columns=recomputed_cols, errors="ignore")

    enriched = (
        summary_base
        .merge(base_stats, on="cluster_id", how="left")
        .merge(daily_stats, on="cluster_id", how="left")
    )

    enriched["n_posts"] = pd.to_numeric(enriched["n_posts"], errors="coerce").fillna(
        enriched["n_posts_recomputed"]
    )
    enriched["peak_count"] = pd.to_numeric(
        enriched["peak_count"],
        errors="coerce",
    ).fillna(0)
    enriched["avg_score"] = pd.to_numeric(
        enriched["avg_score"],
        errors="coerce",
    ).fillna(0).clip(lower=0)
    enriched["avg_comments"] = pd.to_numeric(
        enriched["avg_comments"],
        errors="coerce",
    ).fillna(0).clip(lower=0)
    enriched["mean_daily_posts"] = pd.to_numeric(
        enriched["mean_daily_posts"],
        errors="coerce",
    ).fillna(0)
    enriched["std_daily_posts"] = pd.to_numeric(
        enriched["std_daily_posts"],
        errors="coerce",
    ).fillna(0)
    enriched["active_days"] = pd.to_numeric(
        enriched["active_days"],
        errors="coerce",
    ).fillna(0).astype(int)

    enriched["burst_share"] = np.where(
        enriched["n_posts"] > 0,
        enriched["peak_count"] / enriched["n_posts"],
        0.0,
    )
    enriched["peak_ratio"] = enriched["burst_share"]
    enriched["peak_prominence"] = np.where(
        enriched["mean_daily_posts"] > 0,
        enriched["peak_count"] / enriched["mean_daily_posts"],
        0.0,
    )
    enriched["engagement"] = enriched["avg_score"] + enriched["avg_comments"]
    enriched["S1_baseline"] = (
        enriched["peak_count"] * np.log1p(enriched["n_posts"])
    )
    enriched["S4_full_metadata"] = (
        enriched["peak_count"]
        * np.log1p(enriched["n_posts"])
        * enriched["burst_share"]
        * np.log1p(enriched["engagement"])
    )
    enriched["eventness_score"] = enriched["burst_share"]
    enriched["importance_score"] = enriched["peak_count"]
    enriched["event_candidate_score"] = enriched["S4_full_metadata"]
    enriched["score_method"] = SCORE_METHOD

    return enriched.drop(columns=["n_posts_recomputed"], errors="ignore")


def normalize_event_terms(row: pd.Series) -> set[str]:
    text = " ".join(
        str(row.get(column, "") or "")
        for column in ["interpreted_label_ctfidf", "ctfidf_top_terms"]
    ).lower()
    text = re.sub(r"[^a-z0-9\s|,]", " ", text)
    terms = re.split(r"[|,]", text)
    return {re.sub(r"\s+", " ", term).strip() for term in terms if term.strip()}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def deduplicate_event_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    ranked = candidates.copy().reset_index(drop=True)
    ranked["peak_date"] = pd.to_datetime(
        ranked["peak_date"],
        utc=True,
        errors="coerce",
    )
    terms = [normalize_event_terms(row) for _, row in ranked.iterrows()]
    parent = list(range(len(ranked)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in combinations(range(len(ranked)), 2):
        left_date = ranked.at[left, "peak_date"]
        right_date = ranked.at[right, "peak_date"]
        if pd.isna(left_date) or pd.isna(right_date):
            continue
        date_diff = abs((left_date - right_date).days)
        if date_diff > DEDUP_PEAK_DATE_WINDOW_DAYS:
            continue
        if jaccard_similarity(terms[left], terms[right]) >= DEDUP_JACCARD_THRESHOLD:
            union(left, right)

    ranked["auto_macro_event_id"] = [f"event_{find(idx)}" for idx in range(len(ranked))]
    deduplicated = (
        ranked
        .sort_values(SCORE_METHOD, ascending=False)
        .drop_duplicates(subset=["auto_macro_event_id"], keep="first")
        .sort_values(SCORE_METHOD, ascending=False)
        .reset_index(drop=True)
    )
    deduplicated["rank"] = np.arange(1, len(deduplicated) + 1)
    return deduplicated


def run_ctfidf_labeling(cfg: PipelineConfig) -> None:
    """Label clusters with c-TF-IDF terms and rank event candidates."""
    if not cfg.posts_with_clusters_path.exists():
        raise FileNotFoundError(
            f"Posts with clusters file not found: {cfg.posts_with_clusters_path}"
        )

    if not cfg.cluster_summary_path.exists():
        raise FileNotFoundError(
            f"Cluster summary file not found: {cfg.cluster_summary_path}"
        )

    posts = pd.read_parquet(cfg.posts_with_clusters_path)
    cluster_summary = pd.read_csv(cfg.cluster_summary_path)

    required_post_cols = {"cluster_id", "title"}
    missing_post_cols = required_post_cols - set(posts.columns)
    if missing_post_cols:
        raise ValueError(f"Missing required post columns: {sorted(missing_post_cols)}")

    required_summary_cols = {"cluster_id", "n_posts", "peak_count"}
    missing_summary_cols = required_summary_cols - set(cluster_summary.columns)
    if missing_summary_cols:
        raise ValueError(
            f"Missing required cluster summary columns: {sorted(missing_summary_cols)}"
        )

    text_column = select_text_column(posts)
    valid_posts = posts[posts["cluster_id"] != -1].copy()

    cluster_docs = (
        valid_posts
        .groupby("cluster_id")[text_column]
        .apply(lambda values: " ".join(values.fillna("").astype(str)))
        .reset_index(name="cluster_text")
    )

    if cluster_docs.empty:
        raise ValueError("No non-noise clusters found for c-TF-IDF labeling.")

    n_clusters = len(cluster_docs)
    if n_clusters < 10:
        min_df = 1
        max_df = 1.0
    else:
        min_df = 3
        max_df = 0.8

    print("c-TF-IDF vectorizer parameters:")
    print("n_clusters:", n_clusters)
    print("min_df:", min_df)
    print("max_df:", max_df)

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=min_df,
        max_df=max_df,
    )

    X = vectorizer.fit_transform(cluster_docs["cluster_text"])
    words = vectorizer.get_feature_names_out()

    doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log((X.shape[0] + 1) / (doc_freq + 1))
    c_tf_idf = X.multiply(idf)

    labels = []
    top_terms_all = []

    for row_idx in range(c_tf_idf.shape[0]):
        row = c_tf_idf.getrow(row_idx).toarray().ravel()
        top_idx = row.argsort()[-TOP_N_TERMS:][::-1]
        top_terms = [words[idx] for idx in top_idx if row[idx] > 0]

        labels.append(" | ".join(top_terms[:3]))
        top_terms_all.append(", ".join(top_terms))

    cluster_labels = pd.DataFrame({
        "cluster_id": cluster_docs["cluster_id"],
        "interpreted_label_ctfidf": labels,
        "ctfidf_top_terms": top_terms_all,
    })

    labeled_summary = cluster_summary.merge(
        cluster_labels,
        on="cluster_id",
        how="left",
    )
    labeled_summary = add_s4_features(posts, labeled_summary)

    event_candidates = (
        labeled_summary
        .sort_values(SCORE_METHOD, ascending=False)
        .reset_index(drop=True)
    )
    event_candidates["rank"] = np.arange(1, len(event_candidates) + 1)
    event_candidates_deduplicated = deduplicate_event_candidates(event_candidates)
    event_candidates_top30_deduplicated = event_candidates_deduplicated.head(
        TOP_K_DEDUP
    ).copy()

    cfg.run_processed_dir.mkdir(parents=True, exist_ok=True)
    event_candidates_path(cfg).parent.mkdir(parents=True, exist_ok=True)

    labeled_summary.to_csv(labeled_summary_path(cfg), index=False)
    event_candidates.to_csv(event_candidates_path(cfg), index=False)
    event_candidates_deduplicated.to_csv(
        event_candidates_deduplicated_path(cfg),
        index=False,
    )
    event_candidates_top30_deduplicated.to_csv(
        event_candidates_top30_deduplicated_path(cfg),
        index=False,
    )

    print("Clusters labeled:", cluster_labels["cluster_id"].nunique())
    print("Saved labeled summary:")
    print(labeled_summary_path(cfg))
    print("Saved event candidates:")
    print(event_candidates_path(cfg))
    print(event_candidates_deduplicated_path(cfg))
    print(event_candidates_top30_deduplicated_path(cfg))
    print("\nTop 10 event candidates:")
    preview_cols = [
        "cluster_id",
        "interpreted_label_ctfidf",
        "n_posts",
        "peak_count",
        "burst_share",
        "engagement",
        "S4_full_metadata",
        "event_candidate_score",
    ]
    preview_cols = [col for col in preview_cols if col in event_candidates.columns]
    print(event_candidates[preview_cols].head(10).to_string(index=False))


def main(cfg: PipelineConfig | None = None) -> None:
    run_ctfidf_labeling(cfg or DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
