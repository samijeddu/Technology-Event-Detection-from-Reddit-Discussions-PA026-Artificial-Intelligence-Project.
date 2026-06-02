from __future__ import annotations

import json
import re
import time
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from typing import Iterable

import hdbscan
import numpy as np
import optuna
import pandas as pd
import umap
from optuna.samplers import TPESampler
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aug_2025_oct_2025"
    / "reddit_tech_embeddings.npy"
)
METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aug_2025_oct_2025"
    / "reddit_tech_metadata.parquet"
)
REAL_EVENTS_FILE = (
    PROJECT_ROOT / "data" / "external" / "real_events_aug_oct_2025_top30.csv"
)

GRID_SEARCH_DIR = PROJECT_ROOT / "outputs" / "grid_search_s4_top30_threshold055"
HPO_DIR = GRID_SEARCH_DIR
RESULTS_FILE = GRID_SEARCH_DIR / "grid_search_results.csv"
BEST_PARAMS_FILE = GRID_SEARCH_DIR / "best_params.json"
LATEST_TRIAL_FILE = GRID_SEARCH_DIR / "latest_trial.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MATCH_THRESHOLD = 0.55
TOP_K = 30
N_TRIALS = 40
N_TRIALS_DEFAULT = N_TRIALS
RANDOM_STATE = 42
MAX_SILHOUETTE_SAMPLE = 5000
SCORE_METHOD = "S4_full_metadata"
DEDUP_JACCARD_THRESHOLD = 0.30
DEDUP_PEAK_DATE_WINDOW_DAYS = 2
GRID_SELECTED_TRIALS = 50
GRID_N_NEIGHBORS = [10, 15, 20, 25]
GRID_N_COMPONENTS = [10, 15]
GRID_MIN_DIST = [0.0, 0.03, 0.05]
GRID_MIN_CLUSTER_SIZE = [60, 80, 100]
GRID_MIN_SAMPLES = [15, 25]

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "you", "your", "are",
    "was", "were", "have", "has", "had", "but", "not", "can", "could", "would",
    "should", "will", "just", "like", "about", "into", "than", "then", "they",
    "them", "their", "what", "when", "where", "why", "how", "all", "any",
    "new", "use", "using", "get", "got", "one", "out", "more", "some",
    "really", "does", "did", "been", "because", "after", "before",
    "chatgpt", "openai", "model", "https", "http", "com", "www", "removed",
    "deleted", "now", "there", "here", "want", "people", "time", "who",
    "only", "make", "need",
}


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def select_text_column(df: pd.DataFrame) -> str:
    for column in ["text_for_embedding", "full_text", "title"]:
        if column in df.columns:
            return column
    raise ValueError("No text column found. Expected text_for_embedding, full_text or title.")


def get_top_keywords(texts: pd.Series, top_k: int = 12) -> str:
    text = " ".join(texts.fillna("").astype(str).tolist()).lower()
    words = [
        word
        for word in pd.Series(text.split()).str.extractall(r"([a-zA-Z][a-zA-Z0-9\-]{2,})")[0]
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


def get_sample_titles(group: pd.DataFrame, n: int = 8) -> str:
    if "cluster_probability" in group.columns:
        group = group.sort_values("cluster_probability", ascending=False)
    titles = group["title"].fillna("").astype(str).str.strip().tolist()
    titles = [title for title in titles if title]
    return " || ".join(titles[:n])


def build_cluster_summary(posts: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"cluster_id", "date", "subreddit", "title"}
    missing_cols = required_cols - set(posts.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    text_col = select_text_column(posts)
    df = posts.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.floor("D")
    valid_df = df[df["cluster_id"] != -1].copy()

    rows = []
    for cluster_id, group in valid_df.groupby("cluster_id"):
        n_posts = len(group)
        daily_counts = group.groupby("date").size().sort_values(ascending=False)
        peak_date = daily_counts.index[0] if len(daily_counts) else None
        peak_count = int(daily_counts.iloc[0]) if len(daily_counts) else 0
        mean_daily_posts = float(daily_counts.mean()) if len(daily_counts) else 0.0
        std_daily_posts = float(daily_counts.std()) if len(daily_counts) > 1 else 0.0
        active_days = int(daily_counts.count())
        avg_score = (
            float(pd.to_numeric(group["score"], errors="coerce").mean())
            if "score" in group.columns
            else 0.0
        )
        avg_comments = (
            float(pd.to_numeric(group["num_comments"], errors="coerce").mean())
            if "num_comments" in group.columns
            else 0.0
        )
        avg_score = max(avg_score if np.isfinite(avg_score) else 0.0, 0.0)
        avg_comments = max(avg_comments if np.isfinite(avg_comments) else 0.0, 0.0)
        engagement = avg_score + avg_comments
        burst_share = peak_count / n_posts if n_posts else 0.0
        peak_prominence = (
            peak_count / mean_daily_posts
            if mean_daily_posts > 0
            else 0.0
        )

        rows.append({
            "cluster_id": cluster_id,
            "n_posts": n_posts,
            "date_start": group["date"].min(),
            "date_end": group["date"].max(),
            "peak_date": peak_date,
            "peak_count": peak_count,
            "peak_ratio": burst_share,
            "burst_share": burst_share,
            "active_days": active_days,
            "mean_daily_posts": mean_daily_posts,
            "std_daily_posts": std_daily_posts,
            "peak_prominence": peak_prominence,
            "avg_score": avg_score,
            "avg_comments": avg_comments,
            "engagement": engagement,
            "top_subreddits": str(group["subreddit"].value_counts().head(5).to_dict()),
            "top_keywords": get_top_keywords(group[text_col]),
            "sample_titles": get_sample_titles(group),
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("n_posts", ascending=False).reset_index(drop=True)
    return summary


def add_s4_score(labeled: pd.DataFrame) -> pd.DataFrame:
    scored = labeled.copy()
    scored["n_posts"] = pd.to_numeric(scored["n_posts"], errors="coerce").fillna(0)
    scored["peak_count"] = pd.to_numeric(
        scored["peak_count"],
        errors="coerce",
    ).fillna(0)
    if "burst_share" not in scored.columns:
        scored["burst_share"] = np.where(
            scored["n_posts"] > 0,
            scored["peak_count"] / scored["n_posts"],
            0.0,
        )
    scored["burst_share"] = pd.to_numeric(
        scored["burst_share"],
        errors="coerce",
    ).fillna(0)
    scored["peak_ratio"] = scored["burst_share"]
    for column in ["avg_score", "avg_comments"]:
        if column not in scored.columns:
            scored[column] = 0.0
        scored[column] = pd.to_numeric(scored[column], errors="coerce").fillna(0).clip(
            lower=0
        )
    if "engagement" not in scored.columns:
        scored["engagement"] = scored["avg_score"] + scored["avg_comments"]
    scored["engagement"] = pd.to_numeric(
        scored["engagement"],
        errors="coerce",
    ).fillna(0).clip(lower=0)
    if "mean_daily_posts" in scored.columns:
        scored["mean_daily_posts"] = pd.to_numeric(
            scored["mean_daily_posts"],
            errors="coerce",
        ).fillna(0)
    else:
        scored["mean_daily_posts"] = 0.0
    scored["peak_prominence"] = np.where(
        scored["mean_daily_posts"] > 0,
        scored["peak_count"] / scored["mean_daily_posts"],
        0.0,
    )
    scored["S1_baseline"] = scored["peak_count"] * np.log1p(scored["n_posts"])
    scored["S4_full_metadata"] = (
        scored["peak_count"]
        * np.log1p(scored["n_posts"])
        * scored["burst_share"]
        * np.log1p(scored["engagement"])
    )
    scored["event_candidate_score"] = scored["S4_full_metadata"]
    scored["score_method"] = SCORE_METHOD
    return scored


def run_ctfidf_labeling(posts: pd.DataFrame, cluster_summary: pd.DataFrame) -> pd.DataFrame:
    text_col = select_text_column(posts)
    valid_posts = posts[posts["cluster_id"] != -1].copy()

    cluster_docs = (
        valid_posts
        .groupby("cluster_id")[text_col]
        .apply(lambda values: " ".join(values.fillna("").astype(str)))
        .reset_index(name="cluster_text")
    )
    if cluster_docs.empty:
        raise ValueError("No non-noise clusters found for c-TF-IDF labeling.")

    n_clusters = len(cluster_docs)
    min_df = 1 if n_clusters < 10 else 3
    max_df = 1.0 if n_clusters < 10 else 0.8

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
        top_idx = row.argsort()[-5:][::-1]
        top_terms = [words[idx] for idx in top_idx if row[idx] > 0]
        labels.append(" | ".join(top_terms[:3]))
        top_terms_all.append(", ".join(top_terms))

    cluster_labels = pd.DataFrame({
        "cluster_id": cluster_docs["cluster_id"],
        "interpreted_label_ctfidf": labels,
        "ctfidf_top_terms": top_terms_all,
    })

    labeled = cluster_summary.merge(cluster_labels, on="cluster_id", how="left")
    labeled = add_s4_score(labeled)
    ranked = labeled.sort_values(SCORE_METHOD, ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


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
        if abs((left_date - right_date).days) > DEDUP_PEAK_DATE_WINDOW_DAYS:
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


def build_real_event_texts(real_events: pd.DataFrame) -> pd.DataFrame:
    real_events = real_events.copy()
    texts = []
    for _, row in real_events.iterrows():
        parts = []
        for col in ["event_name", "event_description", "keywords", "category"]:
            value = row.get(col)
            if pd.notna(value):
                value = str(value).strip()
                if value:
                    parts.append(value)
        texts.append(" ".join(parts))
    real_events["evaluation_text"] = texts
    real_events = real_events[
        real_events["evaluation_text"].fillna("").astype(str).str.strip() != ""
    ].copy()
    if real_events.empty:
        raise ValueError("No real events left after filtering empty evaluation text.")
    return real_events


def build_candidate_text(row: pd.Series) -> str:
    parts = []
    for col in [
        "interpreted_label_ctfidf",
        "top_keywords",
        "sample_titles",
        "top_subreddits",
    ]:
        if col in row.index and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value:
                parts.append(value)
    return " ".join(parts)


def evaluate_top30(
    event_candidates: pd.DataFrame,
    real_events: pd.DataFrame,
    model: SentenceTransformer,
    trial_dir: Path,
) -> dict[str, float | int]:
    top_candidates = event_candidates.head(TOP_K).copy()
    real_texts = real_events["evaluation_text"].tolist()
    candidate_texts = top_candidates.apply(build_candidate_text, axis=1).tolist()

    if any(not text.strip() for text in candidate_texts):
        raise ValueError("At least one predicted candidate has empty evaluation text.")

    real_embeddings = model.encode(
        real_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    candidate_embeddings = model.encode(
        candidate_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    similarities = np.matmul(real_embeddings, candidate_embeddings.T)
    best_candidate_indices = similarities.argmax(axis=1)
    best_similarities = similarities.max(axis=1)

    event_id_col = first_existing_column(real_events, ["event_id", "id"])
    event_name_col = first_existing_column(real_events, ["event_name", "name", "title"])
    event_date_col = first_existing_column(real_events, ["event_date", "date"])
    reference_url_col = first_existing_column(
        real_events,
        ["reference_url", "url", "source_url", "reference"],
    )

    rows = []
    matched_candidate_indices = set()
    for real_idx, candidate_idx in enumerate(best_candidate_indices):
        real_row = real_events.iloc[real_idx]
        candidate_row = top_candidates.iloc[int(candidate_idx)]
        similarity = float(best_similarities[real_idx])
        is_match = similarity >= MATCH_THRESHOLD
        if is_match:
            matched_candidate_indices.add(int(candidate_idx))

        rows.append({
            "event_id": real_row[event_id_col] if event_id_col else real_idx + 1,
            "event_name": real_row[event_name_col] if event_name_col else "",
            "event_date": real_row[event_date_col] if event_date_col else "",
            "best_candidate_rank": int(candidate_idx) + 1,
            "best_candidate_cluster_id": candidate_row.get("cluster_id", ""),
            "best_candidate_label": candidate_row.get("interpreted_label_ctfidf", ""),
            "similarity": similarity,
            "is_match": is_match,
            "reference_url": real_row[reference_url_col] if reference_url_col else "",
        })

    matching = pd.DataFrame(rows)
    matching.to_csv(trial_dir / "event_candidate_matching.csv", index=False)

    matched_real_events = int(matching["is_match"].sum())
    n_real_events = len(real_events)
    n_candidates_eval = min(TOP_K, len(top_candidates))
    recall_at_30 = matched_real_events / n_real_events if n_real_events else 0.0
    precision_at_30 = (
        len(matched_candidate_indices) / n_candidates_eval
        if n_candidates_eval
        else 0.0
    )
    f1_at_30 = (
        2 * precision_at_30 * recall_at_30 / (precision_at_30 + recall_at_30)
        if precision_at_30 + recall_at_30 > 0
        else 0.0
    )
    matched_similarities = matching.loc[matching["is_match"], "similarity"]

    return {
        "matched_real_events": matched_real_events,
        "recall_at_30": recall_at_30,
        "precision_at_30": precision_at_30,
        "f1_at_30": f1_at_30,
        "mean_similarity_matched": (
            float(matched_similarities.mean())
            if not matched_similarities.empty
            else 0.0
        ),
        "mean_similarity_all": float(matching["similarity"].mean()),
    }


def build_grid_configs() -> list[dict[str, int | float]]:
    configs = [
        {
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_dist": min_dist,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
        }
        for (
            n_neighbors,
            n_components,
            min_dist,
            min_cluster_size,
            min_samples,
        ) in product(
            GRID_N_NEIGHBORS,
            GRID_N_COMPONENTS,
            GRID_MIN_DIST,
            GRID_MIN_CLUSTER_SIZE,
            GRID_MIN_SAMPLES,
        )
    ]
    return sorted(
        configs,
        key=lambda item: (
            item["n_neighbors"],
            item["n_components"],
            item["min_dist"],
            item["min_cluster_size"],
            item["min_samples"],
        ),
    )


def select_grid_configs(
    configs: list[dict[str, int | float]],
    target_count: int = GRID_SELECTED_TRIALS,
) -> list[dict[str, int | float]]:
    if len(configs) <= target_count:
        return configs

    rng = np.random.default_rng(RANDOM_STATE)
    step = len(configs) / target_count
    offset = float(rng.uniform(0, step))
    selected_indices = [
        min(int(offset + idx * step), len(configs) - 1)
        for idx in range(target_count)
    ]
    selected_indices = sorted(dict.fromkeys(selected_indices))

    # Extremely unlikely, but keep the requested count stable if rounding collided.
    if len(selected_indices) < target_count:
        remaining = [idx for idx in range(len(configs)) if idx not in selected_indices]
        fill = rng.choice(
            remaining,
            size=target_count - len(selected_indices),
            replace=False,
        )
        selected_indices = sorted(selected_indices + [int(idx) for idx in fill])

    return [configs[idx] for idx in selected_indices[:target_count]]


def apply_objective_penalties(result: dict[str, object]) -> float:
    objective_score = float(result["f1_at_30"])
    if int(result["n_clusters"]) < 20:
        objective_score *= 0.25
    if int(result["n_clusters"]) > 300:
        objective_score *= 0.5
    if float(result["noise_ratio"]) > 0.75:
        objective_score *= 0.5
    return objective_score


def initial_trial_result(
    trial_number: int,
    params: dict[str, int | float],
) -> dict[str, object]:
    return {
        "trial_number": trial_number,
        **params,
        "f1_at_30": 0.0,
        "precision_at_30": 0.0,
        "recall_at_30": 0.0,
        "matched_real_events": 0,
        "mean_similarity_matched": 0.0,
        "mean_similarity_all": 0.0,
        "noise_ratio": 1.0,
        "n_clusters": 0,
        "silhouette_score": None,
        "runtime_seconds": 0.0,
        "status": "failed",
        "error": "",
        "objective_score": 0.0,
    }


def run_grid_trial(
    trial_number: int,
    total_trials: int,
    params: dict[str, int | float],
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    real_events: pd.DataFrame,
    model: SentenceTransformer,
) -> dict[str, object]:
    start = time.time()
    trial_dir = GRID_SEARCH_DIR / f"trial_{trial_number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    result = initial_trial_result(trial_number, params)

    try:
        umap_model = umap.UMAP(
            n_neighbors=int(params["n_neighbors"]),
            n_components=int(params["n_components"]),
            min_dist=float(params["min_dist"]),
            metric="cosine",
            random_state=RANDOM_STATE,
        )
        umap_embeddings = np.asarray(
            umap_model.fit_transform(embeddings),
            dtype=np.float32,
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(params["min_cluster_size"]),
            min_samples=int(params["min_samples"]),
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        labels = clusterer.fit_predict(umap_embeddings)

        posts = metadata.copy()
        posts["cluster_id"] = labels
        posts["cluster_probability"] = clusterer.probabilities_

        n_total = len(posts)
        n_noise = int((labels == -1).sum())
        n_clusters = int(pd.Series(labels[labels != -1]).nunique())
        noise_ratio = n_noise / n_total if n_total else 1.0

        cluster_summary = build_cluster_summary(posts)
        event_candidates = run_ctfidf_labeling(posts, cluster_summary)
        event_candidates.to_csv(trial_dir / "event_candidates_ranked.csv", index=False)

        event_candidates_deduplicated = deduplicate_event_candidates(event_candidates)
        event_candidates_deduplicated.to_csv(
            trial_dir / "event_candidates_ranked_deduplicated.csv",
            index=False,
        )
        event_candidates_deduplicated.head(TOP_K).to_csv(
            trial_dir / "event_candidates_top30_deduplicated.csv",
            index=False,
        )

        metrics = evaluate_top30(
            event_candidates_deduplicated,
            real_events,
            model,
            trial_dir,
        )
        (trial_dir / "event_candidate_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        result.update({
            **metrics,
            "noise_ratio": noise_ratio,
            "n_clusters": n_clusters,
            "silhouette_score": compute_silhouette(umap_embeddings, labels),
            "status": "ok",
        })

    except Exception as exc:
        result["error"] = repr(exc)
        print(f"Trial {trial_number} failed: {exc}")

    result["objective_score"] = apply_objective_penalties(result)
    result["runtime_seconds"] = round(time.time() - start, 3)
    metrics_payload = {
        "matched_real_events": result["matched_real_events"],
        "recall_at_30": result["recall_at_30"],
        "precision_at_30": result["precision_at_30"],
        "f1_at_30": result["f1_at_30"],
        "mean_similarity_matched": result["mean_similarity_matched"],
        "mean_similarity_all": result["mean_similarity_all"],
        "objective_score": result["objective_score"],
        "status": result["status"],
        "error": result["error"],
        "match_threshold": MATCH_THRESHOLD,
        "top_k": TOP_K,
    }
    (trial_dir / "event_candidate_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nTrial {trial_number + 1} / {total_trials}")
    print("params:", params)
    print("n_clusters:", result["n_clusters"])
    print("noise_ratio:", result["noise_ratio"])
    print("f1_at_30:", result["f1_at_30"])
    print("precision_at_30:", result["precision_at_30"])
    print("recall_at_30:", result["recall_at_30"])
    print("matched_real_events:", result["matched_real_events"])
    print("mean_similarity_all:", result["mean_similarity_all"])
    print("final objective_score:", result["objective_score"])
    print("runtime_seconds:", result["runtime_seconds"])

    return result


def compute_silhouette(umap_embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    mask = labels != -1
    assigned_labels = labels[mask]
    if assigned_labels.size < 2 or np.unique(assigned_labels).size < 2:
        return None

    X = umap_embeddings[mask]
    if X.shape[0] > MAX_SILHOUETTE_SAMPLE:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(X.shape[0], size=MAX_SILHOUETTE_SAMPLE, replace=False)
        X = X[idx]
        assigned_labels = assigned_labels[idx]

    if np.unique(assigned_labels).size < 2:
        return None

    return float(silhouette_score(X, assigned_labels))


def run_trial(
    trial: optuna.Trial,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    real_events: pd.DataFrame,
    model: SentenceTransformer,
    results: list[dict[str, object]],
) -> float:
    start = time.time()
    trial_dir = HPO_DIR / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "n_neighbors": trial.suggest_categorical("n_neighbors", [10, 15, 20, 25]),
        "n_components": trial.suggest_categorical("n_components", [10, 15]),
        "min_dist": trial.suggest_categorical("min_dist", [0.0, 0.03, 0.05, 0.1]),
        "min_cluster_size": trial.suggest_categorical(
            "min_cluster_size",
            [40, 60, 80, 100],
        ),
        "min_samples": trial.suggest_categorical("min_samples", [15, 20, 25, 35]),
    }

    result: dict[str, object] = {
        "trial_number": trial.number,
        **params,
        "f1_at_30": 0.0,
        "precision_at_30": 0.0,
        "recall_at_30": 0.0,
        "mean_similarity_matched": 0.0,
        "mean_similarity_all": 0.0,
        "noise_ratio": 1.0,
        "n_clusters": 0,
        "silhouette_score": None,
        "runtime_seconds": 0.0,
        "status": "failed",
        "error": "",
        "objective_score": 0.0,
    }

    try:
        umap_model = umap.UMAP(
            n_neighbors=params["n_neighbors"],
            n_components=params["n_components"],
            min_dist=params["min_dist"],
            metric="cosine",
            random_state=RANDOM_STATE,
        )
        umap_embeddings = np.asarray(
            umap_model.fit_transform(embeddings),
            dtype=np.float32,
        )
        np.save(trial_dir / "umap_embeddings.npy", umap_embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        labels = clusterer.fit_predict(umap_embeddings)
        np.save(trial_dir / "cluster_labels.npy", labels)

        posts = metadata.copy()
        posts["cluster_id"] = labels
        posts["cluster_probability"] = clusterer.probabilities_
        posts.to_parquet(trial_dir / "posts_with_clusters.parquet", index=False)

        n_total = len(posts)
        n_noise = int((labels == -1).sum())
        n_clusters = int(pd.Series(labels[labels != -1]).nunique())
        noise_ratio = n_noise / n_total if n_total else 1.0

        cluster_summary = build_cluster_summary(posts)
        cluster_summary.to_csv(trial_dir / "cluster_summary.csv", index=False)

        event_candidates = run_ctfidf_labeling(posts, cluster_summary)
        event_candidates.to_csv(trial_dir / "event_candidates_ranked.csv", index=False)
        event_candidates_deduplicated = deduplicate_event_candidates(event_candidates)
        event_candidates_deduplicated.to_csv(
            trial_dir / "event_candidates_ranked_deduplicated.csv",
            index=False,
        )
        event_candidates_deduplicated.head(TOP_K).to_csv(
            trial_dir / "event_candidates_top30_deduplicated.csv",
            index=False,
        )

        metrics = evaluate_top30(
            event_candidates_deduplicated,
            real_events,
            model,
            trial_dir,
        )
        sil_score = compute_silhouette(umap_embeddings, labels)

        result.update({
            **metrics,
            "noise_ratio": noise_ratio,
            "n_clusters": n_clusters,
            "silhouette_score": sil_score,
            "status": "ok",
        })

    except Exception as exc:
        result["error"] = repr(exc)
        print(f"Trial {trial.number} failed: {exc}")

    objective_score = float(result["f1_at_30"])
    if int(result["n_clusters"]) < 20:
        objective_score *= 0.25
    if int(result["n_clusters"]) > 300:
        objective_score *= 0.5
    if float(result["noise_ratio"]) > 0.75:
        objective_score *= 0.5

    result["objective_score"] = objective_score
    result["runtime_seconds"] = round(time.time() - start, 3)
    results.append(result)
    pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

    print("\nTrial", trial.number)
    print("params:", params)
    print("n_clusters:", result["n_clusters"])
    print("noise_ratio:", result["noise_ratio"])
    print("f1_at_30:", result["f1_at_30"])
    print("mean_similarity_all:", result["mean_similarity_all"])
    print("final objective score:", objective_score)
    print("runtime_seconds:", result["runtime_seconds"])

    return objective_score


def run_hpo(n_trials: int = N_TRIALS_DEFAULT) -> None:
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
    if not REAL_EVENTS_FILE.exists():
        raise FileNotFoundError(f"Real events file not found: {REAL_EVENTS_FILE}")

    HPO_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(EMBEDDINGS_FILE, mmap_mode="r")
    metadata = pd.read_parquet(METADATA_FILE)
    if embeddings.shape[0] != metadata.shape[0]:
        raise ValueError(
            f"Embeddings rows {embeddings.shape[0]} do not match metadata rows {metadata.shape[0]}"
        )

    real_events = build_real_event_texts(pd.read_csv(REAL_EVENTS_FILE))
    model = SentenceTransformer(MODEL_NAME)

    results: list[dict[str, object]] = []
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: run_trial(trial, embeddings, metadata, real_events, model, results),
        n_trials=n_trials,
    )

    best_payload = {
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
    }
    BEST_PARAMS_FILE.write_text(
        json.dumps(best_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nBest trial")
    print("number:", study.best_trial.number)
    print("f1_at_30:", study.best_value)
    print("params:", study.best_trial.params)
    print("\nSaved results:", RESULTS_FILE)
    print("Saved best params:", BEST_PARAMS_FILE)


def save_latest_trial(
    current_trial: int,
    results: list[dict[str, object]],
    total_selected_trials: int,
) -> None:
    completed = len(results)
    ok_results = [row for row in results if row.get("status") == "ok"]
    candidate_results = ok_results if ok_results else results
    best_result = max(
        candidate_results,
        key=lambda row: float(row.get("objective_score", 0.0)),
    )
    best_params = {
        key: best_result[key]
        for key in [
            "n_neighbors",
            "n_components",
            "min_dist",
            "min_cluster_size",
            "min_samples",
        ]
    }
    payload = {
        "current_trial": current_trial,
        "best_score_so_far": float(best_result.get("objective_score", 0.0)),
        "best_params_so_far": best_params,
        "completed_trials": completed,
        "total_selected_trials": total_selected_trials,
    }
    LATEST_TRIAL_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_grid_search() -> None:
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
    if not REAL_EVENTS_FILE.exists():
        raise FileNotFoundError(f"Real events file not found: {REAL_EVENTS_FILE}")

    GRID_SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    all_configs = build_grid_configs()
    selected_configs = select_grid_configs(all_configs)
    total_grid_combinations = len(all_configs)
    total_selected_trials = len(selected_configs)

    print("Grid search S4 top30 threshold 0.55")
    print("total_grid_combinations:", total_grid_combinations)
    print("total_selected_trials:", total_selected_trials)
    print("output_dir:", GRID_SEARCH_DIR)

    embeddings = np.load(EMBEDDINGS_FILE, mmap_mode="r")
    metadata = pd.read_parquet(METADATA_FILE)
    if embeddings.shape[0] != metadata.shape[0]:
        raise ValueError(
            f"Embeddings rows {embeddings.shape[0]} do not match metadata rows {metadata.shape[0]}"
        )

    real_events = build_real_event_texts(pd.read_csv(REAL_EVENTS_FILE))
    model = SentenceTransformer(MODEL_NAME)

    results: list[dict[str, object]] = []
    result_columns = [
        "trial_number",
        "n_neighbors",
        "n_components",
        "min_dist",
        "min_cluster_size",
        "min_samples",
        "f1_at_30",
        "precision_at_30",
        "recall_at_30",
        "matched_real_events",
        "mean_similarity_matched",
        "mean_similarity_all",
        "noise_ratio",
        "n_clusters",
        "silhouette_score",
        "runtime_seconds",
        "status",
        "error",
        "objective_score",
    ]

    for trial_number, params in enumerate(selected_configs):
        result = run_grid_trial(
            trial_number=trial_number,
            total_trials=total_selected_trials,
            params=params,
            embeddings=embeddings,
            metadata=metadata,
            real_events=real_events,
            model=model,
        )
        results.append(result)
        pd.DataFrame(results, columns=result_columns).to_csv(RESULTS_FILE, index=False)
        save_latest_trial(trial_number, results, total_selected_trials)

    best_result = max(results, key=lambda row: float(row.get("objective_score", 0.0)))
    best_params = {
        key: best_result[key]
        for key in [
            "n_neighbors",
            "n_components",
            "min_dist",
            "min_cluster_size",
            "min_samples",
        ]
    }
    best_payload = {
        "best_trial_number": int(best_result["trial_number"]),
        "best_objective_score": float(best_result["objective_score"]),
        "best_params": best_params,
        "total_grid_combinations": total_grid_combinations,
        "total_selected_trials": total_selected_trials,
        "match_threshold": MATCH_THRESHOLD,
        "top_k": TOP_K,
        "score_method": SCORE_METHOD,
    }
    BEST_PARAMS_FILE.write_text(
        json.dumps(best_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nBest trial")
    print("number:", best_result["trial_number"])
    print("best params:", best_params)
    print("best objective_score:", best_result["objective_score"])
    print("results:", RESULTS_FILE)
    print("best params file:", BEST_PARAMS_FILE)
    print("latest trial file:", LATEST_TRIAL_FILE)


def main() -> None:
    run_grid_search()


if __name__ == "__main__":
    main()
