from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REAL_EVENTS_FILE = (
    PROJECT_ROOT / "data" / "external" / "real_events_nov_2025_jan_2026_top30.csv"
)
PREDICTED_CANDIDATES_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "nov_2025_jan_2026"
    / "temporal_analysis"
    / "event_candidates_top30_deduplicated.csv"
)
PREDICTED_CANDIDATES_DEDUP_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "nov_2025_jan_2026"
    / "temporal_analysis"
    / "event_candidates_top30_deduplicated.csv"
)
EVALUATION_DIR = (
    PROJECT_ROOT / "outputs" / "nov_2025_jan_2026" / "evaluation"
)
MATCHING_OUTPUT = EVALUATION_DIR / "event_candidate_matching.csv"
METRICS_OUTPUT = EVALUATION_DIR / "event_candidate_metrics.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 30
MATCH_THRESHOLD = 0.55
SCORE_METHOD = "S4_full_metadata"
DEDUP_JACCARD_THRESHOLD = 0.30
DEDUP_PEAK_DATE_WINDOW_DAYS = 2


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def join_available_fields(row: pd.Series, columns: Iterable[str]) -> str:
    parts = []
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            value = str(row[column]).strip()
            if value:
                parts.append(value)
    return " ".join(parts)


def build_real_event_text(row: pd.Series) -> str:
    return join_available_fields(
        row,
        [
            "event_name",
            "event_description",
            "keywords",
            "category",
        ],
    )


def build_candidate_text(row: pd.Series) -> str:
    return join_available_fields(
        row,
        [
            "interpreted_label_ctfidf",
            "top_keywords",
            "sample_titles",
            "top_subreddits",
        ],
    )


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

    if SCORE_METHOD in candidates.columns:
        candidates = candidates.sort_values(SCORE_METHOD, ascending=False)
    elif "event_candidate_score" in candidates.columns:
        candidates = candidates.sort_values("event_candidate_score", ascending=False)

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
    score_col = SCORE_METHOD if SCORE_METHOD in ranked.columns else "event_candidate_score"
    deduplicated = (
        ranked
        .sort_values(score_col, ascending=False)
        .drop_duplicates(subset=["auto_macro_event_id"], keep="first")
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
    )
    deduplicated["rank"] = np.arange(1, len(deduplicated) + 1)
    return deduplicated


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not REAL_EVENTS_FILE.exists():
        raise FileNotFoundError(f"Real events file not found: {REAL_EVENTS_FILE}")

    if not PREDICTED_CANDIDATES_DEDUP_FILE.exists() and not PREDICTED_CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Predicted candidates file not found: {PREDICTED_CANDIDATES_FILE}"
        )

    real_events = pd.read_csv(REAL_EVENTS_FILE)
    if PREDICTED_CANDIDATES_DEDUP_FILE.exists():
        candidates = pd.read_csv(PREDICTED_CANDIDATES_DEDUP_FILE)
    else:
        candidates = deduplicate_event_candidates(pd.read_csv(PREDICTED_CANDIDATES_FILE))
    candidates = candidates.head(TOP_K).copy()

    if real_events.empty:
        raise ValueError("Real events file is empty.")

    if candidates.empty:
        raise ValueError("Predicted candidates file has no top candidates.")

    print("Real events columns found:")
    print(real_events.columns.tolist())

    return real_events, candidates


def evaluate_event_candidates() -> None:
    real_events, candidates = load_inputs()

    real_events = real_events.copy()
    evaluation_texts = []

    for _, row in real_events.iterrows():
        parts = []

        for col in [
            "event_name",
            "event_description",
            "keywords",
            "category",
        ]:
            value = row.get(col)

            if pd.notna(value):
                value = str(value).strip()

                if value:
                    parts.append(value)

        real_text = " ".join(parts)
        row["evaluation_text"] = real_text
        evaluation_texts.append(real_text)

    real_events["evaluation_text"] = evaluation_texts
    print(real_events[["event_name", "evaluation_text"]].head(3))

    event_id_col = first_existing_column(real_events, ["event_id", "id"])
    empty_text_mask = real_events["evaluation_text"].fillna("").astype(str).str.strip() == ""

    for idx, row in real_events.loc[empty_text_mask].iterrows():
        event_identifier = row[event_id_col] if event_id_col is not None else idx
        print(
            "Warning: skipping real event with empty evaluation text "
            f"(event_id/index={event_identifier})"
        )

    real_events = real_events.loc[~empty_text_mask].copy()

    if real_events.empty:
        raise ValueError("No real events left after filtering empty evaluation text.")

    real_texts = real_events["evaluation_text"].tolist()
    candidate_texts = candidates.apply(build_candidate_text, axis=1).tolist()

    if any(not text.strip() for text in candidate_texts):
        raise ValueError("At least one predicted candidate has empty evaluation text.")

    print("Loading model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    real_embeddings = model.encode(
        real_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    candidate_embeddings = model.encode(
        candidate_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    similarities = np.matmul(real_embeddings, candidate_embeddings.T)
    best_candidate_indices = similarities.argmax(axis=1)
    best_similarities = similarities.max(axis=1)

    event_name_col = first_existing_column(real_events, ["event_name", "name", "title"])
    event_date_col = first_existing_column(real_events, ["event_date", "date"])
    reference_url_col = first_existing_column(
        real_events,
        ["reference_url", "url", "source_url", "reference"],
    )
    cluster_id_col = first_existing_column(candidates, ["cluster_id", "Cluster"])
    label_col = first_existing_column(
        candidates,
        ["interpreted_label_ctfidf", "label", "Label"],
    )
    rank_col = first_existing_column(candidates, ["rank", "Rank"])

    rows = []
    matched_candidate_indices = set()

    for real_idx, candidate_idx in enumerate(best_candidate_indices):
        real_row = real_events.iloc[real_idx]
        candidate_row = candidates.iloc[int(candidate_idx)]
        similarity = float(best_similarities[real_idx])
        is_match = similarity >= MATCH_THRESHOLD

        if is_match:
            matched_candidate_indices.add(int(candidate_idx))

        candidate_rank = (
            int(candidate_row[rank_col])
            if rank_col is not None and pd.notna(candidate_row[rank_col])
            else int(candidate_idx) + 1
        )

        rows.append({
            "event_id": (
                real_row[event_id_col]
                if event_id_col is not None
                else real_idx + 1
            ),
            "event_name": (
                real_row[event_name_col]
                if event_name_col is not None
                else ""
            ),
            "event_date": (
                real_row[event_date_col]
                if event_date_col is not None
                else ""
            ),
            "best_candidate_rank": candidate_rank,
            "best_candidate_cluster_id": (
                candidate_row[cluster_id_col]
                if cluster_id_col is not None
                else ""
            ),
            "best_candidate_label": (
                candidate_row[label_col]
                if label_col is not None
                else ""
            ),
            "similarity": similarity,
            "is_match": is_match,
            "reference_url": (
                real_row[reference_url_col]
                if reference_url_col is not None
                else ""
            ),
        })

    matching = pd.DataFrame(rows)

    matched_real_events = int(matching["is_match"].sum())
    n_real_events = len(real_events)
    n_candidates_eval = min(TOP_K, len(candidates))

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
    metrics = {
        "matched_real_events": matched_real_events,
        "n_real_events": n_real_events,
        "n_candidates_evaluated": n_candidates_eval,
        "match_threshold": MATCH_THRESHOLD,
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

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    matching.to_csv(MATCHING_OUTPUT, index=False)
    METRICS_OUTPUT.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nEvent candidate evaluation metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaved matching table:")
    print(MATCHING_OUTPUT)
    print("Saved metrics:")
    print(METRICS_OUTPUT)


def main() -> None:
    evaluate_event_candidates()


if __name__ == "__main__":
    main()
