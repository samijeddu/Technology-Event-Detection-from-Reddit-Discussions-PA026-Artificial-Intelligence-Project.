from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "grid_search_s4_top30_threshold055"
    / "grid_search_results.csv"
)
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
TABLES_DIR = PROJECT_ROOT / "docs" / "tables"

DPI = 300


def load_results() -> pd.DataFrame:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Grid search results not found: {RESULTS_FILE}")

    df = pd.read_csv(RESULTS_FILE)
    required_cols = {
        "trial_number",
        "n_neighbors",
        "min_cluster_size",
        "f1_at_30",
        "objective_score",
        "n_clusters",
        "noise_ratio",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    return df


def best_row(df: pd.DataFrame) -> pd.Series:
    return df.sort_values("objective_score", ascending=False).iloc[0]


def save_current_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(path)


def plot_f1_vs_trial(df: pd.DataFrame, best: pd.Series) -> Path:
    path = FIGURES_DIR / "f1_vs_trial.png"
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df.sort_values("trial_number"), x="trial_number", y="f1_at_30")
    plt.scatter(
        best["trial_number"],
        best["f1_at_30"],
        color="red",
        s=90,
        label=f"Best trial {int(best['trial_number'])}",
        zorder=5,
    )
    plt.xlabel("Trial number")
    plt.ylabel("F1@30")
    plt.title("Grid Search F1@30 by Trial")
    plt.legend()
    save_current_figure(path)
    return path


def plot_f1_vs_n_clusters(df: pd.DataFrame, best: pd.Series) -> Path:
    path = FIGURES_DIR / "f1_vs_n_clusters.png"
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="n_clusters", y="f1_at_30", s=55)
    plt.scatter(
        best["n_clusters"],
        best["f1_at_30"],
        color="red",
        s=100,
        label=f"Best trial {int(best['trial_number'])}",
        zorder=5,
    )
    plt.xlabel("Number of clusters")
    plt.ylabel("F1@30")
    plt.title("F1@30 vs Number of Clusters")
    plt.legend()
    save_current_figure(path)
    return path


def plot_f1_vs_noise_ratio(df: pd.DataFrame, best: pd.Series) -> Path:
    path = FIGURES_DIR / "f1_vs_noise_ratio.png"
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="noise_ratio", y="f1_at_30", s=55)
    plt.scatter(
        best["noise_ratio"],
        best["f1_at_30"],
        color="red",
        s=100,
        label=f"Best trial {int(best['trial_number'])}",
        zorder=5,
    )
    plt.xlabel("Noise ratio")
    plt.ylabel("F1@30")
    plt.title("F1@30 vs Noise Ratio")
    plt.legend()
    save_current_figure(path)
    return path


def plot_hpo_scatter_clusters_noise(df: pd.DataFrame, best: pd.Series) -> Path:
    path = FIGURES_DIR / "hpo_scatter_clusters_noise.png"
    plt.figure(figsize=(8.5, 5.5))
    scatter = plt.scatter(
        df["n_clusters"],
        df["f1_at_30"],
        c=df["noise_ratio"],
        cmap="viridis",
        s=65,
        alpha=0.85,
    )
    plt.scatter(
        best["n_clusters"],
        best["f1_at_30"],
        color="red",
        edgecolor="black",
        s=120,
        label=f"Best trial {int(best['trial_number'])}",
        zorder=5,
    )
    plt.colorbar(scatter, label="Noise ratio")
    plt.xlabel("Number of clusters")
    plt.ylabel("F1@30")
    plt.title("Grid Search Tradeoff: Clusters, F1@30, and Noise")
    plt.legend()
    save_current_figure(path)
    return path


def plot_heatmap_neighbors_cluster_size(df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "heatmap_neighbors_cluster_size_f1.png"
    pivot = df.pivot_table(
        index="min_cluster_size",
        columns="n_neighbors",
        values="f1_at_30",
        aggfunc="mean",
    ).sort_index(ascending=True)

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Mean F1@30"},
    )
    plt.xlabel("UMAP n_neighbors")
    plt.ylabel("HDBSCAN min_cluster_size")
    plt.title("Mean F1@30 by n_neighbors and min_cluster_size")
    save_current_figure(path)
    return path


def save_top10_table(df: pd.DataFrame) -> Path:
    path = TABLES_DIR / "top10_grid_search_results.csv"
    top10 = df.sort_values("objective_score", ascending=False).head(10)
    top10.to_csv(path, index=False)
    print(path)
    return path


def save_best_params_summary(best: pd.Series) -> Path:
    path = TABLES_DIR / "best_params_summary.csv"
    columns = [
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
        "n_clusters",
        "noise_ratio",
        "silhouette_score",
        "objective_score",
    ]
    pd.DataFrame([best[columns].to_dict()]).to_csv(path, index=False)
    print(path)
    return path


def save_run_summary() -> Path:
    path = TABLES_DIR / "run_summary.csv"
    rows = [
        {
            "run_mode": "smoke_zst_feb_2026",
            "dataset": "Feb 2026",
            "input_type": "raw .zst",
            "scanned_posts": 1_731_207,
            "kept_posts": 5_000,
            "cleaned_posts": 1_980,
            "embeddings_shape": "(1980, 384)",
            "umap_shape": "",
            "clusters": 3,
            "noise_points": "",
            "noise_ratio": "",
            "notes": "End-to-end smoke test from RS_2026-02.zst.",
        },
        {
            "run_mode": "final_aug_oct_2025",
            "dataset": "Aug 2025 - Oct 2025",
            "input_type": "existing parquet + embeddings",
            "scanned_posts": "",
            "kept_posts": "",
            "cleaned_posts": "",
            "embeddings_shape": "(165733, 384)",
            "umap_shape": "(165733, 15)",
            "clusters": 210,
            "noise_points": 87_447,
            "noise_ratio": 0.5276,
            "notes": "Final run with best grid-search parameters.",
        },
        {
            "run_mode": "generalization_nov_jan",
            "dataset": "Nov 2025 - Jan 2026",
            "input_type": "existing parquet + embeddings",
            "scanned_posts": "",
            "kept_posts": "",
            "cleaned_posts": "",
            "embeddings_shape": "(2000, 384)",
            "umap_shape": "",
            "clusters": 2,
            "noise_points": "",
            "noise_ratio": 0.07,
            "notes": "Limited generalization/smoke-style test with 2000 embeddings.",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(path)
    return path


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()
    best = best_row(df)

    print("Created files:")
    plot_f1_vs_trial(df, best)
    plot_f1_vs_n_clusters(df, best)
    plot_f1_vs_noise_ratio(df, best)
    plot_hpo_scatter_clusters_noise(df, best)
    plot_heatmap_neighbors_cluster_size(df)
    save_top10_table(df)
    save_best_params_summary(best)
    save_run_summary()


if __name__ == "__main__":
    main()
