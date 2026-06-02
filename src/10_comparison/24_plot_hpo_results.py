from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "outputs" / "hpo" / "hpo_results.csv"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
DPI = 300


def load_results() -> tuple[pd.DataFrame, str]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"HPO results file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    required_cols = {
        "trial_number",
        "n_neighbors",
        "min_cluster_size",
        "n_clusters",
        "noise_ratio",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if "f1_at_30" in df.columns:
        f1_col = "f1_at_30"
    elif "f1_at_20" in df.columns:
        f1_col = "f1_at_20"
        print(
            "WARNING: f1_at_30 is not present in outputs/hpo/hpo_results.csv; "
            "using legacy f1_at_20 values for the requested plots."
        )
    else:
        raise ValueError("Missing required F1 column: expected f1_at_30 or f1_at_20")

    df = df.copy()
    df["f1_for_plot"] = pd.to_numeric(df[f1_col], errors="coerce")
    df["trial_number"] = pd.to_numeric(df["trial_number"], errors="coerce")
    df["n_clusters"] = pd.to_numeric(df["n_clusters"], errors="coerce")
    df["noise_ratio"] = pd.to_numeric(df["noise_ratio"], errors="coerce")
    df["n_neighbors"] = pd.to_numeric(df["n_neighbors"], errors="coerce")
    df["min_cluster_size"] = pd.to_numeric(df["min_cluster_size"], errors="coerce")
    df = df.dropna(
        subset=[
            "trial_number",
            "f1_for_plot",
            "n_clusters",
            "noise_ratio",
            "n_neighbors",
            "min_cluster_size",
        ]
    )

    if df.empty:
        raise ValueError("No valid HPO rows available after numeric conversion.")

    return df, f1_col


def save_figure(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(path)
    return path


def best_trial(df: pd.DataFrame) -> pd.Series:
    return df.sort_values("f1_for_plot", ascending=False).iloc[0]


def plot_f1_vs_trial(df: pd.DataFrame, f1_label: str, best: pd.Series) -> Path:
    path = FIGURES_DIR / "f1_vs_trial.png"
    plt.figure(figsize=(8.5, 5))
    ordered = df.sort_values("trial_number")
    sns.lineplot(data=ordered, x="trial_number", y="f1_for_plot", marker="o")
    plt.scatter(
        best["trial_number"],
        best["f1_for_plot"],
        color="red",
        s=95,
        label=f"Best trial {int(best['trial_number'])}",
        zorder=5,
    )
    plt.xlabel("Trial number")
    plt.ylabel(f1_label.replace("_", "@").replace("f1@", "F1@"))
    plt.title("HPO F1 by Trial")
    plt.legend()
    return save_figure(path)


def plot_f1_vs_n_clusters(df: pd.DataFrame, f1_label: str, best: pd.Series) -> Path:
    path = FIGURES_DIR / "f1_vs_n_clusters.png"
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="n_clusters", y="f1_for_plot", s=70)
    plt.scatter(
        best["n_clusters"],
        best["f1_for_plot"],
        color="red",
        edgecolor="black",
        s=115,
        label=f"Best trial {int(best['trial_number'])}",
        zorder=5,
    )
    plt.xlabel("Number of clusters")
    plt.ylabel(f1_label.replace("_", "@").replace("f1@", "F1@"))
    plt.title("HPO F1 vs Number of Clusters")
    plt.legend()
    return save_figure(path)


def plot_f1_vs_noise_ratio(df: pd.DataFrame, f1_label: str, best: pd.Series) -> Path:
    path = FIGURES_DIR / "f1_vs_noise_ratio.png"
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="noise_ratio", y="f1_for_plot", s=70)
    plt.scatter(
        best["noise_ratio"],
        best["f1_for_plot"],
        color="red",
        edgecolor="black",
        s=115,
        label=f"Best trial {int(best['trial_number'])}",
        zorder=5,
    )
    plt.xlabel("Noise ratio")
    plt.ylabel(f1_label.replace("_", "@").replace("f1@", "F1@"))
    plt.title("HPO F1 vs Noise Ratio")
    plt.legend()
    return save_figure(path)


def plot_clusters_vs_noise_ratio(df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "clusters_vs_noise_ratio.png"
    plt.figure(figsize=(8.5, 5.5))
    scatter = plt.scatter(
        df["n_clusters"],
        df["noise_ratio"],
        c=df["f1_for_plot"],
        cmap="viridis",
        s=80,
        alpha=0.85,
    )
    plt.colorbar(scatter, label="F1")
    plt.xlabel("Number of clusters")
    plt.ylabel("Noise ratio")
    plt.title("HPO Cluster Count vs Noise Ratio")
    return save_figure(path)


def plot_heatmap_neighbors_cluster_size(df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "heatmap_neighbors_cluster_size.png"
    pivot = df.pivot_table(
        index="min_cluster_size",
        columns="n_neighbors",
        values="f1_for_plot",
        aggfunc="mean",
    ).sort_index()

    plt.figure(figsize=(7.5, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Mean F1"},
    )
    plt.xlabel("UMAP n_neighbors")
    plt.ylabel("HDBSCAN min_cluster_size")
    plt.title("Mean F1 by n_neighbors and min_cluster_size")
    return save_figure(path)


def save_top10_trials(df: pd.DataFrame, f1_col: str) -> Path:
    path = FIGURES_DIR / "top10_trials_table.csv"
    top10 = df.sort_values("f1_for_plot", ascending=False).head(10).copy()
    if f1_col != "f1_at_30":
        top10["metric_used_for_sorting"] = f1_col
    top10 = top10.drop(columns=["f1_for_plot"])
    top10.to_csv(path, index=False)
    print(path)
    return path


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df, f1_col = load_results()
    best = best_trial(df)

    print(f"Input file: {INPUT_FILE}")
    print(f"Rows loaded: {len(df)}")
    print("Generated files:")
    generated = [
        plot_f1_vs_trial(df, f1_col, best),
        plot_f1_vs_n_clusters(df, f1_col, best),
        plot_f1_vs_noise_ratio(df, f1_col, best),
        plot_clusters_vs_noise_ratio(df),
        plot_heatmap_neighbors_cluster_size(df),
        save_top10_trials(df, f1_col),
    ]

    print("Summary:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
