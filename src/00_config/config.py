from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PipelineConfig:
    """Centralized configuration for a single monthly pipeline run."""

    start_month: str
    end_month: str
    output_run_name: str

    # Resolve the repository root from this file location:
    # src/00_config/config.py -> project root.
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    def __post_init__(self) -> None:
        self._validate_month_range()

    @property
    def raw_dir(self) -> Path:
        """Directory containing raw monthly Reddit parquet files."""
        return self.project_root / "data" / "raw"

    @property
    def interim_dir(self) -> Path:
        """Directory for intermediate artifacts shared across pipeline stages."""
        return self.project_root / "data" / "interim"

    @property
    def processed_dir(self) -> Path:
        """Directory for processed, run-specific pipeline artifacts."""
        return self.project_root / "data" / "processed"

    @property
    def outputs_dir(self) -> Path:
        """Directory for analysis outputs, plots, validation files and reports."""
        return self.project_root / "outputs"

    @property
    def run_interim_dir(self) -> Path:
        """Intermediate directory scoped to this configured run."""
        return self.interim_dir / self.output_run_name

    @property
    def run_processed_dir(self) -> Path:
        """Processed-data directory scoped to this configured run."""
        return self.processed_dir / self.output_run_name

    @property
    def run_outputs_dir(self) -> Path:
        """Output directory scoped to this configured run."""
        return self.outputs_dir / self.output_run_name

    @property
    def cleaned_parquet_path(self) -> Path:
        return self.run_interim_dir / "reddit_tech_cleaned_for_embeddings.parquet"

    @property
    def embeddings_path(self) -> Path:
        return self.run_processed_dir / "reddit_tech_embeddings.npy"

    @property
    def metadata_path(self) -> Path:
        return self.run_processed_dir / "reddit_tech_metadata.parquet"

    @property
    def umap_embeddings_path(self) -> Path:
        return self.run_processed_dir / "reddit_tech_umap_embeddings.npy"

    @property
    def posts_with_clusters_path(self) -> Path:
        return self.run_processed_dir / "reddit_tech_posts_with_clusters.parquet"

    @property
    def cluster_labels_path(self) -> Path:
        return self.run_processed_dir / "reddit_tech_cluster_labels.npy"

    @property
    def cluster_summary_path(self) -> Path:
        return self.run_processed_dir / "reddit_tech_cluster_summary.csv"

    @property
    def month_list(self) -> List[str]:
        """Inclusive list of months in YYYY-MM format."""
        start = self._parse_month(self.start_month)
        end = self._parse_month(self.end_month)

        months: List[str] = []
        year = start.year
        month = start.month

        while (year, month) <= (end.year, end.month):
            months.append(f"{year:04d}-{month:02d}")
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1

        return months

    @property
    def month_tags(self) -> List[str]:
        """Inclusive list of months converted to YYYY_MM format."""
        return [self.month_to_tag(month) for month in self.month_list]

    @staticmethod
    def month_to_tag(month: str) -> str:
        """Convert a month string from YYYY-MM to YYYY_MM."""
        parsed = PipelineConfig._parse_month(month)
        return f"{parsed.year:04d}_{parsed.month:02d}"

    @staticmethod
    def _parse_month(month: str) -> date:
        try:
            year_str, month_str = month.split("-", maxsplit=1)
            year = int(year_str)
            month_num = int(month_str)
            return date(year, month_num, 1)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid month '{month}'. Expected format is YYYY-MM."
            ) from exc

    def _validate_month_range(self) -> None:
        start = self._parse_month(self.start_month)
        end = self._parse_month(self.end_month)

        if start > end:
            raise ValueError(
                "Invalid month range: start_month must be earlier than or equal "
                "to end_month."
            )

        if not self.output_run_name or not self.output_run_name.strip():
            raise ValueError("output_run_name must be a non-empty string.")
