from __future__ import annotations

from pathlib import Path
import importlib.util
import shutil
import sys
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_MODE = "generalization_nov_jan"
VALID_RUN_MODES = {
    "smoke_zst_feb_2026",
    "final_aug_oct_2025",
    "generalization_nov_jan",
}

RUN_EXTRACTION = False
RUN_CLEANING = False
RUN_EMBEDDINGS = False
RUN_UMAP = True
RUN_HDBSCAN = True
RUN_CLUSTER_SUMMARY = True
RUN_CTFIDF = True
EXTRACTION_MAX_POSTS: int | None = None
SMOKE_TEST = False
SAMPLE_SIZE: int | None = None


def load_module(module_name: str, relative_path: str) -> ModuleType:
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


config_module = load_module("pipeline_config", "src/00_config/config.py")
extraction_module = load_module(
    "extract_reddit_months",
    "src/01_extraction/01_extract_reddit_months.py",
)
cleaning_module = load_module(
    "clean_for_embeddings",
    "src/02_cleaning/03_clean_for_embeddings.py",
)
embeddings_module = load_module(
    "generate_embeddings",
    "src/03_embeddings/04_generate_embeddings.py",
)
umap_module = load_module(
    "run_umap",
    "src/04_clustering/06_run_umap.py",
)
hdbscan_module = load_module(
    "run_hdbscan",
    "src/04_clustering/07_run_hdbscan.py",
)
cluster_summary_module = load_module(
    "cluster_summary",
    "src/05_labeling/11_cluster_summary.py",
)
ctfidf_labeling_module = load_module(
    "ctfidf_labeling",
    "src/05_labeling/12_ctfidf_labeling.py",
)

PipelineConfig = config_module.PipelineConfig
extract_months = extraction_module.extract_months
clean_for_embeddings = cleaning_module.clean_for_embeddings
generate_embeddings = embeddings_module.generate_embeddings
run_umap = umap_module.run_umap
run_hdbscan = hdbscan_module.run_hdbscan
build_cluster_summary = cluster_summary_module.build_cluster_summary
run_ctfidf_labeling = ctfidf_labeling_module.run_ctfidf_labeling


def configure_run() -> PipelineConfig:
    global RUN_EXTRACTION
    global RUN_CLEANING
    global RUN_EMBEDDINGS
    global RUN_UMAP
    global RUN_HDBSCAN
    global RUN_CLUSTER_SUMMARY
    global RUN_CTFIDF
    global EXTRACTION_MAX_POSTS
    global SMOKE_TEST
    global SAMPLE_SIZE

    if RUN_MODE not in VALID_RUN_MODES:
        raise ValueError(
            f"Invalid RUN_MODE={RUN_MODE!r}. "
            f"Expected one of: {sorted(VALID_RUN_MODES)}"
        )

    if RUN_MODE == "smoke_zst_feb_2026":
        RUN_EXTRACTION = True
        RUN_CLEANING = True
        RUN_EMBEDDINGS = True
        RUN_UMAP = True
        RUN_HDBSCAN = True
        RUN_CLUSTER_SUMMARY = True
        RUN_CTFIDF = True
        EXTRACTION_MAX_POSTS = 5000
        SMOKE_TEST = True
        SAMPLE_SIZE = 2000
        return PipelineConfig(
            start_month="2026-02",
            end_month="2026-02",
            output_run_name="test_feb_2026",
        )

    if RUN_MODE == "final_aug_oct_2025":
        RUN_EXTRACTION = False
        RUN_CLEANING = False
        RUN_EMBEDDINGS = False
        RUN_UMAP = True
        RUN_HDBSCAN = True
        RUN_CLUSTER_SUMMARY = True
        RUN_CTFIDF = True
        EXTRACTION_MAX_POSTS = None
        SMOKE_TEST = False
        SAMPLE_SIZE = None
        return PipelineConfig(
            start_month="2025-08",
            end_month="2025-10",
            output_run_name="aug_2025_oct_2025",
        )

    RUN_EXTRACTION = False
    RUN_CLEANING = False
    RUN_EMBEDDINGS = True
    RUN_UMAP = True
    RUN_HDBSCAN = True
    RUN_CLUSTER_SUMMARY = True
    RUN_CTFIDF = True
    EXTRACTION_MAX_POSTS = None
    SMOKE_TEST = False
    SAMPLE_SIZE = None
    return PipelineConfig(
        start_month="2025-11",
        end_month="2026-01",
        output_run_name="nov_2025_jan_2026",
    )


def enabled_steps() -> list[str]:
    steps = []
    if RUN_EXTRACTION:
        steps.append("extraction")
    if RUN_CLEANING:
        steps.append("cleaning")
    if RUN_EMBEDDINGS:
        steps.append("embeddings")
    if RUN_UMAP:
        steps.append("umap")
    if RUN_HDBSCAN:
        steps.append("hdbscan")
    if RUN_CLUSTER_SUMMARY:
        steps.append("cluster_summary")
    if RUN_CTFIDF:
        steps.append("ctfidf_labeling")
    return steps


def raw_parquet_inputs(cfg: PipelineConfig) -> list[Path]:
    return [
        cfg.raw_dir / f"reddit_tech_month_{month_tag}.parquet"
        for month_tag in cfg.month_tags
    ]


def ensure_file_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input for {description}:\n{path}"
        )


def ensure_files_exist(paths: list[Path], description: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required input files for {description}:\n"
            + "\n".join(str(path) for path in missing)
        )


def ensure_smoke_zst_source(cfg: PipelineConfig) -> None:
    if RUN_MODE != "smoke_zst_feb_2026":
        return

    expected = (
        cfg.project_root
        / "data"
        / "external"
        / "reddit"
        / "submissions"
        / "RS_2026-02.zst"
    )
    source = Path(
        r"C:\Users\Win10\Downloads\reddit\submissions\RS_2026-02.zst"
    )

    if expected.exists():
        print("Smoke .zst source found in project:")
        print(expected)
        return

    if source.exists():
        expected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, expected)
        print("Copied smoke .zst source into project:")
        print("From:", source)
        print("To:", expected)
        return

    raise FileNotFoundError(
        "Missing smoke source .zst file.\n"
        f"Expected project path: {expected}\n"
        f"Fallback source path: {source}"
    )


def print_configuration(cfg: PipelineConfig) -> None:
    print("Pipeline configuration")
    print("RUN_MODE:", RUN_MODE)
    print("Start month:", cfg.start_month)
    print("End month:", cfg.end_month)
    print("Run name:", cfg.output_run_name)
    print("Enabled steps:", ", ".join(enabled_steps()) or "none")
    print("Smoke test:", SMOKE_TEST)
    print("Extraction max posts:", EXTRACTION_MAX_POSTS)
    print("Sample size:", SAMPLE_SIZE)


def validate_generalization_cleaned_input(cfg: PipelineConfig) -> int:
    import pandas as pd

    ensure_file_exists(cfg.cleaned_parquet_path, "generalization embeddings")
    print("Generalization cleaned parquet:")
    print(cfg.cleaned_parquet_path)
    cleaned_rows = len(pd.read_parquet(cfg.cleaned_parquet_path))
    print("Cleaned parquet rows:", cleaned_rows)
    if cleaned_rows < 100_000:
        raise ValueError(
            "Generalization cleaned parquet is too small. "
            f"Expected at least 100000 rows, got {cleaned_rows}."
        )
    return cleaned_rows


def validate_generalization_embeddings(cfg: PipelineConfig, cleaned_rows: int) -> None:
    import numpy as np
    import pandas as pd

    ensure_file_exists(cfg.embeddings_path, "generalization embeddings validation")
    ensure_file_exists(cfg.metadata_path, "generalization metadata validation")

    embeddings = np.load(cfg.embeddings_path, mmap_mode="r")
    metadata = pd.read_parquet(cfg.metadata_path)

    print("Embeddings shape:", embeddings.shape)
    print("Metadata shape:", metadata.shape)

    if embeddings.shape[0] != cleaned_rows:
        raise ValueError(
            "Embeddings rows do not match cleaned parquet rows. "
            f"Embeddings rows: {embeddings.shape[0]}, cleaned rows: {cleaned_rows}."
        )

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            "Embeddings rows do not match metadata rows. "
            f"Embeddings rows: {embeddings.shape[0]}, metadata rows: {len(metadata)}."
        )


def validate_generalization_umap_input(cfg: PipelineConfig) -> None:
    import numpy as np

    ensure_file_exists(cfg.embeddings_path, "generalization UMAP")
    embeddings = np.load(cfg.embeddings_path, mmap_mode="r")
    print("UMAP input embeddings shape:", embeddings.shape)
    if embeddings.shape[0] <= 100_000:
        raise ValueError(
            "Generalization UMAP input is too small. "
            f"Expected more than 100000 rows, got {embeddings.shape[0]}."
        )


def main() -> None:
    cfg = configure_run()
    generalization_cleaned_rows: int | None = None

    print_configuration(cfg)

    if RUN_EXTRACTION:
        print("\n[1/7] Extraction")
        ensure_smoke_zst_source(cfg)
        extract_months(cfg, max_posts=EXTRACTION_MAX_POSTS)
        print("[1/7] Extraction completed")
    else:
        print("\n[1/7] Extraction skipped")

    if RUN_CLEANING:
        print("\n[2/7] Cleaning")
        ensure_files_exist(raw_parquet_inputs(cfg), "cleaning")
        clean_for_embeddings(cfg)
        print("[2/7] Cleaning completed")
    else:
        print("\n[2/7] Cleaning skipped")

    if RUN_EMBEDDINGS:
        print("\n[3/7] Embeddings")
        if RUN_MODE == "generalization_nov_jan":
            generalization_cleaned_rows = validate_generalization_cleaned_input(cfg)
        else:
            ensure_file_exists(cfg.cleaned_parquet_path, "embeddings")
        generate_embeddings(cfg, sample_size=SAMPLE_SIZE if SMOKE_TEST else None)
        if RUN_MODE == "generalization_nov_jan":
            if generalization_cleaned_rows is None:
                raise RuntimeError("Missing generalization cleaned row count.")
            validate_generalization_embeddings(cfg, generalization_cleaned_rows)
        print("[3/7] Embeddings completed")
    else:
        print("\n[3/7] Embeddings skipped")

    if RUN_UMAP:
        print("\n[4/7] UMAP")
        if RUN_MODE == "generalization_nov_jan":
            validate_generalization_umap_input(cfg)
        else:
            ensure_file_exists(cfg.embeddings_path, "UMAP")
        run_umap(cfg)
        print("[4/7] UMAP completed")
    else:
        print("\n[4/7] UMAP skipped")

    if RUN_HDBSCAN:
        print("\n[5/7] HDBSCAN")
        ensure_file_exists(cfg.umap_embeddings_path, "HDBSCAN")
        ensure_file_exists(cfg.metadata_path, "HDBSCAN")
        run_hdbscan(cfg)
        print("[5/7] HDBSCAN completed")
    else:
        print("\n[5/7] HDBSCAN skipped")

    if RUN_CLUSTER_SUMMARY:
        print("\n[6/7] Cluster summary")
        ensure_file_exists(cfg.posts_with_clusters_path, "cluster summary")
        build_cluster_summary(cfg)
        print("[6/7] Cluster summary completed")
    else:
        print("\n[6/7] Cluster summary skipped")

    if RUN_CTFIDF:
        print("\n[7/7] c-TF-IDF labeling and event ranking")
        ensure_file_exists(cfg.posts_with_clusters_path, "c-TF-IDF labeling")
        ensure_file_exists(cfg.cluster_summary_path, "c-TF-IDF labeling")
        run_ctfidf_labeling(cfg)
        print("[7/7] c-TF-IDF labeling and event ranking completed")
    else:
        print("\n[7/7] c-TF-IDF labeling and event ranking skipped")

    print(f"\nPipeline completed for RUN_MODE={RUN_MODE}")
    if RUN_MODE == "generalization_nov_jan":
        print("Generalization run completed on full Nov-Jan dataset.")


if __name__ == "__main__":
    main()
