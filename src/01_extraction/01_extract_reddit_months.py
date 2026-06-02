from pathlib import Path
import importlib.util
import json
import sys
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

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

# Broad technology-related subreddit filter.
# This defines the domain, but does NOT detect events.
TECH_SUBREDDITS = {
    # General technology
    "technology",
    "tech",
    "gadgets",
    "futurology",

    # AI / ML / Data Science
    "artificial",
    "artificialinteligence",
    "artificialintelligence",
    "machinelearning",
    "datascience",
    "singularity",
    "openai",
    "chatgpt",
    "localllama",
    "claudeai",
    "geminiai",
    "bard",
    "generativeai",

    # Programming / software
    "programming",
    "learnprogramming",
    "webdev",
    "software",
    "softwareengineering",
    "coding",
    "python",
    "javascript",
    "linux",
    "opensource",

    # Hardware / chips
    "hardware",
    "nvidia",
    "amd",
    "intel",
    "pcmasterrace",
    "buildapc",

    # Big tech / products
    "apple",
    "iphone",
    "google",
    "microsoft",
    "android",
    "windows",
    "tesla",
    "spacex",

    # Startups / business tech
    "startups",
    "saas",
    "entrepreneur",
}

TECH_SUBREDDITS = {s.lower().strip() for s in TECH_SUBREDDITS}

# Negative filtering:
# remove clearly irrelevant/noisy submissions.
LOW_QUALITY_TEXT_KEYWORDS = [
    "nsfw",
    "porn",
    "sex",
    "dating",
    "relationship",
    "girlfriend",
    "boyfriend",
    "wife",
    "husband",
    "roleplay",
    "r4r",
    "aita",
    "am i the asshole",
    "confession",
    "tifu",
]

EXCLUDED_SUBREDDITS = {
    "dirtypenpals",
    "dirtyr4r",
    "kinktown",
    "indianroleplay",
    "dirtyredditchat",
    "dirtychatpals",
    "roleplaydiscord",
    "nsfw_roleplay",
    "18above_roleplay",
    "r4r",
    "pajerosbi",
    "limitlessrp",
}

EXCLUDED_SUBREDDITS = {s.lower().strip() for s in EXCLUDED_SUBREDDITS}

BATCH_SIZE = 50_000
LOG_EVERY = 500_000
CHUNK_SIZE = 1 << 20  # 1 MB

# ==================================================


def reddit_submissions_dir(cfg: PipelineConfig) -> Path:
    """
    Location for source Reddit .zst dumps inside the project tree.

    Expected filenames follow the Reddit submissions convention:
    RS_YYYY-MM.zst.
    """
    return cfg.project_root / "data" / "external" / "reddit" / "submissions"


def input_files_for_months(cfg: PipelineConfig) -> list[Path]:
    """Return the .zst files matching the configured inclusive month range."""
    input_dir = reddit_submissions_dir(cfg)
    return [input_dir / f"RS_{month}.zst" for month in cfg.month_list]


def stream_zst_jsonl(path: Path, chunk_size: int = CHUNK_SIZE):
    """
    Stream JSON lines from a .zst Reddit dump using low memory.
    """
    dctx = zstd.ZstdDecompressor()

    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buffer = ""

            while True:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break

                buffer += chunk.decode("utf-8", errors="ignore")
                lines = buffer.split("\n")
                buffer = lines.pop()

                for line in lines:
                    line = line.strip()
                    if line:
                        yield line

            tail = buffer.strip()
            if tail:
                yield tail


def month_tag_from_filename(path: Path) -> str:
    """
    Example:
    RS_2025-08.zst -> 2025_08
    """
    return path.stem.replace("RS_", "").replace("-", "_")


def is_tech_subreddit(subreddit: str) -> bool:
    """
    Keep only posts from broad technology-related subreddits.
    This defines the domain but does not detect events.
    """
    if not isinstance(subreddit, str):
        return False

    subreddit_norm = subreddit.lower().strip()
    return subreddit_norm in TECH_SUBREDDITS


def contains_low_quality_text(text: str) -> bool:
    """
    Negative text filter.
    Removes obvious noise, but does not select specific events.
    """
    if not isinstance(text, str):
        return False

    text = text.lower()
    return any(keyword in text for keyword in LOW_QUALITY_TEXT_KEYWORDS)


def is_excluded_subreddit(subreddit: str) -> bool:
    """
    Remove known noisy subreddits.
    """
    if not isinstance(subreddit, str):
        return False

    subreddit_norm = subreddit.lower().strip()
    return subreddit_norm in EXCLUDED_SUBREDDITS


def process_file(input_file: Path, output_dir: Path, max_posts: int | None = None):
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    start_time = time.time()
    month_tag = month_tag_from_filename(input_file)
    output_file = output_dir / f"reddit_tech_month_{month_tag}.parquet"

    output_dir.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        output_file.unlink()

    print(f"\nProcessing file: {input_file.name}")
    print(f"Output file    : {output_file}")

    parquet_writer = None
    batch = []

    total_scanned = 0
    total_kept = 0
    total_skipped_subreddit = 0
    total_skipped_low_quality = 0
    total_parse_errors = 0

    def flush_batch(rows):
        nonlocal parquet_writer

        if not rows:
            return

        df = pd.DataFrame(rows)

        df["created_dt"] = pd.to_datetime(
            df["created_utc"],
            unit="s",
            utc=True,
            errors="coerce"
        )

        table = pa.Table.from_pandas(df, preserve_index=False)

        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(
                str(output_file),
                table.schema,
                compression="snappy"
            )

        parquet_writer.write_table(table)

    for line in stream_zst_jsonl(input_file):
        total_scanned += 1

        if total_scanned % LOG_EVERY == 0:
            print(
                f"[{input_file.name}] "
                f"Scanned: {total_scanned:,} | "
                f"Kept: {total_kept:,} | "
                f"Skipped subreddit: {total_skipped_subreddit:,} | "
                f"Skipped noise: {total_skipped_low_quality:,}",
                flush=True
            )

        try:
            post = json.loads(line)
        except Exception:
            total_parse_errors += 1
            continue

        subreddit = post.get("subreddit", "") or ""
        subreddit_norm = str(subreddit).lower().strip()

        # --------------------------------------------------
        # 1. Broad subreddit-domain filter
        # --------------------------------------------------
        if not is_tech_subreddit(subreddit):
            total_skipped_subreddit += 1
            continue

        # --------------------------------------------------
        # 2. Excluded noisy subreddits
        # --------------------------------------------------
        if is_excluded_subreddit(subreddit):
            total_skipped_subreddit += 1
            continue

        title = post.get("title", "") or ""
        selftext = post.get("selftext", "") or ""
        full_text = f"{title} {selftext}".strip()

        # --------------------------------------------------
        # 3. Negative text filter
        # --------------------------------------------------
        if contains_low_quality_text(full_text):
            total_skipped_low_quality += 1
            continue

        if not full_text:
            total_skipped_low_quality += 1
            continue

        total_kept += 1

        batch.append({
            "id": post.get("id"),
            "created_utc": post.get("created_utc"),
            "subreddit": subreddit,
            "subreddit_norm": subreddit_norm,
            "title": title,
            "selftext": selftext,
            "full_text": full_text,
            "author": post.get("author"),
            "score": post.get("score"),
            "num_comments": post.get("num_comments"),
            "upvote_ratio": post.get("upvote_ratio"),
            "url": post.get("url"),
            "permalink": post.get("permalink"),
            "domain": post.get("domain"),
            "source_month": month_tag,
        })

        if len(batch) >= BATCH_SIZE:
            flush_batch(batch)
            batch = []
            print(
                f"Wrote chunk | "
                f"File: {input_file.name} | "
                f"Kept: {total_kept:,} | "
                f"Scanned: {total_scanned:,}",
                flush=True
            )

        if max_posts is not None and total_kept >= max_posts:
            print(f"Reached max_posts={max_posts:,}. Stopping extraction.")
            break

    if batch:
        flush_batch(batch)

    if parquet_writer is not None:
        parquet_writer.close()

    print("\nDONE")
    print(f"File                  : {input_file.name}")
    print(f"Total scanned          : {total_scanned:,}")
    print(f"Total kept             : {total_kept:,}")
    print(f"Skipped by subreddit   : {total_skipped_subreddit:,}")
    print(f"Skipped low-quality    : {total_skipped_low_quality:,}")
    print(f"Parse errors           : {total_parse_errors:,}")
    print(f"Saved to               : {output_file}")
    print(f"Posts read             : {total_scanned:,}")
    print(f"Posts saved            : {total_kept:,}")
    print(f"Total time seconds     : {time.time() - start_time:.2f}")

    return total_scanned, total_kept


def extract_months(cfg: PipelineConfig, max_posts: int | None = None):
    if max_posts is not None:
        if max_posts <= 0:
            raise ValueError("max_posts must be a positive integer or None.")
        print("Smoke extraction mode enabled")
        print(f"Max posts: {max_posts}")

    start_time = time.time()
    input_dir = reddit_submissions_dir(cfg)
    input_files = input_files_for_months(cfg)

    missing_files = [path for path in input_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing input .zst files for configured months:\n"
            + "\n".join(str(path) for path in missing_files)
            + f"\nExpected source directory: {input_dir}"
        )

    print("Files found:")
    for f in input_files:
        print("-", f.name)

    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    total_read = 0
    total_saved = 0

    for input_file in input_files:
        remaining_posts = None
        if max_posts is not None:
            remaining_posts = max_posts - total_saved
            if remaining_posts <= 0:
                break

        posts_read, posts_saved = process_file(
            input_file,
            cfg.raw_dir,
            max_posts=remaining_posts,
        )
        total_read += posts_read
        total_saved += posts_saved

    print("\nExtraction summary")
    print(f"Posts read          : {total_read:,}")
    print(f"Posts saved         : {total_saved:,}")
    print(f"Total time seconds  : {time.time() - start_time:.2f}")


def main(cfg: PipelineConfig | None = None):
    extract_months(cfg or DEFAULT_CONFIG, max_posts=None)


if __name__ == "__main__":
    main()
