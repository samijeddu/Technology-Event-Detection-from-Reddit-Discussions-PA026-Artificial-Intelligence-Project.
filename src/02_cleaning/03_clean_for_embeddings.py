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

MIN_TEXT_LENGTH_CHARS = 50
MIN_TEXT_LENGTH_WORDS = 10

EXCLUDED_SUBREDDITS = {
    "techsupport",
    "applehelp",
    "buildapc",
    "pcmasterrace",
}

LOW_QUALITY_PATTERN = (
    r"\b(?:aita|relationship|girlfriend|boyfriend|wife|husband|dating|sex|porn|"
    r"nsfw|roleplay|r4r|am i the asshole|tifu|venting|confession)\b"
)

RECURRING_THREAD_PATTERN = (
    r"\b(?:daily thread|weekly thread|monthly thread|daily advice thread|"
    r"support megathread|megathread|free-talk|free talk|simple questions|"
    r"daily discussion|weekly discussion|monthly discussion|daily support thread|"
    r"weekly support thread|question thread|discussion thread|daily questions|"
    r"weekly questions|daily help|weekly help)\b"
)

LOW_INFORMATION_TITLE_PATTERN = (
    r"^(?:help|help!|question|need help|need advice|advice|"
    r"please help|help needed|i need help|quick question|"
    r"what is this|what does this mean|is this normal|"
    r"is this true|how do i fix this|why does this happen|"
    r"what am i doing wrong|am i doing something wrong|"
    r"which one|curious|anyone else|is it just me|"
    r"where to start|performance issue|what do you think|hear me out)$"
)

SUPPORT_TROUBLESHOOTING_PATTERN = (
    r"\b(?:how do i fix|not working|stopped working|issue|problem|"
    r"trouble|broken|crash|crashes|freeze|frozen|black screen|"
    r"battery drain|overheating|password|icloud|backup|restore|"
    r"storage|passcode|connect to itunes|coil whine|artifacts|"
    r"bugged|stuck|can't authenticate|error|errors|out of memory)\b"
)

PURCHASE_ACCESSORY_PATTERN = (
    r"\b(?:which one|should i buy|recommend|recommendation|best case|"
    r"case recommendations|screen protector|charger|storage plan|"
    r"upgrade my iphone|what case|protective case|durable case|"
    r"phone case|cases|case suggestions|best thin case|slim cases|"
    r"best buy|preorder|return my|bought a|selling)\b"
)

CAREER_LEARNING_PATTERN = (
    r"\b(?:career growth|learn programming|learning programming|"
    r"where to start|certifications|placement|master'?s in data science|"
    r"switching languages|newbie|beginner|how to learn|study material|"
    r"starting a programming career|finance major trying to learn coding)\b"
)

GENERIC_BUSINESS_PATTERN = (
    r"\b(?:roast my landing page|roast my website|feedback for my website|"
    r"what are you building|self-promotion|biggest struggle|"
    r"mindset shift|passion project|success stories|failure or success|"
    r"saas founders|what saas|saas idea|cold email|email marketing|"
    r"looking for work|looking for advice|looking for a business partner|"
    r"test your idea|improve your strategy|share your startup|"
    r"first saas|building my saas|built my first saas)\b"
)

LINUX_DISTRO_SUPPORT_PATTERN = (
    r"\b(?:linux distro|best distro|switch to linux|switching to linux|"
    r"linux mint|arch btw|fedora|debian|manjaro|endeavour os|"
    r"new linux user|what distros|distro for a newbie)\b"
)

GPU_SUPPORT_PATTERN = (
    r"\b(?:gpu related|game related|global settings reset|power limit|"
    r"gsync|cpu bottleneck|driver update|nvidia driver|fps cap|"
    r"rx 9070xt|rtx 5070|rtx 5070ti|rtx 5080|rtx 5090|"
    r"9950x3d|idle around|furmark)\b"
)
# ==================================================


def input_files_for_months(cfg: PipelineConfig) -> list[Path]:
    """Return raw parquet inputs matching the configured inclusive month range."""
    return [
        cfg.raw_dir / f"reddit_tech_month_{month_tag}.parquet"
        for month_tag in cfg.month_tags
    ]


def clean_text_for_embeddings(text: str) -> str:
    """
    Light cleaning for sentence embeddings.
    Keeps product names, model versions and useful punctuation.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"\[removed\]|\[deleted\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"&amp;", " and ", text)
    text = re.sub(r"&lt;|&gt;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_removed_or_deleted(text: str) -> bool:
    """
    Detect fully removed or deleted text fields.
    """
    if not isinstance(text, str):
        return True

    text = text.strip().lower()
    return text in {"", "[removed]", "[deleted]", "removed", "deleted"}


def non_latin_ratio(text: str) -> float:
    """
    Approximate ratio of non-ASCII characters.
    Used only to remove obvious non-English spam/support clusters.
    """
    if not isinstance(text, str) or not text:
        return 0.0

    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0

    ascii_chars = sum(1 for c in chars if c.isascii())
    return 1 - (ascii_chars / len(chars))


def clean_for_embeddings(cfg: PipelineConfig):
    input_files = input_files_for_months(cfg)

    missing_files = [path for path in input_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing raw parquet files for configured months:\n"
            + "\n".join(str(path) for path in missing_files)
            + f"\nExpected raw directory: {cfg.raw_dir}"
        )

    print("Files found:")
    for f in input_files:
        print("-", f.name)

    dfs = []

    for input_file in input_files:
        print(f"\nReading file: {input_file.name}")
        df_part = pd.read_parquet(input_file)
        print("Rows:", len(df_part))
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)

    print("\nMerged rows:", len(df))

    required_cols = {
        "id",
        "created_utc",
        "created_dt",
        "subreddit",
        "title",
        "selftext",
        "full_text",
    }

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # --------------------------------------------------
    # Datetime columns
    # --------------------------------------------------
    print("Parsing datetime columns...")
    df["created_dt"] = pd.to_datetime(df["created_dt"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_dt"]).copy()
    df["date"] = df["created_dt"].dt.floor("D")

    # --------------------------------------------------
    # Normalize fields
    # --------------------------------------------------
    print("Normalizing text fields...")
    df["subreddit"] = df["subreddit"].fillna("").astype(str)
    df["subreddit_norm"] = df["subreddit"].str.lower().str.strip()

    df["title"] = df["title"].fillna("").astype(str)
    df["selftext"] = df["selftext"].fillna("").astype(str)

    # --------------------------------------------------
    # Remove noisy support-heavy subreddits
    # --------------------------------------------------
    print("Removing noisy subreddits...")
    before = len(df)
    excluded = {s.lower().strip() for s in EXCLUDED_SUBREDDITS}
    df = df[~df["subreddit_norm"].isin(excluded)].copy()
    print("Removed noisy subreddit rows:", before - len(df))
    print("Rows after subreddit filtering:", len(df))

    # --------------------------------------------------
    # Remove fully deleted / removed posts
    # Keep posts with useful titles even if selftext is removed.
    # --------------------------------------------------
    print("Removing fully deleted/removed posts...")
    before = len(df)

    mask_fully_removed = (
        df["title"].apply(is_removed_or_deleted) &
        df["selftext"].apply(is_removed_or_deleted)
    )

    df = df[~mask_fully_removed].copy()
    print("Removed fully deleted/removed rows:", before - len(df))
    print("Rows after removed/deleted filtering:", len(df))

    # --------------------------------------------------
    # Rebuild full_text
    # --------------------------------------------------
    print("Rebuilding full_text...")
    df["full_text"] = (df["title"] + " " + df["selftext"]).str.strip()
    df = df[df["full_text"].str.len() > 0].copy()
    print("After removing empty full_text:", len(df))

    # --------------------------------------------------
    # Remove recurring / templated community threads
    # --------------------------------------------------
    print("Removing recurring community threads...")
    before = len(df)

    mask_recurring_threads = (
        df["title"].str.contains(
            RECURRING_THREAD_PATTERN,
            flags=re.IGNORECASE,
            regex=True,
            na=False
        )
        |
        df["full_text"].str.contains(
            RECURRING_THREAD_PATTERN,
            flags=re.IGNORECASE,
            regex=True,
            na=False
        )
    )

    df = df[~mask_recurring_threads].copy()
    print("Removed recurring thread rows:", before - len(df))
    print("Rows after recurring thread filtering:", len(df))

    # --------------------------------------------------
    # Remove low-quality textual content
    # --------------------------------------------------
    print("Removing low-quality text...")
    before = len(df)

    mask_low_quality = df["full_text"].str.contains(
        LOW_QUALITY_PATTERN,
        flags=re.IGNORECASE,
        regex=True,
        na=False
    )

    df = df[~mask_low_quality].copy()
    print("Removed low-quality rows:", before - len(df))
    print("Rows after low-quality filtering:", len(df))

    # --------------------------------------------------
    # Remove low-information / support / generic discussion posts
    # --------------------------------------------------
    print("Removing low-information and support-like posts...")
    before = len(df)

    mask_low_information_title = df["title"].str.contains(
        LOW_INFORMATION_TITLE_PATTERN,
        flags=re.IGNORECASE,
        regex=True,
        na=False
    )

    mask_support_troubleshooting = df["full_text"].str.contains(
        SUPPORT_TROUBLESHOOTING_PATTERN,
        flags=re.IGNORECASE,
        regex=True,
        na=False
    )

    mask_purchase_accessory = df["full_text"].str.contains(
        PURCHASE_ACCESSORY_PATTERN,
        flags=re.IGNORECASE,
        regex=True,
        na=False
    )

    mask_career_learning = df["full_text"].str.contains(
        CAREER_LEARNING_PATTERN,
        flags=re.IGNORECASE,
        regex=True,
        na=False
    )

    mask_generic_business = df["full_text"].str.contains(
        GENERIC_BUSINESS_PATTERN,
        flags=re.IGNORECASE,
        regex=True,
        na=False
    )

    mask_linux_support = df["full_text"].str.contains(
        LINUX_DISTRO_SUPPORT_PATTERN,
        flags=re.IGNORECASE,
        regex=True,
        na=False
    )

    mask_gpu_support = df["full_text"].str.contains(
        GPU_SUPPORT_PATTERN,
        flags=re.IGNORECASE,
        regex=True,
        na=False
    )

    df["non_latin_ratio"] = df["full_text"].apply(non_latin_ratio)
    mask_high_non_latin = df["non_latin_ratio"] > 0.35

    mask_extra_noise = (
        mask_low_information_title |
        mask_support_troubleshooting |
        mask_purchase_accessory |
        mask_career_learning |
        mask_generic_business |
        mask_linux_support |
        mask_gpu_support |
        mask_high_non_latin
    )

    df = df[~mask_extra_noise].copy()

    print("Removed low-information/support rows:", before - len(df))
    print("Rows after extra noise filtering:", len(df))

    # --------------------------------------------------
    # Clean text for embeddings
    # --------------------------------------------------
    print("Cleaning text for embeddings...")
    df["text_for_embedding"] = df["full_text"].apply(clean_text_for_embeddings)

    df = df[df["text_for_embedding"].str.len() > 0].copy()
    print("After removing empty text_for_embedding:", len(df))

    # --------------------------------------------------
    # Length features
    # --------------------------------------------------
    print("Computing text lengths...")
    df["text_length_chars"] = df["text_for_embedding"].str.len()
    df["text_length_words"] = df["text_for_embedding"].str.count(r"\S+")

    before = len(df)
    df = df[
        (df["text_length_chars"] >= MIN_TEXT_LENGTH_CHARS) &
        (df["text_length_words"] >= MIN_TEXT_LENGTH_WORDS)
    ].copy()

    print("Removed short text rows:", before - len(df))
    print("After short text filtering:", len(df))

    # --------------------------------------------------
    # Remove duplicates by id
    # --------------------------------------------------
    print("Removing duplicate ids...")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["id"]).copy()
    print("Removed duplicate ids:", before_dedup - len(df))

    # --------------------------------------------------
    # Remove duplicate text
    # --------------------------------------------------
    print("Removing duplicate texts...")
    before_text_dedup = len(df)
    df = df.drop_duplicates(subset=["text_for_embedding"]).copy()
    print("Removed duplicate texts:", before_text_dedup - len(df))

    # --------------------------------------------------
    # Numeric columns
    # --------------------------------------------------
    print("Converting numeric columns...")
    for col in ["score", "num_comments", "upvote_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------
    # Keep relevant columns
    # --------------------------------------------------
    keep_cols = [
        "id",
        "created_utc",
        "created_dt",
        "date",
        "source_month",
        "subreddit",
        "subreddit_norm",
        "title",
        "selftext",
        "full_text",
        "text_for_embedding",
        "text_length_chars",
        "text_length_words",
        "author",
        "score",
        "num_comments",
        "upvote_ratio",
        "url",
        "permalink",
        "domain",
    ]

    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols].copy()

    # --------------------------------------------------
    # Sort rows
    # --------------------------------------------------
    print("Sorting rows...")
    df = df.sort_values(["created_dt", "id"]).reset_index(drop=True)

    # --------------------------------------------------
    # Save output
    # --------------------------------------------------
    cfg.run_interim_dir.mkdir(parents=True, exist_ok=True)

    print("Saving parquet...")
    df.to_parquet(cfg.cleaned_parquet_path, index=False)

    print("\nSaved:", cfg.cleaned_parquet_path)
    print("Final shape:", df.shape)

    print("\nDate range:")
    print(df["created_dt"].min(), "->", df["created_dt"].max())

    print("\nTop subreddits:")
    print(df["subreddit"].value_counts().head(20))

    print("\nPreview:")
    preview_cols = ["created_dt", "subreddit", "title", "text_for_embedding"]
    print(df[preview_cols].head(5))


def process_all_files(cfg: PipelineConfig | None = None):
    clean_for_embeddings(cfg or DEFAULT_CONFIG)


def main(cfg: PipelineConfig | None = None):
    clean_for_embeddings(cfg or DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
