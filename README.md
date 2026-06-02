# Technology Event Detection from Reddit Discussions

Final project for **PA026 Artificial Intelligence Project**.

This repository contains an unsupervised pipeline for detecting emerging technology events from Reddit discussions. The system uses sentence embeddings, UMAP dimensionality reduction, HDBSCAN clustering, c-TF-IDF cluster labeling, S4 event scoring, automatic event deduplication, grid search hyperparameter optimization, and automatic evaluation against a pseudo-ground-truth set of real-world technology events.

The project is an event discovery pipeline, not a supervised classifier. Its goal is to surface candidate technology events from noisy social discussions and rank them for inspection.

## Repository

GitHub:
[Source code repository](https://github.com/samijeddu/Technology-Event-Detection-from-Reddit-Discussions-PA026-Artificial-Intelligence-Project.)

This repository contains:

- Source code
- Evaluation scripts
- Hyperparameter optimization experiments
- Documentation
- Final report figures
- Reproducibility instructions

## Dataset Download

The original Reddit Pushshift `.zst` dumps are not stored in this repository due to their size.

They can be downloaded from:

https://academictorrents.com/details/c5ba00048236b60f819dbf010e9034d24fc291fb

After downloading the dumps, the extraction pipeline can be executed starting from:

```text
src/01_extraction/
```

The smoke-test mode can be executed on a single downloaded month to verify the full pipeline from raw dump extraction through event candidate generation.

## Pipeline Overview

The pipeline supports both raw Reddit `.zst` dumps and existing parquet/embedding artifacts.

1. **Input data**
   - Raw `.zst` Reddit submission dumps for smoke testing.
   - Existing parquet and embedding files for final and generalization runs.
2. **Cleaning**
   - Filters technology-related Reddit posts.
   - Builds text fields used for embedding.
3. **Sentence embeddings**
   - Generates 384-dimensional sentence-transformer embeddings.
4. **UMAP**
   - Reduces embeddings to 15 dimensions using the selected final parameters.
5. **HDBSCAN**
   - Clusters UMAP embeddings and identifies noise points.
6. **Cluster summary**
   - Computes cluster size, temporal peak statistics, subreddit distribution, keywords, and sample titles.
7. **c-TF-IDF labeling**
   - Generates interpretable cluster labels and top terms.
8. **S4 event scoring**
   - Scores candidates using temporal concentration, cluster size, and engagement.
9. **Deduplication**
   - Merges near-duplicate event candidates using label/term Jaccard similarity and peak-date proximity.
10. **Top-30 event candidates**
   - Produces a deduplicated ranked top-30 list.
11. **Evaluation**
   - Matches top-30 candidates against a pseudo-ground-truth list of real-world technology events using semantic similarity.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

On Unix-like systems:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How To Run

Execution is controlled in `main.py` through:

```python
RUN_MODE = "final_aug_oct_2025"
```

Available modes:

- `smoke_zst_feb_2026`
  - Starts from raw `RS_2026-02.zst`.
  - Runs extraction, cleaning, embeddings, UMAP, HDBSCAN, cluster summary, and c-TF-IDF.
  - Intended only to validate that the full raw-dump pipeline works end to end.

- `final_aug_oct_2025`
  - Uses existing parquet and embedding files for August-October 2025.
  - Regenerates UMAP, HDBSCAN, cluster summaries, labels, S4 ranking, and final candidates using the best grid-search parameters.
  - This is the final result mode.

- `generalization_nov_jan`
  - Uses the cleaned parquet for November 2025-January 2026 and regenerates embeddings, UMAP, HDBSCAN, cluster summaries, c-TF-IDF labels, S4 ranking, and final candidates.
  - Applies the Aug-Oct best hyperparameters without modification to a temporally separate dataset.
  - Intended as the main generalization test.

To run any mode:

1. Open `main.py`.
2. Set the desired `RUN_MODE`.
3. Run `python main.py`.

Smoke run from raw `.zst`:

```python
RUN_MODE = "smoke_zst_feb_2026"
```

```powershell
python main.py
```

Final Aug-Oct run:

```python
RUN_MODE = "final_aug_oct_2025"
```

```powershell
python main.py
```

Generalization Nov-Jan run:

```python
RUN_MODE = "generalization_nov_jan"
```

```powershell
python main.py
```

## Data Availability

Large files are intentionally not included in Git:

- raw Reddit `.zst` dumps
- raw monthly parquet files
- cleaned/interim parquet files
- sentence embeddings (`.npy`)
- UMAP embeddings
- generated output directories

The expected local data layout is described below.

## Expected Folder Structure

```text
AI_Progect_final_version/
  main.py
  requirements.txt
  README.md
  docs/
    final_report.md
  notebooks/
    AI_progect_masaryk.ipynb
  src/
    00_config/
    01_extraction/
    02_cleaning/
    03_embeddings/
    04_clustering/
    05_labeling/
    07_validation/
    10_comparison/
  data/
    external/
      reddit/
        submissions/
          RS_2026-02.zst
      real_events_aug_oct_2025_top30.csv
    raw/
    interim/
    processed/
  outputs/
    aug_2025_oct_2025/
    grid_search_s4_top30_threshold055/
```

## Final Parameters

Best grid-search configuration:

| Component | Parameter | Value |
|---|---:|---:|
| UMAP | `n_neighbors` | 10 |
| UMAP | `n_components` | 15 |
| UMAP | `min_dist` | 0.03 |
| HDBSCAN | `min_cluster_size` | 100 |
| HDBSCAN | `min_samples` | 15 |

## Final Results Summary

Grid search setup:

| Item | Value |
|---|---:|
| Search strategy | 50 selected experiments from 144 possible combinations |
| Scoring | S4 scoring |
| Candidate processing | Deduplication + top30 event candidates |
| Matching threshold | 0.55 |

Main evaluation dataset:

| Item | Value |
|---|---:|
| `RUN_MODE` | `final_aug_oct_2025` |
| Dataset | Aug 2025-Oct 2025 |
| Posts | 165,733 |
| Embeddings shape | `(165733, 384)` |
| UMAP shape | `(165733, 15)` |
| HDBSCAN clusters | 210 |
| Noise points | 87,447 |
| Noise ratio | 0.5276 |

Best grid-search evaluation:

| Metric | Value |
|---|---:|
| F1@30 | 0.3111 |
| Precision@30 | 0.2333 |
| Recall@30 | 0.4667 |
| Matched real events | 14 / 30 |
| Number of clusters | 210 |
| Noise ratio | 0.5276 |
| Silhouette score | 0.4947 |

Results comparison:

| Dataset | Posts | Clusters | Noise ratio | Precision@30 | Recall@30 | F1@30 | Matched events |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aug 2025-Oct 2025 | 165,733 | 210 | 0.5276 | 0.2333 | 0.4667 | 0.3111 | 14 / 30 |
| Nov 2025-Jan 2026 | 140,669 | 187 | 0.5429 | 0.2000 | 0.3000 | 0.2400 | 9 / 30 |

Smoke test:

| Item | Value |
|---|---:|
| `RUN_MODE` | `smoke_zst_feb_2026` |
| Raw file | `RS_2026-02.zst` |
| Scanned posts | 1,731,207 |
| Kept posts | 5,000 |
| Cleaned posts | 1,980 |
| Embeddings shape | `(1980, 384)` |
| Clusters | 3 |
| Result | Pipeline completed successfully from raw `.zst` |

Generalization run:

| Item | Value |
|---|---:|
| `RUN_MODE` | `generalization_nov_jan` |
| Dataset | Nov 2025-Jan 2026 |
| Date range | 2025-11-01 to 2026-01-31 |
| Cleaned posts | 140,669 |
| Posts | 140,669 |
| Embeddings shape | `(140669, 384)` |
| UMAP shape | `(140669, 15)` |
| HDBSCAN clusters | 187 |
| Noise points | 76,370 |
| Noise ratio | 0.5429 |
| Precision@30 | 0.2000 |
| Recall@30 | 0.3000 |
| F1@30 | 0.2400 |
| Matched events | 9 / 30 |
| Note | Full temporally separated generalization test using Aug-Oct best parameters |

## Limitations

- Evaluation uses automatic semantic matching against a pseudo-ground-truth event list, not manual expert validation.
- Ground truth construction is manual and can affect measured performance.
- The system is unsupervised and ranks event candidates; it is not a perfect event classifier.
- The method is unsupervised event discovery, not supervised event classification.
- Reddit discussions are noisy, duplicated, and biased toward active subreddits.
- HDBSCAN can either over-fragment events or merge broad themes depending on density parameters.
- Deduplication uses simple label/term similarity and peak-date proximity; it may miss semantic duplicates with different wording.
- Some real events are broader than Reddit cluster labels, which makes semantic matching imperfect.
- The generalization run is temporally separated and supports robustness analysis, but it is still evaluated automatically and should be complemented with manual inspection.

## Main Outputs

Typical final output files:

```text
outputs/aug_2025_oct_2025/temporal_analysis/event_candidates_ranked.csv
outputs/aug_2025_oct_2025/temporal_analysis/event_candidates_ranked_deduplicated.csv
outputs/aug_2025_oct_2025/temporal_analysis/event_candidates_top30_deduplicated.csv
outputs/aug_2025_oct_2025/evaluation/
outputs/nov_2025_jan_2026/evaluation/
outputs/grid_search_s4_top30_threshold055/grid_search_results.csv
outputs/grid_search_s4_top30_threshold055/best_params.json
```
