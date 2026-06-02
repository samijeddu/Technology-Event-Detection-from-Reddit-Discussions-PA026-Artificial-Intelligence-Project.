# GitHub Submission Checklist

Use this checklist before creating the final GitHub commit.

## Include

- `README.md`
- `requirements.txt`
- `src/`
- `notebooks/AI_progect_masaryk.ipynb`
- `docs/final_report.md`
- `docs/final_report.pdf`, if exported locally
- `docs/figures/*.png`
- `docs/tables/*.csv`, if needed for the report
- `data/external/*.csv` ground-truth event tables

## Exclude

- raw Reddit dumps (`*.zst`)
- raw/interim/processed datasets
- parquet files
- embedding arrays (`*.npy`)
- generated HPO and pipeline outputs
- virtual environments
- Python cache files
- model/cache folders

## Final Checks

Run:

```powershell
git status --short
Get-ChildItem -Recurse -File | Where-Object { $_.Length -gt 10MB } | Select-Object FullName,Length
```

Do not run `git add`, `git commit`, or `git push` until the status output has been reviewed.
