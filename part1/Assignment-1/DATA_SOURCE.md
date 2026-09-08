# Data Source and Reproduction Contract

## Source

- **Title:** 1982–2022 NBA Player Statistics with MVP Votes
- **Publisher:** Robert Sunderhaft
- **Kaggle URL:** https://www.kaggle.com/datasets/robertsunderhaft/nba-player-season-statistics-with-mvp-win-share
- **Expected member:** `NBA_Dataset.csv`
- **Observed coverage:** 17,697 rows, 55 columns, seasons 1982–2022
- **Target:** `award_share`
- **SHA-256 of the uploaded ZIP used for this analysis:** `f012ada68853a8fa62a72d01b25283866d874184943e9af3448e67f2351c3637`

The dataset is intentionally not committed to the repository. Download it directly from Kaggle and review the license/usage terms displayed by Kaggle before use or redistribution.

## Download

With a configured Kaggle API client:

```bash
mkdir -p data/raw
kaggle datasets download \
  -d robertsunderhaft/nba-player-season-statistics-with-mvp-win-share \
  -p data/raw
```

Rename the downloaded ZIP to the repository’s expected filename:

```bash
mv data/raw/nba-player-season-statistics-with-mvp-win-share.zip \
  data/raw/nba_mvp_stats.zip
```

Alternatively, download the ZIP in a browser and save it as `data/raw/nba_mvp_stats.zip`.

## Verify

```bash
python src/future_data_schema_audit.py data/raw/nba_mvp_stats.zip \
  --output results/source_schema_audit.json
```

The original source passes the model-input and logical-range gates with one review warning: two rows share the 1989/Charles Jones/WSB key. They are not exact duplicates and are retained. The audit also reports 1,954 `TOT` rows and 44 rows involved in repeated player-season names; these are investigated rather than automatically deleted.

If your checksum differs, do not assume the data are invalid: Kaggle may have published a new version. Record the dataset version and rerun the full schema and overlap audit before comparing results.
