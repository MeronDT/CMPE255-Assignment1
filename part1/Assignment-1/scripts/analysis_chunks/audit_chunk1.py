import os
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Reproducible display and plotting settings
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")
sns.set_theme(style="whitegrid", context="notebook")

ZIP_PATH = Path(os.environ.get("NBA_MVP_DATA_PATH", "data/raw/nba_mvp_stats.zip"))
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk1_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


section("1. SOFTWARE VERSIONS")
print(f"pandas:     {pd.__version__}")
print(f"NumPy:      {np.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"seaborn:    {sns.__version__}")

section("2. ARCHIVE CONTENTS AND DATA IMPORT")
with ZipFile(ZIP_PATH) as archive:
    members = [m for m in archive.namelist() if not m.endswith("/")]
    csv_members = [m for m in members if m.lower().endswith(".csv")]
    print("Archive members:", members)
    if len(csv_members) != 1:
        raise ValueError(
            f"Expected exactly one CSV file, but found {len(csv_members)}: {csv_members}"
        )
    with archive.open(csv_members[0]) as csv_file:
        df = pd.read_csv(csv_file)

print(f"Loaded file: {csv_members[0]}")
print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
print("Columns:")
print(df.columns.tolist())

section("3. FIRST FIVE ROWS")
print(df.head().to_string(index=False))

section("4. DATA TYPES AND NON-MISSING COUNTS")
schema = pd.DataFrame(
    {
        "dtype": df.dtypes.astype(str),
        "non_missing": df.notna().sum(),
        "missing": df.isna().sum(),
        "unique": df.nunique(dropna=True),
    }
)
schema["missing_pct"] = 100 * schema["missing"] / len(df)
print(schema.to_string())

section("5. BASIC DATA-QUALITY CHECKS")
print(f"Exact duplicate rows: {df.duplicated().sum():,}")

player_col = next((c for c in ["player", "Player", "name", "Name"] if c in df), None)
season_col = next((c for c in ["year", "Year", "season", "Season"] if c in df), None)
if player_col and season_col:
    duplicate_player_seasons = df.duplicated([player_col, season_col], keep=False).sum()
    print(
        f"Rows involved in duplicate {player_col}-{season_col} keys: "
        f"{duplicate_player_seasons:,}"
    )
else:
    print("Player-season key check: skipped because a standard player/season pair was not found.")

missing_summary = (
    schema.loc[schema["missing"] > 0, ["missing", "missing_pct"]]
    .sort_values(["missing_pct", "missing"], ascending=False)
)
print("\nColumns with missing values:")
print("None" if missing_summary.empty else missing_summary.to_string())

section("6. SEASON COVERAGE")
if season_col is None:
    print("No standard season column was found.")
else:
    season_numeric = pd.to_numeric(df[season_col], errors="coerce")
    print(f"Season column: {season_col}")
    print(f"Earliest season value: {season_numeric.min():.0f}")
    print(f"Latest season value:   {season_numeric.max():.0f}")
    print(f"Distinct seasons:      {season_numeric.nunique():,}")
    season_counts = df.groupby(season_col).size()
    print(f"Rows per season — min: {season_counts.min():,}, median: {season_counts.median():.1f}, max: {season_counts.max():,}")

section("7. TARGET AUDIT: award_share")
TARGET = "award_share"
if TARGET not in df.columns:
    raise KeyError(f"Required target column '{TARGET}' was not found.")

target = pd.to_numeric(df[TARGET], errors="coerce")
target_summary = target.describe(percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]).to_frame("award_share")
print(target_summary.to_string())
print(f"Missing target values:       {target.isna().sum():,}")
print(f"Zero target values:          {target.eq(0).sum():,} ({target.eq(0).mean():.2%})")
print(f"Positive target values:      {target.gt(0).sum():,} ({target.gt(0).mean():.2%})")
print(f"Negative target values:      {target.lt(0).sum():,}")
print(f"Values greater than 1:       {target.gt(1).sum():,}")

if season_col is not None:
    target_by_season = (
        df.assign(_target=target)
        .groupby(season_col)
        .agg(
            players=(TARGET, "size"),
            positive_vote_getters=("_target", lambda s: s.gt(0).sum()),
            total_award_share=("_target", "sum"),
            maximum_award_share=("_target", "max"),
        )
    )
    print("\nSeason-level target diagnostics (first 5 and last 5 seasons):")
    print(pd.concat([target_by_season.head(), target_by_season.tail()]).to_string())
    print("\nAcross-season diagnostics:")
    print(target_by_season.describe().to_string())

section("8. INITIAL DIAGNOSTIC FIGURES")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

sns.histplot(target, bins=40, ax=axes[0], color="#2a6fbb")
axes[0].set_title("Distribution of MVP vote share (all player-seasons)")
axes[0].set_xlabel("award_share")
axes[0].set_ylabel("Player-season count")

sns.histplot(target[target > 0], bins=30, ax=axes[1], color="#dd6b20")
axes[1].set_title("Distribution among vote-getters only")
axes[1].set_xlabel("award_share | award_share > 0")
axes[1].set_ylabel("Player-season count")

fig.tight_layout()
target_plot_path = OUTPUT_DIR / "target_distribution.png"
fig.savefig(target_plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {target_plot_path}")

if season_col is not None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(
        target_by_season.index,
        target_by_season["positive_vote_getters"],
        marker="o",
        linewidth=1.7,
        markersize=3.5,
        color="#2f855a",
    )
    ax.set_title("Number of players receiving MVP vote share by season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Players with award_share > 0")
    fig.tight_layout()
    season_plot_path = OUTPUT_DIR / "positive_vote_getters_by_season.png"
    fig.savefig(season_plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {season_plot_path}")

print("\nChunk 1 audit completed without modifying the source dataset.")
