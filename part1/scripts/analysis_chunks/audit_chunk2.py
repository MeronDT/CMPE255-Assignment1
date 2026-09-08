import os
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")
sns.set_theme(style="whitegrid", context="notebook")

ZIP_PATH = Path(os.environ.get("NBA_MVP_DATA_PATH", "data/raw/nba_mvp_stats.zip"))
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk2_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")


# Load the sole CSV directly from the ZIP without altering the source archive.
with ZipFile(ZIP_PATH) as archive:
    csv_members = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and not name.endswith("/")
    ]
    if len(csv_members) != 1:
        raise ValueError(f"Expected exactly one CSV; found {csv_members}")
    with archive.open(csv_members[0]) as csv_file:
        df = pd.read_csv(csv_file)

section("1. DATA RECONFIRMATION")
print(f"Loaded: {csv_members[0]}")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

section("2. DUPLICATE PLAYER-SEASON GROUPS")
key = ["season", "player"]
duplicate_mask = df.duplicated(key, keep=False)
duplicate_rows = df.loc[duplicate_mask].copy()
duplicate_group_sizes = duplicate_rows.groupby(key).size().rename("rows_in_group")

print(f"Rows involved in duplicate player-season keys: {duplicate_mask.sum():,}")
print(f"Distinct duplicated player-season keys: {len(duplicate_group_sizes):,}")
print("Duplicate-group size distribution:")
print(duplicate_group_sizes.value_counts().sort_index().rename_axis("rows_in_group").to_string())

duplicate_display_columns = [
    "season", "player", "team_id", "g", "gs", "mp", "mp_per_g",
    "pts_per_g", "trb_per_g", "ast_per_g", "ws", "vorp",
    "award_share", "mov", "win_loss_pct",
]
duplicate_display = duplicate_rows[duplicate_display_columns].sort_values(key + ["team_id"])
print("\nAll rows belonging to duplicated player-season keys:")
print(duplicate_display.to_string(index=False))

duplicate_profile = (
    duplicate_rows.groupby(key)
    .agg(
        row_count=("team_id", "size"),
        distinct_teams=("team_id", "nunique"),
        teams=("team_id", lambda s: ", ".join(sorted(s.astype(str).unique()))),
        distinct_award_shares=("award_share", "nunique"),
        award_share_min=("award_share", "min"),
        award_share_max=("award_share", "max"),
    )
    .reset_index()
)
print("\nDuplicate-key diagnostic summary:")
print(duplicate_profile.to_string(index=False))
print(f"\nGroups spanning more than one team: {(duplicate_profile['distinct_teams'] > 1).sum():,}")
print(f"Groups with inconsistent award_share values: {(duplicate_profile['distinct_award_shares'] > 1).sum():,}")

section("3. DO DUPLICATE KEYS DISTORT SEASON TARGET TOTALS?")
all_season_sums = df.groupby("season")["award_share"].sum().rename("all_rows_sum")
deduplicated_season_sums = (
    df.sort_values(key)
    .drop_duplicates(key, keep="first")
    .groupby("season")["award_share"]
    .sum()
    .rename("one_row_per_player_sum")
)
season_sum_comparison = pd.concat(
    [all_season_sums, deduplicated_season_sums], axis=1
).assign(difference=lambda x: x["all_rows_sum"] - x["one_row_per_player_sum"])

affected_seasons = season_sum_comparison.loc[
    ~np.isclose(season_sum_comparison["difference"], 0)
]
print(f"Seasons whose target total changes after naive key deduplication: {len(affected_seasons):,}")
print("None" if affected_seasons.empty else affected_seasons.to_string())

section("4. MISSING PERCENTAGES VERSUS THEIR ATTEMPT DENOMINATORS")
percentage_denominators = {
    "fg_pct": "fga_per_g",
    "fg2_pct": "fg2a_per_g",
    "fg3_pct": "fg3a_per_g",
    "ft_pct": "fta_per_g",
}

missing_mechanism_rows = []
for percentage, attempts in percentage_denominators.items():
    missing = df[percentage].isna()
    zero_attempts = df[attempts].eq(0)
    missing_mechanism_rows.append(
        {
            "percentage": percentage,
            "attempt_column": attempts,
            "missing_count": int(missing.sum()),
            "zero_attempt_count": int(zero_attempts.sum()),
            "missing_and_zero_attempts": int((missing & zero_attempts).sum()),
            "missing_with_positive_attempts": int((missing & df[attempts].gt(0)).sum()),
            "defined_despite_zero_attempts": int((~missing & zero_attempts).sum()),
            "missing_explained_by_zero_attempts_pct": (
                100 * (missing & zero_attempts).sum() / missing.sum()
                if missing.sum() else np.nan
            ),
        }
    )

missing_mechanisms = pd.DataFrame(missing_mechanism_rows).set_index("percentage")
print(missing_mechanisms.to_string())

section("5. REMAINING MISSING-VALUE PATTERNS")
missing_columns = df.columns[df.isna().any()].tolist()
missing_pattern_labels = (
    df[missing_columns]
    .isna()
    .apply(lambda row: ", ".join(row.index[row].tolist()) if row.any() else "Complete row", axis=1)
)
pattern_counts = missing_pattern_labels.value_counts().rename("row_count")
print("Ten most common row-level missingness patterns:")
print(pattern_counts.head(10).to_string())

advanced_columns = [
    "per", "orb_pct", "drb_pct", "trb_pct", "ast_pct", "stl_pct",
    "blk_pct", "usg_pct", "ws_per_48",
]
advanced_missing_mask = df[advanced_columns].isna().any(axis=1)
advanced_missing_display_columns = [
    "season", "player", "team_id", "g", "gs", "mp", "mp_per_g",
    "fga_per_g", "fta_per_g", *advanced_columns,
]
print("\nRows missing one or more selected advanced metrics:")
print(
    df.loc[advanced_missing_mask, advanced_missing_display_columns]
    .sort_values(["season", "player"])
    .to_string(index=False)
)

section("6. ERA PATTERN IN THREE-POINT-PERCENTAGE MISSINGNESS")
fg3_missing_by_season = (
    df.assign(fg3_pct_missing=df["fg3_pct"].isna())
    .groupby("season")
    .agg(
        player_rows=("player", "size"),
        missing_fg3_pct=("fg3_pct_missing", "sum"),
        missing_rate=("fg3_pct_missing", "mean"),
        zero_3pa_rate=("fg3a_per_g", lambda s: s.eq(0).mean()),
    )
)
print("First five seasons:")
print(fg3_missing_by_season.head().to_string())
print("\nLast five seasons:")
print(fg3_missing_by_season.tail().to_string())
print("\nSeasons with the highest fg3_pct missingness rates:")
print(fg3_missing_by_season.nlargest(10, "missing_rate").to_string())

section("7. DIAGNOSTIC FIGURES")
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

missing_pct = (100 * df[missing_columns].isna().mean()).sort_values(ascending=False)
sns.barplot(
    x=missing_pct.values,
    y=missing_pct.index,
    ax=axes[0],
    color="#3b82b7",
)
axes[0].set_title("Missing values by column")
axes[0].set_xlabel("Missing rows (%)")
axes[0].set_ylabel("")

axes[1].plot(
    fg3_missing_by_season.index,
    100 * fg3_missing_by_season["missing_rate"],
    label="fg3_pct missing",
    color="#c05621",
    linewidth=2,
)
axes[1].plot(
    fg3_missing_by_season.index,
    100 * fg3_missing_by_season["zero_3pa_rate"],
    label="fg3a_per_g = 0",
    color="#2f855a",
    linewidth=2,
    linestyle="--",
)
axes[1].set_title("Three-point missingness follows zero attempts")
axes[1].set_xlabel("Season")
axes[1].set_ylabel("Player rows (%)")
axes[1].legend()

fig.tight_layout()
figure_path = OUTPUT_DIR / "duplicate_and_missingness_diagnostics.png"
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {figure_path}")

print("\nChunk 2 diagnostics completed without altering the source dataset.")
