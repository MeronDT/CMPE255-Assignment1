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
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk3_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 94}\n{title}\n{'=' * 94}")


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

section("2. BASIC NUMERIC RANGES")
range_columns = [
    "season", "age", "g", "gs", "mp", "mp_per_g", "pts_per_g",
    "fg_pct", "fg2_pct", "fg3_pct", "ft_pct", "efg_pct", "ts_pct",
    "fg3a_per_fga_pct", "fta_per_fga_pct", "win_loss_pct",
    "per", "ws", "ws_per_48", "bpm", "vorp", "award_share",
]
range_summary = df[range_columns].agg(["min", "max"]).T
range_summary["missing"] = df[range_columns].isna().sum()
range_summary["non_finite"] = [
    (~np.isfinite(pd.to_numeric(df[column], errors="coerce").dropna())).sum()
    for column in range_columns
]
print(range_summary.to_string())

section("3. HARD LOGICAL-CONSTRAINT CHECKS")
bounded_zero_one = [
    "fg_pct", "fg2_pct", "fg3_pct", "ft_pct",
    "fg3a_per_fga_pct", "win_loss_pct", "award_share",
]
constraint_results = []
for column in bounded_zero_one:
    values = df[column].dropna()
    constraint_results.append(
        {
            "check": f"0 <= {column} <= 1",
            "violations": int(((values < 0) | (values > 1)).sum()),
        }
    )

additional_checks = {
    "season is an integer": np.isclose(df["season"], np.round(df["season"])),
    "age > 0": df["age"] > 0,
    "games >= 0": df["g"] >= 0,
    "games_started >= 0": df["gs"] >= 0,
    "games_started <= games": df["gs"] <= df["g"],
    "minutes >= 0": df["mp"] >= 0,
    "minutes_per_game >= 0": df["mp_per_g"] >= 0,
    "points_per_game >= 0": df["pts_per_g"] >= 0,
}
for name, valid_mask in additional_checks.items():
    constraint_results.append(
        {"check": name, "violations": int((~valid_mask).sum())}
    )

constraint_table = pd.DataFrame(constraint_results)
print(constraint_table.to_string(index=False))

section("4. ARITHMETIC RECONCILIATION WITH ROUNDING TOLERANCES")
# Per-game and advanced statistics are rounded in the source. Tolerances below
# allow for the accumulated rounding error of the displayed component fields.
reconciliation_specs = {
    "rebounds: trb vs orb + drb": {
        "residual": df["trb_per_g"] - (df["orb_per_g"] + df["drb_per_g"]),
        "tolerance": 0.21,
    },
    "field goals: fg vs fg2 + fg3": {
        "residual": df["fg_per_g"] - (df["fg2_per_g"] + df["fg3_per_g"]),
        "tolerance": 0.21,
    },
    "points: pts vs 2*fg2 + 3*fg3 + ft": {
        "residual": df["pts_per_g"]
        - (2 * df["fg2_per_g"] + 3 * df["fg3_per_g"] + df["ft_per_g"]),
        "tolerance": 0.41,
    },
    "win shares: ws vs ows + dws": {
        "residual": df["ws"] - (df["ows"] + df["dws"]),
        "tolerance": 0.21,
    },
    "box plus-minus: bpm vs obpm + dbpm": {
        "residual": df["bpm"] - (df["obpm"] + df["dbpm"]),
        "tolerance": 0.21,
    },
}

reconciliation_rows = []
residual_long = []
for name, spec in reconciliation_specs.items():
    absolute_residual = spec["residual"].abs()
    reconciliation_rows.append(
        {
            "relationship": name,
            "mean_abs_error": absolute_residual.mean(),
            "95th_pct_abs_error": absolute_residual.quantile(0.95),
            "max_abs_error": absolute_residual.max(),
            "tolerance": spec["tolerance"],
            "rows_over_tolerance": int((absolute_residual > spec["tolerance"]).sum()),
        }
    )
    residual_long.extend(
        {"relationship": name, "absolute_residual": value}
        for value in absolute_residual.dropna()
    )

reconciliation_table = pd.DataFrame(reconciliation_rows)
print(reconciliation_table.to_string(index=False))

minute_residual = (df["mp"] - df["g"] * df["mp_per_g"]).abs()
minute_tolerance = 0.05 * df["g"] + 0.51
print("\nMinutes reconciliation: mp versus g * mp_per_g")
print(f"Mean absolute difference:   {minute_residual.mean():.4f}")
print(f"Maximum absolute difference:{minute_residual.max():.4f}")
print(f"Rows above rounding tolerance: {(minute_residual > minute_tolerance).sum():,}")

section("5. TEAM-FIELD CONSISTENCY WITHIN SEASON AND TEAM")
team_fields = ["mov", "mov_adj", "win_loss_pct"]
non_tot = df.loc[df["team_id"] != "TOT"].copy()
team_unique_counts = non_tot.groupby(["season", "team_id"])[team_fields].nunique(dropna=False)
inconsistent_team_groups = team_unique_counts.loc[(team_unique_counts > 1).any(axis=1)]
print(f"Non-TOT season-team groups checked: {len(team_unique_counts):,}")
print(f"Groups with inconsistent team fields: {len(inconsistent_team_groups):,}")
print("None" if inconsistent_team_groups.empty else inconsistent_team_groups.to_string())

section("6. TOT RECORD AUDIT")
tot_rows = df.loc[df["team_id"] == "TOT"].copy()
print(f"TOT player-season rows: {len(tot_rows):,} ({len(tot_rows) / len(df):.2%} of all rows)")
print(f"Seasons containing TOT rows: {tot_rows['season'].nunique():,}")
print(f"Earliest / latest TOT season: {tot_rows['season'].min()} / {tot_rows['season'].max()}")
print(f"TOT rows with positive award_share: {tot_rows['award_share'].gt(0).sum():,}")

tot_team_field_summary = tot_rows[team_fields].agg(["min", "max", "mean", "nunique"]).T
print("\nTeam-field values among TOT rows:")
print(tot_team_field_summary.to_string())

print("\nTOT rows with positive MVP vote share:")
positive_tot_columns = [
    "season", "player", "g", "gs", "mp", "pts_per_g", "ws", "vorp",
    "award_share", "mov", "mov_adj", "win_loss_pct",
]
print(
    tot_rows.loc[tot_rows["award_share"] > 0, positive_tot_columns]
    .sort_values(["season", "award_share"], ascending=[True, False])
    .to_string(index=False)
)

print("\nTOT rows with the largest minutes totals:")
tot_display_columns = [
    "season", "player", "g", "gs", "mp", "pts_per_g", "ws", "vorp",
    "award_share", "mov", "mov_adj", "win_loss_pct",
]
print(
    tot_rows.nlargest(15, "mp")[tot_display_columns]
    .sort_values(["season", "player"])
    .to_string(index=False)
)

tot_by_season = tot_rows.groupby("season").size().rename("tot_rows")
print("\nTOT-row counts by season:")
print(tot_by_season.to_string())

section("7. DIAGNOSTIC FIGURES")
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

axes[0].bar(
    tot_by_season.index,
    tot_by_season.values,
    color="#2f855a",
    width=0.8,
)
axes[0].set_title("Combined-team (TOT) rows by season")
axes[0].set_xlabel("Season")
axes[0].set_ylabel("TOT player-season rows")

residual_frame = pd.DataFrame(residual_long)
short_labels = {
    "rebounds: trb vs orb + drb": "Rebounds",
    "field goals: fg vs fg2 + fg3": "Field goals",
    "points: pts vs 2*fg2 + 3*fg3 + ft": "Points",
    "win shares: ws vs ows + dws": "Win shares",
    "box plus-minus: bpm vs obpm + dbpm": "BPM",
}
residual_frame["relationship"] = residual_frame["relationship"].map(short_labels)
sns.boxplot(
    data=residual_frame,
    x="relationship",
    y="absolute_residual",
    showfliers=False,
    color="#4c78a8",
    ax=axes[1],
)
axes[1].set_title("Arithmetic residuals caused by displayed rounding")
axes[1].set_xlabel("")
axes[1].set_ylabel("Absolute residual")
axes[1].tick_params(axis="x", rotation=25)

fig.tight_layout()
figure_path = OUTPUT_DIR / "logical_consistency_and_tot.png"
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {figure_path}")

print("\nChunk 3 audit completed without altering the source dataset.")
