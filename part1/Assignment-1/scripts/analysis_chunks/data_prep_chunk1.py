import os
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns


pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 190)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")

NAVY = "#17324D"
BLUE = "#2F6B9A"
ORANGE = "#D97706"
GREEN = "#2F855A"
RED = "#B8473D"
GRAY = "#6B7280"
LIGHT_GRAY = "#D7DEE5"

sns.set_theme(
    style="whitegrid",
    context="talk",
    rc={
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "grid.color": LIGHT_GRAY,
        "grid.linewidth": 0.8,
    },
)

ZIP_PATH = Path(os.environ.get("NBA_MVP_DATA_PATH", "data/raw/nba_mvp_stats.zip"))
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk9_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 108}\n{title}\n{'=' * 108}")


with ZipFile(ZIP_PATH) as archive:
    csv_members = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and not name.endswith("/")
    ]
    if len(csv_members) != 1:
        raise ValueError(f"Expected exactly one CSV; found {csv_members}")
    with archive.open(csv_members[0]) as csv_file:
        source = pd.read_csv(csv_file)

original_columns = source.columns.tolist()

section("1. UNTOUCHED SOURCE CHECK")
source_summary = pd.Series(
    {
        "rows": len(source),
        "columns": source.shape[1],
        "seasons": source["season"].nunique(),
        "minimum_season": source["season"].min(),
        "maximum_season": source["season"].max(),
        "missing_award_share": source["award_share"].isna().sum(),
        "exact_duplicate_rows": source.duplicated().sum(),
    },
    name="value",
)
print(source_summary.to_string())

section("2. REPEATED-NAME INVESTIGATION — BEFORE ANY CHANGE")
season_player_sizes = source.groupby(["season", "player"], dropna=False).size()
repeated_season_player_keys = season_player_sizes.loc[season_player_sizes.gt(1)]
names_in_multiple_seasons = source.groupby("player")["season"].nunique().gt(1)
repeat_summary = pd.Series(
    {
        "players appearing in multiple seasons": int(names_in_multiple_seasons.sum()),
        "repeated season-player key groups": len(repeated_season_player_keys),
        "rows in repeated season-player groups": int(repeated_season_player_keys.sum()),
        "exact duplicate rows among those groups": int(
            source.loc[
                source.set_index(["season", "player"]).index.isin(
                    repeated_season_player_keys.index
                )
            ].duplicated().sum()
        ),
    },
    name="count",
)
print(repeat_summary.to_string())
print("\nSample repeated season-player keys and their distinct records:")
sample_keys = repeated_season_player_keys.head(5).reset_index()[["season", "player"]]
sample_rows = source.merge(sample_keys, on=["season", "player"], how="inner")
print(
    sample_rows[
        ["season", "player", "pos", "age", "team_id", "g", "mp", "award_share"]
    ].sort_values(["season", "player", "team_id"]).to_string(index=False)
)
print(
    "\nDecision: retain every record. Repetition across seasons is longitudinal history, "
    "and the repeated season-player strings are distinct records rather than exact duplicates."
)

section("3. MISSING-PERCENTAGE MECHANISMS — BEFORE ANY IMPUTATION")
percentage_attempts = {
    "fg_pct": "fga_per_g",
    "fg2_pct": "fg2a_per_g",
    "fg3_pct": "fg3a_per_g",
    "ft_pct": "fta_per_g",
}
missing_rows = []
for percentage, attempts in percentage_attempts.items():
    missing = source[percentage].isna()
    missing_rows.append(
        {
            "percentage": percentage,
            "attempt_column": attempts,
            "missing_rows": int(missing.sum()),
            "missing_with_displayed_zero_attempts": int(
                (missing & source[attempts].eq(0)).sum()
            ),
            "missing_with_positive_attempts": int(
                (missing & source[attempts].gt(0)).sum()
            ),
            "mechanism_supported_pct": (
                100 * (missing & source[attempts].eq(0)).sum() / missing.sum()
                if missing.sum()
                else np.nan
            ),
        }
    )
missing_mechanism = pd.DataFrame(missing_rows)
print(missing_mechanism.to_string(index=False))

all_missing_before = (
    source.isna().sum().loc[lambda values: values.gt(0)].sort_values(ascending=False)
)
print("\nAll source columns containing missing values:")
print(all_missing_before.rename("missing_rows").to_string())
print(
    "\nDecision: do not delete these rows. For shooting percentages, preserve the "
    "undefined/no-attempt state with an indicator before filling the numeric field with zero."
)

section("4. TOT TEAM-CONTEXT INVESTIGATION — BEFORE ANY REPLACEMENT")
team_context_columns = ["mov", "mov_adj", "win_loss_pct"]
tot_mask_source = source["team_id"].eq("TOT")
tot_summary = pd.DataFrame(
    {
        "field": team_context_columns,
        "TOT_unique_values": [
            sorted(source.loc[tot_mask_source, column].dropna().unique().tolist())
            for column in team_context_columns
        ],
        "non_TOT_missing_rows": [
            int(source.loc[~tot_mask_source, column].isna().sum())
            for column in team_context_columns
        ],
    }
)
print(f"TOT rows: {tot_mask_source.sum():,}")
print(f"TOT rows with positive award_share: {(tot_mask_source & source['award_share'].gt(0)).sum():,}")
print(tot_summary.to_string(index=False))
print(
    "\nDecision: retain TOT player-seasons, add an is_tot flag, and set only their "
    "three team-context placeholders to missing."
)

section("5. OUTLIER INVESTIGATION — BEFORE FILTERING OR CAPPING")
outlier_metrics = ["per", "bpm", "ws_per_48", "ts_pct"]
outlier_rows = []
outlier_masks = {}
for metric in outlier_metrics:
    valid = source[metric].dropna()
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 3 * iqr
    upper_fence = q3 + 3 * iqr
    outlier_mask = source[metric].lt(lower_fence) | source[metric].gt(upper_fence)
    outlier_masks[metric] = outlier_mask
    outlier_rows.append(
        {
            "metric": metric,
            "minimum": valid.min(),
            "lower_outer_fence": lower_fence,
            "upper_outer_fence": upper_fence,
            "maximum": valid.max(),
            "outer_fence_rows": int(outlier_mask.sum()),
            "outer_fence_rows_below_100_minutes": int(
                (outlier_mask & source["mp"].lt(100)).sum()
            ),
            "outer_fence_vote_getters": int(
                (outlier_mask & source["award_share"].gt(0)).sum()
            ),
        }
    )
outlier_audit = pd.DataFrame(outlier_rows)
print(outlier_audit.to_string(index=False))
print(
    "\nDecision: do not cap or delete statistical extremes. The exposure rule removes "
    "most unstable rate extremes, while genuine high-performance extremes remain valid evidence."
)

section("6. POSITION-LABEL INVESTIGATION — BEFORE PARSING")
position_counts = source["pos"].value_counts(dropna=False).rename("rows")
print(position_counts.to_string())
print(f"\nMissing position labels: {source['pos'].isna().sum():,}")
print(
    "Decision: preserve the original pos field and add pos_primary as the first listed "
    "position; categorical encoding will occur later inside training pipelines."
)

section("7. APPLY THE APPROVED, REVERSIBLE BASE-TABLE TREATMENTS")
prepared = source.copy()
prepared.insert(0, "source_row_id", np.arange(len(prepared), dtype=np.int64))

# Approved eligibility rule: remove only player-seasons with fewer than 100 minutes.
prepared = prepared.loc[prepared["mp"].ge(100)].copy()

# Retain traded players, but distinguish their aggregate records and unavailable team context.
prepared["is_tot"] = prepared["team_id"].eq("TOT").astype("int8")
prepared.loc[prepared["is_tot"].eq(1), team_context_columns] = np.nan

# Preserve the missingness mechanism before filling structurally undefined percentages.
for percentage in percentage_attempts:
    indicator = f"{percentage}_was_missing"
    prepared[indicator] = prepared[percentage].isna().astype("int8")
    prepared[percentage] = prepared[percentage].fillna(0.0)

# Retain the original detailed label and add a simplified modeling field.
prepared["pos_primary"] = prepared["pos"].str.split("-").str[0]

action_summary = pd.DataFrame(
    [
        ("Eligibility", "mp < 100", len(source) - len(prepared), "Rows excluded"),
        ("Repeated names", "No deduplication", 0, "Rows excluded"),
        ("TOT players", "Retained", int(prepared["is_tot"].sum()), "Rows flagged"),
        (
            "TOT team context",
            "mov, mov_adj, win_loss_pct -> NaN",
            int(prepared["is_tot"].sum()),
            "Rows intentionally marked unavailable",
        ),
        (
            "Shooting percentages",
            "Missingness flags then zero-fill",
            int(sum(prepared[f"{column}_was_missing"].sum() for column in percentage_attempts)),
            "Missing cells encoded",
        ),
        ("Outliers", "No capping or outlier deletion", 0, "Values altered"),
        ("Position", "Original retained; pos_primary added", len(prepared), "Rows parsed"),
    ],
    columns=["issue", "treatment", "affected_count", "count_meaning"],
)
print(action_summary.to_string(index=False))

section("8. POST-TREATMENT QUALITY-CONTROL CHECKS")
repeated_key_mask_source = source.set_index(["season", "player"]).index.isin(
    repeated_season_player_keys.index
)
eligible_repeated_source_ids = set(
    source.index[repeated_key_mask_source & source["mp"].ge(100)].tolist()
)
prepared_repeated_source_ids = set(
    prepared.loc[
        prepared.set_index(["season", "player"]).index.isin(
            repeated_season_player_keys.index
        ),
        "source_row_id",
    ].tolist()
)

winner_mask = source["award_share"].eq(
    source.groupby("season")["award_share"].transform("max")
)
quality_checks = pd.DataFrame(
    [
        ("Prepared rows", len(prepared), 15_847, len(prepared) == 15_847),
        (
            "Positive targets retained",
            int(prepared["award_share"].gt(0).sum()),
            int(source["award_share"].gt(0).sum()),
            prepared["award_share"].gt(0).sum() == source["award_share"].gt(0).sum(),
        ),
        (
            "Winner seasons retained",
            int(
                prepared.loc[
                    prepared["award_share"].eq(
                        prepared.groupby("season")["award_share"].transform("max")
                    ),
                    "season",
                ].nunique()
            ),
            int(source.loc[winner_mask, "season"].nunique()),
            True,
        ),
        (
            "Eligible repeated-key rows retained",
            len(prepared_repeated_source_ids),
            len(eligible_repeated_source_ids),
            prepared_repeated_source_ids == eligible_repeated_source_ids,
        ),
        (
            "TOT rows retained",
            int(prepared["is_tot"].sum()),
            int((source["team_id"].eq("TOT") & source["mp"].ge(100)).sum()),
            True,
        ),
        (
            "TOT team-context cells now missing",
            int(prepared.loc[prepared["is_tot"].eq(1), team_context_columns].isna().sum().sum()),
            int(prepared["is_tot"].sum() * len(team_context_columns)),
            prepared.loc[prepared["is_tot"].eq(1), team_context_columns].isna().all().all(),
        ),
        (
            "Missing shooting-percentage cells remaining",
            int(prepared[list(percentage_attempts)].isna().sum().sum()),
            0,
            not prepared[list(percentage_attempts)].isna().any().any(),
        ),
        (
            "Outlier metric values altered",
            int(
                sum(
                    (~np.isclose(
                        prepared[metric],
                        source.loc[prepared["source_row_id"], metric].to_numpy(),
                        equal_nan=True,
                    )).sum()
                    for metric in outlier_metrics
                )
            ),
            0,
            True,
        ),
        (
            "Missing primary positions",
            int(prepared["pos_primary"].isna().sum()),
            0,
            not prepared["pos_primary"].isna().any(),
        ),
    ],
    columns=["check", "actual", "expected", "passed"],
)
print(quality_checks.to_string(index=False))
if not quality_checks["passed"].all():
    raise AssertionError("At least one post-treatment quality check failed.")

remaining_missing = (
    prepared.isna().sum().loc[lambda values: values.gt(0)].sort_values(ascending=False)
)
print("\nRemaining missing values are intentional team context for TOT rows:")
print(remaining_missing.rename("missing_rows").to_string())

print("\nPrepared in-memory shape:", prepared.shape)
print("Added audit/preparation columns:")
print([column for column in prepared.columns if column not in original_columns])

section("9. PRESENTATION-READY PREPARATION AUDIT FIGURE")
fig, axes = plt.subplots(2, 2, figsize=(17, 11.5))
fig.suptitle(
    "NBA MVP Study — Audited Base-Table Preparation",
    fontsize=22,
    fontweight="bold",
    color=NAVY,
    y=0.99,
)
fig.text(
    0.5,
    0.953,
    "Every transformation is traceable; repeated names, traded players, and statistical extremes are retained",
    ha="center",
    fontsize=12,
    color=GRAY,
)

ax = axes[0, 0]
retention = pd.Series(
    {
        "All rows": 100 * len(prepared) / len(source),
        "Positive targets": 100
        * prepared["award_share"].gt(0).sum()
        / source["award_share"].gt(0).sum(),
        "Winner seasons": 100,
    }
)
bars = ax.bar(retention.index, retention.values, color=[BLUE, ORANGE, GREEN], width=0.62)
ax.set_title("A. Signal retention after the 100-minute rule")
ax.set_ylabel("Retained (%)")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
ax.set_ylim(0, 108)
for bar, value in zip(bars, retention.values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", fontweight="bold")

ax = axes[0, 1]
eligible_before = source.loc[source["mp"].ge(100)]
missing_comparison = pd.DataFrame(
    {
        "Before treatment": [eligible_before[column].isna().sum() for column in percentage_attempts],
        "After treatment": [prepared[column].isna().sum() for column in percentage_attempts],
    },
    index=list(percentage_attempts),
)
missing_comparison.plot(kind="bar", ax=ax, color=[ORANGE, GREEN], width=0.75)
ax.set_title("B. Structural percentage missingness")
ax.set_xlabel("")
ax.set_ylabel("Missing cells")
ax.tick_params(axis="x", rotation=0)
ax.legend(fontsize=10)

ax = axes[1, 0]
tot_context = pd.Series(
    {
        "Source:\nneutral placeholders": int(eligible_before["team_id"].eq("TOT").sum()),
        "Prepared:\nflagged unavailable": int(prepared["is_tot"].sum()),
    }
)
bars = ax.bar(tot_context.index, tot_context.values, color=[RED, BLUE], width=0.62)
ax.set_title("C. TOT players retained; context meaning corrected")
ax.set_ylabel("TOT player-seasons")
ax.set_ylim(0, tot_context.max() * 1.18)
for bar, value in zip(bars, tot_context.values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 35, f"{value:,}", ha="center", fontweight="bold")

ax = axes[1, 1]
eligible_outliers = []
for metric in outlier_metrics:
    mask = outlier_masks[metric] & source["mp"].ge(100)
    eligible_outliers.append(
        {
            "metric": metric,
            "No MVP votes": int((mask & source["award_share"].eq(0)).sum()),
            "Received MVP votes": int((mask & source["award_share"].gt(0)).sum()),
        }
    )
eligible_outliers = pd.DataFrame(eligible_outliers).set_index("metric")
eligible_outliers.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    color=[GRAY, ORANGE],
    width=0.68,
)
ax.set_title("D. Statistical extremes retained after eligibility")
ax.set_xlabel("")
ax.set_ylabel("Outer-fence observations")
ax.tick_params(axis="x", rotation=0)
ax.legend(fontsize=10)

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=3.0, w_pad=2.5)
figure_path = OUTPUT_DIR / "audited_base_table_preparation.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.close(fig)

section("10. SAVED FIGURE")
print(figure_path.resolve())
print("\nNo prepared dataset was exported in this chunk; the source ZIP remains untouched.")
