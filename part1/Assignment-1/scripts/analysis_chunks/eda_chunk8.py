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
pd.set_option("display.width", 180)
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
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk8_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 104}\n{title}\n{'=' * 104}")


with ZipFile(ZIP_PATH) as archive:
    csv_members = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and not name.endswith("/")
    ]
    if len(csv_members) != 1:
        raise ValueError(f"Expected exactly one CSV; found {csv_members}")
    with archive.open(csv_members[0]) as csv_file:
        df = pd.read_csv(csv_file)

section("1. SOURCE AND DATA-UNDERSTANDING READINESS CHECK")
winner_mask = df["award_share"].eq(
    df.groupby("season")["award_share"].transform("max")
)
readiness = pd.DataFrame(
    {
        "check": [
            "Rows",
            "Columns",
            "Seasons",
            "Missing target values",
            "Exact duplicate rows",
            "Positive-target rows",
            "Zero-target rows",
            "Season winners represented",
            "TOT aggregate rows",
        ],
        "result": [
            len(df),
            df.shape[1],
            df["season"].nunique(),
            df["award_share"].isna().sum(),
            df.duplicated().sum(),
            df["award_share"].gt(0).sum(),
            df["award_share"].eq(0).sum(),
            df.loc[winner_mask, "season"].nunique(),
            df["team_id"].eq("TOT").sum(),
        ],
        "interpretation": [
            "Complete source population",
            "Mixed identifiers, context, predictors, and target",
            f"{df['season'].min()}–{df['season'].max()}",
            "Pass",
            "Pass",
            "Rare candidate class",
            "Expected non-candidate majority",
            "All seasons have an observed winner",
            "Retain player; team context requires special handling",
        ],
    }
)
print(readiness.to_string(index=False))

section("2. ELIGIBILITY-FILTER TRADE-OFF")
thresholds = [0, 1, 50, 100, 250, 500, 750, 1000]
positive_total = df["award_share"].gt(0).sum()
winner_seasons_total = df.loc[winner_mask, "season"].nunique()
filter_rows = []
for threshold in thresholds:
    eligible = df["mp"].ge(threshold)
    retained_winner_seasons = df.loc[eligible & winner_mask, "season"].nunique()
    filter_rows.append(
        {
            "minimum_minutes": threshold,
            "rows": eligible.sum(),
            "rows_retained_pct": 100 * eligible.mean(),
            "positive_targets": (eligible & df["award_share"].gt(0)).sum(),
            "positive_targets_retained_pct": 100
            * (eligible & df["award_share"].gt(0)).sum()
            / positive_total,
            "winner_seasons": retained_winner_seasons,
            "winner_seasons_retained_pct": 100
            * retained_winner_seasons
            / winner_seasons_total,
        }
    )
filter_table = pd.DataFrame(filter_rows)
print(filter_table.to_string(index=False))

# This is an analysis view used only to quantify the proposed rule.
# It is not saved as a prepared modeling dataset.
proposed_view = df.loc[df["mp"].ge(100)].copy()
proposed_winner_mask = proposed_view["award_share"].eq(
    proposed_view.groupby("season")["award_share"].transform("max")
)

section("3. REMAINING CONDITIONS UNDER THE PROPOSED 100-MINUTE ANALYSIS POPULATION")
condition_counts = pd.DataFrame(
    {
        "condition": [
            "Missing 3P percentage",
            "Missing FT percentage",
            "Missing 2P percentage",
            "Missing FG percentage",
            "TOT rows with placeholder team context",
            "Missing award_share",
        ],
        "rows": [
            proposed_view["fg3_pct"].isna().sum(),
            proposed_view["ft_pct"].isna().sum(),
            proposed_view["fg2_pct"].isna().sum(),
            proposed_view["fg_pct"].isna().sum(),
            proposed_view["team_id"].eq("TOT").sum(),
            proposed_view["award_share"].isna().sum(),
        ],
    }
)
condition_counts["percent_of_proposed_rows"] = (
    100 * condition_counts["rows"] / len(proposed_view)
)
print(f"Proposed-view rows: {len(proposed_view):,}")
print(
    f"Positive targets retained: "
    f"{proposed_view['award_share'].gt(0).sum():,}/{positive_total:,}"
)
print(
    f"Winner seasons retained: "
    f"{proposed_view.loc[proposed_winner_mask, 'season'].nunique():,}/"
    f"{winner_seasons_total:,}"
)
print(condition_counts.to_string(index=False))

section("4. PROPOSED CHRONOLOGICAL EVALUATION WINDOWS — COUNTS ONLY")
phase = pd.Series(index=proposed_view.index, dtype="object")
phase.loc[proposed_view["season"].le(2014)] = "Development: 1982–2014"
phase.loc[proposed_view["season"].between(2015, 2018)] = "Validation: 2015–2018"
phase.loc[proposed_view["season"].between(2019, 2022)] = "Final test: 2019–2022"
proposed_view["evaluation_window"] = phase

window_order = [
    "Development: 1982–2014",
    "Validation: 2015–2018",
    "Final test: 2019–2022",
]
window_rows = []
for label in window_order:
    subset = proposed_view.loc[proposed_view["evaluation_window"].eq(label)]
    subset_winners = subset["award_share"].eq(
        subset.groupby("season")["award_share"].transform("max")
    )
    window_rows.append(
        {
            "window": label,
            "seasons": subset["season"].nunique(),
            "rows": len(subset),
            "positive_targets": subset["award_share"].gt(0).sum(),
            "positive_target_rate_pct": 100 * subset["award_share"].gt(0).mean(),
            "winner_seasons": subset.loc[subset_winners, "season"].nunique(),
        }
    )
window_table = pd.DataFrame(window_rows)
print(window_table.to_string(index=False))
print(
    "\nImportant: these are proposed calendar windows, not a completed split. "
    "All fitting and feature-selection operations must later occur inside each "
    "training fold."
)

section("5. FINAL DATA-UNDERSTANDING DECISION REGISTER")
decision_register = pd.DataFrame(
    [
        ("Population", "Primary: mp >= 100; sensitivity: mp >= 500"),
        ("Target", "Keep continuous award_share and all zero-target rows"),
        ("Identifiers", "Exclude player and team_id from primary predictors"),
        ("TOT records", "Keep; add is_tot and mark team context unavailable"),
        ("Structural missingness", "Zero-fill undefined shooting percentages plus indicators"),
        ("Outliers", "No global winsorization; filter low exposure and use robust models/scaling"),
        ("Era", "Keep raw performance plus selected within-season percentile features"),
        ("Redundancy", "Domain-pruned linear set; broader regularized/nonlinear comparison set"),
        ("Position", "Primary-position one-hot encoding with a no-position sensitivity model"),
        ("Validation", "Chronological, season-grouped evaluation; untouched 2019–2022 final test"),
        ("Ranking", "Evaluate predictions within season, not only row-level regression error"),
        ("Clustering", "Optional descriptive archetypes; predictive use only if validated"),
    ],
    columns=["decision_area", "provisional_choice"],
)
print(decision_register.to_string(index=False))

# Presentation-ready synthesis dashboard.
fig, axes = plt.subplots(2, 2, figsize=(17, 12))
fig.suptitle(
    "NBA MVP Study — Data Understanding Readiness Dashboard",
    fontsize=22,
    fontweight="bold",
    color=NAVY,
    y=0.99,
)
fig.text(
    0.5,
    0.955,
    "Descriptive evidence for the proposed preparation design; no preprocessing has been fitted",
    ha="center",
    fontsize=12,
    color=GRAY,
)

ax = axes[0, 0]
ax.plot(
    filter_table["minimum_minutes"],
    filter_table["rows_retained_pct"],
    marker="o",
    linewidth=2.5,
    color=BLUE,
    label="All player-seasons",
)
ax.plot(
    filter_table["minimum_minutes"],
    filter_table["positive_targets_retained_pct"],
    marker="o",
    linewidth=2.5,
    color=ORANGE,
    label="Positive award_share rows",
)
ax.plot(
    filter_table["minimum_minutes"],
    filter_table["winner_seasons_retained_pct"],
    marker="o",
    linewidth=2.5,
    color=GREEN,
    label="Winner seasons",
)
ax.axvline(100, color=RED, linestyle="--", linewidth=2, label="Proposed primary cutoff")
ax.set_title("A. Exposure-filter trade-off")
ax.set_xlabel("Minimum regular-season minutes")
ax.set_ylabel("Retained (%)")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
ax.set_ylim(50, 103)
ax.legend(fontsize=9, loc="lower left")

ax = axes[0, 1]
target_counts = pd.Series(
    {
        "Zero award share": proposed_view["award_share"].eq(0).sum(),
        "Positive award share": proposed_view["award_share"].gt(0).sum(),
    }
)
bars = ax.bar(target_counts.index, target_counts.values, color=[BLUE, ORANGE], width=0.62)
ax.set_yscale("log")
ax.set_title("B. Target imbalance remains by design")
ax.set_ylabel("Player-seasons (log scale)")
ax.tick_params(axis="x", rotation=8)
for bar, value in zip(bars, target_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value * 1.12,
        f"{value:,}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
ax.text(
    0.02,
    0.95,
    f"Positive rate = {100 * proposed_view['award_share'].gt(0).mean():.2f}%",
    transform=ax.transAxes,
    va="top",
    fontsize=11,
    color=NAVY,
)

ax = axes[1, 0]
plot_conditions = condition_counts.loc[condition_counts["rows"].gt(0)].sort_values("rows")
condition_labels = {
    "Missing 3P percentage": "Undefined 3P%",
    "Missing FT percentage": "Undefined FT%",
    "Missing 2P percentage": "Undefined 2P%",
    "Missing FG percentage": "Undefined FG%",
    "TOT rows with placeholder team context": "TOT team context",
    "Missing award_share": "Missing target",
}
labels = plot_conditions["condition"].map(condition_labels)
bars = ax.barh(labels, plot_conditions["rows"], color=[ORANGE, BLUE, BLUE, BLUE, RED][:len(labels)])
ax.set_title("C. Conditions preparation must encode")
ax.set_xlabel("Affected player-seasons")
for bar, value in zip(bars, plot_conditions["rows"]):
    ax.text(value + 25, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", fontsize=10)
ax.set_xlim(0, max(plot_conditions["rows"]) * 1.18)

ax = axes[1, 1]
short_labels = ["Development", "Validation", "Final test"]
bars = ax.bar(short_labels, window_table["rows"], color=[BLUE, ORANGE, GREEN], width=0.62)
ax.set_title("D. Proposed chronological evaluation windows")
ax.set_ylabel("Eligible player-seasons")
for bar, row in zip(bars, window_table.itertuples(index=False)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 140,
        f"{row.rows:,} rows\n{row.positive_targets} positive\n{row.seasons} seasons",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
ax.set_ylim(0, window_table["rows"].max() * 1.18)

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=3.0, w_pad=2.5)
figure_path = OUTPUT_DIR / "data_understanding_readiness_dashboard.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.close(fig)

section("6. SAVED FIGURE")
print(figure_path.resolve())
