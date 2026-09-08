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
LIGHT_BLUE = "#9EC3DB"
GRAY = "#6B7280"

sns.set_theme(
    style="whitegrid",
    context="talk",
    rc={
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.frameon": False,
        "grid.color": "#D7DEE5",
        "grid.linewidth": 0.8,
    },
)

ZIP_PATH = Path(os.environ.get("NBA_MVP_DATA_PATH", "data/raw/nba_mvp_stats.zip"))
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk4_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 96}\n{title}\n{'=' * 96}")


with ZipFile(ZIP_PATH) as archive:
    csv_members = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and not name.endswith("/")
    ]
    if len(csv_members) != 1:
        raise ValueError(f"Expected exactly one CSV; found {csv_members}")
    with archive.open(csv_members[0]) as csv_file:
        df = pd.read_csv(csv_file)

df = df.assign(
    vote_status=np.where(df["award_share"] > 0, "Received MVP votes", "No MVP votes")
)

section("1. PARTICIPATION SUMMARY")
participation_summary = (
    df.groupby("vote_status")[["g", "gs", "mp", "mp_per_g"]]
    .describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])
)
print(participation_summary.to_string())

positive_vote_getters = df.loc[df["award_share"] > 0].copy()
print("\nLowest-minute positive MVP vote-getters:")
low_vote_columns = [
    "season", "player", "team_id", "g", "gs", "mp", "mp_per_g",
    "pts_per_g", "ws", "vorp", "award_share",
]
print(
    positive_vote_getters.nsmallest(15, "mp")[low_vote_columns]
    .sort_values(["mp", "season", "player"])
    .to_string(index=False)
)

section("2. MINUTES-THRESHOLD TRADEOFFS")
minute_thresholds = [0, 1, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
threshold_rows = []
total_rows = len(df)
total_vote_getters = df["award_share"].gt(0).sum()
total_vote_share = df["award_share"].sum()
season_winner_index = df.groupby("season")["award_share"].idxmax()
season_winners = df.loc[season_winner_index]

for threshold in minute_thresholds:
    keep = df["mp"] >= threshold
    retained_vote_getters = (keep & df["award_share"].gt(0)).sum()
    retained_vote_share = df.loc[keep, "award_share"].sum()
    retained_winners = (season_winners["mp"] >= threshold).sum()
    threshold_rows.append(
        {
            "minimum_minutes": threshold,
            "rows_retained": int(keep.sum()),
            "rows_retained_pct": 100 * keep.mean(),
            "vote_getters_retained": int(retained_vote_getters),
            "vote_getters_retained_pct": 100 * retained_vote_getters / total_vote_getters,
            "vote_share_retained_pct": 100 * retained_vote_share / total_vote_share,
            "season_winners_retained": int(retained_winners),
            "season_winners_retained_pct": 100 * retained_winners / len(season_winners),
        }
    )

threshold_table = pd.DataFrame(threshold_rows)
print(threshold_table.to_string(index=False))

section("3. PRACTICAL FILTER OPTIONS")
filter_definitions = {
    "No filter": pd.Series(True, index=df.index),
    "Positive minutes": df["mp"] > 0,
    "At least 100 minutes": df["mp"] >= 100,
    "At least 250 minutes": df["mp"] >= 250,
    "At least 500 minutes": df["mp"] >= 500,
    "20+ games and 250+ minutes": (df["g"] >= 20) & (df["mp"] >= 250),
}
filter_rows = []
for label, keep in filter_definitions.items():
    lost_positive = df.loc[~keep & df["award_share"].gt(0)]
    filter_rows.append(
        {
            "filter": label,
            "rows_kept": int(keep.sum()),
            "rows_removed": int((~keep).sum()),
            "dataset_kept_pct": 100 * keep.mean(),
            "vote_getters_removed": int(len(lost_positive)),
            "vote_share_removed": lost_positive["award_share"].sum(),
            "winners_removed": int((season_winners["mp"].index.isin(lost_positive.index)).sum()),
        }
    )

filter_table = pd.DataFrame(filter_rows)
print(filter_table.to_string(index=False))

section("4. EXPOSURE BANDS AND ADVANCED-METRIC STABILITY")
minute_bins = [-0.001, 99, 249, 499, 999, 1999, np.inf]
minute_labels = ["0–99", "100–249", "250–499", "500–999", "1,000–1,999", "2,000+"]
df["minute_band"] = pd.cut(
    df["mp"], bins=minute_bins, labels=minute_labels, include_lowest=True, ordered=True
)

advanced_metrics = ["per", "bpm", "ws_per_48", "ts_pct"]
band_summary_rows = []
for metric in advanced_metrics:
    grouped = df.groupby("minute_band", observed=False)[metric]
    for band, values in grouped:
        band_summary_rows.append(
            {
                "metric": metric,
                "minute_band": str(band),
                "n": int(values.notna().sum()),
                "median": values.median(),
                "p05": values.quantile(0.05),
                "p95": values.quantile(0.95),
            }
        )

band_summary = pd.DataFrame(band_summary_rows)
print(band_summary.to_string(index=False))

section("5. EXTREME ADVANCED-METRIC OBSERVATIONS")
extreme_rows = []
for metric in advanced_metrics:
    valid = df[metric].dropna()
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    extreme = df[metric].lt(lower) | df[metric].gt(upper)
    extreme_rows.append(
        {
            "metric": metric,
            "lower_outer_fence": lower,
            "upper_outer_fence": upper,
            "extreme_rows": int(extreme.sum()),
            "extreme_with_mp_below_100": int((extreme & df["mp"].lt(100)).sum()),
            "extreme_with_mp_below_500": int((extreme & df["mp"].lt(500)).sum()),
            "extreme_vote_getters": int((extreme & df["award_share"].gt(0)).sum()),
        }
    )

extreme_summary = pd.DataFrame(extreme_rows)
print(extreme_summary.to_string(index=False))

for metric in advanced_metrics:
    print(f"\nLowest and highest five observations for {metric}:")
    display_columns = [
        "season", "player", "team_id", "g", "mp", "mp_per_g",
        metric, "award_share",
    ]
    extremes = pd.concat(
        [df.nsmallest(5, metric), df.nlargest(5, metric)], ignore_index=True
    )[display_columns]
    print(extremes.to_string(index=False))

section("6. PROFESSIONAL EDA FIGURES")
# Figure 1: participation and filtering tradeoffs.
fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))

for status, color in [("No MVP votes", GRAY), ("Received MVP votes", ORANGE)]:
    subset = df.loc[df["vote_status"] == status, "mp"] + 1
    sns.ecdfplot(
        x=subset,
        ax=axes[0],
        label=status,
        color=color,
        linewidth=2.5,
    )
axes[0].set_xscale("log")
axes[0].set_title("MVP vote-getters have substantially more playing time", loc="left")
axes[0].set_xlabel("Total season minutes + 1 (log scale)")
axes[0].set_ylabel("Cumulative share of player-seasons")
axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
axes[0].legend(loc="upper left")

axes[1].plot(
    threshold_table["minimum_minutes"],
    threshold_table["rows_retained_pct"],
    marker="o",
    linewidth=2.5,
    color=BLUE,
    label="All player-seasons retained",
)
axes[1].plot(
    threshold_table["minimum_minutes"],
    threshold_table["vote_getters_retained_pct"],
    marker="o",
    linewidth=2.5,
    color=ORANGE,
    label="MVP vote-getters retained",
)
axes[1].plot(
    threshold_table["minimum_minutes"],
    threshold_table["season_winners_retained_pct"],
    marker="o",
    linewidth=2,
    linestyle="--",
    color=GREEN,
    label="Season winners retained",
)
axes[1].set_ylim(0, 104)
axes[1].set_title("Higher minute thresholds trade realism for a smaller dataset", loc="left")
axes[1].set_xlabel("Minimum total minutes")
axes[1].set_ylabel("Observations retained")
axes[1].yaxis.set_major_formatter(PercentFormatter(100))
axes[1].legend(loc="lower left")

fig.suptitle(
    "Participation and candidate-filter tradeoffs",
    fontsize=20,
    fontweight="bold",
    color=NAVY,
    x=0.06,
    ha="left",
)
fig.text(
    0.06,
    0.93,
    "NBA player-seasons, 1982–2022. Thresholds use only regular-season minutes, not MVP outcomes.",
    fontsize=11,
    color=GRAY,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
participation_path = OUTPUT_DIR / "participation_and_filter_tradeoffs.png"
fig.savefig(participation_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {participation_path}")

# Figure 2: advanced-metric stability by minutes band.
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
metric_titles = {
    "per": "Player Efficiency Rating (PER)",
    "bpm": "Box Plus/Minus (BPM)",
    "ws_per_48": "Win shares per 48 minutes",
    "ts_pct": "True shooting percentage",
}

for ax, metric in zip(axes.flat, advanced_metrics):
    metric_data = band_summary.loc[band_summary["metric"] == metric].copy()
    x = np.arange(len(metric_data))
    lower_error = metric_data["median"] - metric_data["p05"]
    upper_error = metric_data["p95"] - metric_data["median"]
    ax.errorbar(
        x,
        metric_data["median"],
        yerr=np.vstack([lower_error, upper_error]),
        fmt="o-",
        color=BLUE,
        ecolor=LIGHT_BLUE,
        elinewidth=6,
        capsize=5,
        linewidth=2,
        markersize=7,
    )
    ax.axhline(metric_data["median"].iloc[-1], color=GRAY, linestyle=":", linewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_data["minute_band"], rotation=25, ha="right")
    ax.set_title(metric_titles[metric], loc="left")
    ax.set_xlabel("Total-minute band")
    ax.set_ylabel("Median and 5th–95th percentile")

fig.suptitle(
    "Low-minute samples produce unstable rate and efficiency statistics",
    fontsize=20,
    fontweight="bold",
    color=NAVY,
    x=0.06,
    ha="left",
)
fig.text(
    0.06,
    0.95,
    "Dots show medians; vertical ranges show the middle 90% of observations within each playing-time band.",
    fontsize=11,
    color=GRAY,
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
stability_path = OUTPUT_DIR / "advanced_metric_stability_by_minutes.png"
fig.savefig(stability_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {stability_path}")

print("\nChunk 4 EDA completed without filtering or altering the source dataset.")
