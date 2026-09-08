import os
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
GRAY = "#6B7280"
LIGHT_GRAY = "#D7DEE5"
POSITION_COLORS = {
    "PG": "#325D88",
    "SG": "#5B8DB8",
    "SF": "#D18A28",
    "PF": "#61A17A",
    "C": "#27705B",
}

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
        "grid.color": LIGHT_GRAY,
        "grid.linewidth": 0.8,
    },
)

ZIP_PATH = Path(os.environ.get("NBA_MVP_DATA_PATH", "data/raw/nba_mvp_stats.zip"))
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk6_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


with ZipFile(ZIP_PATH) as archive:
    csv_members = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and not name.endswith("/")
    ]
    if len(csv_members) != 1:
        raise ValueError(f"Expected exactly one CSV; found {csv_members}")
    with archive.open(csv_members[0]) as csv_file:
        df = pd.read_csv(csv_file)

# Descriptive EDA view only; this does not change the uploaded dataset.
eda = df.loc[df["mp"] >= 100].copy()
eda["received_votes"] = eda["award_share"].gt(0).astype(int)
eda["primary_position"] = eda["pos"].str.split("-").str[0]
eda["era"] = pd.cut(
    eda["season"],
    bins=[1981, 1991, 2001, 2011, 2022],
    labels=["1982–1991", "1992–2001", "2002–2011", "2012–2022"],
    include_lowest=True,
)

# As established previously, TOT team fields are placeholders rather than observations.
team_columns = ["mov", "mov_adj", "win_loss_pct"]
eda.loc[eda["team_id"].eq("TOT"), team_columns] = np.nan

section("1. POSITION FIELD AND PRIMARY-POSITION MAPPING")
position_counts = eda["pos"].value_counts().rename("player_seasons")
print("Original position labels:")
print(position_counts.to_string())
print("\nPrimary-position counts after taking the first listed position:")
print(eda["primary_position"].value_counts().reindex(["PG", "SG", "SF", "PF", "C"]).to_string())

section("2. MVP OUTCOMES BY PRIMARY POSITION")
winner_index = eda.groupby("season")["award_share"].idxmax()
winners = eda.loc[winner_index].copy()

position_summary = (
    eda.groupby("primary_position")
    .agg(
        player_seasons=("player", "size"),
        vote_getters=("received_votes", "sum"),
        vote_getter_rate=("received_votes", "mean"),
        total_award_share=("award_share", "sum"),
        median_positive_share=(
            "award_share",
            lambda s: s[s > 0].median() if (s > 0).any() else np.nan,
        ),
    )
    .reindex(["PG", "SG", "SF", "PF", "C"])
)
position_summary["vote_getter_rate_pct"] = 100 * position_summary["vote_getter_rate"]
position_summary["mvp_winners"] = winners["primary_position"].value_counts()
position_summary["mvp_winners"] = position_summary["mvp_winners"].fillna(0).astype(int)
print(position_summary.to_string())

section("3. VOTE-GETTER POSITION COMPOSITION BY ERA")
positive = eda.loc[eda["received_votes"].eq(1)].copy()
era_position_counts = pd.crosstab(positive["era"], positive["primary_position"])
era_position_counts = era_position_counts.reindex(
    index=["1982–1991", "1992–2001", "2002–2011", "2012–2022"],
    columns=["PG", "SG", "SF", "PF", "C"],
    fill_value=0,
)
era_position_pct = 100 * era_position_counts.div(era_position_counts.sum(axis=1), axis=0)
print("Vote-getter counts:")
print(era_position_counts.to_string())
print("\nWithin-era percentage composition:")
print(era_position_pct.to_string())

section("4. SEASON-BY-SEASON STATISTICAL ENVIRONMENT")
trend_features = ["pts_per_g", "fg3a_per_g", "ts_pct", "bpm"]
trend_rows = []
for season, season_frame in eda.groupby("season"):
    candidate_frame = season_frame.loc[season_frame["received_votes"].eq(1)]
    row = {"season": season}
    for feature in trend_features:
        row[f"league_median_{feature}"] = season_frame[feature].median()
        row[f"league_p90_{feature}"] = season_frame[feature].quantile(0.90)
        row[f"vote_getter_median_{feature}"] = candidate_frame[feature].median()
    trend_rows.append(row)

season_trends = pd.DataFrame(trend_rows).set_index("season")
endpoint_comparison = pd.concat(
    [season_trends.loc[[1982]], season_trends.loc[[2022]]]
)
print("Selected environment measures in the first and last seasons:")
print(endpoint_comparison.to_string())

section("5. RAW VERSUS WITHIN-SEASON ASSOCIATIONS")
normalization_features = [
    "pts_per_g", "trb_per_g", "ast_per_g", "mp_per_g", "per",
    "ts_pct", "ws", "ws_per_48", "bpm", "vorp", "win_loss_pct",
]

normalization_rows = []
for feature in normalization_features:
    percentile_column = f"{feature}_season_pct"
    zscore_column = f"{feature}_season_z"
    eda[percentile_column] = eda.groupby("season")[feature].transform(
        lambda s: s.rank(method="average", pct=True)
    )
    eda[zscore_column] = eda.groupby("season")[feature].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else np.nan
    )

    all_frame = eda[[feature, percentile_column, zscore_column, "award_share"]].dropna()
    positive_frame = eda.loc[
        eda["received_votes"].eq(1),
        [feature, percentile_column, zscore_column, "award_share"],
    ].dropna()

    normalization_rows.append(
        {
            "feature": feature,
            "n_all": len(all_frame),
            "spearman_all_raw": all_frame[feature].corr(
                all_frame["award_share"], method="spearman"
            ),
            "spearman_all_percentile": all_frame[percentile_column].corr(
                all_frame["award_share"], method="spearman"
            ),
            "pearson_all_raw": all_frame[feature].corr(
                all_frame["award_share"], method="pearson"
            ),
            "pearson_all_zscore": all_frame[zscore_column].corr(
                all_frame["award_share"], method="pearson"
            ),
            "n_positive": len(positive_frame),
            "spearman_positive_raw": positive_frame[feature].corr(
                positive_frame["award_share"], method="spearman"
            ),
            "spearman_positive_percentile": positive_frame[percentile_column].corr(
                positive_frame["award_share"], method="spearman"
            ),
        }
    )

normalization_table = pd.DataFrame(normalization_rows)
normalization_table["positive_percentile_gain"] = (
    normalization_table["spearman_positive_percentile"]
    - normalization_table["spearman_positive_raw"]
)
normalization_table["all_percentile_gain"] = (
    normalization_table["spearman_all_percentile"]
    - normalization_table["spearman_all_raw"]
)
print(normalization_table.to_string(index=False))

section("6. PROFESSIONAL EDA FIGURES")
# Figure 1: the statistical environment through time.
fig, axes = plt.subplots(2, 2, figsize=(16, 10.5))
trend_titles = {
    "pts_per_g": "Points per game",
    "fg3a_per_g": "Three-point attempts per game",
    "ts_pct": "True shooting percentage",
    "bpm": "Box Plus/Minus",
}

for ax, feature in zip(axes.flat, trend_features):
    league_series = season_trends[f"league_median_{feature}"]
    candidate_series = season_trends[f"vote_getter_median_{feature}"]
    ax.plot(
        season_trends.index,
        league_series,
        color=GRAY,
        linewidth=1.2,
        alpha=0.45,
    )
    ax.plot(
        season_trends.index,
        league_series.rolling(3, center=True, min_periods=1).median(),
        color=GRAY,
        linewidth=2.5,
        label="League median",
    )
    ax.plot(
        season_trends.index,
        candidate_series,
        color=ORANGE,
        linewidth=1.2,
        alpha=0.35,
    )
    ax.plot(
        season_trends.index,
        candidate_series.rolling(3, center=True, min_periods=1).median(),
        color=ORANGE,
        linewidth=2.5,
        label="Vote-getter median",
    )
    ax.set_title(trend_titles[feature], loc="left")
    ax.set_xlabel("Season")
    ax.set_ylabel(trend_titles[feature])
    if feature == "ts_pct":
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

legend_handles = [
    Line2D([0], [0], color=GRAY, linewidth=3, label="League median"),
    Line2D([0], [0], color=ORANGE, linewidth=3, label="Vote-getter median"),
]
fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.94, 0.925), ncol=2)
fig.suptitle(
    "The statistical environment changed substantially from 1982 to 2022",
    fontsize=20,
    fontweight="bold",
    color=NAVY,
    x=0.06,
    y=0.995,
    ha="left",
)
fig.text(
    0.06,
    0.93,
    "Thin lines show annual values; heavy lines show centered three-season medians. The EDA population uses at least 100 minutes.",
    fontsize=10.5,
    color=GRAY,
)
fig.tight_layout(rect=[0, 0, 1, 0.87])
trend_path = OUTPUT_DIR / "statistical_environment_by_season.png"
fig.savefig(trend_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {trend_path}")

# Figure 2: position representation and historical composition.
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

bottom = np.zeros(len(era_position_pct))
x = np.arange(len(era_position_pct))
for position in ["PG", "SG", "SF", "PF", "C"]:
    values = era_position_pct[position].to_numpy()
    axes[0].bar(
        x,
        values,
        bottom=bottom,
        label=position,
        color=POSITION_COLORS[position],
        width=0.68,
    )
    bottom += values
axes[0].set_xticks(x)
axes[0].set_xticklabels(era_position_pct.index)
axes[0].set_ylim(0, 100)
axes[0].yaxis.set_major_formatter(PercentFormatter(100))
axes[0].set_title("Vote-getter position mix changed across eras", loc="left")
axes[0].set_xlabel("Era")
axes[0].set_ylabel("Share of positive vote-getters")
axes[0].legend(title="Primary position", ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))

ordered_positions = ["PG", "SG", "SF", "PF", "C"]
bars = axes[1].bar(
    ordered_positions,
    position_summary.loc[ordered_positions, "vote_getter_rate_pct"],
    color=[POSITION_COLORS[p] for p in ordered_positions],
    width=0.65,
)
for bar, value in zip(bars, position_summary.loc[ordered_positions, "vote_getter_rate_pct"]):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.12,
        f"{value:.1f}%",
        ha="center",
        fontsize=10.5,
        color=NAVY,
        fontweight="bold",
    )
axes[1].set_title("Point guards have the highest historical vote-getter rate", loc="left")
axes[1].set_xlabel("Primary listed position")
axes[1].set_ylabel("Player-seasons receiving MVP votes")
axes[1].yaxis.set_major_formatter(PercentFormatter(100))
axes[1].set_ylim(0, position_summary["vote_getter_rate_pct"].max() * 1.18)

fig.suptitle(
    "Position shapes the historical MVP candidate pool",
    fontsize=20,
    fontweight="bold",
    color=NAVY,
    x=0.06,
    y=0.995,
    ha="left",
)
fig.text(
    0.06,
    0.925,
    "Hybrid labels are summarized by their first listed position for this descriptive view.",
    fontsize=10.5,
    color=GRAY,
)
fig.tight_layout(rect=[0, 0.06, 1, 0.87])
position_path = OUTPUT_DIR / "position_and_mvp_representation.png"
fig.savefig(position_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {position_path}")

# Figure 3: raw versus within-season percentile association.
plot_table = normalization_table.copy().sort_values("spearman_positive_percentile")
y = np.arange(len(plot_table))
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

panels = [
    ("spearman_all_raw", "spearman_all_percentile", "All player-seasons"),
    ("spearman_positive_raw", "spearman_positive_percentile", "Positive vote-getters only"),
]
for ax, (raw_col, pct_col, title) in zip(axes, panels):
    ax.axvline(0, color=NAVY, linewidth=1)
    for position, (_, row) in zip(y, plot_table.iterrows()):
        ax.plot(
            [row[raw_col], row[pct_col]],
            [position, position],
            color=LIGHT_GRAY,
            linewidth=3,
        )
        ax.scatter(row[raw_col], position, s=70, color=BLUE, zorder=3)
        ax.scatter(
            row[pct_col],
            position,
            s=70,
            facecolor="white",
            edgecolor=ORANGE,
            linewidth=2.2,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_table["feature"] if ax is axes[0] else [])
    ax.set_xlabel("Spearman correlation with award_share")
    ax.set_title(title, loc="left")

legend_items = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
           markersize=8, label="Raw feature"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
           markeredgecolor=ORANGE, markeredgewidth=2, markersize=8,
           label="Within-season percentile"),
]
fig.legend(handles=legend_items, loc="upper right", bbox_to_anchor=(0.94, 0.925), ncol=2)
fig.suptitle(
    "Within-season percentiles modestly improve several cross-era relationships",
    fontsize=20,
    fontweight="bold",
    color=NAVY,
    x=0.06,
    y=0.995,
    ha="left",
)
fig.text(
    0.06,
    0.925,
    "Percentiles compare each player only with other players in the same season and use no MVP target information.",
    fontsize=10.5,
    color=GRAY,
)
fig.tight_layout(rect=[0, 0, 1, 0.87])
normalization_path = OUTPUT_DIR / "raw_vs_season_percentile_associations.png"
fig.savefig(normalization_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {normalization_path}")

print("\nChunk 6 EDA completed without creating the final preparation pipeline or modifying the source dataset.")
