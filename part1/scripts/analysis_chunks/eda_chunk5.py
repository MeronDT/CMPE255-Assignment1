import os
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
GROUP_COLORS = {
    "Traditional / participation": BLUE,
    "Advanced": ORANGE,
    "Team": GREEN,
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
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk5_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 98}\n{title}\n{'=' * 98}")


def feature_group(column: str) -> str:
    team_features = {"mov", "mov_adj", "win_loss_pct"}
    advanced_features = {
        "per", "ts_pct", "fg3a_per_fga_pct", "fta_per_fga_pct",
        "orb_pct", "drb_pct", "trb_pct", "ast_pct", "stl_pct",
        "blk_pct", "tov_pct", "usg_pct", "ows", "dws", "ws",
        "ws_per_48", "obpm", "dbpm", "bpm", "vorp",
    }
    if column in team_features:
        return "Team"
    if column in advanced_features:
        return "Advanced"
    return "Traditional / participation"


def binned_medians(frame: pd.DataFrame, feature: str, bins: int = 10) -> pd.DataFrame:
    working = frame[[feature, "award_share"]].dropna().copy()
    distinct_values = working[feature].nunique()
    q = min(bins, distinct_values)
    if q < 2:
        return pd.DataFrame(columns=[feature, "award_share"])
    working["bin"] = pd.qcut(working[feature], q=q, duplicates="drop")
    return (
        working.groupby("bin", observed=False)
        .agg(**{feature: (feature, "median")}, award_share=("award_share", "median"))
        .reset_index(drop=True)
    )


with ZipFile(ZIP_PATH) as archive:
    csv_members = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and not name.endswith("/")
    ]
    if len(csv_members) != 1:
        raise ValueError(f"Expected exactly one CSV; found {csv_members}")
    with archive.open(csv_members[0]) as csv_file:
        df = pd.read_csv(csv_file)

# Use the provisional 100-minute EDA population. This is a view, not a source edit.
eda = df.loc[df["mp"] >= 100].copy()
eda["received_votes"] = eda["award_share"].gt(0).astype(int)
eda["vote_status"] = np.where(
    eda["received_votes"].eq(1), "Received MVP votes", "No MVP votes"
)

# TOT team values were verified as placeholders. Mark them unavailable only for
# analyses involving team context; player statistics and targets remain intact.
team_columns = ["mov", "mov_adj", "win_loss_pct"]
eda.loc[eda["team_id"].eq("TOT"), team_columns] = np.nan

section("1. EDA POPULATION")
print(f"Original rows: {len(df):,}")
print(f"Rows with at least 100 minutes: {len(eda):,} ({len(eda) / len(df):.2%})")
print(f"Positive MVP vote-getters retained: {eda['received_votes'].sum():,} of {df['award_share'].gt(0).sum():,}")
print(f"TOT rows with team fields treated as unavailable: {eda['team_id'].eq('TOT').sum():,}")

section("2. NUMERIC ASSOCIATIONS WITH AWARD SHARE")
excluded_numeric = {"season", "award_share", "received_votes"}
numeric_features = [
    column for column in eda.select_dtypes(include=np.number).columns
    if column not in excluded_numeric
]

association_rows = []
positive = eda.loc[eda["received_votes"].eq(1)]
for feature in numeric_features:
    all_pair = eda[[feature, "award_share", "received_votes"]].dropna()
    positive_pair = positive[[feature, "award_share"]].dropna()
    association_rows.append(
        {
            "feature": feature,
            "group": feature_group(feature),
            "n_all": len(all_pair),
            "spearman_all": all_pair[feature].corr(all_pair["award_share"], method="spearman"),
            "pearson_all": all_pair[feature].corr(all_pair["award_share"], method="pearson"),
            "corr_with_vote_indicator": all_pair[feature].corr(all_pair["received_votes"], method="pearson"),
            "n_positive": len(positive_pair),
            "spearman_positive": positive_pair[feature].corr(
                positive_pair["award_share"], method="spearman"
            ),
        }
    )

associations = pd.DataFrame(association_rows)
associations["abs_spearman_all"] = associations["spearman_all"].abs()
associations = associations.sort_values("abs_spearman_all", ascending=False)

print("Top 20 features by absolute Spearman association with award_share:")
print(
    associations[
        [
            "feature", "group", "n_all", "spearman_all", "pearson_all",
            "corr_with_vote_indicator", "n_positive", "spearman_positive",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nTop 15 associations among positive vote-getters only:")
print(
    associations.assign(abs_positive=associations["spearman_positive"].abs())
    .sort_values("abs_positive", ascending=False)[
        ["feature", "group", "n_positive", "spearman_positive", "spearman_all"]
    ]
    .head(15)
    .to_string(index=False)
)

section("3. DESCRIPTIVE CANDIDATE PROFILE")
profile_features = [
    "pts_per_g", "trb_per_g", "ast_per_g", "mp_per_g",
    "per", "ws", "bpm", "vorp", "win_loss_pct",
]
profile_rows = []
for feature in profile_features:
    for status, group in eda.groupby("vote_status"):
        values = group[feature].dropna()
        profile_rows.append(
            {
                "feature": feature,
                "vote_status": status,
                "n": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "p25": values.quantile(0.25),
                "p75": values.quantile(0.75),
            }
        )

profile_table = pd.DataFrame(profile_rows)
print(profile_table.to_string(index=False))

section("4. TEAM SUCCESS AND MVP VOTING")
team_analysis = eda.dropna(subset=["win_loss_pct"]).copy()
win_bins = [0, 0.400, 0.500, 0.600, 0.700, 1.001]
win_labels = ["Below .400", ".400–.499", ".500–.599", ".600–.699", ".700+"]
team_analysis["team_win_band"] = pd.cut(
    team_analysis["win_loss_pct"],
    bins=win_bins,
    labels=win_labels,
    right=False,
    include_lowest=True,
)
team_vote_summary = (
    team_analysis.groupby("team_win_band", observed=False)
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
)
team_vote_summary["vote_getter_rate_pct"] = 100 * team_vote_summary["vote_getter_rate"]
print(team_vote_summary.to_string())

section("5. PROFESSIONAL EDA FIGURES")
# Figure 1: association ranking, separating all-player and positive-only effects.
top_features = associations.head(14).sort_values("spearman_all")
y = np.arange(len(top_features))

fig, ax = plt.subplots(figsize=(12.5, 8.5))
ax.axvline(0, color=NAVY, linewidth=1)
for position, (_, row) in zip(y, top_features.iterrows()):
    ax.plot(
        [row["spearman_all"], row["spearman_positive"]],
        [position, position],
        color=LIGHT_GRAY,
        linewidth=3,
        zorder=1,
    )
    ax.scatter(
        row["spearman_all"],
        position,
        s=85,
        color=GROUP_COLORS[row["group"]],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.scatter(
        row["spearman_positive"],
        position,
        s=75,
        facecolor="white",
        edgecolor=GROUP_COLORS[row["group"]],
        linewidth=2,
        zorder=3,
    )

ax.set_yticks(y)
ax.set_yticklabels(top_features["feature"])
ax.set_xlabel("Spearman rank correlation with award_share")
ax.set_ylabel("")
ax.set_title("Volume and all-in-one impact metrics are most associated with MVP share", loc="left", pad=18)
ax.text(
    0,
    1.02,
    "Filled circles: all player-seasons. Open circles: positive vote-getters only.",
    transform=ax.transAxes,
    fontsize=11,
    color=GRAY,
)

legend_items = [
    Patch(facecolor=GROUP_COLORS[label], label=label)
    for label in GROUP_COLORS
]
legend_items.extend(
    [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NAVY,
               markeredgecolor="white", markersize=9, label="All player-seasons"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=NAVY, markeredgewidth=2, markersize=9,
               label="Vote-getters only"),
    ]
)
ax.legend(handles=legend_items, loc="lower right", fontsize=10)
fig.suptitle(
    "Which statistics move with MVP vote share?",
    fontsize=21,
    fontweight="bold",
    color=NAVY,
    x=0.07,
    y=0.995,
    ha="left",
)
fig.text(
    0.07,
    0.925,
    "EDA population: 15,847 player-seasons with at least 100 minutes; team placeholders excluded from team-feature correlations.",
    fontsize=10.5,
    color=GRAY,
)
fig.tight_layout(rect=[0, 0, 1, 0.87])
association_path = OUTPUT_DIR / "mvp_share_association_ranking.png"
fig.savefig(association_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {association_path}")

# Figure 2: candidate versus non-candidate distributions.
comparison_features = ["pts_per_g", "ws", "vorp", "win_loss_pct"]
comparison_titles = {
    "pts_per_g": "Points per game",
    "ws": "Win shares",
    "vorp": "Value over replacement player",
    "win_loss_pct": "Team win percentage",
}
fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
palette = {"No MVP votes": "#A8B2BD", "Received MVP votes": ORANGE}

for ax, feature in zip(axes.flat, comparison_features):
    plot_data = eda.dropna(subset=[feature])
    sns.boxplot(
        data=plot_data,
        x="vote_status",
        y=feature,
        hue="vote_status",
        order=["No MVP votes", "Received MVP votes"],
        palette=palette,
        showfliers=False,
        width=0.55,
        legend=False,
        ax=ax,
    )
    ax.set_title(comparison_titles[feature], loc="left")
    ax.set_xlabel("")
    ax.set_ylabel(comparison_titles[feature])
    ax.tick_params(axis="x", labelsize=10)
    if feature == "win_loss_pct":
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

fig.suptitle(
    "MVP vote-getters separate sharply on production, impact, and team success",
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
    "Boxes show medians and interquartile ranges; extreme display outliers are hidden for readability, not removed from analysis.",
    fontsize=10.5,
    color=GRAY,
)
fig.tight_layout(rect=[0, 0, 1, 0.87])
profile_path = OUTPUT_DIR / "candidate_profile_comparison.png"
fig.savefig(profile_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {profile_path}")

# Figure 3: magnitude of vote share among actual vote-getters.
positive_plot = positive.copy()
positive_plot.loc[positive_plot["team_id"].eq("TOT"), team_columns] = np.nan
scatter_features = ["pts_per_g", "ws", "vorp", "win_loss_pct"]

fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
for ax, feature in zip(axes.flat, scatter_features):
    plot_data = positive_plot.dropna(subset=[feature, "award_share"])
    ax.scatter(
        plot_data[feature],
        plot_data["award_share"],
        s=28,
        alpha=0.32,
        color=BLUE,
        edgecolors="none",
    )
    trend = binned_medians(plot_data, feature)
    ax.plot(
        trend[feature],
        trend["award_share"],
        color=ORANGE,
        marker="o",
        linewidth=2.5,
        markersize=5,
        label="Decile median",
    )
    correlation = plot_data[feature].corr(plot_data["award_share"], method="spearman")
    ax.text(
        0.03,
        0.94,
        f"Spearman ρ = {correlation:.2f}",
        transform=ax.transAxes,
        fontsize=11,
        color=NAVY,
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": LIGHT_GRAY},
    )
    ax.set_title(comparison_titles[feature], loc="left")
    ax.set_xlabel(comparison_titles[feature])
    ax.set_ylabel("MVP award share")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    if feature == "win_loss_pct":
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))

fig.suptitle(
    "Among vote-getters, stronger performance generally corresponds to larger vote share",
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
    "Blue points are positive vote-getters; orange lines connect within-feature decile medians. Relationships are descriptive, not causal.",
    fontsize=10.5,
    color=GRAY,
)
fig.tight_layout(rect=[0, 0, 1, 0.87])
positive_path = OUTPUT_DIR / "positive_vote_share_relationships.png"
fig.savefig(positive_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {positive_path}")

# Figure 4: team-win band and probability of receiving votes.
fig, ax = plt.subplots(figsize=(11.5, 6.5))
bars = ax.bar(
    team_vote_summary.index.astype(str),
    team_vote_summary["vote_getter_rate_pct"],
    color=["#A8B2BD", "#8DA5B8", "#6C91AC", "#4A7FA3", GREEN],
    width=0.68,
)
for bar, value in zip(bars, team_vote_summary["vote_getter_rate_pct"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.25,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        color=NAVY,
        fontweight="bold",
    )
ax.set_title("Players on winning teams are much more likely to receive MVP votes", loc="left", pad=16)
ax.set_xlabel("Team regular-season win percentage")
ax.set_ylabel("Player-seasons receiving MVP votes")
ax.yaxis.set_major_formatter(PercentFormatter(100))
ax.set_ylim(0, team_vote_summary["vote_getter_rate_pct"].max() * 1.18)
fig.suptitle(
    "Team success and MVP consideration",
    fontsize=20,
    fontweight="bold",
    color=NAVY,
    x=0.08,
    ha="left",
)
fig.text(
    0.08,
    0.92,
    "TOT rows are excluded because their team records are neutral placeholders.",
    fontsize=10.5,
    color=GRAY,
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
team_path = OUTPUT_DIR / "team_success_and_mvp_votes.png"
fig.savefig(team_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {team_path}")

print("\nChunk 5 EDA completed without modifying the source dataset.")
