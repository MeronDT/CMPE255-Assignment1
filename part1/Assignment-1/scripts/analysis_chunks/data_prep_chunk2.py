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
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")

NAVY = "#17324D"
BLUE = "#2F6B9A"
ORANGE = "#D97706"
GREEN = "#2F855A"
GRAY = "#6B7280"
LIGHT_GRAY = "#D7DEE5"

sns.set_theme(
    style="whitegrid",
    context="talk",
    rc={
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "grid.color": LIGHT_GRAY,
        "grid.linewidth": 0.8,
    },
)

ZIP_PATH = Path(os.environ.get("NBA_MVP_DATA_PATH", "data/raw/nba_mvp_stats.zip"))
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk10_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 112}\n{title}\n{'=' * 112}")


def add_season_percentiles(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return a copy with same-season percentile ranks; the target is never referenced."""
    result = frame.copy()
    for feature in features:
        result[f"{feature}_season_pct"] = result.groupby("season")[feature].rank(
            method="average",
            pct=True,
            na_option="keep",
        )
    return result


with ZipFile(ZIP_PATH) as archive:
    csv_members = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and not name.endswith("/")
    ]
    if len(csv_members) != 1:
        raise ValueError(f"Expected exactly one CSV; found {csv_members}")
    with archive.open(csv_members[0]) as csv_file:
        source = pd.read_csv(csv_file)

# Reconstruct the approved base-table treatments from the preceding chunk.
prepared = source.copy()
prepared.insert(0, "source_row_id", np.arange(len(prepared), dtype=np.int64))
prepared = prepared.loc[prepared["mp"].ge(100)].copy()
prepared["is_tot"] = prepared["team_id"].eq("TOT").astype("int8")
prepared.loc[
    prepared["is_tot"].eq(1),
    ["mov", "mov_adj", "win_loss_pct"],
] = np.nan

percentage_attempts = {
    "fg_pct": "fga_per_g",
    "fg2_pct": "fg2a_per_g",
    "fg3_pct": "fg3a_per_g",
    "ft_pct": "fta_per_g",
}
for percentage in percentage_attempts:
    prepared[f"{percentage}_was_missing"] = prepared[percentage].isna().astype("int8")
    prepared[percentage] = prepared[percentage].fillna(0.0)
prepared["pos_primary"] = prepared["pos"].str.split("-").str[0]

base_columns = prepared.columns.tolist()

section("1. PROPOSED ERA-AWARE FEATURE CANDIDATES — AUDIT BEFORE CREATION")
candidate_features = [
    "pts_per_g",
    "mp_per_g",
    "ts_pct",
    "per",
    "ws",
    "ws_per_48",
    "bpm",
    "vorp",
    "win_loss_pct",
]
feature_rationales = {
    "pts_per_g": "Scoring prominence relative to contemporaries",
    "mp_per_g": "Role and playing-time prominence",
    "ts_pct": "Scoring efficiency relative to the season environment",
    "per": "Box-score rate productivity",
    "ws": "Cumulative contribution",
    "ws_per_48": "Contribution rate independent of raw minutes",
    "bpm": "All-around impact rate",
    "vorp": "Cumulative value above replacement",
    "win_loss_pct": "Team success relative to same-season competition",
}
candidate_audit = pd.DataFrame(
    {
        "raw_feature": candidate_features,
        "missing_rows": [prepared[feature].isna().sum() for feature in candidate_features],
        "unique_values": [prepared[feature].nunique(dropna=True) for feature in candidate_features],
        "minimum": [prepared[feature].min() for feature in candidate_features],
        "maximum": [prepared[feature].max() for feature in candidate_features],
        "basketball_rationale": [feature_rationales[feature] for feature in candidate_features],
    }
)
print(candidate_audit.to_string(index=False))
print(
    "\nThe 1,769 missing team win percentages are intentional TOT records. "
    "They will remain missing in their season-percentile counterpart."
)

section("2. INVESTIGATE THREE ERA-NORMALIZATION OPTIONS")
comparison_rows = []
temporary_percentiles = add_season_percentiles(prepared, candidate_features)

for feature in candidate_features:
    grouped = prepared.groupby("season")[feature]
    season_mean = grouped.transform("mean")
    season_std = grouped.transform("std").replace(0, np.nan)
    season_z = (prepared[feature] - season_mean) / season_std

    season_median = grouped.transform("median")
    season_q1 = grouped.transform(lambda values: values.quantile(0.25))
    season_q3 = grouped.transform(lambda values: values.quantile(0.75))
    season_iqr = (season_q3 - season_q1).replace(0, np.nan)
    season_robust_z = (prepared[feature] - season_median) / season_iqr

    percentile = temporary_percentiles[f"{feature}_season_pct"]
    comparison_rows.append(
        {
            "feature": feature,
            "raw_to_percentile_spearman": prepared[feature].corr(
                percentile, method="spearman"
            ),
            "percentile_minimum": percentile.min(),
            "percentile_maximum": percentile.max(),
            "max_absolute_season_z": season_z.abs().max(),
            "max_absolute_robust_z": season_robust_z.abs().max(),
            "percentile_missing_rows": percentile.isna().sum(),
        }
    )

normalization_comparison = pd.DataFrame(comparison_rows)
print(normalization_comparison.to_string(index=False))
print(
    "\nDecision: use percentile ranks. They are bounded, interpretable, resistant to "
    "extreme rate statistics, and retain the ordering of players within each season. "
    "Raw features will also be retained so absolute magnitude is not lost."
)

section("3. QUANTIFY OBSERVED ERA SHIFTS")
era_metrics = ["pts_per_g", "fg3a_per_g", "ts_pct"]
era_rows = []
for season in [1982, 1990, 2000, 2010, 2022]:
    season_rows = prepared.loc[prepared["season"].eq(season)]
    positive_rows = season_rows.loc[season_rows["award_share"].gt(0)]
    for metric in era_metrics:
        era_rows.append(
            {
                "season": season,
                "metric": metric,
                "league_median": season_rows[metric].median(),
                "league_90th_percentile": season_rows[metric].quantile(0.90),
                "vote_getter_median": positive_rows[metric].median(),
            }
        )
era_table = pd.DataFrame(era_rows)
print(era_table.to_string(index=False))

section("4. RAW VERSUS WITHIN-SEASON PERCENTILE ASSOCIATION WITH MVP VOTE SHARE")
association_rows = []
positive_mask = prepared["award_share"].gt(0)
for feature in candidate_features:
    percentile_feature = f"{feature}_season_pct"
    raw_all = prepared[feature].corr(prepared["award_share"], method="spearman")
    pct_all = temporary_percentiles[percentile_feature].corr(
        temporary_percentiles["award_share"], method="spearman"
    )
    raw_positive = prepared.loc[positive_mask, feature].corr(
        prepared.loc[positive_mask, "award_share"], method="spearman"
    )
    pct_positive = temporary_percentiles.loc[positive_mask, percentile_feature].corr(
        temporary_percentiles.loc[positive_mask, "award_share"], method="spearman"
    )
    association_rows.append(
        {
            "feature": feature,
            "raw_spearman_all_rows": raw_all,
            "percentile_spearman_all_rows": pct_all,
            "raw_spearman_vote_getters": raw_positive,
            "percentile_spearman_vote_getters": pct_positive,
            "vote_getter_change": pct_positive - raw_positive,
        }
    )
association_table = pd.DataFrame(association_rows).sort_values(
    "percentile_spearman_vote_getters", ascending=False
)
print(association_table.to_string(index=False))
print(
    "\nThese correlations are descriptive evidence, not a fitted feature-selection step. "
    "The target is not used in the percentile calculation."
)

section("5. ADD THE APPROVED WITHIN-SEASON PERCENTILE FEATURES")
prepared_era = add_season_percentiles(prepared, candidate_features)
new_percentile_features = [f"{feature}_season_pct" for feature in candidate_features]

feature_catalog = pd.DataFrame(
    {
        "engineered_feature": new_percentile_features,
        "source_feature": candidate_features,
        "rationale": [feature_rationales[feature] for feature in candidate_features],
    }
)
print(feature_catalog.to_string(index=False))

section("6. FEATURE-ENGINEERING QUALITY AND LEAKAGE CHECKS")
raw_values_unchanged = all(
    np.allclose(
        prepared_era[feature],
        prepared[feature],
        equal_nan=True,
    )
    for feature in candidate_features
)
percentiles_in_bounds = all(
    prepared_era[feature].dropna().between(0, 1, inclusive="both").all()
    for feature in new_percentile_features
)
no_infinite_values = not np.isinf(
    prepared_era[new_percentile_features].to_numpy(dtype=float)
).any()

# Explicit leakage test: change the target and independently recompute the features.
# Because add_season_percentiles never reads award_share, results must remain identical.
target_perturbed = prepared.copy()
target_perturbed["award_share"] = target_perturbed["award_share"].iloc[::-1].to_numpy()
perturbed_features = add_season_percentiles(target_perturbed, candidate_features)
target_independence = all(
    np.allclose(
        prepared_era[feature],
        perturbed_features[feature],
        equal_nan=True,
    )
    for feature in new_percentile_features
)

non_tot_wlp_complete = prepared_era.loc[
    prepared_era["is_tot"].eq(0), "win_loss_pct_season_pct"
].notna().all()
tot_wlp_missing = prepared_era.loc[
    prepared_era["is_tot"].eq(1), "win_loss_pct_season_pct"
].isna().all()

quality_checks = pd.DataFrame(
    [
        ("Rows unchanged", len(prepared_era), len(prepared), len(prepared_era) == len(prepared)),
        (
            "Source row order unchanged",
            prepared_era["source_row_id"].equals(prepared["source_row_id"]),
            True,
            prepared_era["source_row_id"].equals(prepared["source_row_id"]),
        ),
        ("Raw candidate values unchanged", raw_values_unchanged, True, raw_values_unchanged),
        (
            "Engineered percentile columns added",
            len(new_percentile_features),
            len(candidate_features),
            len(new_percentile_features) == len(candidate_features),
        ),
        ("All nonmissing percentiles in [0, 1]", percentiles_in_bounds, True, percentiles_in_bounds),
        ("No infinite engineered values", no_infinite_values, True, no_infinite_values),
        ("Features independent of award_share", target_independence, True, target_independence),
        ("Non-TOT team percentiles complete", non_tot_wlp_complete, True, non_tot_wlp_complete),
        ("TOT team percentiles remain missing", tot_wlp_missing, True, tot_wlp_missing),
        (
            "Target unchanged",
            prepared_era["award_share"].equals(prepared["award_share"]),
            True,
            prepared_era["award_share"].equals(prepared["award_share"]),
        ),
    ],
    columns=["check", "actual", "expected", "passed"],
)
print(quality_checks.to_string(index=False))
if not quality_checks["passed"].all():
    raise AssertionError("At least one era-feature quality check failed.")

engineered_summary = prepared_era[new_percentile_features].agg(
    ["count", "min", "median", "max"]
).T
engineered_summary["missing"] = prepared_era[new_percentile_features].isna().sum()
print("\nEngineered-feature summary:")
print(engineered_summary.to_string())
print(f"\nPrepared shape before era features: {prepared.shape}")
print(f"Prepared shape after era features:  {prepared_era.shape}")

section("7. FIGURE 1 — READABLE ERA-SHIFT TRENDS")
annual_rows = []
for season, season_rows in prepared.groupby("season"):
    vote_getters = season_rows.loc[season_rows["award_share"].gt(0)]
    annual_rows.append(
        {
            "season": season,
            "league_median_3pa": season_rows["fg3a_per_g"].median(),
            "league_p90_3pa": season_rows["fg3a_per_g"].quantile(0.90),
            "vote_getter_median_3pa": vote_getters["fg3a_per_g"].median(),
            "league_median_ts": season_rows["ts_pct"].median(),
            "league_p90_ts": season_rows["ts_pct"].quantile(0.90),
            "vote_getter_median_ts": vote_getters["ts_pct"].median(),
        }
    )
annual = pd.DataFrame(annual_rows)

fig, axes = plt.subplots(2, 1, figsize=(17, 11), sharex=True)
fig.suptitle(
    "Why Era-Aware Features Are Necessary",
    fontsize=23,
    fontweight="bold",
    color=NAVY,
    y=0.99,
)

ax = axes[0]
ax.plot(annual["season"], annual["league_median_3pa"], color=BLUE, linewidth=2.5, label="League median")
ax.plot(annual["season"], annual["league_p90_3pa"], color=GREEN, linewidth=2.5, label="League 90th percentile")
ax.plot(
    annual["season"],
    annual["vote_getter_median_3pa"],
    color=ORANGE,
    linewidth=2.8,
    linestyle="--",
    label="MVP vote-getter median",
)
ax.set_title("A. Three-point attempts per game changed dramatically")
ax.set_ylabel("3-point attempts per game")
ax.legend(loc="upper left", ncol=3)

ax = axes[1]
ax.plot(annual["season"], annual["league_median_ts"], color=BLUE, linewidth=2.5, label="League median")
ax.plot(annual["season"], annual["league_p90_ts"], color=GREEN, linewidth=2.5, label="League 90th percentile")
ax.plot(
    annual["season"],
    annual["vote_getter_median_ts"],
    color=ORANGE,
    linewidth=2.8,
    linestyle="--",
    label="MVP vote-getter median",
)
ax.set_title("B. True-shooting efficiency also shifted upward")
ax.set_xlabel("Season")
ax.set_ylabel("True-shooting percentage")
ax.legend(loc="upper left", ncol=3)
ax.set_xticks(np.arange(1982, 2023, 5))

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
era_figure_path = OUTPUT_DIR / "era_shift_trends.png"
fig.savefig(era_figure_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(era_figure_path.resolve())

section("8. FIGURE 2 — RAW VERSUS SEASON-PERCENTILE ASSOCIATIONS")
friendly_labels = {
    "pts_per_g": "Points per game",
    "mp_per_g": "Minutes per game",
    "ts_pct": "True-shooting percentage",
    "per": "Player efficiency rating",
    "ws": "Win shares",
    "ws_per_48": "Win shares per 48",
    "bpm": "Box plus/minus",
    "vorp": "Value over replacement",
    "win_loss_pct": "Team win percentage",
}
plot_associations = association_table.sort_values("percentile_spearman_vote_getters")
y_positions = np.arange(len(plot_associations))
bar_height = 0.36

fig, ax = plt.subplots(figsize=(15, 9))
ax.barh(
    y_positions - bar_height / 2,
    plot_associations["raw_spearman_vote_getters"],
    height=bar_height,
    color=BLUE,
    label="Raw statistic",
)
ax.barh(
    y_positions + bar_height / 2,
    plot_associations["percentile_spearman_vote_getters"],
    height=bar_height,
    color=ORANGE,
    label="Within-season percentile",
)
ax.set_yticks(y_positions)
ax.set_yticklabels(plot_associations["feature"].map(friendly_labels))
ax.set_xlabel("Spearman association with award_share among vote getters")
ax.set_title(
    "Era Adjustment Usually Preserves or Improves Candidate Ordering",
    fontsize=20,
    color=NAVY,
    pad=16,
)
ax.set_xlim(0, 0.72)
ax.legend(loc="lower right")

for index, row in enumerate(plot_associations.itertuples(index=False)):
    ax.text(
        row.raw_spearman_vote_getters + 0.008,
        index - bar_height / 2,
        f"{row.raw_spearman_vote_getters:.2f}",
        va="center",
        fontsize=10,
        color=NAVY,
    )
    ax.text(
        row.percentile_spearman_vote_getters + 0.008,
        index + bar_height / 2,
        f"{row.percentile_spearman_vote_getters:.2f}",
        va="center",
        fontsize=10,
        color=NAVY,
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
association_figure_path = OUTPUT_DIR / "raw_vs_season_percentile_associations.png"
fig.savefig(association_figure_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(association_figure_path.resolve())

section("9. CHUNK BOUNDARY")
print("No scaling, imputer fitting, one-hot encoding, feature selection, or data splitting was performed.")
print("No prepared dataset was exported; the source ZIP remains untouched.")
