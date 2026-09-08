import os
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 90)
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
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk12_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "award_share"


def section(title: str) -> None:
    print(f"\n{'=' * 120}\n{title}\n{'=' * 120}")


def add_season_percentiles(
    feature_frame_without_target: pd.DataFrame,
    raw_features: list[str],
) -> pd.DataFrame:
    """Create within-season percentiles from predictors only."""
    if TARGET in feature_frame_without_target.columns:
        raise ValueError("Target must be removed before feature engineering.")

    result = feature_frame_without_target.copy()
    for feature in raw_features:
        result[f"{feature}_season_pct"] = result.groupby("season")[feature].rank(
            method="average",
            pct=True,
            na_option="keep",
        )
    return result


def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", RobustScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


with ZipFile(ZIP_PATH) as archive:
    csv_members = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and not name.endswith("/")
    ]
    if len(csv_members) != 1:
        raise ValueError(f"Expected exactly one CSV; found {csv_members}")
    with archive.open(csv_members[0]) as csv_file:
        source = pd.read_csv(csv_file)

# Approved base-table preparation. The filter references minutes only, not the target.
base = source.copy()
base.insert(0, "source_row_id", np.arange(len(base), dtype=np.int64))
base = base.loc[base["mp"].ge(100)].copy()
base["is_tot"] = base["team_id"].eq("TOT").astype("int8")
base.loc[base["is_tot"].eq(1), ["mov", "mov_adj", "win_loss_pct"]] = np.nan

structural_percentages = ["fg_pct", "fg2_pct", "fg3_pct", "ft_pct"]
for percentage in structural_percentages:
    base[f"{percentage}_was_missing"] = base[percentage].isna().astype("int8")
    base[percentage] = base[percentage].fillna(0.0)
base["pos_primary"] = base["pos"].str.split("-").str[0]

# Remove the target before any era-aware feature engineering.
target_series = base[TARGET].copy()
feature_only_base = base.drop(columns=[TARGET])
era_raw_features = [
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
prepared_features = add_season_percentiles(feature_only_base, era_raw_features)
prepared = prepared_features.copy()
prepared[TARGET] = target_series

section("1. FINAL TARGET CONTRACT")
target_contract = pd.DataFrame(
    [
        ("Name", TARGET),
        ("Analytical role", "Continuous numerical regression target and within-season ranking relevance"),
        ("Meaning", "Share of observed MVP voting support for a player-season"),
        ("Allowed range in source", f"{source[TARGET].min():.3f} to {source[TARGET].max():.3f}"),
        ("Missing values in source", int(source[TARGET].isna().sum())),
        ("Modeling implication", "Zero-inflated continuous outcome; not ordinary binary classification"),
        ("Ranking unit", "Season"),
    ],
    columns=["property", "definition"],
)
print(target_contract.to_string(index=False))

# Development and validation targets may support model development.
# The 2019–2022 target is deliberately not summarized or assigned to y_test here.
development_mask = prepared["season"].le(2014)
validation_mask = prepared["season"].between(2015, 2018)
final_test_mask = prepared["season"].between(2019, 2022)

development = prepared.loc[development_mask].copy()
validation = prepared.loc[validation_mask].copy()
final_test_features_only = prepared.loc[final_test_mask].drop(columns=[TARGET]).copy()

target_rows = []
for label, frame in [("Development", development), ("Model-selection validation", validation)]:
    target_rows.append(
        {
            "window": label,
            "seasons": frame["season"].nunique(),
            "rows": len(frame),
            "target_minimum": frame[TARGET].min(),
            "target_mean": frame[TARGET].mean(),
            "target_maximum": frame[TARGET].max(),
            "zero_targets": frame[TARGET].eq(0).sum(),
            "positive_targets": frame[TARGET].gt(0).sum(),
            "positive_target_rate_pct": 100 * frame[TARGET].gt(0).mean(),
        }
    )
target_summary = pd.DataFrame(target_rows)
print("\nTarget distributions available for development decisions:")
print(target_summary.to_string(index=False))
print(
    f"\nFinal-test feature rows sealed for later evaluation: {len(final_test_features_only):,} "
    "across seasons 2019–2022. No y_test object is created in this script."
)

section("2. FINAL FEATURE CONTRACT")
compact_numeric = [
    "age",
    "g",
    "mp_per_g",
    "pts_per_g",
    "trb_per_g",
    "ast_per_g",
    "stl_per_g",
    "blk_per_g",
    "tov_per_g",
    "ts_pct",
    "usg_pct",
    "per",
    "ws",
    "bpm",
    "win_loss_pct",
    "is_tot",
    "pts_per_g_season_pct",
    "mp_per_g_season_pct",
    "ts_pct_season_pct",
    "per_season_pct",
    "ws_season_pct",
    "bpm_season_pct",
    "win_loss_pct_season_pct",
]
categorical_features = ["pos_primary"]

broad_exclusions = {
    "season",
    TARGET,
    "source_row_id",
    "mp",
    "fg_per_g",
    "fg2_per_g",
    "fg3_per_g",
    "ft_per_g",
    "orb_per_g",
    "drb_per_g",
    "orb_pct",
    "drb_pct",
    "efg_pct",
    "ows",
    "dws",
    "obpm",
    "dbpm",
    "mov",
}
broad_numeric = [
    feature
    for feature in prepared.select_dtypes(include=np.number).columns
    if feature not in broad_exclusions
]

compact_feature_groups = pd.DataFrame(
    [
        ("Player context", 3, "age, games, minutes per game"),
        ("Box-score production", 7, "points, rebounds, assists, steals, blocks, turnovers, usage"),
        ("Efficiency and impact", 5, "true shooting, PER, win shares, BPM, team win percentage"),
        ("Data-quality context", 1, "is_tot"),
        ("Era-aware percentiles", 7, "season-relative versions of selected role, impact, and team features"),
        ("Categorical role", 1, "primary position; one-hot encoded"),
    ],
    columns=["feature_group", "feature_count", "contents"],
)
print("Compact feature groups:")
print(compact_feature_groups.to_string(index=False))
print(f"\nCompact numeric predictors ({len(compact_numeric)}):")
print(compact_numeric)
print(f"\nCompact categorical predictors ({len(categorical_features)}):")
print(categorical_features)
print(f"\nBroader domain-pruned numeric predictors ({len(broad_numeric)}):")
print(broad_numeric)

excluded_roles = pd.DataFrame(
    [
        ("player", "Reporting label; would permit player memorization"),
        ("team_id", "Reporting/TOT identification; franchise identity is not the team-performance measure"),
        ("pos", "Detailed audit label; pos_primary is the modeling version"),
        ("season", "Used for grouping and temporal validation, not as a direct predictor"),
        ("source_row_id", "Lineage key with no basketball meaning"),
        (TARGET, "Target; direct inclusion would be leakage"),
    ],
    columns=["excluded_field", "reason"],
)
print("\nFields excluded from every predictor matrix:")
print(excluded_roles.to_string(index=False))

section("3. FINAL PREPROCESSING CONTRACT")
preprocessing_contract = pd.DataFrame(
    [
        ("Eligibility", "Keep mp >= 100", "Removes unstable tiny samples while retaining all observed vote recipients"),
        ("Repeated names", "Retain; no name-based deduplication", "Same strings can represent different people"),
        ("Structural shooting missingness", "Indicator plus zero-fill", "Separates undefined/no-attempt percentages from observed percentages"),
        ("TOT rows", "Retain; is_tot=1", "Traded players remain eligible"),
        ("TOT team context", "Set mov, mov_adj, and win_loss_pct to missing", "Neutral placeholders are not treated as genuine team performance"),
        ("Other numeric missingness", "Training-window median imputation", "Robust and compatible with estimators requiring complete matrices"),
        ("Numeric scaling for linear models", "RobustScaler", "Reduces sensitivity to statistical extremes"),
        ("Numeric scaling for tree models", "None", "Tree splits depend on ordering rather than measurement scale"),
        ("Position", "Development-fitted one-hot encoding; ignore unknown", "Transparent categorical representation without ordinal assumptions"),
        ("Outliers", "No winsorization or automatic deletion", "Exceptional MVP-caliber seasons remain intact"),
        ("Era", "Keep raw values plus within-season percentile ranks", "Preserves absolute magnitude and contemporary standing"),
    ],
    columns=["area", "final_decision", "reason"],
)
print(preprocessing_contract.to_string(index=False))

section("4. CHRONOLOGICAL MODEL-SELECTION CONTRACT")
window_contract = pd.DataFrame(
    [
        ("Development", "1982–2014", len(development), "Expanding-fold tuning and baseline comparison"),
        ("Model-selection validation", "2015–2018", len(validation), "Select the final model specification after development CV"),
        ("Final test", "2019–2022", len(final_test_features_only), "One-time evaluation only after every choice is locked"),
    ],
    columns=["window", "seasons", "rows", "permitted_use"],
)
print(window_contract.to_string(index=False))

fold_definitions = [
    ("Fold 1", 1982, 2002, 2003, 2006),
    ("Fold 2", 1982, 2006, 2007, 2010),
    ("Fold 3", 1982, 2010, 2011, 2014),
]
fold_rows = []
fold_indices = []
for fold_name, train_start, train_end, valid_start, valid_end in fold_definitions:
    train_positions = np.flatnonzero(
        development["season"].between(train_start, train_end).to_numpy()
    )
    valid_positions = np.flatnonzero(
        development["season"].between(valid_start, valid_end).to_numpy()
    )
    fold_indices.append((train_positions, valid_positions))
    fold_rows.append(
        {
            "fold": fold_name,
            "training_seasons": f"{train_start}–{train_end}",
            "training_rows": len(train_positions),
            "fold_validation_seasons": f"{valid_start}–{valid_end}",
            "fold_validation_rows": len(valid_positions),
            "latest_train_before_earliest_validation": train_end < valid_start,
        }
    )
fold_table = pd.DataFrame(fold_rows)
print("\nExpanding development folds:")
print(fold_table.to_string(index=False))

section("5. FIT PREPROCESSING FOR VERIFICATION — NO PREDICTIVE MODEL")
compact_preprocessor_template = make_preprocessor(
    compact_numeric,
    categorical_features,
    scale_numeric=True,
)
broad_linear_preprocessor_template = make_preprocessor(
    broad_numeric,
    categorical_features,
    scale_numeric=True,
)
broad_tree_preprocessor_template = make_preprocessor(
    broad_numeric,
    categorical_features,
    scale_numeric=False,
)

compact_columns = compact_numeric + categorical_features
broad_columns = broad_numeric + categorical_features
X_development_compact = development[compact_columns]
X_validation_compact = validation[compact_columns]
X_development_broad = development[broad_columns]
X_validation_broad = validation[broad_columns]
y_development = development[TARGET].copy()
y_validation = validation[TARGET].copy()

# Development-wide fitting here verifies the external-validation transformation.
# During cross-validation, the entire pipeline must instead be refitted inside each fold.
compact_preprocessor = clone(compact_preprocessor_template)
broad_linear_preprocessor = clone(broad_linear_preprocessor_template)
broad_tree_preprocessor = clone(broad_tree_preprocessor_template)

compact_development_ready = compact_preprocessor.fit_transform(X_development_compact)
compact_validation_ready = compact_preprocessor.transform(X_validation_compact)
broad_linear_development_ready = broad_linear_preprocessor.fit_transform(X_development_broad)
broad_linear_validation_ready = broad_linear_preprocessor.transform(X_validation_broad)
broad_tree_development_ready = broad_tree_preprocessor.fit_transform(X_development_broad)
broad_tree_validation_ready = broad_tree_preprocessor.transform(X_validation_broad)

transformation_summary = pd.DataFrame(
    [
        ("Compact linear", compact_development_ready.shape, compact_validation_ready.shape, True),
        ("Broad linear", broad_linear_development_ready.shape, broad_linear_validation_ready.shape, True),
        ("Broad tree", broad_tree_development_ready.shape, broad_tree_validation_ready.shape, False),
    ],
    columns=["path", "development_shape", "validation_shape", "numeric_scaling"],
)
print(transformation_summary.to_string(index=False))

section("6. VERIFY FOLD-LOCAL PREPROCESSING STATISTICS")
fold_preprocessing_rows = []
win_pct_index = compact_numeric.index("win_loss_pct")
for (fold_name, train_start, train_end, valid_start, valid_end), (train_positions, valid_positions) in zip(
    fold_definitions,
    fold_indices,
):
    fold_preprocessor = clone(compact_preprocessor_template)
    fold_X_train = X_development_compact.iloc[train_positions]
    fold_preprocessor.fit(fold_X_train)
    fitted_median = (
        fold_preprocessor.named_transformers_["numeric"]
        .named_steps["imputer"]
        .statistics_[win_pct_index]
    )
    fold_preprocessing_rows.append(
        {
            "fold": fold_name,
            "training_seasons": f"{train_start}–{train_end}",
            "fold_validation_seasons": f"{valid_start}–{valid_end}",
            "training_win_pct_median": fold_X_train["win_loss_pct"].median(),
            "fitted_imputer_value": fitted_median,
            "values_match": np.isclose(
                fitted_median,
                fold_X_train["win_loss_pct"].median(),
            ),
        }
    )
fold_preprocessing_audit = pd.DataFrame(fold_preprocessing_rows)
print(fold_preprocessing_audit.to_string(index=False))

section("7. TARGET-LEAKAGE AND FINAL-TEST-SEPARATION AUDIT")
compact_output_names = compact_preprocessor.get_feature_names_out().tolist()
broad_output_names = broad_linear_preprocessor.get_feature_names_out().tolist()
forbidden_predictors = {TARGET, "player", "team_id", "pos", "season", "source_row_id"}

all_fold_train_seasons = set()
all_fold_validation_seasons = set()
for train_positions, valid_positions in fold_indices:
    all_fold_train_seasons.update(development.iloc[train_positions]["season"].unique().tolist())
    all_fold_validation_seasons.update(development.iloc[valid_positions]["season"].unique().tolist())

leakage_checks = pd.DataFrame(
    [
        ("Target absent from compact input", TARGET not in compact_columns),
        ("Target absent from broad input", TARGET not in broad_columns),
        ("Forbidden fields absent from compact input", forbidden_predictors.isdisjoint(compact_columns)),
        ("Forbidden fields absent from broad input", forbidden_predictors.isdisjoint(broad_columns)),
        ("Target absent after compact transformation", TARGET not in compact_output_names),
        ("Target absent after broad transformation", TARGET not in broad_output_names),
        ("Era features constructed while target column was absent", TARGET not in feature_only_base.columns),
        ("Development and model-selection validation rows do not overlap", set(development["source_row_id"]).isdisjoint(validation["source_row_id"])),
        ("Development and final-test feature rows do not overlap", set(development["source_row_id"]).isdisjoint(final_test_features_only["source_row_id"])),
        ("Model-selection validation and final-test rows do not overlap", set(validation["source_row_id"]).isdisjoint(final_test_features_only["source_row_id"])),
        ("Every fold validates strictly after its training seasons", fold_table["latest_train_before_earliest_validation"].all()),
        ("2015–2018 absent from development folds", all(season < 2015 for season in all_fold_train_seasons | all_fold_validation_seasons)),
        ("2019–2022 absent from all model-selection folds", all(season < 2019 for season in all_fold_train_seasons | all_fold_validation_seasons)),
        ("Final-test target object was not created", "y_test" not in locals()),
        ("Development-wide imputer fitted to development median", np.isclose(
            compact_preprocessor.named_transformers_["numeric"].named_steps["imputer"].statistics_[win_pct_index],
            development["win_loss_pct"].median(),
        )),
        ("Every fold-local imputer matches its own training median", fold_preprocessing_audit["values_match"].all()),
        ("Compact development matrix contains no missing values", not np.isnan(compact_development_ready).any()),
        ("Compact validation matrix contains no missing values", not np.isnan(compact_validation_ready).any()),
        ("No predictive estimator exists in any preprocessing template", all(
            not hasattr(template, "predict")
            for template in [
                compact_preprocessor_template,
                broad_linear_preprocessor_template,
                broad_tree_preprocessor_template,
            ]
        )),
    ],
    columns=["check", "passed"],
)
print(leakage_checks.to_string(index=False))
print(f"\nAll leakage and separation checks passed: {leakage_checks['passed'].all()}")
if not leakage_checks["passed"].all():
    raise AssertionError("At least one leakage or separation check failed.")

section("8. READABLE PRE-MODELING CONTRACT FIGURE")
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
fig.suptitle(
    "Pre-Modeling Data Contract",
    fontsize=23,
    fontweight="bold",
    color=NAVY,
    y=0.99,
)

ax = axes[0]
window_labels = ["Development\n1982–2014", "Validation\n2015–2018", "Final test\n2019–2022"]
window_counts = [len(development), len(validation), len(final_test_features_only)]
bars = ax.bar(window_labels, window_counts, color=[BLUE, ORANGE, GREEN], width=0.60)
ax.set_title("A. Chronological data windows")
ax.set_ylabel("Eligible player-seasons")
ax.set_ylim(0, max(window_counts) * 1.18)
for bar, value in zip(bars, window_counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 180,
        f"{value:,}",
        ha="center",
        fontweight="bold",
    )
ax.text(
    2,
    window_counts[2] * 0.52,
    "Sealed until\nfinal evaluation",
    ha="center",
    va="center",
    color="white",
    fontweight="bold",
    fontsize=11,
)

ax = axes[1]
family_labels = ["Compact\ninterpretable", "Broader\ndomain-pruned"]
numeric_counts = [len(compact_numeric), len(broad_numeric)]
categorical_counts = [len(categorical_features), len(categorical_features)]
ax.bar(family_labels, numeric_counts, color=BLUE, width=0.60, label="Numeric")
ax.bar(
    family_labels,
    categorical_counts,
    bottom=numeric_counts,
    color=ORANGE,
    width=0.60,
    label="Categorical before one-hot encoding",
)
ax.set_title("B. Final predictor families")
ax.set_ylabel("Input feature count")
ax.set_ylim(0, (max(numeric_counts) + 1) * 1.20)
for index, (numeric_count, categorical_count) in enumerate(zip(numeric_counts, categorical_counts)):
    ax.text(
        index,
        numeric_count + categorical_count + 1,
        f"{numeric_count} numeric + {categorical_count} categorical",
        ha="center",
        fontweight="bold",
        fontsize=11,
    )
ax.legend(loc="upper left")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=3.0)
figure_path = OUTPUT_DIR / "pre_modeling_data_contract.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(figure_path.resolve())

section("9. STOP CONFIRMATION")
print("No regression, ranking, classification, clustering, or ensemble estimator was trained.")
print("No predictions, model coefficients, feature importances, or model-selection scores were produced.")
print("The 2019–2022 target was not assigned to a y_test object or used in any decision.")
