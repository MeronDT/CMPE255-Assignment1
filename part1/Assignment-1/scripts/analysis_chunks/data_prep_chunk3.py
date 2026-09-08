import os
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 210)
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
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk11_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 116}\n{title}\n{'=' * 116}")


def add_season_percentiles(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for feature in features:
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

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def high_correlation_pairs(frame: pd.DataFrame, features: list[str], threshold: float = 0.90) -> pd.DataFrame:
    correlation = frame[features].corr(method="spearman")
    rows = []
    for index, left in enumerate(features):
        for right in features[index + 1:]:
            rho = correlation.loc[left, right]
            if pd.notna(rho) and abs(rho) >= threshold:
                rows.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "spearman_rho": rho,
                        "absolute_rho": abs(rho),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["feature_1", "feature_2", "spearman_rho", "absolute_rho"])
    return pd.DataFrame(rows).sort_values("absolute_rho", ascending=False)


with ZipFile(ZIP_PATH) as archive:
    csv_members = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and not name.endswith("/")
    ]
    if len(csv_members) != 1:
        raise ValueError(f"Expected exactly one CSV; found {csv_members}")
    with archive.open(csv_members[0]) as csv_file:
        source = pd.read_csv(csv_file)

# Reconstruct the previously approved, auditable base table.
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
prepared = add_season_percentiles(prepared, era_raw_features)

section("1. DEFINE CHRONOLOGICAL HOLDOUT WINDOWS")
development_mask = prepared["season"].le(2014)
validation_mask = prepared["season"].between(2015, 2018)
test_mask = prepared["season"].between(2019, 2022)

development = prepared.loc[development_mask].copy()
validation = prepared.loc[validation_mask].copy()
test = prepared.loc[test_mask].copy()

window_rows = []
for name, frame in [
    ("Development", development),
    ("Model-selection validation", validation),
    ("Untouched final test", test),
]:
    window_rows.append(
        {
            "window": name,
            "first_season": frame["season"].min(),
            "last_season": frame["season"].max(),
            "seasons": frame["season"].nunique(),
            "rows": len(frame),
            "positive_targets": frame["award_share"].gt(0).sum(),
            "positive_target_rate_pct": 100 * frame["award_share"].gt(0).mean(),
            "winner_seasons": frame.loc[
                frame["award_share"].eq(
                    frame.groupby("season")["award_share"].transform("max")
                ),
                "season",
            ].nunique(),
        }
    )
window_table = pd.DataFrame(window_rows)
print(window_table.to_string(index=False))

split_integrity = pd.DataFrame(
    [
        ("All prepared rows assigned once", len(development) + len(validation) + len(test), len(prepared)),
        ("Development ends before validation", development["season"].max(), validation["season"].min() - 1),
        ("Validation ends before test", validation["season"].max(), test["season"].min() - 1),
        (
            "No source-row overlap",
            len(
                set(development["source_row_id"])
                & (set(validation["source_row_id"]) | set(test["source_row_id"]))
            ),
            0,
        ),
    ],
    columns=["check", "actual", "expected"],
)
split_integrity["passed"] = split_integrity["actual"].eq(split_integrity["expected"])
print("\nSplit-integrity checks:")
print(split_integrity.to_string(index=False))

section("2. DEFINE EXPANDING-WINDOW DEVELOPMENT FOLDS")
fold_definitions = [
    ("Fold 1", 1982, 2002, 2003, 2006),
    ("Fold 2", 1982, 2006, 2007, 2010),
    ("Fold 3", 1982, 2010, 2011, 2014),
]
fold_rows = []
fold_indices = []
for fold_name, train_start, train_end, valid_start, valid_end in fold_definitions:
    fold_train_mask = development["season"].between(train_start, train_end)
    fold_valid_mask = development["season"].between(valid_start, valid_end)
    train_positions = np.flatnonzero(fold_train_mask.to_numpy())
    valid_positions = np.flatnonzero(fold_valid_mask.to_numpy())
    fold_indices.append((train_positions, valid_positions))
    fold_rows.append(
        {
            "fold": fold_name,
            "training_seasons": f"{train_start}–{train_end}",
            "training_rows": len(train_positions),
            "validation_seasons": f"{valid_start}–{valid_end}",
            "validation_rows": len(valid_positions),
            "temporal_order_valid": train_end < valid_start,
        }
    )
fold_table = pd.DataFrame(fold_rows)
print(fold_table.to_string(index=False))

section("3. INVESTIGATE FEATURE-FAMILY INCLUSIONS AND EXCLUSIONS")
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

broad_exclusions = {
    "season": "Grouping and chronological evaluation only",
    "award_share": "Target; including it would be direct leakage",
    "source_row_id": "Audit key with no basketball meaning",
    "mp": "Approximately determined by games and minutes per game",
    "fg_per_g": "Scoring total represented by points and shot components",
    "fg2_per_g": "Made-shot component removed to reduce arithmetic redundancy",
    "fg3_per_g": "Made-shot component removed to reduce arithmetic redundancy",
    "ft_per_g": "Made-shot component removed to reduce arithmetic redundancy",
    "orb_per_g": "Use total rebounds rather than total plus components",
    "drb_per_g": "Use total rebounds rather than total plus components",
    "orb_pct": "Use total rebound percentage rather than components",
    "drb_pct": "Use total rebound percentage rather than components",
    "efg_pct": "Use true-shooting percentage as the broader efficiency summary",
    "ows": "Use total win shares rather than total plus components",
    "dws": "Use total win shares rather than total plus components",
    "obpm": "Use total BPM rather than total plus components",
    "dbpm": "Use total BPM rather than total plus components",
    "mov": "Use adjusted margin and win percentage rather than both margins",
}

all_numeric_candidates = prepared.select_dtypes(include=np.number).columns.tolist()
broad_numeric = [
    feature for feature in all_numeric_candidates
    if feature not in broad_exclusions
]
categorical_features = ["pos_primary"]

feature_family_table = pd.DataFrame(
    [
        (
            "Compact interpretable",
            len(compact_numeric),
            len(categorical_features),
            "Regularized linear models and explanation",
        ),
        (
            "Broader domain-pruned",
            len(broad_numeric),
            len(categorical_features),
            "Nonlinear models and robustness comparison",
        ),
    ],
    columns=["feature_family", "numeric_features", "categorical_features", "intended_use"],
)
print(feature_family_table.to_string(index=False))

print("\nCompact numeric features:")
print(compact_numeric)

print("\nBroad-set exclusions and reasons:")
exclusion_table = pd.DataFrame(
    broad_exclusions.items(), columns=["excluded_feature", "reason"]
)
print(exclusion_table.to_string(index=False))

identifier_audit = pd.DataFrame(
    {
        "field": ["player", "team_id", "pos", "season", "source_row_id", "award_share"],
        "role": [
            "Reporting only",
            "Reporting and TOT identification only",
            "Audit-only detailed position label",
            "Grouping and chronological splitting only",
            "Row lineage only",
            "Target only",
        ],
        "included_as_predictor": [False, False, False, False, False, False],
    }
)
print("\nIdentifier and target audit:")
print(identifier_audit.to_string(index=False))

section("4. REDUNDANCY REMAINING WITHIN EACH PROPOSED FEATURE FAMILY")
compact_pairs = high_correlation_pairs(development, compact_numeric, threshold=0.90)
broad_pairs = high_correlation_pairs(development, broad_numeric, threshold=0.90)
redundancy_summary = pd.DataFrame(
    [
        ("Compact interpretable", len(compact_numeric), len(compact_pairs)),
        ("Broader domain-pruned", len(broad_numeric), len(broad_pairs)),
    ],
    columns=["feature_family", "numeric_features", "pairs_with_abs_spearman_at_least_0_90"],
)
print(redundancy_summary.to_string(index=False))
print("\nCompact-set highly correlated pairs:")
print("None" if compact_pairs.empty else compact_pairs.to_string(index=False))
print("\nTop 20 broader-set highly correlated pairs:")
print("None" if broad_pairs.empty else broad_pairs.head(20).to_string(index=False))
print(
    "\nDecision: retain raw-plus-percentile pairs intentionally, then use regularization "
    "for linear models. The broader set is reserved for models that can represent nonlinearities."
)

section("5. CONSTRUCT AND FIT DEVELOPMENT-ONLY PREPROCESSORS")
compact_linear_preprocessor = make_preprocessor(
    compact_numeric,
    categorical_features,
    scale_numeric=True,
)
broad_linear_preprocessor = make_preprocessor(
    broad_numeric,
    categorical_features,
    scale_numeric=True,
)
broad_tree_preprocessor = make_preprocessor(
    broad_numeric,
    categorical_features,
    scale_numeric=False,
)

compact_input_columns = compact_numeric + categorical_features
broad_input_columns = broad_numeric + categorical_features

X_development_compact = development[compact_input_columns]
X_validation_compact = validation[compact_input_columns]
X_test_compact = test[compact_input_columns]

X_development_broad = development[broad_input_columns]
X_validation_broad = validation[broad_input_columns]
X_test_broad = test[broad_input_columns]

# FIT occurs only here, using seasons 1982–2014.
X_development_compact_ready = compact_linear_preprocessor.fit_transform(X_development_compact)
X_validation_compact_ready = compact_linear_preprocessor.transform(X_validation_compact)
X_test_compact_ready = compact_linear_preprocessor.transform(X_test_compact)

X_development_broad_ready = broad_linear_preprocessor.fit_transform(X_development_broad)
X_validation_broad_ready = broad_linear_preprocessor.transform(X_validation_broad)
X_test_broad_ready = broad_linear_preprocessor.transform(X_test_broad)

X_development_tree_ready = broad_tree_preprocessor.fit_transform(X_development_broad)
X_validation_tree_ready = broad_tree_preprocessor.transform(X_validation_broad)
X_test_tree_ready = broad_tree_preprocessor.transform(X_test_broad)

transformed_shapes = pd.DataFrame(
    [
        ("Compact + robust scaling", X_development_compact_ready.shape, X_validation_compact_ready.shape, X_test_compact_ready.shape),
        ("Broad + robust scaling", X_development_broad_ready.shape, X_validation_broad_ready.shape, X_test_broad_ready.shape),
        ("Broad + no scaling", X_development_tree_ready.shape, X_validation_tree_ready.shape, X_test_tree_ready.shape),
    ],
    columns=["preprocessor", "development_shape", "validation_shape", "test_shape"],
)
print(transformed_shapes.to_string(index=False))

compact_output_names = compact_linear_preprocessor.get_feature_names_out().tolist()
broad_output_names = broad_linear_preprocessor.get_feature_names_out().tolist()
print(f"\nCompact output features ({len(compact_output_names)}):")
print(compact_output_names)
print(f"\nBroad output feature count: {len(broad_output_names)}")
print("Broad output feature-name sample:")
print(broad_output_names[:20] + ["..."] + broad_output_names[-10:])

section("6. AUDIT TRAIN-ONLY IMPUTATION AND CATEGORICAL LEARNING")
compact_imputer = compact_linear_preprocessor.named_transformers_["numeric"].named_steps["imputer"]
imputation_rows = []
for feature, fitted_statistic in zip(compact_numeric, compact_imputer.statistics_):
    if development[feature].isna().any() or validation[feature].isna().any() or test[feature].isna().any():
        imputation_rows.append(
            {
                "feature": feature,
                "development_missing": development[feature].isna().sum(),
                "validation_missing": validation[feature].isna().sum(),
                "test_missing": test[feature].isna().sum(),
                "development_median": development[feature].median(),
                "all_periods_median": prepared[feature].median(),
                "fitted_imputer_value": fitted_statistic,
                "fit_equals_development_median": np.isclose(
                    fitted_statistic,
                    development[feature].median(),
                ),
            }
        )
imputation_audit = pd.DataFrame(imputation_rows)
print(imputation_audit.to_string(index=False))

position_encoder = compact_linear_preprocessor.named_transformers_["categorical"].named_steps["one_hot"]
learned_positions = position_encoder.categories_[0].tolist()
print(f"\nPosition categories learned from development only: {learned_positions}")
print(f"Validation positions: {sorted(validation['pos_primary'].unique().tolist())}")
print(f"Test positions: {sorted(test['pos_primary'].unique().tolist())}")

section("7. EXPLICIT FUTURE-DATA INDEPENDENCE TEST")
future_perturbed = prepared.copy()
future_mask = future_perturbed["season"].ge(2015)
future_perturbed.loc[future_mask, "win_loss_pct"] = 9.999
future_perturbed.loc[future_mask, "win_loss_pct_season_pct"] = 9.999
future_perturbed.loc[future_mask, "pos_primary"] = "FUTURE_ONLY_POSITION"

refitted_on_same_development = clone(compact_linear_preprocessor)
refitted_on_same_development.fit(
    future_perturbed.loc[future_perturbed["season"].le(2014), compact_input_columns]
)

original_imputer_stats = compact_linear_preprocessor.named_transformers_["numeric"].named_steps["imputer"].statistics_
perturbed_imputer_stats = refitted_on_same_development.named_transformers_["numeric"].named_steps["imputer"].statistics_
original_categories = compact_linear_preprocessor.named_transformers_["categorical"].named_steps["one_hot"].categories_[0]
perturbed_categories = refitted_on_same_development.named_transformers_["categorical"].named_steps["one_hot"].categories_[0]

future_independence_checks = pd.DataFrame(
    [
        (
            "Imputer statistics unchanged after extreme future perturbation",
            np.allclose(original_imputer_stats, perturbed_imputer_stats, equal_nan=True),
        ),
        (
            "Learned position categories unchanged after future-only category injection",
            np.array_equal(original_categories, perturbed_categories),
        ),
        (
            "FUTURE_ONLY_POSITION absent from learned categories",
            "FUTURE_ONLY_POSITION" not in perturbed_categories,
        ),
        (
            "award_share absent from compact predictors",
            "award_share" not in compact_input_columns,
        ),
        (
            "award_share absent from broad predictors",
            "award_share" not in broad_input_columns,
        ),
    ],
    columns=["check", "passed"],
)
print(future_independence_checks.to_string(index=False))

section("8. FINAL PREPROCESSING QUALITY CHECKS")
arrays = {
    "compact_development": X_development_compact_ready,
    "compact_validation": X_validation_compact_ready,
    "compact_test": X_test_compact_ready,
    "broad_linear_development": X_development_broad_ready,
    "broad_linear_validation": X_validation_broad_ready,
    "broad_linear_test": X_test_broad_ready,
    "broad_tree_development": X_development_tree_ready,
    "broad_tree_validation": X_validation_tree_ready,
    "broad_tree_test": X_test_tree_ready,
}
array_checks = []
for name, values in arrays.items():
    array_checks.append(
        {
            "array": name,
            "rows": values.shape[0],
            "columns": values.shape[1],
            "missing_values": int(np.isnan(values).sum()),
            "infinite_values": int(np.isinf(values).sum()),
        }
    )
array_check_table = pd.DataFrame(array_checks)
print(array_check_table.to_string(index=False))

all_checks_passed = (
    split_integrity["passed"].all()
    and fold_table["temporal_order_valid"].all()
    and future_independence_checks["passed"].all()
    and array_check_table["missing_values"].eq(0).all()
    and array_check_table["infinite_values"].eq(0).all()
)
print(f"\nAll split, leakage, and preprocessing checks passed: {all_checks_passed}")
if not all_checks_passed:
    raise AssertionError("At least one final preprocessing check failed.")

section("9. FIGURE 1 — CHRONOLOGICAL EVALUATION DESIGN")
fig, axes = plt.subplots(2, 1, figsize=(17, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.7]})
fig.suptitle(
    "Chronological Evaluation Design",
    fontsize=23,
    fontweight="bold",
    color=NAVY,
    y=0.99,
)

ax = axes[0]
main_windows = [
    (1982, 2014, "Development", BLUE),
    (2015, 2018, "Validation", ORANGE),
    (2019, 2022, "Final test", GREEN),
]
for start, end, label, color in main_windows:
    ax.barh(0, end - start + 1, left=start, height=0.55, color=color)
    ax.text((start + end + 1) / 2, 0, f"{label}\n{start}–{end}", ha="center", va="center", color="white", fontweight="bold", fontsize=12)
ax.set_yticks([])
ax.set_title("A. Model-development and holdout windows", loc="left")
ax.set_ylim(-0.55, 0.55)

ax = axes[1]
for row_index, (_, train_start, train_end, valid_start, valid_end) in enumerate(fold_definitions):
    ax.barh(row_index, train_end - train_start + 1, left=train_start, height=0.50, color=BLUE)
    ax.barh(row_index, valid_end - valid_start + 1, left=valid_start, height=0.50, color=ORANGE)
    ax.text(train_start + 1, row_index, f"Train through {train_end}", va="center", color="white", fontweight="bold", fontsize=11)
    ax.text((valid_start + valid_end + 1) / 2, row_index, f"{valid_start}–{valid_end}", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
ax.set_yticks(np.arange(len(fold_definitions)))
ax.set_yticklabels([definition[0] for definition in fold_definitions])
ax.invert_yaxis()
ax.set_title("B. Expanding-window folds inside development", loc="left")
ax.set_xlabel("Season")
ax.set_xticks(np.arange(1982, 2023, 4))
ax.legend(
    handles=[Patch(color=BLUE, label="Training seasons"), Patch(color=ORANGE, label="Fold validation seasons")],
    loc="lower right",
    ncol=2,
)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.4)
timeline_path = OUTPUT_DIR / "chronological_evaluation_design.png"
fig.savefig(timeline_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(timeline_path.resolve())

section("10. FIGURE 2 — FEATURE-FAMILY COMPLEXITY AND REDUNDANCY")
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
fig.suptitle(
    "Feature-Family Audit Before Modeling",
    fontsize=22,
    fontweight="bold",
    color=NAVY,
    y=0.99,
)

family_labels = ["Compact\ninterpretable", "Broader\ndomain-pruned"]
feature_counts = [len(compact_numeric), len(broad_numeric)]
pair_counts = [len(compact_pairs), len(broad_pairs)]

ax = axes[0]
bars = ax.bar(family_labels, feature_counts, color=[BLUE, GREEN], width=0.58)
ax.set_title("A. Numeric predictors")
ax.set_ylabel("Feature count")
ax.set_ylim(0, max(feature_counts) * 1.22)
for bar, value in zip(bars, feature_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, str(value), ha="center", fontweight="bold")

ax = axes[1]
bars = ax.bar(family_labels, pair_counts, color=[BLUE, GREEN], width=0.58)
ax.set_title("B. Remaining highly correlated pairs")
ax.set_ylabel("Pairs with |Spearman correlation| ≥ 0.90")
ax.set_ylim(0, max(pair_counts) * 1.22 if max(pair_counts) else 1)
for bar, value in zip(bars, pair_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, str(value), ha="center", fontweight="bold")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=3.0)
feature_audit_path = OUTPUT_DIR / "feature_family_audit.png"
fig.savefig(feature_audit_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(feature_audit_path.resolve())

section("11. PRE-MODELING STOP")
print("Preprocessing objects were fitted only to development predictors from 1982–2014.")
print("No baseline, regression model, ranking model, clustering model, or hyperparameter search was fitted.")
print("No predictions were generated, and the 2019–2022 target values remain unused for model decisions.")
print("No prepared dataset was exported; the uploaded source ZIP remains untouched.")
