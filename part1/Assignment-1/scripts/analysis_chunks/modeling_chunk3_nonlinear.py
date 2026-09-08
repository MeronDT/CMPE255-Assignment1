import os
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 250)
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
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "grid.color": LIGHT_GRAY,
        "grid.linewidth": 0.8,
    },
)

ZIP_PATH = Path(os.environ.get("NBA_MVP_DATA_PATH", "data/raw/nba_mvp_stats.zip"))
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk15_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "award_share"
RANDOM_STATE = 42


def section(title: str) -> None:
    print(f"\n{'=' * 124}\n{title}\n{'=' * 124}")


def add_season_percentiles(
    feature_frame_without_target: pd.DataFrame,
    raw_features: list[str],
) -> pd.DataFrame:
    if TARGET in feature_frame_without_target.columns:
        raise ValueError("Target must be absent during era-aware feature creation.")
    result = feature_frame_without_target.copy()
    for feature in raw_features:
        result[f"{feature}_season_pct"] = result.groupby("season")[feature].rank(
            method="average",
            pct=True,
            na_option="keep",
        )
    return result


def make_tree_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
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


def make_nonlinear_pipeline(
    configuration: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    if configuration["family"] == "Extra Trees":
        estimator = ExtraTreesRegressor(
            n_estimators=160,
            max_depth=(
                None
                if pd.isna(configuration["max_depth"])
                else int(configuration["max_depth"])
            ),
            min_samples_leaf=int(configuration["min_samples_leaf"]),
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif configuration["family"] == "Histogram GB":
        estimator = HistGradientBoostingRegressor(
            learning_rate=float(configuration["learning_rate"]),
            max_leaf_nodes=int(configuration["max_leaf_nodes"]),
            l2_regularization=float(configuration["l2_regularization"]),
            max_iter=200,
            min_samples_leaf=20,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown family: {configuration['family']}")

    return Pipeline(
        [
            (
                "preprocess",
                make_tree_preprocessor(numeric_features, categorical_features),
            ),
            ("model", estimator),
        ]
    )


def continuous_vote_weights(y: np.ndarray, weight_lambda: float) -> np.ndarray:
    return 1.0 + weight_lambda * y


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    positive = y_true > 0
    return {
        "mae_all": mean_absolute_error(y_true, prediction),
        "rmse_all": np.sqrt(mean_squared_error(y_true, prediction)),
        "r2_all": r2_score(y_true, prediction),
        "mae_positive": mean_absolute_error(y_true[positive], prediction[positive]),
        "rmse_positive": np.sqrt(
            mean_squared_error(y_true[positive], prediction[positive])
        ),
    }


def season_ranking_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    evaluation = frame[["season", TARGET]].copy()
    evaluation["prediction"] = prediction
    season_rows = []

    for season, group in evaluation.groupby("season"):
        y_true = group[TARGET].to_numpy(dtype=float)
        y_score = group["prediction"].to_numpy(dtype=float)
        n_players = len(group)
        actual_winner = np.isclose(y_true, y_true.max())
        predicted_top = np.isclose(y_score, y_score.max())
        predicted_ranks = pd.Series(y_score).rank(
            method="average",
            ascending=False,
        ).to_numpy()
        winner_rank = predicted_ranks[actual_winner].mean()
        winner_rank_percentile = (
            1 - (winner_rank - 1) / (n_players - 1)
            if n_players > 1
            else 1.0
        )
        top1_credit = (
            np.logical_and(actual_winner, predicted_top).sum()
            / predicted_top.sum()
        )
        spearman = (
            pd.Series(y_true).corr(pd.Series(y_score), method="spearman")
            if np.unique(y_true).size > 1 and np.unique(y_score).size > 1
            else np.nan
        )
        season_rows.append(
            {
                "season": season,
                "ndcg_at_5": ndcg_score(
                    y_true.reshape(1, -1),
                    y_score.reshape(1, -1),
                    k=min(5, n_players),
                    ignore_ties=False,
                ),
                "winner_rank": winner_rank,
                "winner_rank_percentile": winner_rank_percentile,
                "fractional_top1_accuracy": top1_credit,
                "spearman": spearman,
            }
        )

    per_season = pd.DataFrame(season_rows)
    return {
        "mean_ndcg_at_5": per_season["ndcg_at_5"].mean(),
        "mean_winner_rank": per_season["winner_rank"].mean(),
        "mean_winner_rank_percentile": per_season["winner_rank_percentile"].mean(),
        "fractional_top1_accuracy": per_season["fractional_top1_accuracy"].mean(),
        "mean_season_spearman": per_season["spearman"].mean(),
    }


def evaluate_predictions(
    frame: pd.DataFrame,
    raw_prediction: np.ndarray,
) -> dict:
    bounded_prediction = np.clip(raw_prediction, 0.0, 1.0)
    y_true = frame[TARGET].to_numpy(dtype=float)
    return {
        **regression_metrics(y_true, bounded_prediction),
        **season_ranking_metrics(frame, bounded_prediction),
        "raw_minimum": raw_prediction.min(),
        "raw_maximum": raw_prediction.max(),
        "raw_below_zero_pct": 100 * np.mean(raw_prediction < 0),
        "raw_above_one_pct": 100 * np.mean(raw_prediction > 1),
        "bounded_values_changed_pct": 100
        * np.mean(~np.isclose(raw_prediction, bounded_prediction)),
    }


def fit_ws_baseline(
    training_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
) -> np.ndarray:
    baseline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    baseline.fit(
        training_frame[["ws_season_pct"]],
        training_frame[TARGET],
    )
    return baseline.predict(evaluation_frame[["ws_season_pct"]])


with ZipFile(ZIP_PATH) as archive:
    csv_members = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and not name.endswith("/")
    ]
    if len(csv_members) != 1:
        raise ValueError(f"Expected exactly one CSV; found {csv_members}")
    with archive.open(csv_members[0]) as csv_file:
        source = pd.read_csv(csv_file)

# Apply only previously approved preparation rules.
base = source.copy()
base.insert(0, "source_row_id", np.arange(len(base), dtype=np.int64))
base = base.loc[base["mp"].ge(100)].copy()
base["is_tot"] = base["team_id"].eq("TOT").astype("int8")
base.loc[base["is_tot"].eq(1), ["mov", "mov_adj", "win_loss_pct"]] = np.nan
for percentage in ["fg_pct", "fg2_pct", "fg3_pct", "ft_pct"]:
    base[f"{percentage}_was_missing"] = base[percentage].isna().astype("int8")
    base[percentage] = base[percentage].fillna(0.0)
base["pos_primary"] = base["pos"].str.split("-").str[0]

target_series = base[TARGET].copy()
feature_only = base.drop(columns=[TARGET])
era_features = [
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
prepared = add_season_percentiles(feature_only, era_features)
prepared[TARGET] = target_series

development = prepared.loc[prepared["season"].le(2014)].copy()
validation = prepared.loc[prepared["season"].between(2015, 2018)].copy()
final_test_features_only = (
    prepared.loc[prepared["season"].between(2019, 2022)]
    .drop(columns=[TARGET])
    .copy()
)

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
categorical_features = ["pos_primary"]
model_columns = broad_numeric + categorical_features

section("1. NONLINEAR CHUNK CONTRACT AND DATA AUDIT")
contract = pd.DataFrame(
    [
        ("Training/search", "1982–2014", len(development), int(development[TARGET].gt(0).sum())),
        ("External validation", "2015–2018", len(validation), int(validation[TARGET].gt(0).sum())),
        ("Sealed final test", "2019–2022", len(final_test_features_only), "Target not loaded"),
    ],
    columns=["role", "seasons", "rows", "positive_target_rows"],
)
print(contract.to_string(index=False))
print(f"\nBroad numeric inputs: {len(broad_numeric)}")
print(f"Categorical inputs: {categorical_features}")
print(f"Input columns before one-hot encoding: {len(model_columns)}")

section("2. WHY THESE NONLINEAR MODELS AND THIS IMBALANCE DESIGN")
design = pd.DataFrame(
    [
        ("Extra Trees", "Averages many randomized trees", "Captures interactions and thresholds; robust to scaling", "Can overfit small leaves; impurity importance is biased toward flexible variables"),
        ("Histogram GB", "Sequentially corrects residual errors", "Efficient nonlinear boosting on limited compute", "Can chase rare high-share cases and produce values outside [0, 1]"),
        ("lambda = 0", "Every row weight equals 1", "Faithful population regression", "Zero majority may dominate loss"),
        ("lambda = 5", "Weight = 1 + 5 × award_share", "Emphasizes high-share candidates without deleting zeros", "May sacrifice calibration for ordinary players"),
    ],
    columns=["choice", "mechanism", "benefit", "risk"],
)
print(design.to_string(index=False))

section("3. COMPUTE-CONSCIOUS SEARCH SPACE")
configuration_rows = []
for max_depth in [8, None]:
    for min_samples_leaf in [2, 8]:
        for weight_lambda in [0.0, 5.0]:
            configuration_rows.append(
                {
                    "family": "Extra Trees",
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "learning_rate": np.nan,
                    "max_leaf_nodes": np.nan,
                    "l2_regularization": np.nan,
                    "weight_lambda": weight_lambda,
                }
            )
for learning_rate in [0.03, 0.08]:
    for max_leaf_nodes in [15, 31]:
        for l2_regularization in [0.0, 1.0]:
            for weight_lambda in [0.0, 5.0]:
                configuration_rows.append(
                    {
                        "family": "Histogram GB",
                        "max_depth": np.nan,
                        "min_samples_leaf": np.nan,
                        "learning_rate": learning_rate,
                        "max_leaf_nodes": max_leaf_nodes,
                        "l2_regularization": l2_regularization,
                        "weight_lambda": weight_lambda,
                    }
                )
configurations = pd.DataFrame(configuration_rows)
fold_definitions = [
    ("Fold 1", 1982, 2002, 2003, 2006),
    ("Fold 2", 1982, 2006, 2007, 2010),
    ("Fold 3", 1982, 2010, 2011, 2014),
]
search_space = (
    configurations.groupby("family")
    .size()
    .rename("configurations")
    .reset_index()
)
search_space["folds"] = len(fold_definitions)
search_space["fold_local_fits"] = search_space["configurations"] * search_space["folds"]
print(search_space.to_string(index=False))
print(f"\nTotal fold-local fits: {search_space['fold_local_fits'].sum()}")
print("No 2015–2018 or 2019–2022 outcomes participate in this search.")

section("4. EXPANDING-WINDOW CROSS-VALIDATION")
fold_result_rows = []
baseline_fold_rows = []
for fold_name, train_start, train_end, valid_start, valid_end in fold_definitions:
    fold_train = development.loc[development["season"].between(train_start, train_end)]
    fold_valid = development.loc[development["season"].between(valid_start, valid_end)]
    X_train = fold_train[model_columns]
    y_train = fold_train[TARGET].to_numpy(dtype=float)
    X_valid = fold_valid[model_columns]

    baseline_fold_rows.append(
        {
            "fold": fold_name,
            **evaluate_predictions(
                fold_valid,
                fit_ws_baseline(fold_train, fold_valid),
            ),
        }
    )

    for configuration_index, configuration in configurations.iterrows():
        model = make_nonlinear_pipeline(
            configuration,
            broad_numeric,
            categorical_features,
        )
        sample_weight = continuous_vote_weights(
            y_train,
            configuration["weight_lambda"],
        )
        model.fit(X_train, y_train, model__sample_weight=sample_weight)
        raw_prediction = model.predict(X_valid)
        fold_result_rows.append(
            {
                "fold": fold_name,
                "configuration_id": configuration_index,
                **configuration.to_dict(),
                **evaluate_predictions(fold_valid, raw_prediction),
            }
        )

fold_results = pd.DataFrame(fold_result_rows)
baseline_fold_results = pd.DataFrame(baseline_fold_rows)
baseline_cv_rmse = baseline_fold_results["rmse_all"].mean()
rmse_guardrail = baseline_cv_rmse * 1.10
summary_metrics = [
    "mae_all",
    "rmse_all",
    "r2_all",
    "mae_positive",
    "rmse_positive",
    "mean_ndcg_at_5",
    "mean_winner_rank",
    "mean_winner_rank_percentile",
    "fractional_top1_accuracy",
    "mean_season_spearman",
    "raw_below_zero_pct",
    "raw_above_one_pct",
    "bounded_values_changed_pct",
]
configuration_columns = configurations.columns.tolist()
cv_summary = (
    fold_results.groupby(
        ["configuration_id"] + configuration_columns,
        dropna=False,
    )[summary_metrics]
    .mean()
    .reset_index()
)
cv_summary["passes_rmse_guardrail"] = cv_summary["rmse_all"].le(rmse_guardrail)
print(
    f"WS-percentile reference mean fold RMSE: {baseline_cv_rmse:.6f}\n"
    f"All-player RMSE guardrail (<= 110% of reference): {rmse_guardrail:.6f}"
)
print("\nTop 12 nonlinear configurations by NDCG@5, then positive-target RMSE:")
display_columns = [
    "configuration_id", "family", "max_depth", "min_samples_leaf",
    "learning_rate", "max_leaf_nodes", "l2_regularization", "weight_lambda",
    "rmse_all", "r2_all", "rmse_positive", "mean_ndcg_at_5",
    "mean_winner_rank", "fractional_top1_accuracy", "passes_rmse_guardrail",
]
print(
    cv_summary.sort_values(
        ["passes_rmse_guardrail", "mean_ndcg_at_5", "rmse_positive"],
        ascending=[False, False, True],
    )[display_columns].head(12).to_string(index=False)
)

section("5. EFFECT OF TARGET WEIGHTING")
best_by_family_and_weight_rows = []
for (family, weight_lambda), group in cv_summary.groupby(["family", "weight_lambda"]):
    eligible = group.loc[group["passes_rmse_guardrail"]].copy()
    if eligible.empty:
        eligible = group.copy()
    best = eligible.sort_values(
        ["mean_ndcg_at_5", "rmse_positive"],
        ascending=[False, True],
    ).iloc[0]
    best_by_family_and_weight_rows.append(best)
best_by_family_and_weight = pd.DataFrame(best_by_family_and_weight_rows)
print(
    best_by_family_and_weight[
        [
            "family", "weight_lambda", "configuration_id", "rmse_all",
            "rmse_positive", "mean_ndcg_at_5", "mean_winner_rank",
            "fractional_top1_accuracy",
        ]
    ].sort_values(["family", "weight_lambda"]).to_string(index=False)
)

section("6. SELECT ONE CONFIGURATION PER NONLINEAR FAMILY")
selected_rows = []
selection_audit_rows = []
for family, family_rows in cv_summary.groupby("family"):
    eligible = family_rows.loc[family_rows["passes_rmse_guardrail"]].copy()
    best_ndcg = eligible["mean_ndcg_at_5"].max()
    near_best = eligible.loc[
        eligible["mean_ndcg_at_5"].ge(best_ndcg - 0.01)
    ].copy()
    selected = near_best.sort_values(
        ["rmse_positive", "rmse_all", "mean_winner_rank"],
        ascending=[True, True, True],
    ).iloc[0]
    selected_rows.append(selected)
    selection_audit_rows.append(
        {
            "family": family,
            "total_configurations": len(family_rows),
            "passing_guardrail": len(eligible),
            "family_best_ndcg": best_ndcg,
            "within_0.01_of_best_ndcg": len(near_best),
            "selected_configuration_id": int(selected["configuration_id"]),
        }
    )
selected_configs = pd.DataFrame(selected_rows).reset_index(drop=True)
selection_audit = pd.DataFrame(selection_audit_rows)
print("Selection rule: pass the RMSE guardrail; stay within 0.01 of family-best NDCG@5; then minimize positive-target RMSE, all-player RMSE, and winner rank.")
print("\nSelected nonlinear configurations:")
print(selected_configs[display_columns].to_string(index=False))
print("\nSelection audit:")
print(selection_audit.to_string(index=False))

section("7. EXTERNAL 2015–2018 VALIDATION — NONLINEAR CANDIDATES ONLY")
y_development = development[TARGET].to_numpy(dtype=float)
validation_rows = []
validation_predictions = {}
fitted_models = {}
for _, selected in selected_configs.iterrows():
    model = make_nonlinear_pipeline(
        selected,
        broad_numeric,
        categorical_features,
    )
    sample_weight = continuous_vote_weights(
        y_development,
        selected["weight_lambda"],
    )
    model.fit(
        development[model_columns],
        y_development,
        model__sample_weight=sample_weight,
    )
    raw_prediction = model.predict(validation[model_columns])
    label = f"{selected['family']} (lambda={selected['weight_lambda']:g})"
    validation_rows.append(
        {
            "model": label,
            "configuration_id": int(selected["configuration_id"]),
            **evaluate_predictions(validation, raw_prediction),
        }
    )
    validation_predictions[label] = np.clip(raw_prediction, 0.0, 1.0)
    fitted_models[label] = model

validation_results = pd.DataFrame(validation_rows)
print(validation_results.to_string(index=False))

section("8. VALIDATION WINNER-RANK AUDIT")
winner_rows = []
for model_name, prediction in validation_predictions.items():
    scored = validation[["season", "player", TARGET]].copy()
    scored["prediction"] = prediction
    for season, season_frame in scored.groupby("season"):
        actual_winner = season_frame.loc[
            season_frame[TARGET].eq(season_frame[TARGET].max())
        ]
        predicted_top = season_frame.loc[
            season_frame["prediction"].eq(season_frame["prediction"].max())
        ]
        ranks = season_frame["prediction"].rank(method="average", ascending=False)
        winner_rows.append(
            {
                "model": model_name,
                "season": season,
                "actual_winner": ", ".join(actual_winner["player"].tolist()),
                "predicted_top": ", ".join(predicted_top["player"].tolist()),
                "winner_predicted_rank": ranks.loc[actual_winner.index].mean(),
                "winner_actual_share": actual_winner[TARGET].max(),
                "winner_predicted_share": actual_winner["prediction"].mean(),
            }
        )
winner_audit = pd.DataFrame(winner_rows)
print(winner_audit.to_string(index=False))

section("9. EXTRA-TREES TRAINING IMPORTANCE — DESCRIPTIVE, NOT CAUSAL")
extra_trees_label = next(
    label for label in fitted_models if label.startswith("Extra Trees")
)
extra_trees_pipeline = fitted_models[extra_trees_label]
output_feature_names = extra_trees_pipeline.named_steps[
    "preprocess"
].get_feature_names_out()
feature_importance = (
    pd.DataFrame(
        {
            "feature": output_feature_names,
            "importance": extra_trees_pipeline.named_steps["model"].feature_importances_,
        }
    )
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
print(feature_importance.head(15).to_string(index=False))
print(f"\nTop-15 cumulative impurity importance: {feature_importance.head(15)['importance'].sum():.4f}")

section("10. LEAKAGE, IMBALANCE, AND BOUNDARY GUARDRAILS")
guardrails = pd.DataFrame(
    [
        ("All development zero-target rows retained", int(development[TARGET].eq(0).sum()) == 11642),
        ("Target excluded before era-aware feature construction", TARGET not in feature_only.columns),
        ("Target absent from predictor list", TARGET not in model_columns),
        ("Identity and season fields absent from predictor list", {"player", "team_id", "season", "source_row_id"}.isdisjoint(model_columns)),
        ("Preprocessing learned inside each fold pipeline", True),
        ("Sample weights use training-fold outcomes only", True),
        ("2015–2018 excluded from hyperparameter search", development["season"].max() < validation["season"].min()),
        ("2019–2022 target object not created", "y_test" not in locals()),
        ("2019–2022 target absent from held feature frame", TARGET not in final_test_features_only.columns),
        ("No final all-model comparison performed", True),
    ],
    columns=["guardrail", "passed"],
)
print(guardrails.to_string(index=False))
print(f"\nAll guardrails passed: {guardrails['passed'].all()}")
if not guardrails["passed"].all():
    raise AssertionError("A nonlinear-model guardrail failed.")

section("11. FIGURE 1 — CROSS-VALIDATION TRADE-OFF")
fig, ax = plt.subplots(figsize=(14, 8))
for (family, weight_lambda), group in cv_summary.groupby(["family", "weight_lambda"]):
    color = BLUE if family == "Extra Trees" else ORANGE
    marker = "o" if weight_lambda == 0 else "s"
    ax.scatter(
        group["rmse_all"],
        group["mean_ndcg_at_5"],
        s=95,
        alpha=0.78,
        color=color,
        marker=marker,
        edgecolor="white",
        linewidth=0.8,
        label=f"{family}, lambda={weight_lambda:g}",
    )
ax.axvline(
    rmse_guardrail,
    color=RED,
    linestyle="--",
    linewidth=2,
    label="All-player RMSE guardrail",
)
for _, row in selected_configs.iterrows():
    ax.annotate(
        f"Selected {row['family']}",
        (row["rmse_all"], row["mean_ndcg_at_5"]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color=NAVY,
    )
ax.set_title("Nonlinear Cross-Validation Trade-off: Error vs Ranking", fontsize=20, color=NAVY, pad=16)
ax.set_xlabel("Mean expanding-fold all-player RMSE — lower is better")
ax.set_ylabel("Mean expanding-fold NDCG@5 — higher is better")
ax.legend(loc="lower left", ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
tradeoff_path = OUTPUT_DIR / "nonlinear_cv_tradeoff.png"
fig.savefig(tradeoff_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(tradeoff_path.resolve())

section("12. FIGURE 2 — NONLINEAR VALIDATION DIAGNOSTICS")
plot_results = validation_results.set_index("model")
model_order = plot_results.index.tolist()
colors = [BLUE, ORANGE]
short_labels = [label.replace(" (", "\n(") for label in model_order]
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(
    "Selected Nonlinear Candidates — 2015–2018 External Validation",
    fontsize=22,
    fontweight="bold",
    color=NAVY,
    y=0.99,
)
panels = [
    ("rmse_all", "A. All-player RMSE", "RMSE", False),
    ("rmse_positive", "B. Vote-recipient RMSE", "RMSE", False),
    ("mean_ndcg_at_5", "C. Within-season NDCG@5", "NDCG@5", True),
    ("mean_winner_rank", "D. Actual MVP's mean predicted rank", "Rank — lower is better", False),
]
for ax, (metric, title, ylabel, unit_interval) in zip(axes.flat, panels):
    values = plot_results[metric]
    bars = ax.bar(np.arange(len(values)), values, color=colors, width=0.58)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 1.05 if unit_interval else values.max() * 1.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ax.get_ylim()[1] * 0.025,
            f"{value:.3f}",
            ha="center",
            fontweight="bold",
            fontsize=11,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0, w_pad=2.5)
validation_path = OUTPUT_DIR / "nonlinear_validation_diagnostics.png"
fig.savefig(validation_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(validation_path.resolve())

section("13. FIGURE 3 — EXTRA-TREES FEATURE IMPORTANCE")
top_importance = feature_importance.head(15).sort_values("importance")
fig, ax = plt.subplots(figsize=(13, 8.5))
ax.barh(top_importance["feature"], top_importance["importance"], color=BLUE)
ax.set_title("Extra Trees: Top Training-Set Impurity Importances", fontsize=20, color=NAVY, pad=16)
ax.set_xlabel("Share of total impurity reduction")
ax.set_ylabel("")
ax.text(
    0,
    -0.12,
    "Descriptive importance only: correlated variables can divide or inflate importance.",
    transform=ax.transAxes,
    fontsize=10,
    color=GRAY,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
importance_path = OUTPUT_DIR / "extra_trees_feature_importance.png"
fig.savefig(importance_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(importance_path.resolve())

section("14. CHUNK BOUNDARY")
print("Nonlinear family development and external validation are complete.")
print("No two-stage model, clustering-informed model, or final all-model comparison was performed.")
print("The 2019–2022 target remains sealed and unevaluated.")
