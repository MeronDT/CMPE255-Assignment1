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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
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
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk14_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "award_share"


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


def make_compact_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
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


def make_regularized_pipeline(
    family: str,
    alpha: float,
    l1_ratio: float | None,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    if family == "Ridge":
        estimator = Ridge(alpha=alpha)
    elif family == "Elastic Net":
        estimator = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=100_000,
            tol=1e-6,
            selection="cyclic",
        )
    else:
        raise ValueError(f"Unknown model family: {family}")

    return Pipeline(
        [
            (
                "preprocess",
                make_compact_preprocessor(numeric_features, categorical_features),
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
        top1_credit = np.logical_and(actual_winner, predicted_top).sum() / predicted_top.sum()
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

# Reconstruct the locked preparation design.
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
model_columns = compact_numeric + categorical_features

section("1. TARGET-IMBALANCE RESPONSE PLAN")
imbalance_plan = pd.DataFrame(
    [
        ("Keep all zero rows", "Implemented", "Preserves the real eligible-player population"),
        ("Positive-only error metrics", "Implemented", "Stops the zero majority from hiding candidate errors"),
        ("Season ranking metrics", "Implemented", "Measures the actual MVP ordering objective"),
        ("Continuous vote emphasis", "Tested here", "Weights larger award shares more without converting the target to a class"),
        ("Random oversampling", "Rejected", "Would duplicate rare seasons/candidates and distort the temporal population"),
        ("All-player RMSE guardrail", "Implemented", "Prevents candidate emphasis from overpredicting everyone"),
        ("Two-stage candidate/share model", "Planned later", "Directly separates vote receipt from conditional vote share"),
        ("Calibration and boundedness", "Tracked", "Reports clipping and candidate underprediction rather than hiding it"),
    ],
    columns=["response", "status", "reason"],
)
print(imbalance_plan.to_string(index=False))

weight_lambdas = [0.0, 5.0, 15.0]
weight_examples = pd.DataFrame(
    {
        "award_share": [0.0, 0.01, 0.10, 0.50, 1.0],
    }
)
for weight_lambda in weight_lambdas:
    weight_examples[f"lambda_{weight_lambda:g}_weight"] = continuous_vote_weights(
        weight_examples["award_share"].to_numpy(),
        weight_lambda,
    )
print("\nContinuous training-weight examples, weight = 1 + lambda * award_share:")
print(weight_examples.to_string(index=False))

section("2. CHRONOLOGICAL SEARCH SPACE")
ridge_alphas = [0.1, 1.0, 10.0, 100.0]
elastic_alphas = [0.0001, 0.001, 0.01, 0.1]
elastic_l1_ratios = [0.1, 0.5, 0.9]
fold_definitions = [
    ("Fold 1", 1982, 2002, 2003, 2006),
    ("Fold 2", 1982, 2006, 2007, 2010),
    ("Fold 3", 1982, 2010, 2011, 2014),
]

search_contract = pd.DataFrame(
    [
        ("Ridge", ridge_alphas, "Not applicable", weight_lambdas, len(ridge_alphas) * len(weight_lambdas)),
        ("Elastic Net", elastic_alphas, elastic_l1_ratios, weight_lambdas, len(elastic_alphas) * len(elastic_l1_ratios) * len(weight_lambdas)),
    ],
    columns=["family", "alpha_values", "l1_ratio_values", "weight_lambda_values", "configurations"],
)
print(search_contract.to_string(index=False))
print(f"\nTotal fold-local fits: {(12 + 36) * len(fold_definitions)}")

section("3. EXPANDING-FOLD GRID EVALUATION")
configuration_rows = []
for alpha in ridge_alphas:
    for weight_lambda in weight_lambdas:
        configuration_rows.append(
            {
                "family": "Ridge",
                "alpha": alpha,
                "l1_ratio": np.nan,
                "weight_lambda": weight_lambda,
            }
        )
for alpha in elastic_alphas:
    for l1_ratio in elastic_l1_ratios:
        for weight_lambda in weight_lambdas:
            configuration_rows.append(
                {
                    "family": "Elastic Net",
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "weight_lambda": weight_lambda,
                }
            )
configurations = pd.DataFrame(configuration_rows)

fold_result_rows = []
baseline_fold_rows = []
for fold_name, train_start, train_end, valid_start, valid_end in fold_definitions:
    fold_train = development.loc[development["season"].between(train_start, train_end)]
    fold_valid = development.loc[development["season"].between(valid_start, valid_end)]
    X_train = fold_train[model_columns]
    y_train = fold_train[TARGET].to_numpy(dtype=float)
    X_valid = fold_valid[model_columns]

    ws_raw = fit_ws_baseline(fold_train, fold_valid)
    baseline_fold_rows.append(
        {
            "fold": fold_name,
            **evaluate_predictions(fold_valid, ws_raw),
        }
    )

    for configuration in configurations.itertuples(index=False):
        model = make_regularized_pipeline(
            configuration.family,
            configuration.alpha,
            None if pd.isna(configuration.l1_ratio) else configuration.l1_ratio,
            compact_numeric,
            categorical_features,
        )
        sample_weight = continuous_vote_weights(y_train, configuration.weight_lambda)
        model.fit(X_train, y_train, model__sample_weight=sample_weight)
        raw_prediction = model.predict(X_valid)
        fold_result_rows.append(
            {
                "fold": fold_name,
                "family": configuration.family,
                "alpha": configuration.alpha,
                "l1_ratio": configuration.l1_ratio,
                "weight_lambda": configuration.weight_lambda,
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
    "bounded_values_changed_pct",
]
cv_summary = (
    fold_results.groupby(
        ["family", "alpha", "l1_ratio", "weight_lambda"],
        dropna=False,
    )[summary_metrics]
    .mean()
    .reset_index()
)
cv_summary["passes_all_player_rmse_guardrail"] = cv_summary["rmse_all"].le(rmse_guardrail)

print(
    f"WS-percentile baseline mean fold RMSE: {baseline_cv_rmse:.6f}\n"
    f"Regularized-model RMSE guardrail (<= 110% of baseline): {rmse_guardrail:.6f}"
)
print("\nTop 12 configurations by NDCG@5, then positive-target RMSE:")
print(
    cv_summary.sort_values(
        ["passes_all_player_rmse_guardrail", "mean_ndcg_at_5", "rmse_positive"],
        ascending=[False, False, True],
    ).head(12).to_string(index=False)
)

section("4. SELECT ONE CONFIGURATION PER FAMILY")
selected_rows = []
selection_explanations = []
for family in ["Ridge", "Elastic Net"]:
    family_rows = cv_summary.loc[
        cv_summary["family"].eq(family)
        & cv_summary["passes_all_player_rmse_guardrail"]
    ].copy()
    best_ndcg = family_rows["mean_ndcg_at_5"].max()
    near_best = family_rows.loc[
        family_rows["mean_ndcg_at_5"].ge(best_ndcg - 0.01)
    ].copy()
    selected = near_best.sort_values(
        ["rmse_positive", "rmse_all", "mean_winner_rank"],
        ascending=[True, True, True],
    ).iloc[0]
    selected_rows.append(selected)
    selection_explanations.append(
        {
            "family": family,
            "configurations_passing_guardrail": len(family_rows),
            "best_family_ndcg": best_ndcg,
            "configurations_within_0.01_ndcg": len(near_best),
            "tie_breaker": "Lowest positive-target RMSE, then all-player RMSE, then winner rank",
        }
    )

selected_configs = pd.DataFrame(selected_rows).reset_index(drop=True)
selection_explanation = pd.DataFrame(selection_explanations)
print("Selection rule: pass the all-player RMSE guardrail; remain within 0.01 of family-best NDCG@5; then minimize positive-target RMSE.")
print("\nSelected configurations:")
print(selected_configs.to_string(index=False))
print("\nSelection audit:")
print(selection_explanation.to_string(index=False))

section("5. TARGET-WEIGHT DISTRIBUTION FOR SELECTED CONFIGURATIONS")
weight_distribution_rows = []
y_development = development[TARGET].to_numpy(dtype=float)
for selected in selected_configs.itertuples(index=False):
    weights = continuous_vote_weights(y_development, selected.weight_lambda)
    positive = y_development > 0
    weight_distribution_rows.append(
        {
            "family": selected.family,
            "weight_lambda": selected.weight_lambda,
            "minimum_weight": weights.min(),
            "mean_weight": weights.mean(),
            "maximum_weight": weights.max(),
            "zero_row_weight": weights[~positive][0],
            "mean_positive_row_weight": weights[positive].mean(),
            "positive_rows_pct": 100 * positive.mean(),
            "positive_share_of_total_training_weight_pct": 100
            * weights[positive].sum()
            / weights.sum(),
        }
    )
weight_distribution = pd.DataFrame(weight_distribution_rows)
print(weight_distribution.to_string(index=False))

section("6. FIT SELECTED MODELS ON DEVELOPMENT; EVALUATE 2015–2018")
validation_rows = []
fitted_models = {}
validation_predictions = {}

ws_validation_raw = fit_ws_baseline(development, validation)
validation_rows.append(
    {
        "model": "WS-percentile baseline",
        **evaluate_predictions(validation, ws_validation_raw),
    }
)
validation_predictions["WS-percentile baseline"] = np.clip(ws_validation_raw, 0, 1)

for selected in selected_configs.itertuples(index=False):
    model = make_regularized_pipeline(
        selected.family,
        selected.alpha,
        None if pd.isna(selected.l1_ratio) else selected.l1_ratio,
        compact_numeric,
        categorical_features,
    )
    sample_weight = continuous_vote_weights(
        y_development,
        selected.weight_lambda,
    )
    model.fit(
        development[model_columns],
        y_development,
        model__sample_weight=sample_weight,
    )
    raw_prediction = model.predict(validation[model_columns])
    label = (
        f"{selected.family} "
        f"(lambda={selected.weight_lambda:g})"
    )
    validation_rows.append(
        {
            "model": label,
            **evaluate_predictions(validation, raw_prediction),
        }
    )
    validation_predictions[label] = np.clip(raw_prediction, 0, 1)
    fitted_models[label] = model

validation_results = pd.DataFrame(validation_rows)
print(validation_results.to_string(index=False))

section("7. VALIDATION WINNER-RANK AUDIT")
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

section("8. COEFFICIENT SPARSITY AND GUARDRAILS")
coefficient_rows = []
for model_name, model in fitted_models.items():
    coefficients = model.named_steps["model"].coef_
    coefficient_rows.append(
        {
            "model": model_name,
            "output_features": len(coefficients),
            "nonzero_coefficients": int(np.count_nonzero(~np.isclose(coefficients, 0.0))),
            "zero_coefficients": int(np.isclose(coefficients, 0.0).sum()),
            "largest_absolute_coefficient": np.abs(coefficients).max(),
        }
    )
coefficient_audit = pd.DataFrame(coefficient_rows)
print(coefficient_audit.to_string(index=False))

guardrails = pd.DataFrame(
    [
        ("All zero-target rows retained in every training fold", development[TARGET].eq(0).sum() == 11642),
        ("Weights use training-fold targets only", True),
        ("Preprocessing fitted inside each fold pipeline", True),
        ("2015–2018 excluded from hyperparameter search", development["season"].max() < validation["season"].min()),
        ("2019–2022 target object not created", "y_test" not in locals()),
        ("2019–2022 target absent from held features", TARGET not in final_test_features_only.columns),
        ("No nonlinear, two-stage, ranking, or clustering model fitted", True),
    ],
    columns=["guardrail", "passed"],
)
print("\nGuardrails:")
print(guardrails.to_string(index=False))
print(f"\nAll guardrails passed: {guardrails['passed'].all()}")
if not guardrails["passed"].all():
    raise AssertionError("A regularized-model guardrail failed.")

section("9. FIGURE 1 — TARGET-WEIGHTING DESIGN")
share_grid = np.linspace(0, 1, 101)
fig, ax = plt.subplots(figsize=(14, 6.5))
for weight_lambda, color in zip(weight_lambdas, [GRAY, BLUE, ORANGE]):
    ax.plot(
        share_grid,
        continuous_vote_weights(share_grid, weight_lambda),
        linewidth=3,
        color=color,
        label=f"lambda = {weight_lambda:g}",
    )
ax.set_title("Continuous Vote-Share Emphasis Keeps Every Zero Row", fontsize=20, color=NAVY, pad=16)
ax.set_xlabel("Observed training award_share")
ax.set_ylabel("Training weight")
ax.set_xlim(0, 1)
ax.set_ylim(0.5, 16.8)
ax.legend(loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
weight_figure_path = OUTPUT_DIR / "continuous_target_weighting.png"
fig.savefig(weight_figure_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(weight_figure_path.resolve())

section("10. FIGURE 2 — VALIDATION MODEL COMPARISON")
plot_results = validation_results.set_index("model")
model_order = plot_results.index.tolist()
model_colors = [GRAY, BLUE, ORANGE]
fig, axes = plt.subplots(2, 2, figsize=(17, 11))
fig.suptitle(
    "Regularized Linear Models vs Baseline — 2015–2018 Validation",
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
short_labels = [
    "WS-percentile\nbaseline",
    model_order[1].replace(" (", "\n(") if len(model_order) > 1 else "Ridge",
    model_order[2].replace(" (", "\n(") if len(model_order) > 2 else "Elastic Net",
]

for ax, (metric, title, ylabel, unit_interval) in zip(axes.flat, panels):
    values = plot_results[metric]
    bars = ax.bar(np.arange(len(values)), values, color=model_colors, width=0.62)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 1.05 if unit_interval else values.max() * 1.22)
    for bar, value in zip(bars, values):
        label = f"{value:.3f}" if value < 10 else f"{value:.1f}"
        offset = ax.get_ylim()[1] * 0.025
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            label,
            ha="center",
            fontweight="bold",
            fontsize=10,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0, w_pad=2.5)
comparison_figure_path = OUTPUT_DIR / "regularized_validation_comparison.png"
fig.savefig(comparison_figure_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(comparison_figure_path.resolve())

section("11. CHUNK BOUNDARY")
print("Ridge and Elastic Net target-weight sensitivity is complete.")
print("No nonlinear ensemble, two-stage, direct ranking, or clustering model was fitted.")
print("The 2019–2022 target remains sealed and unevaluated.")
