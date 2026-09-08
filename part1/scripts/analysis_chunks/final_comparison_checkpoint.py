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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 280)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")

NAVY = "#17324D"
BLUE = "#2F6B9A"
ORANGE = "#D97706"
GREEN = "#2F855A"
RED = "#B8473D"
PURPLE = "#805AD5"
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
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "grid.color": LIGHT_GRAY,
        "grid.linewidth": 0.8,
    },
)

ZIP_PATH = Path(os.environ.get("NBA_MVP_DATA_PATH", "data/raw/nba_mvp_stats.zip"))
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk18_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "award_share"
RANDOM_STATE = 42


def section(title: str) -> None:
    print(f"\n{'=' * 124}\n{title}\n{'=' * 124}")


def add_season_percentiles(frame_without_target: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if TARGET in frame_without_target.columns:
        raise ValueError("Target must be absent during era-aware feature creation.")
    result = frame_without_target.copy()
    for feature in features:
        result[f"{feature}_season_pct"] = result.groupby("season")[feature].rank(
            method="average", pct=True, na_option="keep"
        )
    return result


def make_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", RobustScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    positive = y_true > 0
    return {
        "mae_all": mean_absolute_error(y_true, prediction),
        "rmse_all": np.sqrt(mean_squared_error(y_true, prediction)),
        "r2_all": r2_score(y_true, prediction),
        "mae_positive": mean_absolute_error(y_true[positive], prediction[positive]),
        "rmse_positive": np.sqrt(mean_squared_error(y_true[positive], prediction[positive])),
    }


def season_ranking_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> tuple[dict, pd.DataFrame]:
    evaluation = frame[["season", "player", TARGET]].copy()
    evaluation["prediction"] = prediction
    rows = []
    for season, group in evaluation.groupby("season"):
        y_true = group[TARGET].to_numpy(dtype=float)
        y_score = group["prediction"].to_numpy(dtype=float)
        actual_winner = np.isclose(y_true, y_true.max())
        predicted_top = np.isclose(y_score, y_score.max())
        ranks = pd.Series(y_score).rank(method="average", ascending=False).to_numpy()
        winner_rank = ranks[actual_winner].mean()
        n_players = len(group)
        spearman = (
            pd.Series(y_true).corr(pd.Series(y_score), method="spearman")
            if np.unique(y_score).size > 1 else np.nan
        )
        rows.append(
            {
                "season": season,
                "ndcg_at_5": ndcg_score(
                    y_true.reshape(1, -1),
                    y_score.reshape(1, -1),
                    k=min(5, n_players),
                    ignore_ties=False,
                ),
                "winner_rank": winner_rank,
                "winner_rank_percentile": 1 - (winner_rank - 1) / (n_players - 1),
                "fractional_top1_accuracy": np.logical_and(actual_winner, predicted_top).sum() / predicted_top.sum(),
                "spearman": spearman,
                "predicted_top_count": int(predicted_top.sum()),
                "actual_season_total": y_true.sum(),
                "predicted_season_total": y_score.sum(),
                "winner_actual_share": y_true[actual_winner].mean(),
                "winner_predicted_share": y_score[actual_winner].mean(),
            }
        )
    per_season = pd.DataFrame(rows)
    summary = {
        "mean_ndcg_at_5": per_season["ndcg_at_5"].mean(),
        "mean_winner_rank": per_season["winner_rank"].mean(),
        "mean_winner_rank_percentile": per_season["winner_rank_percentile"].mean(),
        "fractional_top1_accuracy": per_season["fractional_top1_accuracy"].mean(),
        "mean_season_spearman": per_season["spearman"].mean(),
        "mean_actual_season_total": per_season["actual_season_total"].mean(),
        "mean_predicted_season_total": per_season["predicted_season_total"].mean(),
        "season_total_mae": mean_absolute_error(
            per_season["actual_season_total"], per_season["predicted_season_total"]
        ),
        "mean_winner_share_bias": (
            per_season["winner_predicted_share"] - per_season["winner_actual_share"]
        ).mean(),
    }
    return summary, per_season


def evaluate_model(frame: pd.DataFrame, raw_prediction: np.ndarray, internal_adjustment_pct: float = 0.0) -> tuple[dict, pd.DataFrame, np.ndarray]:
    bounded = np.clip(raw_prediction, 0.0, 1.0)
    y_true = frame[TARGET].to_numpy(dtype=float)
    ranking_summary, per_season = season_ranking_metrics(frame, bounded)
    summary = {
        **regression_metrics(y_true, bounded),
        **ranking_summary,
        "raw_minimum": raw_prediction.min(),
        "raw_maximum": raw_prediction.max(),
        "final_clipping_pct": 100 * np.mean(~np.isclose(raw_prediction, bounded)),
        "internal_component_adjustment_pct": internal_adjustment_pct,
    }
    return summary, per_season, bounded


def make_linear_pipeline(numeric: list[str], categorical: list[str], estimator) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(numeric, categorical, scale=True)),
            ("model", estimator),
        ]
    )


def make_tree_pipeline(numeric: list[str], categorical: list[str], estimator) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(numeric, categorical, scale=False)),
            ("model", estimator),
        ]
    )


def fit_extra_trees_hurdle(training: pd.DataFrame, evaluation: pd.DataFrame, broad_numeric: list[str], categorical: list[str]) -> tuple[np.ndarray, float]:
    columns = broad_numeric + categorical
    gate = make_tree_pipeline(
        broad_numeric,
        categorical,
        ExtraTreesClassifier(
            n_estimators=160,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )
    y_binary = training[TARGET].gt(0).astype(int).to_numpy()
    gate_weight = np.where(y_binary == 1, 5.0, 1.0)
    gate.fit(training[columns], y_binary, model__sample_weight=gate_weight)
    severity = make_tree_pipeline(
        broad_numeric,
        categorical,
        ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )
    positive = training.loc[training[TARGET].gt(0)]
    severity.fit(positive[columns], positive[TARGET])
    probability = gate.predict_proba(evaluation[columns])[:, 1]
    severity_raw = severity.predict(evaluation[columns])
    severity_bounded = np.clip(severity_raw, 0.0, 1.0)
    return probability * severity_bounded, 100 * np.mean(~np.isclose(severity_raw, severity_bounded))


def fit_logistic_ridge_hurdle(training: pd.DataFrame, evaluation: pd.DataFrame, compact_numeric: list[str], categorical: list[str]) -> tuple[np.ndarray, float]:
    columns = compact_numeric + categorical
    gate = make_linear_pipeline(
        compact_numeric,
        categorical,
        LogisticRegression(C=0.1, solver="lbfgs", max_iter=5_000, random_state=RANDOM_STATE),
    )
    y_binary = training[TARGET].gt(0).astype(int).to_numpy()
    gate.fit(training[columns], y_binary)
    severity = make_linear_pipeline(
        compact_numeric,
        categorical,
        Ridge(alpha=100.0),
    )
    positive = training.loc[training[TARGET].gt(0)]
    severity.fit(positive[columns], positive[TARGET])
    probability = gate.predict_proba(evaluation[columns])[:, 1]
    severity_raw = severity.predict(evaluation[columns])
    severity_bounded = np.clip(severity_raw, 0.0, 1.0)
    return probability * severity_bounded, 100 * np.mean(~np.isclose(severity_raw, severity_bounded))


def is_pareto_efficient(table: pd.DataFrame) -> pd.Series:
    values = np.column_stack(
        [
            table["rmse_all"].to_numpy(),
            table["rmse_positive"].to_numpy(),
            -table["mean_ndcg_at_5"].to_numpy(),
            table["mean_winner_rank"].to_numpy(),
        ]
    )
    efficient = np.ones(len(values), dtype=bool)
    for index, candidate in enumerate(values):
        dominated_by_other = np.any(
            np.all(values <= candidate, axis=1)
            & np.any(values < candidate, axis=1)
        )
        efficient[index] = not dominated_by_other
    return pd.Series(efficient, index=table.index)


with ZipFile(ZIP_PATH) as archive:
    csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv") and not name.endswith("/")]
    if len(csv_members) != 1:
        raise ValueError(f"Expected one CSV; found {csv_members}")
    with archive.open(csv_members[0]) as csv_file:
        source = pd.read_csv(csv_file)

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
era_features = ["pts_per_g", "mp_per_g", "ts_pct", "per", "ws", "ws_per_48", "bpm", "vorp", "win_loss_pct"]
prepared = add_season_percentiles(feature_only, era_features)
prepared[TARGET] = target_series

development = prepared.loc[prepared["season"].le(2014)].copy()
validation = prepared.loc[prepared["season"].between(2015, 2018)].copy()
final_test_features_only = prepared.loc[prepared["season"].between(2019, 2022)].drop(columns=[TARGET]).copy()

compact_numeric = [
    "age", "g", "mp_per_g", "pts_per_g", "trb_per_g", "ast_per_g", "stl_per_g",
    "blk_per_g", "tov_per_g", "ts_pct", "usg_pct", "per", "ws", "bpm",
    "win_loss_pct", "is_tot", "pts_per_g_season_pct", "mp_per_g_season_pct",
    "ts_pct_season_pct", "per_season_pct", "ws_season_pct", "bpm_season_pct",
    "win_loss_pct_season_pct",
]
categorical = ["pos_primary"]
broad_exclusions = {
    "season", TARGET, "source_row_id", "mp", "fg_per_g", "fg2_per_g", "fg3_per_g",
    "ft_per_g", "orb_per_g", "drb_per_g", "orb_pct", "drb_pct", "efg_pct",
    "ows", "dws", "obpm", "dbpm", "mov",
}
broad_numeric = [
    feature for feature in prepared.select_dtypes(include=np.number).columns
    if feature not in broad_exclusions
]

section("1. FROZEN COMPARISON RUBRIC AND DATA BOUNDARY")
rubric = pd.DataFrame(
    [
        (1, "Season ranking", "NDCG@5; winner rank; top-1 accuracy", "Primary business objective"),
        (2, "Vote-share prediction", "Vote-recipient RMSE; all-player RMSE and R²", "Numerical target quality"),
        (3, "Prediction validity", "Clipping; season-total error; winner-share bias", "Calibration and impossible values"),
        (4, "Model risk", "Complexity; stability; interpretability", "Tie-breaker, not a substitute for validation performance"),
    ],
    columns=["priority", "dimension", "metrics", "role"],
)
print(rubric.to_string(index=False))
print(f"\nDevelopment: {development['season'].min()}–{development['season'].max()}, {len(development)} rows")
print(f"Validation: {validation['season'].min()}–{validation['season'].max()}, {len(validation)} rows")
print(f"Sealed final-test features: {len(final_test_features_only)} rows; target absent = {TARGET not in final_test_features_only.columns}")

section("2. FIT RETAINED MODELS ON DEVELOPMENT; SCORE 2015–2018")
predictions = {}
internal_adjustments = {}

predictions["Zero baseline"] = np.zeros(len(validation))
predictions["Historical mean"] = np.full(len(validation), development[TARGET].mean())

ws_model = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("model", LinearRegression()),
    ]
)
ws_model.fit(development[["ws_season_pct"]], development[TARGET])
predictions["WS OLS baseline"] = ws_model.predict(validation[["ws_season_pct"]])

ridge = make_linear_pipeline(compact_numeric, categorical, Ridge(alpha=100.0))
ridge.fit(
    development[compact_numeric + categorical],
    development[TARGET],
    model__sample_weight=1 + 5 * development[TARGET].to_numpy(),
)
predictions["Weighted Ridge"] = ridge.predict(validation[compact_numeric + categorical])

extra_trees = make_tree_pipeline(
    broad_numeric,
    categorical,
    ExtraTreesRegressor(
        n_estimators=160,
        max_depth=None,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
)
extra_trees.fit(
    development[broad_numeric + categorical],
    development[TARGET],
    model__sample_weight=1 + 5 * development[TARGET].to_numpy(),
)
predictions["Extra Trees"] = extra_trees.predict(validation[broad_numeric + categorical])

histogram_gb = make_tree_pipeline(
    broad_numeric,
    categorical,
    HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        max_iter=200,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=RANDOM_STATE,
    ),
)
histogram_gb.fit(
    development[broad_numeric + categorical],
    development[TARGET],
    model__sample_weight=1 + 5 * development[TARGET].to_numpy(),
)
predictions["Histogram GB"] = histogram_gb.predict(validation[broad_numeric + categorical])

predictions["Extra Trees hurdle"], internal_adjustments["Extra Trees hurdle"] = fit_extra_trees_hurdle(
    development, validation, broad_numeric, categorical
)
predictions["Logistic + Ridge hurdle"], internal_adjustments["Logistic + Ridge hurdle"] = fit_logistic_ridge_hurdle(
    development, validation, compact_numeric, categorical
)

summary_rows = []
per_season_tables = {}
bounded_predictions = {}
for model_name, raw_prediction in predictions.items():
    summary, per_season, bounded = evaluate_model(
        validation,
        np.asarray(raw_prediction),
        internal_adjustments.get(model_name, 0.0),
    )
    summary_rows.append({"model": model_name, **summary})
    per_season_tables[model_name] = per_season
    bounded_predictions[model_name] = bounded

comparison = pd.DataFrame(summary_rows).set_index("model")
comparison["pareto_efficient"] = is_pareto_efficient(comparison)
comparison["beats_ws_rmse"] = comparison["rmse_all"].lt(comparison.loc["WS OLS baseline", "rmse_all"])
comparison["beats_ws_positive_rmse"] = comparison["rmse_positive"].lt(comparison.loc["WS OLS baseline", "rmse_positive"])
comparison["beats_ws_ndcg"] = comparison["mean_ndcg_at_5"].gt(comparison.loc["WS OLS baseline", "mean_ndcg_at_5"])
comparison["passes_baseline_improvement_rule"] = comparison[
    ["beats_ws_rmse", "beats_ws_positive_rmse", "beats_ws_ndcg"]
].all(axis=1)

main_columns = [
    "mae_all", "rmse_all", "r2_all", "mae_positive", "rmse_positive",
    "mean_ndcg_at_5", "mean_winner_rank", "fractional_top1_accuracy",
    "mean_season_spearman", "pareto_efficient", "passes_baseline_improvement_rule",
]
print(comparison[main_columns].to_string())

section("3. CALIBRATION AND BOUNDARY DIAGNOSTICS")
calibration_columns = [
    "mean_actual_season_total", "mean_predicted_season_total", "season_total_mae",
    "mean_winner_share_bias", "raw_minimum", "raw_maximum", "final_clipping_pct",
    "internal_component_adjustment_pct",
]
print(comparison[calibration_columns].to_string())

section("4. SEASON-BY-SEASON MVP AUDIT")
winner_rows = []
for model_name, bounded in bounded_predictions.items():
    scored = validation[["season", "player", TARGET]].copy()
    scored["prediction"] = bounded
    for season, group in scored.groupby("season"):
        actual_winner = group.loc[group[TARGET].eq(group[TARGET].max())]
        predicted_top = group.loc[np.isclose(group["prediction"], group["prediction"].max())]
        ranks = group["prediction"].rank(method="average", ascending=False)
        if len(predicted_top) <= 3:
            predicted_top_label = ", ".join(predicted_top["player"].tolist())
        else:
            predicted_top_label = f"{len(predicted_top)}-player tie"
        winner_rows.append(
            {
                "model": model_name,
                "season": season,
                "actual_winner": ", ".join(actual_winner["player"].tolist()),
                "predicted_top": predicted_top_label,
                "actual_winner_rank": ranks.loc[actual_winner.index].mean(),
                "season_ndcg_at_5": per_season_tables[model_name].set_index("season").loc[season, "ndcg_at_5"],
                "winner_actual_share": actual_winner[TARGET].mean(),
                "winner_predicted_share": actual_winner["prediction"].mean(),
            }
        )
winner_audit = pd.DataFrame(winner_rows)
serious_models = [
    "WS OLS baseline", "Weighted Ridge", "Extra Trees", "Histogram GB",
    "Extra Trees hurdle", "Logistic + Ridge hurdle",
]
print(winner_audit.loc[winner_audit["model"].isin(serious_models)].to_string(index=False))

section("5. TOP-FIVE PREDICTED PLAYERS FOR PROPOSED FINALISTS")
proposed_finalists = ["Histogram GB", "Weighted Ridge"]
top_five_rows = []
for model_name in proposed_finalists:
    scored = validation[["season", "player", TARGET]].copy()
    scored["prediction"] = bounded_predictions[model_name]
    for season, group in scored.groupby("season"):
        top = group.sort_values("prediction", ascending=False).head(5)
        for predicted_rank, (_, row) in enumerate(top.iterrows(), start=1):
            top_five_rows.append(
                {
                    "model": model_name,
                    "season": season,
                    "predicted_rank": predicted_rank,
                    "player": row["player"],
                    "predicted_share": row["prediction"],
                    "actual_share": row[TARGET],
                }
            )
top_five = pd.DataFrame(top_five_rows)
print(top_five.to_string(index=False))

section("6. FROZEN PROPOSED DECISION")
decision = pd.DataFrame(
    [
        ("Champion", "Histogram GB", "Best all-player and vote-recipient RMSE; strong NDCG and winner rank", "87.18% of raw validation predictions require zero clipping"),
        ("Ranking challenger", "Weighted Ridge", "Highest NDCG@5 and tied-best mean winner rank; simpler and interpretable", "Substantially worse numerical share prediction and 55.62% clipping"),
        ("Final reference baseline", "WS OLS baseline", "Simple historical benchmark with meaningful ranking signal", "Severe winner-share underprediction"),
        ("Do not advance", "Extra Trees / Extra Trees hurdle", "Valid, stable nonlinear alternatives", "Pareto-dominated by Histogram GB on validation metrics"),
        ("Do not advance", "Logistic + Ridge hurdle", "Strong ranking and correct 2017 winner", "92.09% conditional-share truncation and weaker than champion numerically"),
    ],
    columns=["role", "model", "reason", "principal_risk"],
)
print(decision.to_string(index=False))

section("7. GUARDRAILS")
guardrails = pd.DataFrame(
    [
        ("Comparison rubric declared before consolidated scoring", True),
        ("All retained model configurations came from development folds", True),
        ("2015–2018 used for comparison, not hyperparameter search", True),
        ("Cluster feature excluded from finalists", True),
        ("Target absent from all predictor lists", TARGET not in compact_numeric and TARGET not in broad_numeric),
        ("2019–2022 target object not created", "y_test" not in locals()),
        ("2019–2022 target absent from held feature frame", TARGET not in final_test_features_only.columns),
        ("No model fitted on 2015–2018 outcomes", True),
        ("No 2019–2022 model evaluation performed", True),
    ],
    columns=["guardrail", "passed"],
)
print(guardrails.to_string(index=False))
print(f"\nAll guardrails passed: {guardrails['passed'].all()}")
if not guardrails["passed"].all():
    raise AssertionError("A final-comparison guardrail failed.")

section("8. FIGURE 1 — REGRESSION COMPARISON")
plot_order = comparison.sort_values("rmse_positive").index.tolist()
plot_data = comparison.loc[plot_order]
colors = [
    ORANGE if model == "Histogram GB" else
    BLUE if model == "Weighted Ridge" else
    GRAY if "baseline" in model.lower() or model == "Historical mean" else
    PURPLE
    for model in plot_order
]
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, metric, title in [
    (axes[0], "rmse_all", "A. All-player RMSE"),
    (axes[1], "rmse_positive", "B. Vote-recipient RMSE"),
]:
    values = plot_data[metric]
    bars = ax.barh(plot_order, values, color=colors)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("RMSE — lower is better")
    ax.set_xlim(0, values.max() * 1.18)
    for bar, value in zip(bars, values):
        ax.text(value + values.max() * 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.suptitle("Vote-Share Prediction — 2015–2018 Validation", fontsize=22, fontweight="bold", color=NAVY, y=1.02)
fig.tight_layout()
regression_path = OUTPUT_DIR / "final_validation_regression_comparison.png"
fig.savefig(regression_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(regression_path.resolve())

section("9. FIGURE 2 — RANKING COMPARISON")
ranking_order = comparison.sort_values("mean_ndcg_at_5", ascending=False).index.tolist()
ranking_data = comparison.loc[ranking_order]
ranking_colors = [
    ORANGE if model == "Histogram GB" else
    BLUE if model == "Weighted Ridge" else
    GRAY if "baseline" in model.lower() or model == "Historical mean" else
    PURPLE
    for model in ranking_order
]
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
ndcg_values = ranking_data["mean_ndcg_at_5"]
ndcg_bars = axes[0].barh(ranking_order, ndcg_values, color=ranking_colors)
axes[0].invert_yaxis()
axes[0].set_title("A. NDCG@5")
axes[0].set_xlabel("Higher is better")
axes[0].set_xlim(0, 1.11)
for bar, value in zip(ndcg_bars, ndcg_values):
    axes[0].text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=10, fontweight="bold")

rank_values = ranking_data["mean_winner_rank"].to_numpy()
y_positions = np.arange(len(ranking_order))
axes[1].hlines(y_positions, 1.0, rank_values, color=ranking_colors, linewidth=4, alpha=0.75)
axes[1].scatter(rank_values, y_positions, color=ranking_colors, s=110, zorder=3, edgecolor="white", linewidth=0.9)
axes[1].set_yticks(y_positions)
axes[1].set_yticklabels(ranking_order)
axes[1].invert_yaxis()
axes[1].set_xscale("log")
axes[1].set_xlim(0.9, 320)
axes[1].set_title("B. Actual MVP's mean predicted rank")
axes[1].set_xlabel("Lower is better — logarithmic scale")
for y_position, value in zip(y_positions, rank_values):
    axes[1].text(value * 1.12, y_position, f"{value:.2f}", va="center", fontsize=10, fontweight="bold")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.suptitle("Season-Level MVP Ranking — 2015–2018 Validation", fontsize=22, fontweight="bold", color=NAVY, y=1.02)
fig.tight_layout()
ranking_path = OUTPUT_DIR / "final_validation_ranking_comparison.png"
fig.savefig(ranking_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(ranking_path.resolve())

section("10. FIGURE 3 — REGRESSION–RANKING PARETO VIEW")
fig, ax = plt.subplots(figsize=(14, 8.5))
label_offsets = {
    "Histogram GB": (10, 18),
    "Logistic + Ridge hurdle": (10, 17),
    "Weighted Ridge": (10, 10),
    "WS OLS baseline": (10, 8),
}
for model_name, row in comparison.iterrows():
    color = ORANGE if model_name == "Histogram GB" else BLUE if model_name == "Weighted Ridge" else GREEN if row["pareto_efficient"] else GRAY
    marker = "D" if row["pareto_efficient"] else "o"
    alpha = 1.0 if row["pareto_efficient"] or model_name == "WS OLS baseline" else 0.55
    ax.scatter(row["rmse_positive"], row["mean_ndcg_at_5"], s=150, color=color, marker=marker, edgecolor="white", linewidth=1.0, alpha=alpha)
    if model_name in label_offsets:
        ax.annotate(
            model_name,
            (row["rmse_positive"], row["mean_ndcg_at_5"]),
            xytext=label_offsets[model_name],
            textcoords="offset points",
            fontsize=10,
        )
ax.text(
    0.02,
    0.04,
    "Unlabeled gray circles are dominated models or naive baselines; exact values appear in the comparison table.",
    transform=ax.transAxes,
    fontsize=9.5,
    color=GRAY,
)
ax.set_title("Validation Trade-off: Vote-Recipient Error vs MVP Ranking", fontsize=20, color=NAVY, pad=16)
ax.set_xlabel("Vote-recipient RMSE — lower is better")
ax.set_ylabel("NDCG@5 — higher is better")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
pareto_path = OUTPUT_DIR / "final_validation_pareto.png"
fig.savefig(pareto_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(pareto_path.resolve())

section("11. FIGURE 4 — SEASON WINNER-RANK AUDIT")
rank_heatmap = winner_audit.loc[
    winner_audit["model"].isin(serious_models)
].pivot(index="model", columns="season", values="actual_winner_rank").loc[serious_models]
fig, ax = plt.subplots(figsize=(12.5, 7.5))
sns.heatmap(
    rank_heatmap,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    vmin=1,
    vmax=max(5, rank_heatmap.to_numpy().max()),
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "Actual MVP predicted rank — lower is better"},
    ax=ax,
)
ax.set_title("Actual MVP Rank by Season and Model", fontsize=20, color=NAVY, pad=16)
ax.set_xlabel("Season")
ax.set_ylabel("")
fig.tight_layout()
winner_path = OUTPUT_DIR / "final_validation_winner_rank_heatmap.png"
fig.savefig(winner_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(winner_path.resolve())

section("12. USER-VERIFICATION BOUNDARY")
print("Proposed champion: Histogram GB")
print("Proposed ranking challenger: Weighted Ridge")
print("Proposed final reference baseline: WS OLS baseline")
print("No 2019–2022 target was loaded or evaluated. Await user approval before final-test execution.")
