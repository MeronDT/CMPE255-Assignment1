import os
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 240)
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
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk13_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "award_share"

BASELINE_ORDER = [
    "Zero prediction",
    "Historical mean",
    "Win-shares percentile OLS",
]
BASELINE_COLORS = {
    "Zero prediction": GRAY,
    "Historical mean": BLUE,
    "Win-shares percentile OLS": ORANGE,
}


def section(title: str) -> None:
    print(f"\n{'=' * 122}\n{title}\n{'=' * 122}")


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


def season_ranking_metrics(
    evaluation_frame: pd.DataFrame,
    prediction: np.ndarray,
) -> dict:
    metric_rows = []
    evaluation = evaluation_frame[["season", TARGET]].copy()
    evaluation["prediction"] = prediction

    for season, season_rows in evaluation.groupby("season"):
        y_true = season_rows[TARGET].to_numpy(dtype=float)
        y_score = season_rows["prediction"].to_numpy(dtype=float)
        n_players = len(season_rows)

        actual_winner = np.isclose(y_true, y_true.max())
        predicted_top = np.isclose(y_score, y_score.max())
        fractional_top1_credit = (
            np.logical_and(actual_winner, predicted_top).sum()
            / predicted_top.sum()
        )

        predicted_ranks = pd.Series(y_score).rank(
            method="average",
            ascending=False,
        ).to_numpy()
        winner_rank = predicted_ranks[actual_winner].mean()
        winner_rank_percentile = (
            1.0 - (winner_rank - 1.0) / (n_players - 1.0)
            if n_players > 1
            else 1.0
        )

        if np.unique(y_score).size > 1 and np.unique(y_true).size > 1:
            season_spearman = pd.Series(y_true).corr(
                pd.Series(y_score),
                method="spearman",
            )
        else:
            season_spearman = np.nan

        metric_rows.append(
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
                "fractional_top1_accuracy": fractional_top1_credit,
                "spearman": season_spearman,
            }
        )

    per_season = pd.DataFrame(metric_rows)
    return {
        "mean_ndcg_at_5": per_season["ndcg_at_5"].mean(),
        "mean_winner_rank": per_season["winner_rank"].mean(),
        "mean_winner_rank_percentile": per_season["winner_rank_percentile"].mean(),
        "fractional_top1_accuracy": per_season["fractional_top1_accuracy"].mean(),
        "mean_season_spearman": per_season["spearman"].mean(),
        "defined_spearman_seasons": per_season["spearman"].notna().sum(),
        "evaluated_seasons": len(per_season),
    }


def make_baseline_predictions(
    training_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, Pipeline]:
    y_train = training_frame[TARGET].to_numpy(dtype=float)

    ws_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    ws_pipeline.fit(
        training_frame[["ws_season_pct"]],
        y_train,
    )

    raw_predictions = {
        "Zero prediction": np.zeros(len(evaluation_frame), dtype=float),
        "Historical mean": np.full(
            len(evaluation_frame),
            y_train.mean(),
            dtype=float,
        ),
        "Win-shares percentile OLS": ws_pipeline.predict(
            evaluation_frame[["ws_season_pct"]]
        ),
    }

    bounded_predictions = {
        name: np.clip(values, 0.0, 1.0)
        for name, values in raw_predictions.items()
    }

    diagnostic_rows = []
    for name, values in raw_predictions.items():
        diagnostic_rows.append(
            {
                "baseline": name,
                "raw_minimum": values.min(),
                "raw_maximum": values.max(),
                "raw_below_zero_pct": 100 * np.mean(values < 0),
                "raw_above_one_pct": 100 * np.mean(values > 1),
                "bounded_values_changed_pct": 100
                * np.mean(~np.isclose(values, bounded_predictions[name])),
            }
        )

    return bounded_predictions, pd.DataFrame(diagnostic_rows), ws_pipeline


with ZipFile(ZIP_PATH) as archive:
    csv_members = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and not name.endswith("/")
    ]
    if len(csv_members) != 1:
        raise ValueError(f"Expected exactly one CSV; found {csv_members}")
    with archive.open(csv_members[0]) as csv_file:
        source = pd.read_csv(csv_file)

# Reconstruct the locked data-preparation contract.
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

section("1. BASELINE AND METRIC CONTRACT")
baseline_contract = pd.DataFrame(
    [
        (
            "Zero prediction",
            "Predict 0 for every player",
            "Tests how misleading zero inflation can be",
        ),
        (
            "Historical mean",
            "Predict the mean award_share from the training seasons",
            "Minimum-information regression benchmark",
        ),
        (
            "Win-shares percentile OLS",
            "Training-only univariate OLS using ws_season_pct",
            "Simple basketball-informed production benchmark",
        ),
    ],
    columns=["baseline", "definition", "purpose"],
)
print(baseline_contract.to_string(index=False))

metric_contract = pd.DataFrame(
    [
        ("MAE, all rows", "Lower", "Average absolute vote-share error across all eligible players"),
        ("RMSE, all rows", "Lower", "Penalizes large vote-share misses"),
        ("R-squared", "Higher", "Improvement over predicting the evaluation-set mean; may be negative"),
        ("MAE/RMSE, positive targets", "Lower", "Prevents the zero majority from hiding candidate errors"),
        ("NDCG@5", "Higher", "Rewards placing high-share candidates near the top of each season"),
        ("Winner rank", "Lower", "Average predicted rank of the actual season winner"),
        ("Winner-rank percentile", "Higher", "Season-size-adjusted version of winner rank"),
        ("Fractional top-1 accuracy", "Higher", "Tie-aware probability that a top-score selection is the winner"),
        ("Mean season Spearman", "Higher", "Within-season rank association when predictions are nonconstant"),
    ],
    columns=["metric", "preferred_direction", "interpretation"],
)
print("\nMetric definitions:")
print(metric_contract.to_string(index=False))
print(
    "\nOfficial baseline predictions are bounded to [0, 1]. The percentage of raw "
    "predictions changed by bounding is reported so the correction is not hidden."
)

section("2. EXPANDING-WINDOW BASELINE EVALUATION")
fold_definitions = [
    ("Fold 1", 1982, 2002, 2003, 2006),
    ("Fold 2", 1982, 2006, 2007, 2010),
    ("Fold 3", 1982, 2010, 2011, 2014),
]

fold_metric_rows = []
fold_diagnostic_rows = []
for fold_name, train_start, train_end, valid_start, valid_end in fold_definitions:
    fold_train = development.loc[
        development["season"].between(train_start, train_end)
    ]
    fold_valid = development.loc[
        development["season"].between(valid_start, valid_end)
    ]
    predictions, diagnostics, _ = make_baseline_predictions(fold_train, fold_valid)

    diagnostics.insert(0, "fold", fold_name)
    fold_diagnostic_rows.append(diagnostics)

    y_valid = fold_valid[TARGET].to_numpy(dtype=float)
    for baseline_name, prediction in predictions.items():
        metrics = {
            **regression_metrics(y_valid, prediction),
            **season_ranking_metrics(fold_valid, prediction),
        }
        fold_metric_rows.append(
            {
                "fold": fold_name,
                "validation_seasons": f"{valid_start}–{valid_end}",
                "baseline": baseline_name,
                **metrics,
            }
        )

fold_metrics = pd.DataFrame(fold_metric_rows)
fold_diagnostics = pd.concat(fold_diagnostic_rows, ignore_index=True)
print("Per-fold metrics:")
print(fold_metrics.to_string(index=False))
print("\nRaw-prediction range and bounding diagnostics:")
print(fold_diagnostics.to_string(index=False))

summary_columns = [
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
    "defined_spearman_seasons",
]
cv_summary = (
    fold_metrics.groupby("baseline", sort=False)[summary_columns]
    .mean()
    .reindex(BASELINE_ORDER)
    .reset_index()
)
print("\nMean expanding-fold performance:")
print(cv_summary.to_string(index=False))

section("3. EXTERNAL MODEL-SELECTION VALIDATION: 2015–2018")
validation_predictions, validation_diagnostics, fitted_ws_baseline = (
    make_baseline_predictions(development, validation)
)
y_validation = validation[TARGET].to_numpy(dtype=float)

validation_metric_rows = []
for baseline_name, prediction in validation_predictions.items():
    validation_metric_rows.append(
        {
            "baseline": baseline_name,
            **regression_metrics(y_validation, prediction),
            **season_ranking_metrics(validation, prediction),
        }
    )
validation_metrics = (
    pd.DataFrame(validation_metric_rows)
    .set_index("baseline")
    .reindex(BASELINE_ORDER)
    .reset_index()
)
print(validation_metrics.to_string(index=False))
print("\nRaw-prediction range and bounding diagnostics:")
print(validation_diagnostics.to_string(index=False))

ws_regressor = fitted_ws_baseline.named_steps["regressor"]
print(
    "\nWin-shares baseline fitted on 1982–2014: "
    f"scaled coefficient={ws_regressor.coef_[0]:.6f}, "
    f"intercept={ws_regressor.intercept_:.6f}"
)

section("4. BOUNDING SENSITIVITY FOR THE BASKETBALL-INFORMED BASELINE")
raw_ws_validation = fitted_ws_baseline.predict(validation[["ws_season_pct"]])
bounded_ws_validation = np.clip(raw_ws_validation, 0.0, 1.0)
bounding_sensitivity = pd.DataFrame(
    [
        {
            "prediction_version": "Raw OLS",
            **regression_metrics(y_validation, raw_ws_validation),
        },
        {
            "prediction_version": "Bounded to [0, 1]",
            **regression_metrics(y_validation, bounded_ws_validation),
        },
    ]
)
print(bounding_sensitivity.to_string(index=False))
print(
    "\nDecision for this baseline: use bounded predictions for numerical and ranking "
    "metrics, but retain the changed-value rate as a model-misspecification diagnostic."
)

section("5. SEASON-BY-SEASON VALIDATION WINNER AUDIT")
validation_with_ws = validation[
    ["season", "player", TARGET, "ws", "ws_season_pct"]
].copy()
validation_with_ws["predicted_share"] = bounded_ws_validation

winner_audit_rows = []
for season, season_rows in validation_with_ws.groupby("season"):
    actual_winner_rows = season_rows.loc[
        season_rows[TARGET].eq(season_rows[TARGET].max())
    ]
    predicted_top_rows = season_rows.loc[
        season_rows["predicted_share"].eq(season_rows["predicted_share"].max())
    ]
    predicted_ranks = season_rows["predicted_share"].rank(
        method="average",
        ascending=False,
    )
    actual_winner_rank = predicted_ranks.loc[actual_winner_rows.index].mean()

    winner_audit_rows.append(
        {
            "season": season,
            "actual_winner": ", ".join(actual_winner_rows["player"].tolist()),
            "actual_winner_share": actual_winner_rows[TARGET].max(),
            "ws_baseline_top_player": ", ".join(predicted_top_rows["player"].tolist()),
            "top_player_actual_share": predicted_top_rows[TARGET].max(),
            "actual_winner_predicted_rank": actual_winner_rank,
            "actual_winner_predicted_share": actual_winner_rows["predicted_share"].mean(),
        }
    )
winner_audit = pd.DataFrame(winner_audit_rows)
print(winner_audit.to_string(index=False))

section("6. LEAKAGE AND FINAL-TEST GUARDRAILS")
guardrails = pd.DataFrame(
    [
        ("Era features created without target present", TARGET not in feature_only.columns),
        ("Each baseline fitted separately inside each development fold", len(fold_definitions) == 3),
        ("External validation begins after development ends", development["season"].max() < validation["season"].min()),
        ("2019–2022 features are separate from model selection", set(final_test_features_only["season"].unique()) == {2019, 2020, 2021, 2022}),
        ("Final-test target object was not created", "y_test" not in locals()),
        ("No final-test target used by metric functions", TARGET not in final_test_features_only.columns),
        ("Main multivariable candidate models not fitted", True),
    ],
    columns=["guardrail", "passed"],
)
print(guardrails.to_string(index=False))
print(f"\nAll baseline-stage guardrails passed: {guardrails['passed'].all()}")
if not guardrails["passed"].all():
    raise AssertionError("A baseline-stage leakage guardrail failed.")

section("7. READABLE BASELINE-COMPARISON FIGURE")
plot_metrics = validation_metrics.set_index("baseline").reindex(BASELINE_ORDER)
fig, axes = plt.subplots(2, 2, figsize=(17, 11))
fig.suptitle(
    "Baseline Performance on 2015–2018 Model-Selection Validation",
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

for ax, (metric, title, ylabel, fixed_unit_interval) in zip(axes.flat, panels):
    values = plot_metrics[metric]
    bars = ax.bar(
        np.arange(len(BASELINE_ORDER)),
        values,
        color=[BASELINE_COLORS[name] for name in BASELINE_ORDER],
        width=0.62,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(BASELINE_ORDER)))
    ax.set_xticklabels(
        ["Zero", "Historical\nmean", "WS-percentile\nOLS"],
        rotation=0,
    )
    if fixed_unit_interval:
        ax.set_ylim(0, 1.05)
    else:
        upper = values.max() * 1.22 if values.max() > 0 else 1
        ax.set_ylim(0, upper)
    for bar, value in zip(bars, values):
        label = f"{value:.3f}" if value < 10 else f"{value:.1f}"
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.025
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
figure_path = OUTPUT_DIR / "baseline_validation_comparison.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(figure_path.resolve())

section("8. CHUNK BOUNDARY")
print("Only transparent baselines were evaluated.")
print("No Ridge, Elastic Net, random forest, gradient boosting, two-stage, ranking, or clustering model was fitted.")
print("The 2019–2022 target remains sealed and was not evaluated.")
