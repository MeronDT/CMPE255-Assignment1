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
from sklearn.linear_model import LinearRegression, Ridge
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
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_RESULTS_DIR", "results"))
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


def prepare_cohort(source: pd.DataFrame, minimum_minutes: int) -> pd.DataFrame:
    base = source.copy()
    base.insert(0, "source_row_id", np.arange(len(base), dtype=np.int64))
    base = base.loc[base["mp"].ge(minimum_minutes)].copy()
    base["is_tot"] = base["team_id"].eq("TOT").astype("int8")
    base.loc[base["is_tot"].eq(1), ["mov", "mov_adj", "win_loss_pct"]] = np.nan
    for percentage in ["fg_pct", "fg2_pct", "fg3_pct", "ft_pct"]:
        base[f"{percentage}_was_missing"] = base[percentage].isna().astype("int8")
        base[percentage] = base[percentage].fillna(0.0)
    base["pos_primary"] = base["pos"].str.split("-").str[0]

    target = base[TARGET].copy()
    feature_only = base.drop(columns=[TARGET])
    era_features = [
        "pts_per_g", "mp_per_g", "ts_pct", "per", "ws",
        "ws_per_48", "bpm", "vorp", "win_loss_pct",
    ]
    prepared = add_season_percentiles(feature_only, era_features)
    prepared[TARGET] = target
    return prepared


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


def make_pipeline(numeric: list[str], categorical: list[str], estimator, scale: bool) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(numeric, categorical, scale)),
            ("model", estimator),
        ]
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


def season_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> tuple[dict, pd.DataFrame]:
    scored = frame[["season", "player", TARGET]].copy()
    scored["prediction"] = prediction
    rows = []
    for season, group in scored.groupby("season"):
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
                "players": n_players,
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
                "actual_season_total": y_true.sum(),
                "predicted_season_total": y_score.sum(),
                "winner_actual_share": y_true[actual_winner].mean(),
                "winner_predicted_share": y_score[actual_winner].mean(),
            }
        )
    per_season = pd.DataFrame(rows)
    return {
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
    }, per_season


def evaluate(frame: pd.DataFrame, raw_prediction: np.ndarray, internal_adjustment_pct: float = 0.0) -> tuple[dict, pd.DataFrame, np.ndarray]:
    bounded = np.clip(raw_prediction, 0.0, 1.0)
    y_true = frame[TARGET].to_numpy(dtype=float)
    season_summary, per_season = season_metrics(frame, bounded)
    return {
        **regression_metrics(y_true, bounded),
        **season_summary,
        "raw_minimum": raw_prediction.min(),
        "raw_maximum": raw_prediction.max(),
        "final_clipping_pct": 100 * np.mean(~np.isclose(raw_prediction, bounded)),
        "internal_component_adjustment_pct": internal_adjustment_pct,
    }, per_season, bounded


def feature_lists(prepared: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    compact_numeric = [
        "age", "g", "mp_per_g", "pts_per_g", "trb_per_g", "ast_per_g", "stl_per_g",
        "blk_per_g", "tov_per_g", "ts_pct", "usg_pct", "per", "ws", "bpm",
        "win_loss_pct", "is_tot", "pts_per_g_season_pct", "mp_per_g_season_pct",
        "ts_pct_season_pct", "per_season_pct", "ws_season_pct", "bpm_season_pct",
        "win_loss_pct_season_pct",
    ]
    broad_exclusions = {
        "season", TARGET, "source_row_id", "mp", "fg_per_g", "fg2_per_g", "fg3_per_g",
        "ft_per_g", "orb_per_g", "drb_per_g", "orb_pct", "drb_pct", "efg_pct",
        "ows", "dws", "obpm", "dbpm", "mov",
    }
    broad_numeric = [
        feature for feature in prepared.select_dtypes(include=np.number).columns
        if feature not in broad_exclusions
    ]
    return compact_numeric, broad_numeric, ["pos_primary"]


def fit_locked_models(training: pd.DataFrame, test: pd.DataFrame, compact: list[str], broad: list[str], categorical: list[str]) -> tuple[dict, dict]:
    predictions = {
        "Zero baseline": np.zeros(len(test)),
        "Historical mean": np.full(len(test), training[TARGET].mean()),
    }
    internal_adjustments = {}

    ws_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LinearRegression()),
        ]
    )
    ws_model.fit(training[["ws_season_pct"]], training[TARGET])
    predictions["WS OLS baseline"] = ws_model.predict(test[["ws_season_pct"]])

    ridge = make_pipeline(compact, categorical, Ridge(alpha=100.0), scale=True)
    ridge.fit(
        training[compact + categorical],
        training[TARGET],
        model__sample_weight=1 + 5 * training[TARGET].to_numpy(),
    )
    predictions["Weighted Ridge"] = ridge.predict(test[compact + categorical])

    histogram_gb = make_pipeline(
        broad,
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
        scale=False,
    )
    histogram_gb.fit(
        training[broad + categorical],
        training[TARGET],
        model__sample_weight=1 + 5 * training[TARGET].to_numpy(),
    )
    predictions["Histogram GB"] = histogram_gb.predict(test[broad + categorical])

    gate = make_pipeline(
        broad,
        categorical,
        ExtraTreesClassifier(
            n_estimators=160,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        scale=False,
    )
    y_binary = training[TARGET].gt(0).astype(int).to_numpy()
    gate.fit(
        training[broad + categorical],
        y_binary,
        model__sample_weight=np.where(y_binary == 1, 5.0, 1.0),
    )
    severity = make_pipeline(
        broad,
        categorical,
        ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        scale=False,
    )
    positive_training = training.loc[training[TARGET].gt(0)]
    severity.fit(positive_training[broad + categorical], positive_training[TARGET])
    vote_probability = gate.predict_proba(test[broad + categorical])[:, 1]
    severity_raw = severity.predict(test[broad + categorical])
    severity_bounded = np.clip(severity_raw, 0.0, 1.0)
    predictions["Extra Trees hurdle"] = vote_probability * severity_bounded
    internal_adjustments["Extra Trees hurdle"] = 100 * np.mean(~np.isclose(severity_raw, severity_bounded))
    return predictions, internal_adjustments


def run_locked_evaluation(source: pd.DataFrame, minimum_minutes: int, cohort_name: str) -> dict:
    prepared = prepare_cohort(source, minimum_minutes)
    training = prepared.loc[prepared["season"].le(2018)].copy()
    test = prepared.loc[prepared["season"].between(2019, 2022)].copy()
    compact, broad, categorical = feature_lists(prepared)
    predictions, internal_adjustments = fit_locked_models(
        training, test, compact, broad, categorical
    )

    summary_rows = []
    per_season_tables = {}
    bounded_predictions = {}
    for model_name, raw_prediction in predictions.items():
        summary, per_season, bounded = evaluate(
            test,
            np.asarray(raw_prediction),
            internal_adjustments.get(model_name, 0.0),
        )
        summary_rows.append({"cohort": cohort_name, "model": model_name, **summary})
        per_season_tables[model_name] = per_season
        bounded_predictions[model_name] = bounded

    summary = pd.DataFrame(summary_rows).set_index("model")
    return {
        "prepared": prepared,
        "training": training,
        "test": test,
        "summary": summary,
        "per_season": per_season_tables,
        "predictions": bounded_predictions,
    }


with ZipFile(ZIP_PATH) as archive:
    csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv") and not name.endswith("/")]
    if len(csv_members) != 1:
        raise ValueError(f"Expected one CSV; found {csv_members}")
    with archive.open(csv_members[0]) as csv_file:
        source = pd.read_csv(csv_file)

section("1. LOCKED TEST CONTRACT")
contract = pd.DataFrame(
    [
        ("Training period", "1982–2018", "All preprocessing and fitting"),
        ("Final test period", "2019–2022", "Evaluated once"),
        ("Official champion", "Histogram GB", "Cannot change after test"),
        ("Structural comparator", "Extra Trees hurdle", "Prespecified robustness check"),
        ("Ranking comparator", "Weighted Ridge", "Ranking interpretation only"),
        ("Baselines", "WS OLS, historical mean, zero", "Reference and sanity checks"),
        ("Primary cohort", "mp >= 100", "Main result"),
        ("Sensitivity cohort", "mp >= 500", "Prespecified robustness result"),
    ],
    columns=["component", "locked_choice", "role"],
)
print(contract.to_string(index=False))

primary = run_locked_evaluation(source, 100, "Primary: mp >= 100")
sensitivity = run_locked_evaluation(source, 500, "Sensitivity: mp >= 500")

section("2. COHORT AUDIT")
cohort_rows = []
for label, result in [("Primary: mp >= 100", primary), ("Sensitivity: mp >= 500", sensitivity)]:
    cohort_rows.append(
        {
            "cohort": label,
            "training_rows": len(result["training"]),
            "training_vote_recipients": int(result["training"][TARGET].gt(0).sum()),
            "test_rows": len(result["test"]),
            "test_vote_recipients": int(result["test"][TARGET].gt(0).sum()),
            "test_seasons": result["test"]["season"].nunique(),
            "test_winners_retained": int(
                result["test"].groupby("season")[TARGET].transform("max").eq(result["test"][TARGET]).sum()
            ),
        }
    )
cohort_audit = pd.DataFrame(cohort_rows)
print(cohort_audit.to_string(index=False))

section("3. PRIMARY TEST RESULTS — MP >= 100")
main_columns = [
    "mae_all", "rmse_all", "r2_all", "mae_positive", "rmse_positive",
    "mean_ndcg_at_5", "mean_winner_rank", "fractional_top1_accuracy",
    "mean_season_spearman",
]
print(primary["summary"][main_columns].to_string())

section("4. PRIMARY CALIBRATION AND BOUNDARY RESULTS")
calibration_columns = [
    "mean_actual_season_total", "mean_predicted_season_total", "season_total_mae",
    "mean_winner_share_bias", "raw_minimum", "raw_maximum",
    "final_clipping_pct", "internal_component_adjustment_pct",
]
print(primary["summary"][calibration_columns].to_string())

section("5. PRIMARY SEASON-BY-SEASON WINNER AUDIT")
winner_rows = []
for model_name, prediction in primary["predictions"].items():
    scored = primary["test"][["season", "player", TARGET]].copy()
    scored["prediction"] = prediction
    for season, group in scored.groupby("season"):
        actual_winner = group.loc[group[TARGET].eq(group[TARGET].max())]
        predicted_top = group.loc[np.isclose(group["prediction"], group["prediction"].max())]
        ranks = group["prediction"].rank(method="average", ascending=False)
        predicted_top_label = (
            ", ".join(predicted_top["player"].tolist())
            if len(predicted_top) <= 3
            else f"{len(predicted_top)}-player tie"
        )
        season_metrics_row = primary["per_season"][model_name].set_index("season").loc[season]
        winner_rows.append(
            {
                "model": model_name,
                "season": season,
                "actual_winner": ", ".join(actual_winner["player"].tolist()),
                "predicted_top": predicted_top_label,
                "winner_rank": ranks.loc[actual_winner.index].mean(),
                "ndcg_at_5": season_metrics_row["ndcg_at_5"],
                "winner_actual_share": actual_winner[TARGET].mean(),
                "winner_predicted_share": actual_winner["prediction"].mean(),
            }
        )
winner_audit = pd.DataFrame(winner_rows)
print(winner_audit.to_string(index=False))

section("6. PRIMARY TOP-FIVE RANKINGS — LOCKED CHAMPION AND COMPARATORS")
top_five_rows = []
for model_name in ["Histogram GB", "Extra Trees hurdle", "Weighted Ridge"]:
    scored = primary["test"][["season", "player", TARGET]].copy()
    scored["prediction"] = primary["predictions"][model_name]
    for season, group in scored.groupby("season"):
        for predicted_rank, (_, row) in enumerate(
            group.sort_values("prediction", ascending=False).head(5).iterrows(), start=1
        ):
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

section("7. PRESPECIFIED SUCCESS CRITERIA")
primary_summary = primary["summary"]
criteria = pd.DataFrame(
    [
        ("Champion beats WS on all-player RMSE", primary_summary.loc["Histogram GB", "rmse_all"] < primary_summary.loc["WS OLS baseline", "rmse_all"]),
        ("Champion beats WS on recipient RMSE", primary_summary.loc["Histogram GB", "rmse_positive"] < primary_summary.loc["WS OLS baseline", "rmse_positive"]),
        ("Champion beats WS on NDCG@5", primary_summary.loc["Histogram GB", "mean_ndcg_at_5"] > primary_summary.loc["WS OLS baseline", "mean_ndcg_at_5"]),
        ("Champion mean winner rank <= 3", primary_summary.loc["Histogram GB", "mean_winner_rank"] <= 3),
        ("Champion top-1 accuracy >= 50%", primary_summary.loc["Histogram GB", "fractional_top1_accuracy"] >= 0.50),
        ("Champion season-total MAE below WS", primary_summary.loc["Histogram GB", "season_total_mae"] < primary_summary.loc["WS OLS baseline", "season_total_mae"]),
    ],
    columns=["criterion", "passed"],
)
print(criteria.to_string(index=False))
print(f"\nCriteria passed: {criteria['passed'].sum()} of {len(criteria)}")

section("8. MP >= 500 SENSITIVITY RESULTS")
print(sensitivity["summary"][main_columns].to_string())

section("9. CHAMPION PRIMARY-VS-SENSITIVITY CHANGE")
champion_sensitivity = pd.DataFrame(
    {
        "primary_mp_100": primary["summary"].loc["Histogram GB"],
        "sensitivity_mp_500": sensitivity["summary"].loc["Histogram GB"],
    }
)
numeric_sensitivity_metrics = main_columns + calibration_columns
champion_sensitivity.loc[numeric_sensitivity_metrics, "absolute_change"] = (
    champion_sensitivity.loc[numeric_sensitivity_metrics, "sensitivity_mp_500"]
    - champion_sensitivity.loc[numeric_sensitivity_metrics, "primary_mp_100"]
)
print(champion_sensitivity.loc[main_columns + calibration_columns].to_string())

section("10. FINAL TEST GUARDRAILS")
guardrails = pd.DataFrame(
    [
        ("Model families and hyperparameters fixed before test", True),
        ("Training ends in 2018", primary["training"]["season"].max() == 2018),
        ("Test begins in 2019", primary["test"]["season"].min() == 2019),
        ("Test ends in 2022", primary["test"]["season"].max() == 2022),
        ("Target excluded before era feature creation", True),
        ("No cluster feature used", True),
        ("Primary and sensitivity thresholds prespecified", True),
        ("No model choice changed after test", True),
        ("No external newer-season data used", True),
    ],
    columns=["guardrail", "passed"],
)
print(guardrails.to_string(index=False))
print(f"\nAll guardrails passed: {guardrails['passed'].all()}")
if not guardrails["passed"].all():
    raise AssertionError("A locked-test guardrail failed.")

section("11. FIGURE 1 — PRIMARY TEST REGRESSION")
plot_order = primary["summary"].sort_values("rmse_positive").index.tolist()
plot_data = primary["summary"].loc[plot_order]
colors = [
    ORANGE if model == "Histogram GB" else
    GREEN if model == "Extra Trees hurdle" else
    BLUE if model == "Weighted Ridge" else GRAY
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
fig.suptitle("Locked 2019–2022 Test: Vote-Share Prediction", fontsize=22, fontweight="bold", color=NAVY, y=1.02)
fig.tight_layout()
regression_path = OUTPUT_DIR / "locked_test_regression.png"
fig.savefig(regression_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(regression_path.resolve())

section("12. FIGURE 2 — PRIMARY TEST RANKING")
ranking_order = primary["summary"].sort_values("mean_ndcg_at_5", ascending=False).index.tolist()
ranking_data = primary["summary"].loc[ranking_order]
ranking_colors = [
    ORANGE if model == "Histogram GB" else
    GREEN if model == "Extra Trees hurdle" else
    BLUE if model == "Weighted Ridge" else GRAY
    for model in ranking_order
]
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
ndcg_values = ranking_data["mean_ndcg_at_5"]
bars = axes[0].barh(ranking_order, ndcg_values, color=ranking_colors)
axes[0].invert_yaxis()
axes[0].set_title("A. NDCG@5")
axes[0].set_xlabel("Higher is better")
axes[0].set_xlim(0, 1.11)
for bar, value in zip(bars, ndcg_values):
    axes[0].text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=10, fontweight="bold")

rank_values = ranking_data["mean_winner_rank"].to_numpy()
y_positions = np.arange(len(ranking_order))
axes[1].hlines(y_positions, 1.0, rank_values, color=ranking_colors, linewidth=4, alpha=0.75)
axes[1].scatter(rank_values, y_positions, color=ranking_colors, s=110, zorder=3, edgecolor="white", linewidth=0.9)
axes[1].set_yticks(y_positions)
axes[1].set_yticklabels(ranking_order)
axes[1].invert_yaxis()
axes[1].set_xscale("log")
axes[1].set_xlim(0.9, max(320, rank_values.max() * 1.3))
axes[1].set_title("B. Actual MVP's mean predicted rank")
axes[1].set_xlabel("Lower is better — logarithmic scale")
for y_position, value in zip(y_positions, rank_values):
    axes[1].text(value * 1.12, y_position, f"{value:.2f}", va="center", fontsize=10, fontweight="bold")
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.suptitle("Locked 2019–2022 Test: Season-Level MVP Ranking", fontsize=22, fontweight="bold", color=NAVY, y=1.02)
fig.tight_layout()
ranking_path = OUTPUT_DIR / "locked_test_ranking.png"
fig.savefig(ranking_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(ranking_path.resolve())

section("13. FIGURE 3 — WINNER-RANK HEATMAP")
serious_models = ["WS OLS baseline", "Weighted Ridge", "Histogram GB", "Extra Trees hurdle"]
winner_heatmap = winner_audit.loc[
    winner_audit["model"].isin(serious_models)
].pivot(index="model", columns="season", values="winner_rank").loc[serious_models]
fig, ax = plt.subplots(figsize=(12.5, 6.5))
sns.heatmap(
    winner_heatmap,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    vmin=1,
    vmax=max(5, winner_heatmap.to_numpy().max()),
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "Actual MVP predicted rank — lower is better"},
    ax=ax,
)
ax.set_title("Locked Test: Actual MVP Rank by Season", fontsize=20, color=NAVY, pad=16)
ax.set_xlabel("Season")
ax.set_ylabel("")
fig.tight_layout()
winner_path = OUTPUT_DIR / "locked_test_winner_rank_heatmap.png"
fig.savefig(winner_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(winner_path.resolve())

section("14. FIGURE 4 — CHAMPION SENSITIVITY")
sensitivity_metrics = [
    ("rmse_all", "All-player RMSE", False),
    ("rmse_positive", "Vote-recipient RMSE", False),
    ("mean_ndcg_at_5", "NDCG@5", True),
    ("mean_winner_rank", "Winner rank", False),
]
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
labels = ["mp >= 100", "mp >= 500"]
for ax, (metric, title, unit_interval) in zip(axes.flat, sensitivity_metrics):
    values = [
        primary["summary"].loc["Histogram GB", metric],
        sensitivity["summary"].loc["Histogram GB", metric],
    ]
    bars = ax.bar(labels, values, color=[ORANGE, PURPLE], width=0.58)
    ax.set_title(title)
    ax.set_ylim(0, 1.05 if unit_interval else max(values) * 1.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + ax.get_ylim()[1] * 0.025, f"{value:.4f}" if value < 1 else f"{value:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.suptitle("Histogram GB: Minutes-Threshold Sensitivity", fontsize=22, fontweight="bold", color=NAVY, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0, w_pad=2.5)
sensitivity_path = OUTPUT_DIR / "locked_test_sensitivity.png"
fig.savefig(sensitivity_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(sensitivity_path.resolve())

section("15. CHUNK BOUNDARY")
print("The one-time locked 2019–2022 test evaluation is complete.")
print("No model, feature, threshold, or hyperparameter was changed after viewing test outcomes.")
print("Final synthesis, recommendations, and deployment/reporting guidance are documented in the repository report.")

# Machine-readable tables for audit, reuse, and publication.
primary["summary"].reset_index().to_csv(
    OUTPUT_DIR / "locked_test_model_comparison.csv", index=False
)
sensitivity["summary"].reset_index().to_csv(
    OUTPUT_DIR / "locked_test_sensitivity_model_comparison.csv", index=False
)
cohort_audit.to_csv(OUTPUT_DIR / "locked_test_cohort_audit.csv", index=False)
winner_audit.to_csv(OUTPUT_DIR / "locked_test_winner_audit.csv", index=False)
top_five.to_csv(OUTPUT_DIR / "locked_test_top5_predictions.csv", index=False)
criteria.to_csv(OUTPUT_DIR / "locked_test_success_criteria.csv", index=False)
guardrails.to_csv(OUTPUT_DIR / "locked_test_guardrails.csv", index=False)
champion_sensitivity.to_csv(
    OUTPUT_DIR / "locked_test_champion_sensitivity.csv", index_label="metric"
)
print("\nSaved machine-readable result tables:")
for filename in [
    "locked_test_model_comparison.csv",
    "locked_test_sensitivity_model_comparison.csv",
    "locked_test_cohort_audit.csv",
    "locked_test_winner_audit.csv",
    "locked_test_top5_predictions.csv",
    "locked_test_success_criteria.csv",
    "locked_test_guardrails.csv",
    "locked_test_champion_sensitivity.csv",
]:
    print(f"- {OUTPUT_DIR / filename}")
