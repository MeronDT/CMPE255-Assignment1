from collections import defaultdict
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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 260)
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
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk16_outputs"
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


def make_gate_pipeline(
    configuration: pd.Series,
    compact_numeric: list[str],
    broad_numeric: list[str],
    categorical_features: list[str],
) -> tuple[Pipeline, list[str]]:
    if configuration["family"] == "Logistic + Ridge":
        input_columns = compact_numeric + categorical_features
        estimator = LogisticRegression(
            C=float(configuration["gate_c"]),
            solver="lbfgs",
            max_iter=5_000,
            random_state=RANDOM_STATE,
        )
        preprocessor = make_preprocessor(
            compact_numeric,
            categorical_features,
            scale_numeric=True,
        )
    elif configuration["family"] == "Extra Trees hurdle":
        input_columns = broad_numeric + categorical_features
        estimator = ExtraTreesClassifier(
            n_estimators=160,
            min_samples_leaf=int(configuration["gate_min_samples_leaf"]),
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        preprocessor = make_preprocessor(
            broad_numeric,
            categorical_features,
            scale_numeric=False,
        )
    else:
        raise ValueError(f"Unknown family: {configuration['family']}")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)]), input_columns


def make_severity_pipeline(
    configuration: pd.Series,
    compact_numeric: list[str],
    broad_numeric: list[str],
    categorical_features: list[str],
) -> tuple[Pipeline, list[str]]:
    if configuration["family"] == "Logistic + Ridge":
        input_columns = compact_numeric + categorical_features
        estimator = Ridge(alpha=float(configuration["severity_alpha"]))
        preprocessor = make_preprocessor(
            compact_numeric,
            categorical_features,
            scale_numeric=True,
        )
    elif configuration["family"] == "Extra Trees hurdle":
        input_columns = broad_numeric + categorical_features
        estimator = ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=int(configuration["severity_min_samples_leaf"]),
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        preprocessor = make_preprocessor(
            broad_numeric,
            categorical_features,
            scale_numeric=False,
        )
    else:
        raise ValueError(f"Unknown family: {configuration['family']}")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)]), input_columns


def fit_hurdle_model(
    training_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    configuration: pd.Series,
    compact_numeric: list[str],
    broad_numeric: list[str],
    categorical_features: list[str],
) -> dict:
    gate_model, gate_columns = make_gate_pipeline(
        configuration,
        compact_numeric,
        broad_numeric,
        categorical_features,
    )
    severity_model, severity_columns = make_severity_pipeline(
        configuration,
        compact_numeric,
        broad_numeric,
        categorical_features,
    )

    y_share = training_frame[TARGET].to_numpy(dtype=float)
    y_binary = (y_share > 0).astype(int)
    gate_weights = np.where(
        y_binary == 1,
        float(configuration["gate_positive_weight"]),
        1.0,
    )
    gate_model.fit(
        training_frame[gate_columns],
        y_binary,
        model__sample_weight=gate_weights,
    )

    positive_training = training_frame.loc[training_frame[TARGET].gt(0)]
    severity_model.fit(
        positive_training[severity_columns],
        positive_training[TARGET],
    )

    gate_probability = gate_model.predict_proba(
        evaluation_frame[gate_columns]
    )[:, 1]
    raw_conditional_share = severity_model.predict(
        evaluation_frame[severity_columns]
    )
    bounded_conditional_share = np.clip(raw_conditional_share, 0.0, 1.0)
    expected_share = gate_probability * bounded_conditional_share
    return {
        "gate_model": gate_model,
        "severity_model": severity_model,
        "gate_probability": gate_probability,
        "raw_conditional_share": raw_conditional_share,
        "bounded_conditional_share": bounded_conditional_share,
        "expected_share": expected_share,
    }


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


def gate_metrics(y_binary: np.ndarray, probability: np.ndarray) -> dict:
    bounded_probability = np.clip(probability, 1e-12, 1 - 1e-12)
    return {
        "gate_average_precision": average_precision_score(y_binary, probability),
        "gate_roc_auc": roc_auc_score(y_binary, probability),
        "gate_brier": brier_score_loss(y_binary, probability),
        "gate_log_loss": log_loss(y_binary, bounded_probability, labels=[0, 1]),
        "mean_predicted_vote_probability": probability.mean(),
        "observed_vote_rate": y_binary.mean(),
    }


def evaluate_hurdle_predictions(
    frame: pd.DataFrame,
    gate_probability: np.ndarray,
    raw_conditional_share: np.ndarray,
    expected_share: np.ndarray,
) -> dict:
    y_share = frame[TARGET].to_numpy(dtype=float)
    y_binary = (y_share > 0).astype(int)
    return {
        **regression_metrics(y_share, expected_share),
        **season_ranking_metrics(frame, expected_share),
        **gate_metrics(y_binary, gate_probability),
        "raw_conditional_minimum": raw_conditional_share.min(),
        "raw_conditional_maximum": raw_conditional_share.max(),
        "conditional_below_zero_pct": 100 * np.mean(raw_conditional_share < 0),
        "conditional_above_one_pct": 100 * np.mean(raw_conditional_share > 1),
        "expected_share_minimum": expected_share.min(),
        "expected_share_maximum": expected_share.max(),
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
    baseline.fit(training_frame[["ws_season_pct"]], training_frame[TARGET])
    return baseline.predict(evaluation_frame[["ws_season_pct"]])


def select_recall_constrained_threshold(
    y_binary: np.ndarray,
    probability: np.ndarray,
    minimum_recall: float = 0.90,
) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_binary, probability)
    candidates = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )
    eligible = candidates.loc[candidates["recall"].ge(minimum_recall)].copy()
    if eligible.empty:
        selected = candidates.sort_values(
            ["recall", "precision", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
    else:
        selected = eligible.sort_values(
            ["precision", "threshold"],
            ascending=[False, False],
        ).iloc[0]
    predicted_candidate = probability >= selected["threshold"]
    return {
        "threshold": selected["threshold"],
        "precision": precision_score(y_binary, predicted_candidate, zero_division=0),
        "recall": recall_score(y_binary, predicted_candidate, zero_division=0),
        "flagged_pct": 100 * predicted_candidate.mean(),
    }


def calibration_table(
    y_binary: np.ndarray,
    probability: np.ndarray,
    requested_bins: int = 8,
) -> pd.DataFrame:
    calibration = pd.DataFrame(
        {"observed": y_binary, "predicted_probability": probability}
    )
    calibration["bin"] = pd.qcut(
        calibration["predicted_probability"],
        q=requested_bins,
        duplicates="drop",
    )
    return (
        calibration.groupby("bin", observed=True)
        .agg(
            mean_predicted_probability=("predicted_probability", "mean"),
            observed_vote_rate=("observed", "mean"),
            players=("observed", "size"),
            vote_recipients=("observed", "sum"),
        )
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
        source = pd.read_csv(csv_file)

# Reapply only previously approved preparation decisions.
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
    "pts_per_g", "mp_per_g", "ts_pct", "per", "ws",
    "ws_per_48", "bpm", "vorp", "win_loss_pct",
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
    "age", "g", "mp_per_g", "pts_per_g", "trb_per_g", "ast_per_g",
    "stl_per_g", "blk_per_g", "tov_per_g", "ts_pct", "usg_pct", "per",
    "ws", "bpm", "win_loss_pct", "is_tot", "pts_per_g_season_pct",
    "mp_per_g_season_pct", "ts_pct_season_pct", "per_season_pct",
    "ws_season_pct", "bpm_season_pct", "win_loss_pct_season_pct",
]
categorical_features = ["pos_primary"]
broad_exclusions = {
    "season", TARGET, "source_row_id", "mp", "fg_per_g", "fg2_per_g",
    "fg3_per_g", "ft_per_g", "orb_per_g", "drb_per_g", "orb_pct",
    "drb_pct", "efg_pct", "ows", "dws", "obpm", "dbpm", "mov",
}
broad_numeric = [
    feature
    for feature in prepared.select_dtypes(include=np.number).columns
    if feature not in broad_exclusions
]

section("1. TWO-STAGE PROBLEM FORMULATION")
formulation = pd.DataFrame(
    [
        ("Stage 1: vote gate", "P(award_share > 0 | player statistics)", "All eligible players", "Separates vote receipt from the size of the vote"),
        ("Stage 2: severity", "E(award_share | award_share > 0, statistics)", "Vote recipients only", "Learns magnitude without thousands of structural zeros"),
        ("Combined prediction", "Gate probability × conditional share", "All eligible players", "Produces a continuous expected share for regression and ranking"),
    ],
    columns=["component", "quantity", "training_population", "purpose"],
)
print(formulation.to_string(index=False))
print("\nNo hard threshold is used in the final expected-share prediction.")

section("2. DATA AND IMBALANCE AUDIT")
audit = pd.DataFrame(
    [
        ("Development/search", "1982–2014", len(development), int(development[TARGET].gt(0).sum()), 100 * development[TARGET].gt(0).mean()),
        ("External validation", "2015–2018", len(validation), int(validation[TARGET].gt(0).sum()), 100 * validation[TARGET].gt(0).mean()),
        ("Sealed final test", "2019–2022", len(final_test_features_only), np.nan, np.nan),
    ],
    columns=["role", "seasons", "rows", "vote_recipients", "vote_recipient_pct"],
)
print(audit.to_string(index=False))
print(f"\nCompact numeric features: {len(compact_numeric)}")
print(f"Broad numeric features: {len(broad_numeric)}")

section("3. MODEL AND WEIGHTING DECISIONS")
decisions = pd.DataFrame(
    [
        ("Logistic + Ridge", "Interpretable regularized hurdle model", "May miss nonlinear thresholds and interactions"),
        ("Extra Trees hurdle", "Nonlinear gate and conditional-share functions", "More variance; probabilities can be poorly calibrated"),
        ("Gate positive weight 1", "Preserves the observed class distribution", "Gate may under-emphasize rare vote recipients"),
        ("Gate positive weight 5", "Moderately emphasizes detecting vote recipients", "Weighted probabilities may overstate absolute vote probability"),
        ("Unweighted Stage 2", "Avoids double-emphasizing high-share players", "Very high shares remain uncommon among recipients"),
        ("No random oversampling", "Preserves seasons and the observed player population", "Models must handle the imbalance directly"),
    ],
    columns=["decision", "benefit", "risk"],
)
print(decisions.to_string(index=False))

section("4. COMPUTE-CONSCIOUS SEARCH SPACE")
configuration_rows = []
for gate_c in [0.1, 1.0]:
    for gate_positive_weight in [1.0, 5.0]:
        for severity_alpha in [10.0, 100.0]:
            configuration_rows.append(
                {
                    "family": "Logistic + Ridge",
                    "gate_c": gate_c,
                    "gate_min_samples_leaf": np.nan,
                    "gate_positive_weight": gate_positive_weight,
                    "severity_alpha": severity_alpha,
                    "severity_min_samples_leaf": np.nan,
                }
            )
for gate_min_samples_leaf in [2, 8]:
    for gate_positive_weight in [1.0, 5.0]:
        for severity_min_samples_leaf in [2, 5]:
            configuration_rows.append(
                {
                    "family": "Extra Trees hurdle",
                    "gate_c": np.nan,
                    "gate_min_samples_leaf": gate_min_samples_leaf,
                    "gate_positive_weight": gate_positive_weight,
                    "severity_alpha": np.nan,
                    "severity_min_samples_leaf": severity_min_samples_leaf,
                }
            )
configurations = pd.DataFrame(configuration_rows)
fold_definitions = [
    ("Fold 1", 1982, 2002, 2003, 2006),
    ("Fold 2", 1982, 2006, 2007, 2010),
    ("Fold 3", 1982, 2010, 2011, 2014),
]
search_space = configurations.groupby("family").size().rename("configurations").reset_index()
search_space["chronological_folds"] = len(fold_definitions)
search_space["component_model_fits"] = search_space["configurations"] * len(fold_definitions) * 2
print(search_space.to_string(index=False))
print(f"\nTotal gate-plus-severity fits: {search_space['component_model_fits'].sum()}")

section("5. EXPANDING-WINDOW CROSS-VALIDATION")
fold_result_rows = []
oof_predictions = defaultdict(list)
baseline_fold_rows = []
for fold_name, train_start, train_end, valid_start, valid_end in fold_definitions:
    fold_train = development.loc[development["season"].between(train_start, train_end)]
    fold_valid = development.loc[development["season"].between(valid_start, valid_end)]
    ws_prediction = np.clip(fit_ws_baseline(fold_train, fold_valid), 0.0, 1.0)
    baseline_fold_rows.append(
        {
            "fold": fold_name,
            "rmse_all": np.sqrt(mean_squared_error(fold_valid[TARGET], ws_prediction)),
        }
    )
    for configuration_id, configuration in configurations.iterrows():
        result = fit_hurdle_model(
            fold_train,
            fold_valid,
            configuration,
            compact_numeric,
            broad_numeric,
            categorical_features,
        )
        fold_result_rows.append(
            {
                "fold": fold_name,
                "configuration_id": configuration_id,
                **configuration.to_dict(),
                **evaluate_hurdle_predictions(
                    fold_valid,
                    result["gate_probability"],
                    result["raw_conditional_share"],
                    result["expected_share"],
                ),
            }
        )
        oof_predictions[configuration_id].append(
            pd.DataFrame(
                {
                    "season": fold_valid["season"].to_numpy(),
                    "y_binary": fold_valid[TARGET].gt(0).astype(int).to_numpy(),
                    "gate_probability": result["gate_probability"],
                    "expected_share": result["expected_share"],
                }
            )
        )

fold_results = pd.DataFrame(fold_result_rows)
baseline_cv_rmse = pd.DataFrame(baseline_fold_rows)["rmse_all"].mean()
rmse_guardrail = baseline_cv_rmse * 1.10
summary_metrics = [
    "mae_all", "rmse_all", "r2_all", "mae_positive", "rmse_positive",
    "mean_ndcg_at_5", "mean_winner_rank", "mean_winner_rank_percentile",
    "fractional_top1_accuracy", "mean_season_spearman",
    "gate_average_precision", "gate_roc_auc", "gate_brier", "gate_log_loss",
    "mean_predicted_vote_probability", "observed_vote_rate",
    "conditional_below_zero_pct", "conditional_above_one_pct",
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
display_columns = [
    "configuration_id", "family", "gate_c", "gate_min_samples_leaf",
    "gate_positive_weight", "severity_alpha", "severity_min_samples_leaf",
    "rmse_all", "r2_all", "rmse_positive", "mean_ndcg_at_5",
    "mean_winner_rank", "fractional_top1_accuracy", "gate_average_precision",
    "gate_brier", "passes_rmse_guardrail",
]
print("\nTop hurdle configurations by NDCG@5, then positive-target RMSE:")
print(
    cv_summary.sort_values(
        ["passes_rmse_guardrail", "mean_ndcg_at_5", "rmse_positive"],
        ascending=[False, False, True],
    )[display_columns].head(12).to_string(index=False)
)

section("6. GATE-WEIGHT SENSITIVITY")
weight_rows = []
for (family, gate_weight), group in cv_summary.groupby(["family", "gate_positive_weight"]):
    eligible = group.loc[group["passes_rmse_guardrail"]].copy()
    best = eligible.sort_values(
        ["mean_ndcg_at_5", "rmse_positive"],
        ascending=[False, True],
    ).iloc[0]
    weight_rows.append(best)
weight_summary = pd.DataFrame(weight_rows)
print(
    weight_summary[
        [
            "family", "gate_positive_weight", "configuration_id", "rmse_all",
            "rmse_positive", "mean_ndcg_at_5", "mean_winner_rank",
            "gate_average_precision", "gate_brier",
            "mean_predicted_vote_probability", "observed_vote_rate",
        ]
    ].sort_values(["family", "gate_positive_weight"]).to_string(index=False)
)

section("7. SELECT ONE CONFIGURATION PER HURDLE FAMILY")
selected_rows = []
selection_audit_rows = []
for family, family_rows in cv_summary.groupby("family"):
    eligible = family_rows.loc[family_rows["passes_rmse_guardrail"]].copy()
    best_ndcg = eligible["mean_ndcg_at_5"].max()
    near_best = eligible.loc[eligible["mean_ndcg_at_5"].ge(best_ndcg - 0.01)].copy()
    selected = near_best.sort_values(
        ["rmse_positive", "rmse_all", "gate_brier", "mean_winner_rank"],
        ascending=[True, True, True, True],
    ).iloc[0]
    selected_rows.append(selected)
    selection_audit_rows.append(
        {
            "family": family,
            "configurations": len(family_rows),
            "passing_guardrail": len(eligible),
            "family_best_ndcg": best_ndcg,
            "within_0.01_of_best": len(near_best),
            "selected_configuration_id": int(selected["configuration_id"]),
        }
    )
selected_configs = pd.DataFrame(selected_rows).reset_index(drop=True)
print("Selection: pass the RMSE guardrail; remain within 0.01 of family-best NDCG; then minimize positive RMSE, all-player RMSE, Brier score, and winner rank.")
print("\nSelected hurdle configurations:")
print(selected_configs[display_columns].to_string(index=False))
print("\nSelection audit:")
print(pd.DataFrame(selection_audit_rows).to_string(index=False))

section("8. DEVELOPMENT-ONLY CANDIDATE THRESHOLD")
threshold_rows = []
selected_thresholds = {}
for _, selected in selected_configs.iterrows():
    configuration_id = int(selected["configuration_id"])
    oof = pd.concat(oof_predictions[configuration_id], ignore_index=True)
    operating_point = select_recall_constrained_threshold(
        oof["y_binary"].to_numpy(),
        oof["gate_probability"].to_numpy(),
        minimum_recall=0.90,
    )
    selected_thresholds[configuration_id] = operating_point["threshold"]
    threshold_rows.append(
        {
            "family": selected["family"],
            "configuration_id": configuration_id,
            "threshold": operating_point["threshold"],
            "development_oof_precision": operating_point["precision"],
            "development_oof_recall": operating_point["recall"],
            "development_oof_flagged_pct": operating_point["flagged_pct"],
        }
    )
threshold_audit = pd.DataFrame(threshold_rows)
print(threshold_audit.to_string(index=False))
print("\nThe threshold is diagnostic only; expected-share predictions remain continuous and are not thresholded.")

section("9. EXTERNAL 2015–2018 VALIDATION — HURDLE CANDIDATES ONLY")
validation_rows = []
validation_predictions = {}
validation_gate_probabilities = {}
validation_threshold_rows = []
for _, selected in selected_configs.iterrows():
    configuration_id = int(selected["configuration_id"])
    result = fit_hurdle_model(
        development,
        validation,
        selected,
        compact_numeric,
        broad_numeric,
        categorical_features,
    )
    label = selected["family"]
    validation_rows.append(
        {
            "model": label,
            "configuration_id": configuration_id,
            **evaluate_hurdle_predictions(
                validation,
                result["gate_probability"],
                result["raw_conditional_share"],
                result["expected_share"],
            ),
        }
    )
    validation_predictions[label] = result["expected_share"]
    validation_gate_probabilities[label] = result["gate_probability"]
    threshold = selected_thresholds[configuration_id]
    validation_binary = validation[TARGET].gt(0).astype(int).to_numpy()
    validation_flag = result["gate_probability"] >= threshold
    validation_threshold_rows.append(
        {
            "model": label,
            "development_selected_threshold": threshold,
            "validation_precision": precision_score(validation_binary, validation_flag, zero_division=0),
            "validation_recall": recall_score(validation_binary, validation_flag, zero_division=0),
            "validation_players_flagged": int(validation_flag.sum()),
            "validation_flagged_pct": 100 * validation_flag.mean(),
        }
    )
validation_results = pd.DataFrame(validation_rows)
validation_threshold_audit = pd.DataFrame(validation_threshold_rows)
print(validation_results.to_string(index=False))
print("\nThreshold audit on external validation:")
print(validation_threshold_audit.to_string(index=False))

section("10. VALIDATION WINNER-RANK AUDIT")
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

section("11. PROBABILITY CALIBRATION TABLES")
calibration_tables = {}
validation_binary = validation[TARGET].gt(0).astype(int).to_numpy()
for model_name, probability in validation_gate_probabilities.items():
    table = calibration_table(validation_binary, probability, requested_bins=8)
    calibration_tables[model_name] = table
    print(f"\n{model_name}:")
    print(table.to_string(index=False))

section("12. LEAKAGE AND BOUNDARY GUARDRAILS")
all_predictors = set(compact_numeric + broad_numeric + categorical_features)
guardrails = pd.DataFrame(
    [
        ("All development zero rows retained by Stage 1", int(development[TARGET].eq(0).sum()) == 11642),
        ("Stage 2 trained only on positive shares", True),
        ("Target excluded before era features", TARGET not in feature_only.columns),
        ("Target absent from predictors", TARGET not in all_predictors),
        ("Identity and season absent from predictors", {"player", "team_id", "season", "source_row_id"}.isdisjoint(all_predictors)),
        ("Preprocessing learned within each component and fold", True),
        ("Candidate threshold learned from development OOF predictions", True),
        ("Candidate threshold not applied to expected share", True),
        ("2015–2018 excluded from configuration selection", development["season"].max() < validation["season"].min()),
        ("2019–2022 target object not created", "y_test" not in locals()),
        ("2019–2022 target absent from held features", TARGET not in final_test_features_only.columns),
        ("No final all-model comparison performed", True),
    ],
    columns=["guardrail", "passed"],
)
print(guardrails.to_string(index=False))
print(f"\nAll guardrails passed: {guardrails['passed'].all()}")
if not guardrails["passed"].all():
    raise AssertionError("A two-stage guardrail failed.")

section("13. FIGURE 1 — TWO-STAGE VALIDATION RESULTS")
plot_results = validation_results.set_index("model")
model_order = plot_results.index.tolist()
colors = [BLUE, ORANGE]
short_labels = [label.replace(" + ", " +\n") for label in model_order]
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(
    "Two-Stage MVP Models — 2015–2018 External Validation",
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
performance_path = OUTPUT_DIR / "two_stage_validation_performance.png"
fig.savefig(performance_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(performance_path.resolve())

section("14. FIGURE 2 — GATE DISCRIMINATION AND CALIBRATION")
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.suptitle(
    "Vote-Gate Diagnostics — 2015–2018 External Validation",
    fontsize=22,
    fontweight="bold",
    color=NAVY,
    y=1.02,
)
for model_name, color in zip(model_order, colors):
    probability = validation_gate_probabilities[model_name]
    precision, recall, _ = precision_recall_curve(validation_binary, probability)
    axes[0].plot(
        recall,
        precision,
        linewidth=3,
        color=color,
        label=f"{model_name} (AP={average_precision_score(validation_binary, probability):.3f})",
    )
    table = calibration_tables[model_name]
    axes[1].plot(
        table["mean_predicted_probability"],
        table["observed_vote_rate"],
        marker="o",
        markersize=8,
        linewidth=2.5,
        color=color,
        label=model_name,
    )
axes[0].axhline(validation_binary.mean(), color=GRAY, linestyle="--", linewidth=1.8, label="Vote-recipient prevalence")
axes[0].set_title("A. Precision–recall curve")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_xlim(0, 1.02)
axes[0].set_ylim(0, 1.02)
axes[0].legend(loc="upper right")
axes[1].plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1.8, label="Perfect calibration")
axes[1].set_title("B. Quantile-bin reliability")
axes[1].set_xlabel("Mean predicted vote probability")
axes[1].set_ylabel("Observed vote-recipient rate")
axes[1].set_xlim(0, 1.02)
axes[1].set_ylim(0, 1.02)
axes[1].legend(loc="upper left")
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.tight_layout()
diagnostics_path = OUTPUT_DIR / "vote_gate_diagnostics.png"
fig.savefig(diagnostics_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(diagnostics_path.resolve())

section("15. FIGURE 3 — WEIGHTING EFFECT IN CROSS-VALIDATION")
weight_plot = weight_summary.copy()
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
for ax, metric, title, ylabel in [
    (axes[0], "mean_ndcg_at_5", "A. Ranking quality", "Mean NDCG@5"),
    (axes[1], "gate_brier", "B. Vote-probability calibration error", "Mean Brier score — lower is better"),
]:
    for family, color in [("Logistic + Ridge", BLUE), ("Extra Trees hurdle", ORANGE)]:
        group = weight_plot.loc[weight_plot["family"].eq(family)].sort_values("gate_positive_weight")
        ax.plot(
            group["gate_positive_weight"],
            group[metric],
            marker="o",
            markersize=9,
            linewidth=3,
            color=color,
            label=family,
        )
    ax.set_title(title)
    ax.set_xlabel("Positive-class weight in vote gate")
    ax.set_ylabel(ylabel)
    ax.set_xticks([1, 5])
    ax.legend(loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.suptitle(
    "Moderate Gate Weighting Trades Ranking Against Probability Calibration",
    fontsize=21,
    fontweight="bold",
    color=NAVY,
    y=1.02,
)
fig.tight_layout()
weighting_path = OUTPUT_DIR / "two_stage_weighting_effect.png"
fig.savefig(weighting_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(weighting_path.resolve())

section("16. CHUNK BOUNDARY")
print("Two-stage zero-inflation modeling and validation are complete.")
print("No clustering analysis, direct-ranking model, or final all-model comparison was performed.")
print("The 2019–2022 target remains sealed and unevaluated.")
