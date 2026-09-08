from __future__ import annotations

import json
import os
from pathlib import Path
from zipfile import ZipFile

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, ndcg_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ZIP_PATH = Path(os.environ.get("NBA_MVP_DATA_PATH", "data/raw/nba_mvp_stats.zip"))
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_DEPLOYMENT_DIR", "results/deployment"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path(
    os.environ.get(
        "NBA_MVP_MODEL_PATH",
        "models/nba_mvp_production_models_through_2022.joblib",
    )
)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
TARGET = "award_share"
RANDOM_STATE = 42
PRIMARY_MINUTES = 100

NAVY = "#17324D"
BLUE = "#2F6B9A"
ORANGE = "#D97706"
GREEN = "#2F855A"
GRAY = "#6B7280"
RED = "#B42318"

sns.set_theme(style="whitegrid", context="talk")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")


def section(title: str) -> None:
    print(f"\n{'=' * 110}\n{title}\n{'=' * 110}")


def load_source(path: Path) -> pd.DataFrame:
    with ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one CSV; found {members}")
        with archive.open(members[0]) as stream:
            return pd.read_csv(stream)


def add_season_percentiles(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if TARGET in frame.columns:
        raise ValueError("Target leakage guard: award_share must be removed before percentile creation.")
    result = frame.copy()
    for feature in features:
        result[f"{feature}_season_pct"] = result.groupby("season")[feature].rank(
            method="average", pct=True, na_option="keep"
        )
    return result


def prepare_cohort(raw: pd.DataFrame, minimum_minutes: int = PRIMARY_MINUTES) -> pd.DataFrame:
    required = {
        "season", "player", "team_id", "pos", "mp", "mov", "mov_adj", "win_loss_pct",
        "fg_pct", "fg2_pct", "fg3_pct", "ft_pct", "pts_per_g", "mp_per_g", "ts_pct",
        "per", "ws", "ws_per_48", "bpm", "vorp",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Schema gate failed; missing columns: {missing}")

    base = raw.copy()
    base.insert(0, "source_row_id", np.arange(len(base), dtype=np.int64))
    base = base.loc[base["mp"].ge(minimum_minutes)].copy()
    base["is_tot"] = base["team_id"].eq("TOT").astype("int8")
    base.loc[base["is_tot"].eq(1), ["mov", "mov_adj", "win_loss_pct"]] = np.nan
    for percentage in ["fg_pct", "fg2_pct", "fg3_pct", "ft_pct"]:
        base[f"{percentage}_was_missing"] = base[percentage].isna().astype("int8")
        base[percentage] = base[percentage].fillna(0.0)
    base["pos_primary"] = base["pos"].str.split("-").str[0]

    target = base[TARGET].copy() if TARGET in base.columns else None
    predictors = base.drop(columns=[TARGET], errors="ignore")
    percentile_features = [
        "pts_per_g", "mp_per_g", "ts_pct", "per", "ws", "ws_per_48", "bpm", "vorp",
        "win_loss_pct",
    ]
    prepared = add_season_percentiles(predictors, percentile_features)
    if target is not None:
        prepared[TARGET] = target
    return prepared


def validate_scoring_input(raw: pd.DataFrame, required_predictors: list[str]) -> int:
    missing = sorted(set(required_predictors).difference(raw.columns))
    if missing:
        raise ValueError(f"Production schema gate failed; missing predictors: {missing}")
    seasons = raw["season"].dropna().unique()
    if len(seasons) != 1:
        raise ValueError(f"Production season gate failed; expected one season, found {sorted(seasons.tolist())}")
    return int(seasons[0])


def feature_lists(prepared: pd.DataFrame) -> tuple[list[str], list[str]]:
    exclusions = {
        "season", TARGET, "source_row_id", "mp", "fg_per_g", "fg2_per_g", "fg3_per_g",
        "ft_per_g", "orb_per_g", "drb_per_g", "orb_pct", "drb_pct", "efg_pct", "ows",
        "dws", "obpm", "dbpm", "mov",
    }
    broad_numeric = [
        column for column in prepared.select_dtypes(include=np.number).columns
        if column not in exclusions
    ]
    return broad_numeric, ["pos_primary"]


def make_pipeline(numeric: list[str], categorical: list[str], model) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
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
    return Pipeline([("preprocess", preprocess), ("model", model)])


def fit_locked_specifications(training: pd.DataFrame) -> dict:
    if TARGET not in training.columns:
        raise ValueError("Training requires award_share; scoring does not.")
    numeric, categorical = feature_lists(training)
    x_columns = numeric + categorical
    y = training[TARGET].to_numpy(dtype=float)

    champion = make_pipeline(
        numeric,
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
    champion.fit(training[x_columns], y, model__sample_weight=1 + 5 * y)

    binary = (y > 0).astype(int)
    gate = make_pipeline(
        numeric,
        categorical,
        ExtraTreesClassifier(
            n_estimators=160,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )
    gate.fit(training[x_columns], binary, model__sample_weight=np.where(binary == 1, 5.0, 1.0))

    severity = make_pipeline(
        numeric,
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
    severity.fit(positive[x_columns], positive[TARGET])
    return {
        "champion": champion,
        "challenger_gate": gate,
        "challenger_severity": severity,
        "numeric_features": numeric,
        "categorical_features": categorical,
    }


def score_prepared(models: dict, prepared: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    columns = models["numeric_features"] + models["categorical_features"]
    missing = sorted(set(columns).difference(prepared.columns))
    if missing:
        raise ValueError(f"Prepared-feature gate failed; missing columns: {missing}")
    raw_champion = models["champion"].predict(prepared[columns])
    vote_probability = models["challenger_gate"].predict_proba(prepared[columns])[:, 1]
    severity_raw = models["challenger_severity"].predict(prepared[columns])
    champion = np.clip(raw_champion, 0.0, 1.0)
    challenger = vote_probability * np.clip(severity_raw, 0.0, 1.0)

    keep = [column for column in ["season", "player", "team_id", "mp", TARGET] if column in prepared]
    scored = prepared[keep].copy()
    scored["hgb_award_share"] = champion
    scored["extra_trees_hurdle_share"] = challenger
    health = {
        "hgb_clipping_pct": 100 * np.mean(~np.isclose(raw_champion, champion)),
        "hurdle_severity_clipping_pct": 100 * np.mean(~np.isclose(severity_raw, np.clip(severity_raw, 0, 1))),
        "hgb_season_total": champion.sum(),
        "hurdle_season_total": challenger.sum(),
    }
    return scored, health


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = reference.dropna().to_numpy(dtype=float)
    current = current.dropna().to_numpy(dtype=float)
    if len(reference) == 0 or len(current) == 0:
        return np.nan
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_counts = np.histogram(current, bins=edges)[0] / len(current)
    ref_counts = np.clip(ref_counts, 1e-6, None)
    cur_counts = np.clip(cur_counts, 1e-6, None)
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def winner_rank(y_true: np.ndarray, prediction: np.ndarray) -> float:
    winner = np.isclose(y_true, y_true.max())
    ranks = pd.Series(prediction).rank(method="average", ascending=False).to_numpy()
    return float(ranks[winner].mean())


def post_outcome_metrics(scored: pd.DataFrame, prediction_column: str) -> dict:
    y = scored[TARGET].to_numpy(dtype=float)
    prediction = scored[prediction_column].to_numpy(dtype=float)
    positive = y > 0
    return {
        "all_player_rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "recipient_rmse": float(np.sqrt(mean_squared_error(y[positive], prediction[positive]))),
        "ndcg_at_5": float(ndcg_score(y.reshape(1, -1), prediction.reshape(1, -1), k=5)),
        "winner_rank": winner_rank(y, prediction),
    }


def status(value: float, review: float, investigate: float, lower_is_better: bool = True) -> str:
    if lower_is_better:
        return "PASS" if value < review else ("REVIEW" if value < investigate else "INVESTIGATE")
    return "PASS" if value > review else ("REVIEW" if value > investigate else "INVESTIGATE")


source = load_source(ZIP_PATH)
prepared = prepare_cohort(source)
raw_predictor_columns = [column for column in source.columns if column != TARGET]

section("1. DEPLOYMENT CONTRACT")
contract = {
    "locked_evidence_window": "2019-2022; never recomputed for model selection",
    "production_training_window": "1982-2022",
    "champion": "Weighted Histogram Gradient Boosting",
    "required_challenger": "Extra Trees hurdle",
    "candidate_rule": "mp >= 100; repeat audit at mp >= 500",
    "prediction_bounds": "[0, 1]",
    "scoring_unit": "complete player-season table for exactly one season",
    "automatic_retraining": False,
    "retraining_rule": "schema failure or two consecutive completed seasons with material degradation",
}
print(pd.Series(contract, name="Decision").to_string())

section("2. 2022 SHADOW-DEPLOYMENT DRY RUN")
shadow_training = prepared.loc[prepared["season"].le(2021)].copy()
shadow_raw = source.loc[source["season"].eq(2022)].copy()
validate_scoring_input(shadow_raw, raw_predictor_columns)
shadow_season = prepare_cohort(shadow_raw)
shadow_models = fit_locked_specifications(shadow_training)
shadow_scored, output_health = score_prepared(shadow_models, shadow_season)

top_hgb = shadow_scored.nlargest(5, "hgb_award_share")
top_hurdle = shadow_scored.nlargest(5, "extra_trees_hurdle_share")
hgb_names = set(top_hgb["player"])
hurdle_names = set(top_hurdle["player"])
top5_overlap = len(hgb_names & hurdle_names)
top1_agreement = top_hgb.iloc[0]["player"] == top_hurdle.iloc[0]["player"]

candidate_table = pd.concat(
    [
        top_hgb[["player", "hgb_award_share"]].rename(columns={"hgb_award_share": "predicted_share"}).assign(model="Histogram GB"),
        top_hurdle[["player", "extra_trees_hurdle_share"]].rename(columns={"extra_trees_hurdle_share": "predicted_share"}).assign(model="Extra Trees hurdle"),
    ],
    ignore_index=True,
)
candidate_table["rank"] = candidate_table.groupby("model")["predicted_share"].rank(method="first", ascending=False).astype(int)
print(candidate_table[["model", "rank", "player", "predicted_share"]].sort_values(["model", "rank"]).to_string(index=False))

section("3. INPUT AND OUTPUT MONITORING")
recent_reference = prepared.loc[prepared["season"].between(2017, 2021)]
drift_features = ["pts_per_g", "ts_pct", "ws", "bpm", "win_loss_pct"]
drift = pd.DataFrame(
    {
        "feature": drift_features,
        "psi": [population_stability_index(recent_reference[f], shadow_season[f]) for f in drift_features],
    }
)
drift["status"] = drift["psi"].map(lambda value: status(value, 0.10, 0.25))
drift["recent_missing_pct"] = [100 * recent_reference[f].isna().mean() for f in drift_features]
drift["current_missing_pct"] = [100 * shadow_season[f].isna().mean() for f in drift_features]
drift["missingness_change_pp"] = drift["current_missing_pct"] - drift["recent_missing_pct"]

history_rows = []
for audit_season in range(2017, 2023):
    rolling_reference = prepared.loc[prepared["season"].between(audit_season - 5, audit_season - 1)]
    audit_frame = prepared.loc[prepared["season"].eq(audit_season)]
    history_rows.append(
        {
            "season": audit_season,
            **{
                feature: population_stability_index(rolling_reference[feature], audit_frame[feature])
                for feature in drift_features
            },
        }
    )
psi_history = pd.DataFrame(history_rows)
core_psi_features = ["pts_per_g", "ts_pct", "ws", "bpm"]
high_psi_count = int(drift.loc[drift["feature"].isin(core_psi_features), "psi"].ge(0.25).sum())
psi_policy_status = "PASS" if high_psi_count == 0 else ("REVIEW" if high_psi_count == 1 else "INVESTIGATE")
recent_team_medians = recent_reference.groupby("season")["win_loss_pct"].median()
current_team_median = shadow_season["win_loss_pct"].median()
team_iqr = recent_team_medians.quantile(0.75) - recent_team_medians.quantile(0.25)
team_context_shift = abs(current_team_median - recent_team_medians.median()) / team_iqr if team_iqr > 0 else np.inf
team_context_status = status(team_context_shift, 1.5, 3.0)
recent_counts = recent_reference.groupby("season").size()
candidate_count_deviation = abs(len(shadow_season) - recent_counts.median()) / recent_counts.median()

monitoring = pd.DataFrame(
    [
        ["Schema columns present", 1.0, "must equal 1", "PASS"],
        ["Candidate-count deviation", candidate_count_deviation, "review >= 25%", status(candidate_count_deviation, 0.25, 0.40)],
        ["High-PSI feature count", float(high_psi_count), "1 review; >=2 investigate", psi_policy_status],
        ["Team-context median shift", team_context_shift, "review >= 1.5 recent IQR", team_context_status],
        ["Champion/challenger top-1 agreement", float(top1_agreement), "review if 0", "PASS" if top1_agreement else "REVIEW"],
        ["Top-5 overlap", float(top5_overlap), "review < 3", "PASS" if top5_overlap >= 3 else "REVIEW"],
        ["HGB clipping percentage", output_health["hgb_clipping_pct"] / 100, "known risk; disclose", "REVIEW" if output_health["hgb_clipping_pct"] > 50 else "PASS"],
    ],
    columns=["monitor", "observed", "policy", "status"],
)
print("\nFeature drift versus recent seasons (2017-2021):")
print(drift.to_string(index=False))
print(f"\nTeam-context diagnostic: 2022 median={current_team_median:.4f}; recent-season median={recent_team_medians.median():.4f}; shift={team_context_shift:.3f} IQR")
print("\nRolling five-season PSI history (diagnostic context):")
print(psi_history.to_string(index=False))
print("\nDeployment health checks:")
print(monitoring.to_string(index=False))

section("4. POST-OUTCOME AUDIT — NOT AVAILABLE AT PREDICTION TIME")
outcome_rows = []
for model, column in [("Histogram GB", "hgb_award_share"), ("Extra Trees hurdle", "extra_trees_hurdle_share")]:
    outcome_rows.append({"model": model, **post_outcome_metrics(shadow_scored, column)})
outcomes = pd.DataFrame(outcome_rows)
print(outcomes.to_string(index=False))

section("5. PRODUCTION REFIT THROUGH 2022")
production_models = fit_locked_specifications(prepared)
package = {
    **production_models,
    "contract": contract,
    "trained_through": 2022,
    "minimum_minutes": PRIMARY_MINUTES,
    "required_raw_predictor_columns": raw_predictor_columns,
    "recent_reference_seasons": [2017, 2018, 2019, 2020, 2021, 2022],
}
model_path = MODEL_PATH
joblib.dump(package, model_path)

loaded_package = joblib.load(model_path)
roundtrip_raw = shadow_raw.drop(columns=[TARGET]).copy()
validated_season = validate_scoring_input(roundtrip_raw, loaded_package["required_raw_predictor_columns"])
roundtrip_prepared = prepare_cohort(roundtrip_raw)
roundtrip_scored, _ = score_prepared(loaded_package, roundtrip_prepared)

gate_tests = []
try:
    validate_scoring_input(roundtrip_raw.drop(columns=["bpm"]), loaded_package["required_raw_predictor_columns"])
except ValueError as error:
    gate_tests.append(["Missing-column rejection", "PASS", str(error)])
try:
    validate_scoring_input(source.loc[source["season"].isin([2021, 2022])].drop(columns=[TARGET]), loaded_package["required_raw_predictor_columns"])
except ValueError as error:
    gate_tests.append(["Multiple-season rejection", "PASS", str(error)])
gate_tests.append(["Target-free scoring", "PASS", f"Season {validated_season}; {len(roundtrip_scored)} eligible rows"])
gate_test_table = pd.DataFrame(gate_tests, columns=["test", "result", "detail"])

contract_path = OUTPUT_DIR / "deployment_contract.json"
contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
shadow_scored.to_csv(OUTPUT_DIR / "shadow_2022_predictions.csv", index=False)
drift.to_csv(OUTPUT_DIR / "shadow_2022_feature_drift.csv", index=False)
psi_history.to_csv(OUTPUT_DIR / "rolling_feature_psi_history.csv", index=False)
monitoring.to_csv(OUTPUT_DIR / "shadow_2022_monitoring.csv", index=False)
outcomes.to_csv(OUTPUT_DIR / "shadow_2022_post_outcome_metrics.csv", index=False)
print(f"Saved model package: {model_path} ({model_path.stat().st_size / 1_000_000:.2f} MB)")
print(f"Target present among model features: {TARGET in production_models['numeric_features'] + production_models['categorical_features']}")
print("\nDeployment interface self-tests:")
print(gate_test_table.to_string(index=False))

# Figure 1: readable champion/challenger candidate comparison.
plot_candidates = candidate_table.copy()
order = (
    plot_candidates.groupby("player")["predicted_share"].max().sort_values(ascending=True).index
)
fig, ax = plt.subplots(figsize=(12, 7.2))
sns.barplot(
    data=plot_candidates,
    y="player",
    x="predicted_share",
    hue="model",
    order=order,
    palette={"Histogram GB": ORANGE, "Extra Trees hurdle": GREEN},
    ax=ax,
)
ax.set_title("2022 Shadow Deployment: Champion and Challenger Candidate Lists", pad=15, color=NAVY, weight="bold")
ax.set_xlabel("Predicted MVP vote share")
ax.set_ylabel("")
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=4, fontsize=9)
ax.set_xlim(0, plot_candidates["predicted_share"].max() * 1.16)
ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUTPUT_DIR / "shadow_2022_candidate_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Figure 2: feature-drift audit with standard PSI review bands.
fig, ax = plt.subplots(figsize=(11, 6.5))
colors = [GREEN if value < 0.10 else ORANGE if value < 0.25 else RED for value in drift["psi"]]
ax.barh(drift["feature"], drift["psi"], color=colors)
ax.axvline(0.10, color=ORANGE, linestyle="--", linewidth=2, label="Review threshold (0.10)")
ax.axvline(0.25, color=RED, linestyle="--", linewidth=2, label="Investigate threshold (0.25)")
for index, value in enumerate(drift["psi"]):
    ax.text(value + 0.006, index, f"{value:.3f}", va="center", fontsize=10)
ax.set_xlim(0, drift["psi"].max() * 1.12)
ax.set_title("2022 Input Drift Versus the 2017–2021 Reference", pad=15, color=NAVY, weight="bold")
ax.set_xlabel("Population Stability Index (PSI)")
ax.set_ylabel("")
ax.legend(loc="lower right")
fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.text(
    0.5, 0.035,
    "Note: win_loss_pct PSI is descriptive only; its repeated team-level structure is monitored separately.",
    ha="center", fontsize=10, color=GRAY,
)
fig.savefig(OUTPUT_DIR / "shadow_2022_feature_drift.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Figure 3: operational decision flow.
fig, ax = plt.subplots(figsize=(13, 8))
ax.axis("off")
boxes = [
    (0.50, 0.90, "1. Schema and season gate\nHard stop if incompatible", NAVY),
    (0.50, 0.70, "2. Score complete season\nHGB champion + hurdle challenger", BLUE),
    (0.50, 0.50, "3. Publish top five and warnings\nReview top-candidate disagreement", ORANGE),
    (0.50, 0.30, "4. Add outcomes when available\nRMSE, NDCG@5, winner rank", GREEN),
    (0.50, 0.10, "5. Retrain only after sustained failure\nTwo seasons or schema change", GRAY),
]
for x, y, label, color in boxes:
    ax.text(
        x, y, label, transform=ax.transAxes, ha="center", va="center", color="white",
        fontsize=13, weight="bold", bbox=dict(boxstyle="round,pad=0.8", facecolor=color, edgecolor="none"),
    )
for upper, lower in zip(boxes[:-1], boxes[1:]):
    ax.annotate(
        "", xy=(lower[0], lower[1] + 0.065), xytext=(upper[0], upper[1] - 0.065),
        xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color=NAVY, linewidth=2),
    )
ax.set_title("NBA MVP Model Deployment and Governance Flow", fontsize=19, color=NAVY, weight="bold", pad=20)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "deployment_governance_flow.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("\nFigures saved:")
for name in [
    "shadow_2022_candidate_comparison.png",
    "shadow_2022_feature_drift.png",
    "deployment_governance_flow.png",
]:
    print(f"- {OUTPUT_DIR / name}")
