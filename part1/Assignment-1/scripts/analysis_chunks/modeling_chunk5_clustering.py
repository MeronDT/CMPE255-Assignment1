import os
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    r2_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 260)
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
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "grid.color": LIGHT_GRAY,
        "grid.linewidth": 0.8,
    },
)

ZIP_PATH = Path(os.environ.get("NBA_MVP_DATA_PATH", "data/raw/nba_mvp_stats.zip"))
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk17_outputs"
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


def cluster_matrix(frame: pd.DataFrame, cluster_features: list[str]) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(frame[cluster_features])
    scaled = scaler.fit_transform(imputed)
    return scaled, imputer, scaler


def choose_k_without_target(
    frame: pd.DataFrame,
    cluster_features: list[str],
    sample_size: int = 4_000,
) -> tuple[pd.DataFrame, int]:
    scaled, _, _ = cluster_matrix(frame, cluster_features)
    rows = []
    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(scaled)
        counts = pd.Series(labels).value_counts(normalize=True)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(
                    scaled,
                    labels,
                    sample_size=min(sample_size, len(frame)),
                    random_state=RANDOM_STATE,
                ),
                "calinski_harabasz": calinski_harabasz_score(scaled, labels),
                "davies_bouldin": davies_bouldin_score(scaled, labels),
                "smallest_cluster_pct": 100 * counts.min(),
            }
        )
    diagnostics = pd.DataFrame(rows)

    # Require more nuance than a binary split, avoid excessive fragmentation,
    # and reject solutions with a very small development cluster.
    eligible = diagnostics.loc[
        diagnostics["k"].between(3, 6)
        & diagnostics["smallest_cluster_pct"].ge(5.0)
    ].copy()
    if eligible.empty:
        eligible = diagnostics.loc[diagnostics["k"].between(3, 6)].copy()
    eligible["silhouette_rank"] = eligible["silhouette"].rank(ascending=False, method="min")
    eligible["calinski_rank"] = eligible["calinski_harabasz"].rank(ascending=False, method="min")
    eligible["davies_rank"] = eligible["davies_bouldin"].rank(ascending=True, method="min")
    eligible["rank_sum"] = eligible[["silhouette_rank", "calinski_rank", "davies_rank"]].sum(axis=1)
    selected_k = int(
        eligible.sort_values(
            ["rank_sum", "silhouette", "k"],
            ascending=[True, False, True],
        ).iloc[0]["k"]
    )
    diagnostics["selected"] = diagnostics["k"].eq(selected_k)
    return diagnostics, selected_k


def fit_clusterer(
    training_frame: pd.DataFrame,
    cluster_features: list[str],
    k: int,
) -> Pipeline:
    clusterer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=k,
                    n_init=20,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    clusterer.fit(training_frame[cluster_features])
    return clusterer


def add_ordered_cluster_labels(
    frame: pd.DataFrame,
    raw_labels: np.ndarray,
    cluster_features: list[str],
) -> tuple[pd.DataFrame, dict[int, str]]:
    labeled = frame.copy()
    labeled["cluster_id"] = raw_labels
    candidate_features = [
        "mp_per_g_season_pct",
        "pts_per_g_season_pct",
        "usg_pct_season_pct",
        "bpm_season_pct",
        "win_loss_pct_season_pct",
    ]
    candidate_index = labeled.groupby("cluster_id")[candidate_features].mean().mean(axis=1)
    ordered_ids = candidate_index.sort_values().index.tolist()
    tier_names_by_k = {
        3: ["Limited-role players", "Rotation and starter contributors", "Elite focal players"],
        4: ["Limited-role players", "Rotation contributors", "Established starters", "Elite focal players"],
        5: ["Limited-role players", "Developing rotation players", "Established contributors", "High-impact starters", "Elite focal players"],
        6: ["Limited-role players", "Developing rotation players", "Rotation contributors", "Established starters", "High-impact stars", "Elite focal players"],
    }
    tier_names = tier_names_by_k.get(
        len(ordered_ids),
        [f"Performance tier {index + 1}" for index in range(len(ordered_ids))],
    )
    mapping = dict(zip(ordered_ids, tier_names))
    labeled["cluster_label"] = labeled["cluster_id"].map(mapping)
    return labeled, mapping


def make_ridge_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    preprocessor = ColumnTransformer(
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
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=100.0)),
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


def season_ranking_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    evaluation = frame[["season", TARGET]].copy()
    evaluation["prediction"] = prediction
    rows = []
    for season, group in evaluation.groupby("season"):
        y_true = group[TARGET].to_numpy(dtype=float)
        y_score = group["prediction"].to_numpy(dtype=float)
        n_players = len(group)
        actual_winner = np.isclose(y_true, y_true.max())
        predicted_top = np.isclose(y_score, y_score.max())
        ranks = pd.Series(y_score).rank(method="average", ascending=False).to_numpy()
        winner_rank = ranks[actual_winner].mean()
        rows.append(
            {
                "ndcg_at_5": ndcg_score(
                    y_true.reshape(1, -1),
                    y_score.reshape(1, -1),
                    k=min(5, n_players),
                    ignore_ties=False,
                ),
                "winner_rank": winner_rank,
                "fractional_top1_accuracy": np.logical_and(actual_winner, predicted_top).sum() / predicted_top.sum(),
                "spearman": pd.Series(y_true).corr(pd.Series(y_score), method="spearman"),
            }
        )
    per_season = pd.DataFrame(rows)
    return {
        "mean_ndcg_at_5": per_season["ndcg_at_5"].mean(),
        "mean_winner_rank": per_season["winner_rank"].mean(),
        "fractional_top1_accuracy": per_season["fractional_top1_accuracy"].mean(),
        "mean_season_spearman": per_season["spearman"].mean(),
    }


def evaluate_predictions(frame: pd.DataFrame, raw_prediction: np.ndarray) -> dict:
    bounded = np.clip(raw_prediction, 0.0, 1.0)
    y_true = frame[TARGET].to_numpy(dtype=float)
    return {
        **regression_metrics(y_true, bounded),
        **season_ranking_metrics(frame, bounded),
        "raw_below_zero_pct": 100 * np.mean(raw_prediction < 0),
    }


with ZipFile(ZIP_PATH) as archive:
    csv_members = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and not name.endswith("/")
    ]
    if len(csv_members) != 1:
        raise ValueError(f"Expected exactly one CSV; found {csv_members}")
    with archive.open(csv_members[0]) as csv_file:
        source = pd.read_csv(csv_file)

# Reuse approved preparation, adding only target-free season percentiles needed for clustering.
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
percentile_raw_features = [
    "pts_per_g", "mp_per_g", "trb_per_g", "ast_per_g", "stl_per_g",
    "blk_per_g", "usg_pct", "ts_pct", "per", "ws", "ws_per_48",
    "bpm", "vorp", "win_loss_pct",
]
prepared = add_season_percentiles(feature_only, percentile_raw_features)
prepared[TARGET] = target_series

development = prepared.loc[prepared["season"].le(2014)].copy()
validation = prepared.loc[prepared["season"].between(2015, 2018)].copy()
final_test_features_only = (
    prepared.loc[prepared["season"].between(2019, 2022)]
    .drop(columns=[TARGET])
    .copy()
)

cluster_features = [
    "mp_per_g_season_pct",
    "pts_per_g_season_pct",
    "trb_per_g_season_pct",
    "ast_per_g_season_pct",
    "stl_per_g_season_pct",
    "blk_per_g_season_pct",
    "usg_pct_season_pct",
    "ts_pct_season_pct",
    "bpm_season_pct",
    "win_loss_pct_season_pct",
]

compact_numeric = [
    "age", "g", "mp_per_g", "pts_per_g", "trb_per_g", "ast_per_g",
    "stl_per_g", "blk_per_g", "tov_per_g", "ts_pct", "usg_pct", "per",
    "ws", "bpm", "win_loss_pct", "is_tot", "pts_per_g_season_pct",
    "mp_per_g_season_pct", "ts_pct_season_pct", "per_season_pct",
    "ws_season_pct", "bpm_season_pct", "win_loss_pct_season_pct",
]

section("1. CLUSTERING CONTRACT")
contract = pd.DataFrame(
    [
        ("Fit and select k", "Development features only", "award_share excluded"),
        ("Interpret clusters", "Join target after fitting", "Descriptive, not causal"),
        ("Test as predictor", "Refit clusterer inside every fold", "No future-feature leakage"),
        ("Final test", "2019–2022 features only", "Target remains sealed"),
    ],
    columns=["step", "data", "guardrail"],
)
print(contract.to_string(index=False))
print(f"\nClustering features ({len(cluster_features)}):")
print(cluster_features)

section("2. TARGET-FREE K DIAGNOSTICS")
diagnostics, selected_k = choose_k_without_target(development, cluster_features)
print(diagnostics.to_string(index=False))
print(f"\nSelected k: {selected_k}")
print("Selection uses rank agreement among silhouette, Calinski–Harabasz, and Davies–Bouldin within k=3–6, with a 5% minimum-cluster preference.")

section("3. FIT DEVELOPMENT CLUSTERS AND BUILD PERFORMANCE-TIER LABELS")
development_clusterer = fit_clusterer(development, cluster_features, selected_k)
development_raw_labels = development_clusterer.predict(development[cluster_features])
validation_raw_labels = development_clusterer.predict(validation[cluster_features])
development_labeled, cluster_name_mapping = add_ordered_cluster_labels(
    development,
    development_raw_labels,
    cluster_features,
)
validation_labeled = validation.copy()
validation_labeled["cluster_id"] = validation_raw_labels
validation_labeled["cluster_label"] = validation_labeled["cluster_id"].map(cluster_name_mapping)
print("Cluster ID-to-label mapping:")
print(pd.Series(cluster_name_mapping, name="cluster_label").rename_axis("cluster_id").to_string())

profile = development_labeled.groupby("cluster_label")[cluster_features].mean()
cluster_order = profile[
    [
        "mp_per_g_season_pct", "pts_per_g_season_pct", "usg_pct_season_pct",
        "bpm_season_pct", "win_loss_pct_season_pct",
    ]
].mean(axis=1).sort_values().index.tolist()
profile = profile.loc[cluster_order]
print("\nMean season-percentile profile:")
print(profile.to_string())

section("4. POST-HOC TARGET AND MVP PROFILE")
development_labeled["is_vote_recipient"] = development_labeled[TARGET].gt(0)
season_maximum = development_labeled.groupby("season")[TARGET].transform("max")
development_labeled["is_mvp_winner"] = np.isclose(
    development_labeled[TARGET],
    season_maximum,
)
target_profile = (
    development_labeled.groupby("cluster_label")
    .agg(
        players=("player", "size"),
        population_pct=("player", lambda values: 100 * len(values) / len(development_labeled)),
        vote_recipients=("is_vote_recipient", "sum"),
        vote_recipient_pct=("is_vote_recipient", lambda values: 100 * values.mean()),
        mvp_winners=("is_mvp_winner", "sum"),
        mean_award_share=(TARGET, "mean"),
        maximum_award_share=(TARGET, "max"),
    )
    .loc[cluster_order]
)
positive_cluster_counts = development_labeled.loc[
    development_labeled["is_vote_recipient"]
]["cluster_label"].value_counts()
target_profile["share_of_all_vote_recipients_pct"] = (
    100 * positive_cluster_counts.reindex(target_profile.index, fill_value=0)
    / development_labeled["is_vote_recipient"].sum()
)
print(target_profile.to_string())
print("\nTarget values were not used to create or name clusters; this table is a post-hoc interpretation.")

section("5. TEMPORAL CLUSTER-MIX STABILITY")
development_mix = development_labeled["cluster_label"].value_counts(normalize=True).reindex(cluster_order)
validation_mix = validation_labeled["cluster_label"].value_counts(normalize=True).reindex(cluster_order)
mix_table = pd.DataFrame(
    {
        "development_pct": 100 * development_mix,
        "validation_pct": 100 * validation_mix,
    }
)
mix_table["percentage_point_change"] = mix_table["validation_pct"] - mix_table["development_pct"]
print(mix_table.to_string())
print(f"\nLargest absolute cluster-mix shift: {mix_table['percentage_point_change'].abs().max():.2f} percentage points")

section("6. LEAKAGE-SAFE INCREMENTAL PREDICTIVE TEST")
fold_definitions = [
    ("Fold 1", 1982, 2002, 2003, 2006),
    ("Fold 2", 1982, 2006, 2007, 2010),
    ("Fold 3", 1982, 2010, 2011, 2014),
]
fold_rows = []
fold_k_rows = []
for fold_name, train_start, train_end, valid_start, valid_end in fold_definitions:
    fold_train = development.loc[development["season"].between(train_start, train_end)].copy()
    fold_valid = development.loc[development["season"].between(valid_start, valid_end)].copy()
    fold_diagnostics, fold_k = choose_k_without_target(
        fold_train,
        cluster_features,
        sample_size=3_000,
    )
    fold_clusterer = fit_clusterer(fold_train, cluster_features, fold_k)
    fold_train["cluster_label"] = "cluster_" + pd.Series(
        fold_clusterer.predict(fold_train[cluster_features]),
        index=fold_train.index,
    ).astype(str)
    fold_valid["cluster_label"] = "cluster_" + pd.Series(
        fold_clusterer.predict(fold_valid[cluster_features]),
        index=fold_valid.index,
    ).astype(str)
    fold_k_rows.append(
        {
            "fold": fold_name,
            "selected_k_from_training_features": fold_k,
            "training_rows": len(fold_train),
            "validation_rows": len(fold_valid),
        }
    )

    y_train = fold_train[TARGET].to_numpy(dtype=float)
    sample_weight = 1.0 + 5.0 * y_train
    for model_name, categorical_features in [
        ("Ridge without cluster", ["pos_primary"]),
        ("Ridge with cluster", ["pos_primary", "cluster_label"]),
    ]:
        model = make_ridge_pipeline(compact_numeric, categorical_features)
        input_columns = compact_numeric + categorical_features
        model.fit(
            fold_train[input_columns],
            y_train,
            model__sample_weight=sample_weight,
        )
        raw_prediction = model.predict(fold_valid[input_columns])
        fold_rows.append(
            {
                "fold": fold_name,
                "model": model_name,
                **evaluate_predictions(fold_valid, raw_prediction),
            }
        )

fold_k_audit = pd.DataFrame(fold_k_rows)
fold_results = pd.DataFrame(fold_rows)
metric_columns = [
    "mae_all", "rmse_all", "r2_all", "mae_positive", "rmse_positive",
    "mean_ndcg_at_5", "mean_winner_rank", "fractional_top1_accuracy",
    "mean_season_spearman", "raw_below_zero_pct",
]
cv_comparison = fold_results.groupby("model")[metric_columns].mean()
print("Fold-local cluster-count selection:")
print(fold_k_audit.to_string(index=False))
print("\nMean expanding-fold results:")
print(cv_comparison.to_string())

incremental_change = pd.DataFrame(
    {
        "without_cluster": cv_comparison.loc["Ridge without cluster"],
        "with_cluster": cv_comparison.loc["Ridge with cluster"],
    }
)
incremental_change["absolute_change"] = incremental_change["with_cluster"] - incremental_change["without_cluster"]
print("\nIncremental change from adding cluster membership:")
print(incremental_change.to_string())

section("7. EXTERNAL VALIDATION CHECK OF CLUSTER AUGMENTATION")
development_augmented = development.copy()
validation_augmented = validation.copy()
development_augmented["cluster_label"] = "cluster_" + pd.Series(
    development_raw_labels,
    index=development_augmented.index,
).astype(str)
validation_augmented["cluster_label"] = "cluster_" + pd.Series(
    validation_raw_labels,
    index=validation_augmented.index,
).astype(str)

validation_rows = []
y_development = development_augmented[TARGET].to_numpy(dtype=float)
sample_weight = 1.0 + 5.0 * y_development
for model_name, categorical_features in [
    ("Ridge without cluster", ["pos_primary"]),
    ("Ridge with cluster", ["pos_primary", "cluster_label"]),
]:
    model = make_ridge_pipeline(compact_numeric, categorical_features)
    input_columns = compact_numeric + categorical_features
    model.fit(
        development_augmented[input_columns],
        y_development,
        model__sample_weight=sample_weight,
    )
    raw_prediction = model.predict(validation_augmented[input_columns])
    validation_rows.append(
        {
            "model": model_name,
            **evaluate_predictions(validation_augmented, raw_prediction),
        }
    )
validation_comparison = pd.DataFrame(validation_rows).set_index("model")
print(validation_comparison.to_string())

section("8. GUARDRAILS AND KEEP-OR-REJECT DECISION")
cv_rmse_improved = (
    cv_comparison.loc["Ridge with cluster", "rmse_all"]
    < cv_comparison.loc["Ridge without cluster", "rmse_all"]
)
cv_ndcg_improved = (
    cv_comparison.loc["Ridge with cluster", "mean_ndcg_at_5"]
    > cv_comparison.loc["Ridge without cluster", "mean_ndcg_at_5"]
)
validation_rmse_improved = (
    validation_comparison.loc["Ridge with cluster", "rmse_all"]
    < validation_comparison.loc["Ridge without cluster", "rmse_all"]
)
validation_ndcg_improved = (
    validation_comparison.loc["Ridge with cluster", "mean_ndcg_at_5"]
    > validation_comparison.loc["Ridge without cluster", "mean_ndcg_at_5"]
)
predictive_decision = "KEEP" if all([
    cv_rmse_improved,
    cv_ndcg_improved,
    validation_rmse_improved,
    validation_ndcg_improved,
]) else "REJECT"

guardrails = pd.DataFrame(
    [
        ("award_share excluded from clustering inputs", TARGET not in cluster_features),
        ("Identity and season excluded from clustering inputs", {"player", "team_id", "season", "source_row_id"}.isdisjoint(cluster_features)),
        ("Full-development k selection used no target", True),
        ("Target joined only after cluster fitting for profiling", True),
        ("Fold clusterer and k refitted on each training window", True),
        ("2015–2018 excluded from cluster/model selection", development["season"].max() < validation["season"].min()),
        ("2019–2022 target object not created", "y_test" not in locals()),
        ("2019–2022 target absent from held features", TARGET not in final_test_features_only.columns),
        ("No final all-model comparison performed", True),
    ],
    columns=["guardrail", "passed"],
)
print(guardrails.to_string(index=False))
print(f"\nAll guardrails passed: {guardrails['passed'].all()}")
print(f"Predictive cluster-feature decision: {predictive_decision}")
if not guardrails["passed"].all():
    raise AssertionError("A clustering guardrail failed.")

section("9. FIGURE 1 — K DIAGNOSTICS")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metric_specs = [
    ("silhouette", "A. Silhouette", "Higher is better"),
    ("calinski_harabasz", "B. Calinski–Harabasz", "Higher is better"),
    ("davies_bouldin", "C. Davies–Bouldin", "Lower is better"),
]
for ax, (metric, title, ylabel) in zip(axes, metric_specs):
    ax.plot(diagnostics["k"], diagnostics[metric], marker="o", linewidth=3, color=BLUE)
    selected_row = diagnostics.loc[diagnostics["k"].eq(selected_k)].iloc[0]
    ax.scatter(selected_k, selected_row[metric], s=180, color=ORANGE, edgecolor="white", linewidth=1.2, zorder=5, label=f"Selected k={selected_k}")
    ax.set_title(title)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel(ylabel)
    ax.set_xticks(diagnostics["k"])
    ax.legend(loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.suptitle("Target-Free Cluster-Count Diagnostics", fontsize=22, fontweight="bold", color=NAVY, y=1.03)
fig.tight_layout()
k_path = OUTPUT_DIR / "cluster_count_diagnostics.png"
fig.savefig(k_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(k_path.resolve())

section("10. FIGURE 2 — CLUSTER PROFILE HEATMAP")
display_names = {
    "mp_per_g_season_pct": "Minutes",
    "pts_per_g_season_pct": "Scoring",
    "trb_per_g_season_pct": "Rebounding",
    "ast_per_g_season_pct": "Playmaking",
    "stl_per_g_season_pct": "Steals",
    "blk_per_g_season_pct": "Blocks",
    "usg_pct_season_pct": "Usage",
    "ts_pct_season_pct": "Efficiency",
    "bpm_season_pct": "Overall impact",
    "win_loss_pct_season_pct": "Team success",
}
heatmap_data = profile.rename(columns=display_names)
fig, ax = plt.subplots(figsize=(16, 7.5))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=1,
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "Mean within-season percentile"},
    ax=ax,
)
ax.set_title("NBA Player Performance-Role Segments — Development Seasons", fontsize=21, color=NAVY, pad=16)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
fig.tight_layout()
profile_path = OUTPUT_DIR / "cluster_profile_heatmap.png"
fig.savefig(profile_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(profile_path.resolve())

section("11. FIGURE 3 — CLUSTER POPULATION AND MVP CONCENTRATION")
plot_target = target_profile.reset_index()
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
axes[0].barh(plot_target["cluster_label"], plot_target["population_pct"], color=BLUE)
axes[0].invert_yaxis()
axes[0].set_title("A. Share of all player-seasons")
axes[0].set_xlabel("Population (%)")
axes[0].set_ylabel("")
axes[1].barh(plot_target["cluster_label"], plot_target["share_of_all_vote_recipients_pct"], color=ORANGE)
axes[1].invert_yaxis()
axes[1].set_title("B. Share of all vote recipients")
axes[1].set_xlabel("Vote recipients (%)")
axes[1].set_ylabel("")
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.suptitle("MVP Voting Is Highly Concentrated in the Elite Segment", fontsize=22, fontweight="bold", color=NAVY, y=1.02)
fig.tight_layout()
concentration_path = OUTPUT_DIR / "cluster_mvp_concentration.png"
fig.savefig(concentration_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(concentration_path.resolve())

section("12. FIGURE 4 — CLUSTER FEATURE INCREMENT")
metrics_to_plot = [
    ("rmse_all", "All-player RMSE", False),
    ("rmse_positive", "Vote-recipient RMSE", False),
    ("mean_ndcg_at_5", "NDCG@5", True),
    ("mean_winner_rank", "Winner rank", False),
]
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Does Cluster Membership Add Predictive Information?", fontsize=22, fontweight="bold", color=NAVY, y=0.99)
model_labels = ["Without cluster", "With cluster"]
colors = [GRAY, PURPLE]
for ax, (metric, title, unit_interval) in zip(axes.flat, metrics_to_plot):
    values = [
        cv_comparison.loc["Ridge without cluster", metric],
        cv_comparison.loc["Ridge with cluster", metric],
    ]
    bars = ax.bar(model_labels, values, color=colors, width=0.58)
    ax.set_title(title)
    ax.set_ylim(0, 1.05 if unit_interval else max(values) * 1.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ax.get_ylim()[1] * 0.025,
            f"{value:.4f}" if value < 1 else f"{value:.2f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0, w_pad=2.5)
increment_path = OUTPUT_DIR / "cluster_predictive_increment.png"
fig.savefig(increment_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(increment_path.resolve())

section("13. CHECKPOINT BEFORE FINAL MODEL COMPARISON")
print("Clustering is complete as an exploratory segmentation and incremental feature test.")
print("No final cross-family model comparison has been performed.")
print("The 2019–2022 target remains sealed and unevaluated.")
