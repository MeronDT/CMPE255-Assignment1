import os
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)
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
        "legend.frameon": False,
        "grid.color": LIGHT_GRAY,
        "grid.linewidth": 0.8,
    },
)

ZIP_PATH = Path(os.environ.get("NBA_MVP_DATA_PATH", "data/raw/nba_mvp_stats.zip"))
OUTPUT_DIR = Path(os.environ.get("NBA_MVP_CHUNK_OUTPUT_ROOT", "results/analysis_chunks")) / "chunk7_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 102}\n{title}\n{'=' * 102}")


def connected_components(features: list[str], correlation: pd.DataFrame, threshold: float) -> list[list[str]]:
    adjacency = {feature: set() for feature in features}
    for i, left in enumerate(features):
        for right in features[i + 1:]:
            if abs(correlation.loc[left, right]) >= threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)

    visited = set()
    components = []
    for feature in features:
        if feature in visited:
            continue
        stack = [feature]
        component = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)
        if len(component) > 1:
            components.append(sorted(component))
    return sorted(components, key=lambda values: (-len(values), values))


with ZipFile(ZIP_PATH) as archive:
    csv_members = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and not name.endswith("/")
    ]
    if len(csv_members) != 1:
        raise ValueError(f"Expected exactly one CSV; found {csv_members}")
    with archive.open(csv_members[0]) as csv_file:
        df = pd.read_csv(csv_file)

# Descriptive EDA view; no source rows or columns are altered.
eda = df.loc[df["mp"] >= 100].copy()
eda.loc[eda["team_id"].eq("TOT"), ["mov", "mov_adj", "win_loss_pct"]] = np.nan

excluded_numeric = {"season", "award_share"}
numeric_features = [
    column for column in eda.select_dtypes(include=np.number).columns
    if column not in excluded_numeric
]

section("1. FEATURE SET FOR REDUNDANCY ANALYSIS")
print(f"EDA rows: {len(eda):,}")
print(f"Numeric predictors examined: {len(numeric_features):,}")
print(numeric_features)

section("2. HIGHLY CORRELATED FEATURE PAIRS")
spearman_matrix = eda[numeric_features].corr(method="spearman")
pair_rows = []
for i, left in enumerate(numeric_features):
    for right in numeric_features[i + 1:]:
        rho = spearman_matrix.loc[left, right]
        pair_rows.append(
            {
                "feature_1": left,
                "feature_2": right,
                "spearman_rho": rho,
                "absolute_rho": abs(rho),
            }
        )

pair_table = pd.DataFrame(pair_rows).sort_values("absolute_rho", ascending=False)
print("Top 40 predictor-predictor Spearman correlations:")
print(pair_table.head(40).to_string(index=False))
print(f"\nPairs with |rho| >= 0.95: {(pair_table['absolute_rho'] >= 0.95).sum():,}")
print(f"Pairs with |rho| >= 0.90: {(pair_table['absolute_rho'] >= 0.90).sum():,}")
print(f"Pairs with |rho| >= 0.80: {(pair_table['absolute_rho'] >= 0.80).sum():,}")

section("3. CORRELATION-BASED FEATURE FAMILIES")
clusters_085 = connected_components(numeric_features, spearman_matrix, threshold=0.85)
target_association = {
    feature: eda[feature].corr(eda["award_share"], method="spearman")
    for feature in numeric_features
}

cluster_rows = []
for cluster_number, cluster in enumerate(clusters_085, start=1):
    representative = max(cluster, key=lambda feature: abs(target_association[feature]))
    cluster_rows.append(
        {
            "cluster": cluster_number,
            "size": len(cluster),
            "features": ", ".join(cluster),
            "strongest_target_association": representative,
            "representative_abs_target_rho": abs(target_association[representative]),
        }
    )

cluster_table = pd.DataFrame(cluster_rows)
print("Connected feature families using |rho| >= 0.85:")
print(cluster_table.to_string(index=False))

section("4. KNOWN COMPONENT IDENTITIES")
identity_specs = {
    "trb_per_g = orb_per_g + drb_per_g": (
        eda["trb_per_g"] - (eda["orb_per_g"] + eda["drb_per_g"])
    ),
    "fg_per_g = fg2_per_g + fg3_per_g": (
        eda["fg_per_g"] - (eda["fg2_per_g"] + eda["fg3_per_g"])
    ),
    "pts_per_g = 2*fg2_per_g + 3*fg3_per_g + ft_per_g": (
        eda["pts_per_g"]
        - (2 * eda["fg2_per_g"] + 3 * eda["fg3_per_g"] + eda["ft_per_g"])
    ),
    "ws = ows + dws": eda["ws"] - (eda["ows"] + eda["dws"]),
    "bpm = obpm + dbpm": eda["bpm"] - (eda["obpm"] + eda["dbpm"]),
}
identity_rows = []
for relationship, residual in identity_specs.items():
    identity_rows.append(
        {
            "relationship": relationship,
            "mean_absolute_residual": residual.abs().mean(),
            "maximum_absolute_residual": residual.abs().max(),
        }
    )
identity_table = pd.DataFrame(identity_rows)
print(identity_table.to_string(index=False))

section("5. VARIANCE INFLATION FACTORS FOR A REPRESENTATIVE FEATURE PANEL")
vif_features = [
    "g", "gs", "mp", "mp_per_g",
    "fg_per_g", "fga_per_g", "fg_pct",
    "fg3_per_g", "fg3a_per_g", "fg3_pct",
    "fg2_per_g", "fg2a_per_g", "fg2_pct",
    "ft_per_g", "fta_per_g", "ft_pct",
    "orb_per_g", "drb_per_g", "trb_per_g",
    "ast_per_g", "stl_per_g", "blk_per_g", "tov_per_g", "pts_per_g",
    "per", "ts_pct", "usg_pct", "ows", "dws", "ws", "ws_per_48",
    "obpm", "dbpm", "bpm", "vorp", "mov", "mov_adj", "win_loss_pct",
]

imputed = SimpleImputer(strategy="median").fit_transform(eda[vif_features])
scaled = StandardScaler().fit_transform(imputed)
vif_rows = []
for index, feature in enumerate(vif_features):
    other_columns = np.delete(scaled, index, axis=1)
    target_column = scaled[:, index]
    model = LinearRegression().fit(other_columns, target_column)
    r_squared = model.score(other_columns, target_column)
    vif = np.inf if r_squared >= 0.999999 else 1 / (1 - r_squared)
    vif_rows.append(
        {
            "feature": feature,
            "r_squared_from_other_predictors": r_squared,
            "vif": vif,
            "abs_target_spearman": abs(target_association[feature]),
        }
    )

vif_table = pd.DataFrame(vif_rows).sort_values("vif", ascending=False)
print("Top 25 VIF values:")
print(vif_table.head(25).to_string(index=False))
print(f"\nFeatures with VIF >= 10: {(vif_table['vif'] >= 10).sum():,}")
print(f"Features with VIF >= 5:  {(vif_table['vif'] >= 5).sum():,}")

section("6. PROFESSIONAL EDA FIGURES")
# Figure 1: audience-readable correlation heatmap using representative features.
heatmap_features = [
    "g", "gs", "mp", "mp_per_g",
    "pts_per_g", "fga_per_g", "fta_per_g", "ts_pct", "usg_pct",
    "trb_per_g", "ast_per_g", "stl_per_g", "blk_per_g", "tov_per_g",
    "per", "ows", "dws", "ws", "ws_per_48",
    "obpm", "dbpm", "bpm", "vorp",
    "mov", "mov_adj", "win_loss_pct",
]
heatmap_corr = eda[heatmap_features].corr(method="spearman")
mask = np.triu(np.ones_like(heatmap_corr, dtype=bool), k=1)
diverging = LinearSegmentedColormap.from_list(
    "custom_diverging", ["#B04A5A", "#F7F8FA", "#2F6B9A"], N=256
)

fig, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(
    heatmap_corr,
    mask=mask,
    cmap=diverging,
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.35,
    linecolor="white",
    cbar_kws={"label": "Spearman correlation", "shrink": 0.75},
    ax=ax,
)
ax.set_title("Correlation blocks reveal participation, scoring, impact, and team-stat families", loc="left", pad=18)
ax.tick_params(axis="x", rotation=55, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=9)
fig.suptitle(
    "Many NBA predictors describe overlapping information",
    fontsize=21,
    fontweight="bold",
    color=NAVY,
    x=0.07,
    y=0.995,
    ha="left",
)
fig.text(
    0.07,
    0.94,
    "Lower-triangle Spearman correlations for a representative feature panel; award_share is not included.",
    fontsize=10.5,
    color=GRAY,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
heatmap_path = OUTPUT_DIR / "predictor_correlation_heatmap.png"
fig.savefig(heatmap_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {heatmap_path}")

# Figure 2: strongest pairwise relationships.
top_pairs = pair_table.head(20).sort_values("absolute_rho")
pair_labels = top_pairs["feature_1"] + " ↔ " + top_pairs["feature_2"]
fig, ax = plt.subplots(figsize=(12.5, 8.5))
colors = [ORANGE if value >= 0.95 else BLUE for value in top_pairs["absolute_rho"]]
bars = ax.barh(pair_labels, top_pairs["absolute_rho"], color=colors, height=0.68)
for bar, value in zip(bars, top_pairs["absolute_rho"]):
    ax.text(
        bar.get_width() + 0.004,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        fontsize=9.5,
        color=NAVY,
    )
ax.set_xlim(0.80, 1.025)
ax.set_xlabel("Absolute Spearman correlation")
ax.set_ylabel("")
ax.set_title("The strongest relationships are nearly interchangeable statistically", loc="left", pad=18)
fig.suptitle(
    "Top predictor-predictor correlations",
    fontsize=21,
    fontweight="bold",
    color=NAVY,
    x=0.08,
    y=0.995,
    ha="left",
)
fig.text(
    0.08,
    0.93,
    "Orange bars mark |rho| ≥ 0.95. High correlation does not by itself determine which feature should be retained.",
    fontsize=10.5,
    color=GRAY,
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
pairs_path = OUTPUT_DIR / "strongest_predictor_correlations.png"
fig.savefig(pairs_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {pairs_path}")

# Figure 3: VIF ranking, capped only for display.
vif_plot = vif_table.head(18).sort_values("vif").copy()
finite_values = vif_plot.loc[np.isfinite(vif_plot["vif"]), "vif"]
display_cap = max(100.0, finite_values.max() * 1.15 if len(finite_values) else 100.0)
vif_plot["display_vif"] = vif_plot["vif"].replace(np.inf, display_cap)

fig, ax = plt.subplots(figsize=(11.5, 8))
bar_colors = [ORANGE if value >= 10 else BLUE for value in vif_plot["vif"]]
bars = ax.barh(vif_plot["feature"], vif_plot["display_vif"], color=bar_colors, height=0.68)
for bar, value in zip(bars, vif_plot["vif"]):
    label = "∞" if np.isinf(value) else f"{value:.1f}"
    ax.text(
        bar.get_width() * 1.02,
        bar.get_y() + bar.get_height() / 2,
        label,
        va="center",
        fontsize=9.5,
        color=NAVY,
    )
ax.axvline(5, color=GRAY, linestyle=":", linewidth=1.5, label="VIF = 5")
ax.axvline(10, color=NAVY, linestyle="--", linewidth=1.5, label="VIF = 10")
ax.set_xscale("log")
ax.set_xlabel("Variance inflation factor (log scale)")
ax.set_ylabel("")
ax.set_title("Linear models would face severe coefficient instability without feature control", loc="left", pad=18)
ax.legend(loc="lower right")
fig.suptitle(
    "Multicollinearity diagnostic",
    fontsize=21,
    fontweight="bold",
    color=NAVY,
    x=0.08,
    y=0.995,
    ha="left",
)
fig.text(
    0.08,
    0.93,
    "VIF measures how well each predictor can be reconstructed from the others after median imputation and scaling.",
    fontsize=10.5,
    color=GRAY,
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
vif_path = OUTPUT_DIR / "variance_inflation_factors.png"
fig.savefig(vif_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {vif_path}")

print("\nChunk 7 EDA completed without dropping features or beginning Data Preparation.")
