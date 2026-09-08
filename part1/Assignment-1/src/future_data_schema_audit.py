"""Audit a candidate NBA player-season file before it enters the MVP pipeline.

Examples
--------
python future_data_schema_audit.py candidate.csv
python future_data_schema_audit.py candidate.zip --reference original.zip --season 2022
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


TARGET = "award_share"
IDENTITY_COLUMNS = ["season", "player", "pos", "team_id"]

# Literal raw inputs needed to reproduce the locked Histogram GB pipeline.
MODEL_INPUT_COLUMNS = [
    "season", "player", "pos", "age", "team_id", "g", "gs", "mp_per_g",
    "fga_per_g", "fg_pct", "fg3a_per_g", "fg3_pct", "fg2a_per_g",
    "fg2_pct", "fta_per_g", "ft_pct", "trb_per_g", "ast_per_g",
    "stl_per_g", "blk_per_g", "tov_per_g", "pf_per_g", "pts_per_g", "mp",
    "per", "ts_pct", "fg3a_per_fga_pct", "fta_per_fga_pct", "trb_pct",
    "ast_pct", "stl_pct", "blk_pct", "tov_pct", "usg_pct", "ws",
    "ws_per_48", "bpm", "vorp", "mov", "mov_adj", "win_loss_pct",
]

# The full original schema is checked separately; extra columns are acceptable.
ORIGINAL_COLUMNS = [
    "season", "player", "pos", "age", "team_id", "g", "gs", "mp_per_g",
    "fg_per_g", "fga_per_g", "fg_pct", "fg3_per_g", "fg3a_per_g", "fg3_pct",
    "fg2_per_g", "fg2a_per_g", "fg2_pct", "efg_pct", "ft_per_g", "fta_per_g",
    "ft_pct", "orb_per_g", "drb_per_g", "trb_per_g", "ast_per_g", "stl_per_g",
    "blk_per_g", "tov_per_g", "pf_per_g", "pts_per_g", "mp", "per", "ts_pct",
    "fg3a_per_fga_pct", "fta_per_fga_pct", "orb_pct", "drb_pct", "trb_pct",
    "ast_pct", "stl_pct", "blk_pct", "tov_pct", "usg_pct", "ows", "dws",
    "ws", "ws_per_48", "obpm", "dbpm", "bpm", "vorp", TARGET, "mov",
    "mov_adj", "win_loss_pct",
]

BOUND_ZERO_ONE = [
    "fg_pct", "fg2_pct", "fg3_pct", "ft_pct", "fg3a_per_fga_pct",
    "win_loss_pct", TARGET,
]
BOUND_ZERO_ONE_POINT_FIVE = ["ts_pct", "efg_pct"]
NONNEGATIVE = [
    "age", "g", "gs", "mp_per_g", "mp", "fga_per_g", "fg3a_per_g",
    "fg2a_per_g", "fta_per_g", "trb_per_g", "ast_per_g", "stl_per_g",
    "blk_per_g", "tov_per_g", "pf_per_g", "pts_per_g", "fta_per_fga_pct",
]


def load_table(path: Path) -> tuple[pd.DataFrame, str]:
    """Load a CSV directly or the only/first CSV inside a ZIP archive."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path), path.name
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            csv_members = sorted(
                name for name in archive.namelist() if name.lower().endswith(".csv")
            )
            if not csv_members:
                raise ValueError(f"No CSV file found in {path}")
            member = csv_members[0]
            with archive.open(member) as stream:
                return pd.read_csv(stream), member
    raise ValueError("Candidate must be a .csv or .zip file.")


def numeric_violation_count(series: pd.Series, rule: str) -> int:
    values = pd.to_numeric(series, errors="coerce")
    if rule == "zero_one":
        return int(((values < 0) | (values > 1)).sum())
    if rule == "zero_one_point_five":
        return int(((values < 0) | (values > 1.5)).sum())
    if rule == "nonnegative":
        return int((values < 0).sum())
    raise ValueError(f"Unknown rule: {rule}")


def audit_table(frame: pd.DataFrame, source_label: str) -> dict:
    missing_model = sorted(set(MODEL_INPUT_COLUMNS) - set(frame.columns))
    missing_original = sorted(set(ORIGINAL_COLUMNS) - set(frame.columns))
    extra_original = sorted(set(frame.columns) - set(ORIGINAL_COLUMNS))

    seasons = pd.to_numeric(frame.get("season"), errors="coerce")
    duplicate_key_rows = None
    duplicate_key_groups = None
    duplicate_key_examples = []
    exact_duplicate_rows = int(frame.duplicated(keep=False).sum())
    repeated_player_seasons = None
    if {"season", "player", "team_id"}.issubset(frame.columns):
        duplicate_key_rows = int(
            frame.duplicated(["season", "player", "team_id"], keep=False).sum()
        )
        key_sizes = frame.groupby(
            ["season", "player", "team_id"], dropna=False
        ).size()
        duplicate_key_groups = int(key_sizes.gt(1).sum())
        duplicate_key_examples = [
            {"season": int(key[0]), "player": str(key[1]), "team_id": str(key[2]), "rows": int(size)}
            for key, size in key_sizes[key_sizes.gt(1)].head(10).items()
        ]
    if {"season", "player"}.issubset(frame.columns):
        repeated_player_seasons = int(
            frame.duplicated(["season", "player"], keep=False).sum()
        )

    bounds = {}
    for column in BOUND_ZERO_ONE:
        if column in frame.columns:
            bounds[column] = numeric_violation_count(frame[column], "zero_one")
    for column in BOUND_ZERO_ONE_POINT_FIVE:
        if column in frame.columns:
            bounds[column] = numeric_violation_count(
                frame[column], "zero_one_point_five"
            )
    for column in NONNEGATIVE:
        if column in frame.columns:
            bounds[column] = numeric_violation_count(frame[column], "nonnegative")

    required_missingness = {
        column: round(float(frame[column].isna().mean()), 6)
        for column in MODEL_INPUT_COLUMNS
        if column in frame.columns
    }
    high_missingness = {
        column: rate for column, rate in required_missingness.items() if rate > 0.20
    }

    hard_failures = []
    if missing_model:
        hard_failures.append("missing_locked_model_inputs")
    if any(bounds.values()):
        hard_failures.append("logical_range_violations")
    if high_missingness:
        hard_failures.append("required_column_missingness_above_20pct")

    review_warnings = []
    if duplicate_key_rows:
        review_warnings.append("repeated_season_player_team_keys_require_review")
    if exact_duplicate_rows:
        review_warnings.append("exact_duplicate_rows_require_review")

    if hard_failures:
        candidate_status = "REJECT_OR_REPAIR"
    elif review_warnings:
        candidate_status = "PASS_WITH_REVIEW"
    else:
        candidate_status = "PASS"

    return {
        "source": source_label,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "season_min": None if seasons.isna().all() else int(seasons.min()),
        "season_max": None if seasons.isna().all() else int(seasons.max()),
        "season_count": int(seasons.nunique(dropna=True)),
        "model_input_contract_passed": not missing_model,
        "full_original_schema_present": not missing_original,
        "target_present": TARGET in frame.columns,
        "missing_model_inputs": missing_model,
        "missing_original_columns": missing_original,
        "extra_columns": extra_original,
        "duplicate_season_player_team_rows": duplicate_key_rows,
        "duplicate_season_player_team_groups": duplicate_key_groups,
        "duplicate_key_examples": duplicate_key_examples,
        "exact_duplicate_rows": exact_duplicate_rows,
        "repeated_player_season_rows": repeated_player_seasons,
        "tot_rows": int(frame["team_id"].eq("TOT").sum()) if "team_id" in frame else None,
        "logical_range_violations": {k: v for k, v in bounds.items() if v},
        "required_columns_above_20pct_missing": high_missingness,
        "hard_failures": hard_failures,
        "review_warnings": review_warnings,
        "candidate_status": candidate_status,
    }


def compare_overlap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    season: int,
) -> dict:
    """Compare exact-name numerical columns for one overlapping season."""
    keys = ["season", "player", "team_id"]
    if not set(keys).issubset(candidate.columns) or not set(keys).issubset(reference.columns):
        return {"status": "NOT_COMPARABLE", "reason": "missing overlap keys"}

    left = candidate.loc[pd.to_numeric(candidate["season"], errors="coerce").eq(season)]
    right = reference.loc[pd.to_numeric(reference["season"], errors="coerce").eq(season)]
    common_numeric = sorted(
        (set(left.select_dtypes(include=np.number)) & set(right.select_dtypes(include=np.number)))
        - {"season"}
    )
    merged = left[keys + common_numeric].merge(
        right[keys + common_numeric], on=keys, how="inner", suffixes=("_candidate", "_reference")
    )
    differences = {}
    for column in common_numeric:
        c = merged[f"{column}_candidate"]
        r = merged[f"{column}_reference"]
        comparable = c.notna() & r.notna()
        if comparable.any():
            delta = (c[comparable] - r[comparable]).abs()
            differences[column] = {
                "mean_absolute_difference": round(float(delta.mean()), 10),
                "maximum_absolute_difference": round(float(delta.max()), 10),
            }

    materially_different = [
        column for column, result in differences.items()
        if result["maximum_absolute_difference"] > 1e-8
    ]
    return {
        "status": "PASS" if merged.shape[0] and not materially_different else "REVIEW",
        "season": season,
        "candidate_rows": int(len(left)),
        "reference_rows": int(len(right)),
        "matched_rows": int(len(merged)),
        "common_numeric_columns": len(common_numeric),
        "materially_different_columns": materially_different,
        "differences": differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--season", type=int, default=2022)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidate, candidate_member = load_table(args.candidate)
    report = {"candidate_audit": audit_table(candidate, candidate_member)}
    if args.reference:
        reference, reference_member = load_table(args.reference)
        report["reference_source"] = reference_member
        report["overlap_comparison"] = compare_overlap(
            candidate, reference, args.season
        )

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
