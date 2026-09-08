"""Reproduce the final NBA MVP evaluation and production artifact."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def run(script: Path, env: dict[str, str], arguments: list[str] | None = None) -> None:
    command = [sys.executable, str(script), *(arguments or [])]
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/nba_mvp_stats.zip"),
        help="Path to the Kaggle CSV or ZIP (ZIP expected by final model scripts).",
    )
    parser.add_argument(
        "--skip-deployment",
        action="store_true",
        help="Run the locked evaluation without refitting the production artifact.",
    )
    args = parser.parse_args()

    repository = Path(__file__).resolve().parent
    data = args.data.expanduser().resolve()
    if not data.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data}. See DATA_SOURCE.md for download instructions."
        )
    if data.suffix.lower() != ".zip":
        raise ValueError("The final evaluation and deployment scripts expect a ZIP containing one CSV.")

    results = repository / "results"
    deployment = results / "deployment"
    model = repository / "models" / "nba_mvp_production_models_through_2022.joblib"
    results.mkdir(parents=True, exist_ok=True)
    deployment.mkdir(parents=True, exist_ok=True)
    model.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "NBA_MVP_DATA_PATH": str(data),
            "NBA_MVP_RESULTS_DIR": str(results),
            "NBA_MVP_DEPLOYMENT_DIR": str(deployment),
            "NBA_MVP_MODEL_PATH": str(model),
        }
    )

    run(
        repository / "src" / "future_data_schema_audit.py",
        env,
        [str(data), "--output", str(results / "source_schema_audit.json")],
    )
    run(repository / "src" / "locked_test_evaluation.py", env)
    if not args.skip_deployment:
        run(repository / "src" / "deployment_readiness.py", env)

    print("\nReproduction completed successfully.")
    print(f"Results: {results}")
    if not args.skip_deployment:
        print(f"Model:   {model}")


if __name__ == "__main__":
    main()
