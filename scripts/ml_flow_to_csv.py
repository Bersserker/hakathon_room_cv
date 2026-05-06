import mlflow
import pandas as pd
from pathlib import Path

# подключаемся ТОЛЬКО к sqlite
mlflow.set_tracking_uri(
    "sqlite:///artifacts/logs/mlflow.db"
)

# получаем все эксперименты
experiments = mlflow.search_experiments()

print("\nExperiments found:")
for exp in experiments:
    print(f"{exp.experiment_id}: {exp.name}")

if not experiments:
    raise RuntimeError("No experiments found in mlflow.db")

# все experiment ids
experiment_ids = [
    exp.experiment_id
    for exp in experiments
]

# вытаскиваем runs
runs = mlflow.search_runs(
    experiment_ids=experiment_ids,
    output_format="pandas"
)

print("\nRuns shape:", runs.shape)

# создаем папку
Path("reports").mkdir(exist_ok=True)

# сохраняем
out_path = "reports/mlflow_runs.csv"

runs.to_csv(out_path, index=False)

print(f"\nSaved to: {out_path}")

# leaderboard
cols = [
    c for c in runs.columns
    if (
        c.startswith("metrics.")
        or c.startswith("params.")
        or c == "run_id"
    )
]

print("\nLeaderboard preview:")
print(runs[cols].head())