# MLflow setup

The project logs training runs with MLflow. Dependencies are installed by the standard setup command:

```bash
make setup
```

## Tracking store

The default config uses a local SQLite backend:

```yaml
mlflow:
  tracking_uri: sqlite:///artifacts/logs/mlflow.db
  experiment_name: room_type_image_baseline
  artifact_root: artifacts/logs/mlruns
```

The same locations are centralized under `artifacts.roots`:

```yaml
artifacts:
  roots:
    logs: artifacts/logs
    mlflow: artifacts/logs/mlruns
```

`tracking_uri` stores run metadata and metrics. `artifact_root` stores run artifacts such as checkpoints and prediction files.

## Start the UI

```bash
make mlflow-ui
```

Then open the URL printed by MLflow, usually `http://127.0.0.1:5000`.

## Run a smoke training job

```bash
make smoke-train
```

This uses the normal training entrypoint and the same OmegaConf-based config loader as full training, but passes `--debug` to reduce epochs, samples and workers.

## Troubleshooting

- If the SQLite database directory does not exist, training creates it under `artifacts/logs`.
- If old runs point to moved artifact locations, the training entrypoint normalizes local experiment artifact URIs before logging.
- To reset local tracking state, stop the UI and remove `artifacts/logs/mlflow.db` plus `artifacts/logs/mlruns`.
