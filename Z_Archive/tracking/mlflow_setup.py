import os

import mlflow

EXPERIMENT = "PIBOT_Intradia"  # único para todo el universo


def setup_mlflow(experiment: str = EXPERIMENT, tracking_uri: str | None = None):
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
