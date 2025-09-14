import os
import subprocess

import mlflow
from tracking.mlflow_setup import setup_mlflow

from experiments.evaluate import evaluate_model_for_ticker


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def run_once_mlflow(ticker: str, params: dict, tags: dict):
    setup_mlflow()
    with mlflow.start_run(run_name=f"{ticker}") as run:
        # Params & tags
        mlflow.log_params({"ticker": ticker, **params})
        mlflow.set_tags({**tags, "ticker": ticker, "commit_sha": git_sha(), "universe": "SP500"})

        # Eval
        result = evaluate_model_for_ticker(ticker, params)

        # Metrics
        for k, v in result.metrics.items():
            mlflow.log_metric(k, float(v))

        # Artifacts
        arts = result.artifacts
        if "equity" in arts:
            eq = arts["equity"]
            eq.to_csv("equity.csv")
            mlflow.log_artifact("equity.csv")
        if "trades" in arts:
            tr = arts["trades"]
            tr.to_csv("trades.csv", index=False)
            mlflow.log_artifact("trades.csv")
        if "figs" in arts:
            for name, path in arts["figs"].items():
                mlflow.log_artifact(path)

        # Modelo (si es sklearn / xgb)
        if result.model is not None:
            try:
                import mlflow.sklearn as mls

                mls.log_model(result.model, artifact_path="model")
            except Exception:
                pass


if __name__ == "__main__":
    # Ejemplo de grid mínimo
    ticker = os.environ.get("TICKER", "AAPL")
    base = dict(threshold=0.8, features_set="all", window_main=60)
    run_once_mlflow(
        ticker, base, tags={"freq": "10min", "label_spec": ">0.7%/-0.7%", "rules": "TP/SL=0.5%,EOD"}
    )
