import os

import mlflow
import optuna
from optuna.integration import MLflowCallback
from tracking.mlflow_setup import setup_mlflow

from experiments.evaluate import evaluate_model_for_ticker


def optimize_ticker(ticker: str, n_trials: int = 40):
    setup_mlflow()
    mlflc = MLflowCallback(metric_name="sharpe")  # optimo por Sharpe; cámbialo si prefieres

    def objective(trial: optuna.trial.Trial):
        params = {
            "threshold": trial.suggest_float("threshold", 0.55, 0.9, step=0.05),
            "features_set": trial.suggest_categorical("features_set", ["base", "tech", "all"]),
            "window_main": trial.suggest_int("window_main", 20, 160, step=10),
            "window_alt": trial.suggest_int("window_alt", 5, 60, step=5),
        }
        with mlflow.start_run(nested=True):
            res = evaluate_model_for_ticker(ticker, params)
            # Log en el trial
            mlflow.log_params({"ticker": ticker, **params})
            for k, v in res.metrics.items():
                mlflow.log_metric(k, float(v))
            # artefactos mínimos
            if "equity" in res.artifacts:
                res.artifacts["equity"].to_csv("equity.csv")
                mlflow.log_artifact("equity.csv")
        return res.metrics.get("sharpe", -1e9)

    study = optuna.create_study(direction="maximize", study_name=f"PIBOT_{ticker}_opt")
    study.optimize(objective, n_trials=n_trials, callbacks=[mlflc])
    return study


if __name__ == "__main__":
    ticker = os.environ.get("TICKER", "AAPL")
    optimize_ticker(ticker, n_trials=int(os.environ.get("N_TRIALS", "40")))
