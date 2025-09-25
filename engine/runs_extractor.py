# runs_extractor.py
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pathlib import Path


def extract_mlflow_runs(output_path="data/mlflow_runs.parquet"):
    """Extrae todos los runs de MLflow y los guarda en parquet"""
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()

    experiment = client.get_experiment_by_name("PHIBOT_TRAINING")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=10000,  # Suficiente para tus 4k runs
        order_by=["start_time DESC"]
    )

    data = []
    for run in runs:
        row = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "start_time": run.info.start_time,
            "status": run.info.status,
        }
        # Parámetros y métricas
        row.update(run.data.tags)  # Tags incluyen ticker, model, etc.
        row.update(run.data.params)  # Parámetros
        row.update(run.data.metrics)  # Métricas

        data.append(row)

    df = pd.DataFrame(data)
    return df

    # Convertir tipos de datos
    numeric_cols = ['threshold', 'tp_multiplier', 'sl_multiplier', 'sharpe', 'net_return', 'n_trades']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Guardar
    Path(output_path).parent.mkdir(exist_ok=True)
    df.to_parquet(output_path)

    print(f"Extraídos {len(df)} runs")
    return df


if __name__ == "__main__":
    extract_mlflow_runs()