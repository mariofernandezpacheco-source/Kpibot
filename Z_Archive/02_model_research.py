import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

# Modelos
from xgboost import XGBClassifier

# Config (settings.py en la raíz)
from settings import S

# Utils propios
from utils.A_data_loader import load_data
from utils.B_feature_engineering import add_technical_indicators
from utils.C_label_generator import generate_triple_barrier_labels
from utils.D_backtest_utils import backtest_signals

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from utils.reproducibility import (
    set_global_determinism,
    write_env_versions,
    xgb_deterministic_params,
)

# ---------------------------
# Determinismo global + versiones
# ---------------------------
set_global_determinism(S.seed, set_pythonhash=S.pythonhashseed)
if S.record_versions:
    write_env_versions(S.env_versions_path)

# === Parámetros centralizados ===
N_SPLITS_CV = S.n_splits_cv
THRESHOLD = S.threshold_default
LABEL_WINDOW = S.label_window
DAYS_OF_DATA = S.days_of_data
TIME_LIMIT = S.time_limit_candles

OUTPUT_CONFIG_FILE = S.models_path / "best_models_config.json"


def get_tickers_from_file(file_path: Path) -> list:
    if not file_path.exists():
        raise FileNotFoundError(f"El fichero de tickers no se encontró en: {file_path}")
    with open(file_path) as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"Encontrados {len(tickers)} tickers en {file_path.name}.")
    return tickers


def ensure_atr14(df: pd.DataFrame, *, high="high", low="low", close="close") -> pd.DataFrame:
    """
    Garantiza la columna 'atr_14'. Si no existe, la calcula con TR de Wilder y ATR(14) como EMA(alpha=1/14).
    No depende de librerías externas y es determinista.
    """
    if "atr_14" in df.columns:
        return df.copy()

    needed = {high, low, close}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"No puedo calcular atr_14: faltan columnas {missing}")

    out = df.copy()
    # ordenar temporalmente
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
        out.sort_values("date", inplace=True)
    else:
        out.sort_index(inplace=True)

    prev_close = out[close].shift(1)
    tr1 = (out[high] - out[low]).abs()
    tr2 = (out[high] - prev_close).abs()
    tr3 = (out[low] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["atr_14"] = atr14
    return out


def get_models_to_evaluate():
    xgb_params = dict(
        objective="multi:softprob",
        eval_metric="mlogloss",
        **xgb_deterministic_params(
            seed=S.seed, n_jobs=S.n_jobs_train if not S.deterministic else 1
        ),
    )
    return {
        "XGBoost": XGBClassifier(**xgb_params),
        # Si añades LightGBM:
        # "LightGBM": LGBMClassifier(**lgbm_deterministic_params(seed=S.seed, n_jobs=1 if S.deterministic else S.n_jobs_train),
        #                            objective="multiclass", num_class=3)
    }


def evaluate_model_for_ticker(ticker, timeframe, model_name, model_instance):
    data_folder_path = S.data_path
    try:
        df = load_data(
            ticker=ticker, timeframe=timeframe, use_local=True, base_path=data_folder_path
        )
        df["date"] = pd.to_datetime(df["date"], utc=True)
        # recorte del histórico usado en la investigación
        df = df[df["date"] >= (df["date"].max() - pd.Timedelta(days=DAYS_OF_DATA))]

        # asegurar volatilidad requerida por el etiquetador
        df = ensure_atr14(df)

        # preparar kwargs para pasar label_window SOLO si la función lo acepta
        label_kwargs = {}
        if "label_window" in generate_triple_barrier_labels.__code__.co_varnames:
            label_kwargs["label_window"] = LABEL_WINDOW

        # etiquetas por triple barrera con parámetros desde config_
        df = generate_triple_barrier_labels(
            data=df,
            volatility_col="atr_14",
            tp_multiplier=S.tp_multiplier,
            sl_multiplier=S.sl_multiplier,
            time_limit_candles=TIME_LIMIT,
            **label_kwargs,
        )

        # features
        df = add_technical_indicators(df)
        df = df.dropna()

        features = [
            col
            for col in df.columns
            if col not in ["date", "label", "open", "high", "low", "close", "volume"]
        ]
        X, y = df[features], df["label"]

        if y.nunique() < 2:
            return {"sharpe_ratio": -999}
    except FileNotFoundError:
        return {"sharpe_ratio": -999}

    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    all_metrics = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        mdl = clone(model_instance)
        # XGBoost multiclass usa {0,1,2}; mapeamos si y viene como {-1,0,1}
        mdl.fit(X_train, y_train.map({-1: 0, 0: 1, 1: 2}))

        probs = mdl.predict_proba(X_val)
        class_map = {c: i for i, c in enumerate(mdl.classes_)}
        prob_buy = probs[:, class_map.get(2, -1)] if 2 in class_map else np.zeros(len(probs))
        prob_sell = probs[:, class_map.get(0, -1)] if 0 in class_map else np.zeros(len(probs))

        signals = np.select([prob_buy > THRESHOLD, prob_sell > THRESHOLD], [1, -1], default=0)

        df_val_signals = pd.DataFrame(
            {
                "timestamp": df["date"].iloc[val_idx],
                "close": df["close"].iloc[val_idx],
                "signal": signals,
            }
        )
        metrics, _ = backtest_signals(df_val_signals)
        all_metrics.append(metrics)

    avg_metrics = pd.DataFrame(all_metrics).mean().to_dict()
    print(f"  - {model_name}: Sharpe Avg={avg_metrics.get('sharpe_ratio', 0):.2f}")
    return avg_metrics


def main(tickers_file, timeframe):
    tickers_filepath = (
        Path("../utils") / tickers_file
    )  # si quieres, también lo pasamos a config_.yaml
    models_to_evaluate = get_models_to_evaluate()
    best_models_config = {}

    OUTPUT_CONFIG_FILE.parent.mkdir(exist_ok=True)

    try:
        tickers = get_tickers_from_file(tickers_filepath)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    for ticker in tqdm(tickers, desc="Investigando Tickers"):
        print(f"\n🔎 Analizando ticker: {ticker} ({timeframe})")
        ticker_results = []
        for model_name, model_instance in models_to_evaluate.items():
            performance = evaluate_model_for_ticker(ticker, timeframe, model_name, model_instance)
            ticker_results.append({"model_name": model_name, **performance})

        results_df = pd.DataFrame(ticker_results)
        best_model = results_df.sort_values(by="sharpe_ratio", ascending=False).iloc[0]
        if best_model["sharpe_ratio"] > -999:
            best_models_config[ticker] = best_model["model_name"]
            print(f"🏆 Mejor modelo para {ticker}: {best_model['model_name']}")

    output_filename = f"best_models_config_{timeframe}.json"
    output_path = OUTPUT_CONFIG_FILE.parent / output_filename
    print(f"\n💾 Guardando configuración en {output_path}")
    with open(output_path, "w") as f:
        json.dump(best_models_config, f, indent=4)

    print("\n✅ Proceso de investigación completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Investigación de modelos de trading.")
    parser.add_argument(
        "--tickers_file",
        type=str,
        required=True,
        help="Fichero .txt con la lista de tickers (ej. 'sp500_tickers.txt').",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        required=True,
        choices=["5mins", "10mins"],
        help="Timeframe de velas.",
    )
    args = parser.parse_args()
    main(args.tickers_file, args.timeframe)
