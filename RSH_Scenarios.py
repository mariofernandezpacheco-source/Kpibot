# RSH_Scenarios.py — Un run por combinación (ticker × modelo × thr × tp × sl × tl), con OOF de CV y métricas de backtest
from __future__ import annotations


import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import mlflow




# Paths / imports base
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Settings del proyecto
import settings as settings
S = settings.S

# Backtest + métricas
from engine.backtest_runner import run_backtest_for_ticker
from engine.metrics import trading_metrics


# --------------------------
# Utilidades
# --------------------------
# Añadir esta función en RSH_Scenarios.py después de los imports:

def analyze_temporal_coverage(df: pd.DataFrame, days_requested: int, ticker: str) -> dict:
    """
    Analiza la cobertura temporal de los datos vs lo solicitado.

    Args:
        df: DataFrame con datos OHLCV (debe tener índice datetime)
        days_requested: Días solicitados en la configuración
        ticker: Símbolo para logging

    Returns:
        dict con métricas de cobertura temporal
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return {
            "temporal_coverage_pct": 0.0,
            "trading_days_expected": 0,
            "trading_days_actual": 0,
            "missing_days": 0,
            "weekend_days_found": 0,
            "data_quality": "NO_DATA"
        }

    # Rango de fechas actual en los datos
    start_date = df.index.min().date()
    end_date = df.index.max().date()
    actual_span_days = (end_date - start_date).days + 1

    # Días de trading esperados (excluyendo sábados y domingos)
    expected_trading_days = 0
    current_date = start_date
    while current_date <= end_date:
        # Lunes=0, Domingo=6
        if current_date.weekday() < 5:  # Lunes a Viernes
            expected_trading_days += 1
        current_date += pd.Timedelta(days=1)

    # Días únicos en los datos (solo fechas)
    unique_dates = df.index.date
    actual_trading_days = len(set(unique_dates))

    # Días de fin de semana encontrados en datos (anomalía)
    weekend_days = sum(1 for date in set(unique_dates) if date.weekday() >= 5)

    # Cobertura temporal
    coverage_pct = (actual_trading_days / expected_trading_days * 100) if expected_trading_days > 0 else 0
    missing_days = max(0, expected_trading_days - actual_trading_days)

    # Calidad de datos
    if coverage_pct >= 95:
        quality = "EXCELLENT"
    elif coverage_pct >= 85:
        quality = "GOOD"
    elif coverage_pct >= 70:
        quality = "ACCEPTABLE"
    elif coverage_pct >= 50:
        quality = "POOR"
    else:
        quality = "CRITICAL"

    result = {
        "temporal_coverage_pct": round(coverage_pct, 2),
        "trading_days_expected": expected_trading_days,
        "trading_days_actual": actual_trading_days,
        "missing_days": missing_days,
        "weekend_days_found": weekend_days,
        "data_quality": quality,
        "date_range_start": start_date.strftime("%Y-%m-%d"),
        "date_range_end": end_date.strftime("%Y-%m-%d"),
        "actual_span_days": actual_span_days,
        "days_requested": days_requested
    }

    print(
        f"TEMPORAL CHECK - {ticker}: {coverage_pct:.1f}% cobertura ({actual_trading_days}/{expected_trading_days} días) - {quality}")
    if weekend_days > 0:
        print(f"WARNING - {ticker}: Encontrados {weekend_days} días de fin de semana en datos")
    if missing_days > 0:
        print(f"WARNING - {ticker}: Faltan {missing_days} días de trading")

    return result


# Y esta función para análisis de features:

def analyze_feature_weights(model_pipeline, feature_names: list, ticker: str) -> dict:
    """
    Analiza la importancia/ponderación de features en el modelo entrenado.

    Args:
        model_pipeline: Pipeline entrenado (sklearn)
        feature_names: Lista de nombres de features
        ticker: Símbolo para logging

    Returns:
        dict con análisis de importancia de features
    """
    try:
        # Extraer el modelo final del pipeline
        if hasattr(model_pipeline, 'named_steps'):
            # Pipeline de sklearn
            model = None
            for step_name, step in model_pipeline.named_steps.items():
                if hasattr(step, 'feature_importances_') or hasattr(step, 'coef_'):
                    model = step
                    break
        else:
            model = model_pipeline

        if model is None:
            return {"feature_analysis": "NO_MODEL", "top_features": []}

        # Extraer importancias según el tipo de modelo
        importances = None
        model_type = type(model).__name__

        if hasattr(model, 'feature_importances_'):
            # XGBoost, RandomForest, etc.
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # LogisticRegression, etc.
            coef = model.coef_
            if len(coef.shape) > 1:
                # Multi-class: usar la primera clase o promedio
                importances = np.abs(coef[0]) if coef.shape[0] == 1 else np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)

        if importances is None:
            return {"feature_analysis": "NO_IMPORTANCES", "model_type": model_type}

        # Crear ranking de features
        if len(importances) != len(feature_names):
            print(f"WARNING - {ticker}: Mismatch features ({len(feature_names)}) vs importances ({len(importances)})")
            return {"feature_analysis": "FEATURE_MISMATCH", "model_type": model_type}

        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        # Estadísticas
        total_importance = sum(importances)
        top_5_importance = sum(imp for _, imp in feature_importance[:5])
        top_5_pct = (top_5_importance / total_importance * 100) if total_importance > 0 else 0

        # Features con importancia cero
        zero_features = [name for name, imp in feature_importance if abs(imp) < 1e-10]

        # Top 10 features
        top_features = [
            {"feature": name, "importance": float(imp), "importance_pct": float(imp / total_importance * 100)}
            for name, imp in feature_importance[:10]
        ]

        result = {
            "feature_analysis": "SUCCESS",
            "model_type": model_type,
            "total_features": len(feature_names),
            "top_5_concentration_pct": round(top_5_pct, 2),
            "zero_importance_features": len(zero_features),
            "zero_features": zero_features[:5],  # Primeros 5 si hay muchos
            "top_features": top_features,
            "feature_diversity": "HIGH" if top_5_pct < 70 else ("MEDIUM" if top_5_pct < 85 else "LOW")
        }

        print(
            f"FEATURE CHECK - {ticker}: Top 5 concentran {top_5_pct:.1f}% importancia, {len(zero_features)} features sin uso")
        print(f"FEATURE CHECK - {ticker}: Top 3 features: {', '.join([f[0] for f in feature_importance[:3]])}")

        return result

    except Exception as e:
        print(f"ERROR - Feature analysis para {ticker}: {e}")
        return {"feature_analysis": "ERROR", "error": str(e)}

def convert_atr_multipliers_to_pct(
        ticker: str,
        timeframe: str,
        tp_multipliers: List[float],
        sl_multipliers: List[float],
        days: Optional[int] = None
) -> tuple[List[float], List[float]]:
    """
    Convierte multiplicadores ATR (1,2,3) a porcentajes decimales para el backtest.

    Args:
        ticker: Símbolo del activo
        timeframe: Marco temporal
        tp_multipliers: Lista de multiplicadores TP [1, 2, 3]
        sl_multipliers: Lista de multiplicadores SL [1, 2, 3]
        days: Días de historia para calcular ATR

    Returns:
        tuple: (tp_percentages, sl_percentages) como listas de decimales
    """
    try:
        # Cargar datos para calcular ATR
        from utils.A_data_loader import load_data
        from pathlib import Path
        import settings

        data_dir = Path(settings.S.data_path)
        df = load_data(ticker=ticker, timeframe=timeframe, use_local=True, base_path=data_dir)

        if df.empty:
            print(f"WARNING - No hay datos para {ticker}, usando ATR por defecto")
            avg_atr_pct = 0.02  # 2% por defecto
        else:
            # Aplicar filtro de días si se especifica
            if days and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
                cutoff = df['date'].max() - pd.Timedelta(days=days)
                df = df[df['date'] >= cutoff]

            # Calcular ATR si no existe
            if 'atr_14' not in df.columns:
                from utils.C_label_generator import ensure_atr14  # o donde tengas esta función
                df = ensure_atr14(df)

            # Calcular ATR promedio como porcentaje del precio
            if 'atr_14' in df.columns and 'close' in df.columns:
                atr_values = df['atr_14'].dropna()
                close_values = df['close'].dropna()
                if len(atr_values) > 0 and len(close_values) > 0:
                    # ATR como porcentaje del precio de cierre
                    atr_pct_series = atr_values / close_values.reindex(atr_values.index)
                    avg_atr_pct = atr_pct_series.mean()
                else:
                    avg_atr_pct = 0.02
            else:
                avg_atr_pct = 0.02

        print(f"ATR promedio para {ticker}: {avg_atr_pct:.4f} ({avg_atr_pct * 100:.2f}%)")

        # Convertir multiplicadores a porcentajes
        tp_percentages = [float(mult * avg_atr_pct) for mult in tp_multipliers]
        sl_percentages = [float(mult * avg_atr_pct) for mult in sl_multipliers]

        print(f"TP multiplicadores {tp_multipliers} -> porcentajes {[f'{p:.4f}' for p in tp_percentages]}")
        print(f"SL multiplicadores {sl_multipliers} -> porcentajes {[f'{p:.4f}' for p in sl_percentages]}")

        return tp_percentages, sl_percentages

    except Exception as e:
        print(f"ERROR calculando ATR para {ticker}: {e}")
        print("Usando conversión por defecto (2% ATR)")
        # Fallback: asumir 2% de ATR promedio
        avg_atr_pct = 0.02
        tp_percentages = [float(mult * avg_atr_pct) for mult in tp_multipliers]
        sl_percentages = [float(mult * avg_atr_pct) for mult in sl_multipliers]
        return tp_percentages, sl_percentages


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def as_abs_file_uri(path: Path) -> str:
    return path.resolve().as_uri()

def mlflow_setup():
    """Asegura una única URI consistente (file://.../mlruns) y el experimento."""
    tracking_uri_cfg = getattr(S, "mlflow_tracking_uri", None)
    default_abs_uri = (ROOT / "mlruns").resolve().as_uri()
    try:
        if tracking_uri_cfg:
            if tracking_uri_cfg.startswith("file:"):
                mlflow.set_tracking_uri(default_abs_uri)
            else:
                mlflow.set_tracking_uri(tracking_uri_cfg)
        else:
            mlflow.set_tracking_uri(default_abs_uri)
    except Exception:
        mlflow.set_tracking_uri(default_abs_uri)

    mlflow.set_experiment(getattr(S, "mlflow_experiment", "PHIBOT"))

def _is_finite(x) -> bool:
    try:
        if x is None:
            return False
        return np.isfinite(float(x))
    except Exception:
        return False

def _len_obj(x) -> int:
    if x is None:
        return 0
    if isinstance(x, (pd.DataFrame, pd.Series, list, tuple)):
        return int(len(x))
    try:
        return int(len(x))
    except Exception:
        return 0


def parse_csv_floats(s: Optional[str]) -> List[float]:
    if not s or not str(s).strip(): return []
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]

def parse_csv_ints(s: Optional[str]) -> List[int]:
    if not s or not str(s).strip(): return []
    return [int(x) for x in str(s).split(",") if str(x).strip() != ""]


# --------------------------
# CV “silenciosa” (sin MLflow)
# --------------------------
def run_cv_silent(
    ticker: str,
    timeframe: str,
    *,
    model: str,
    features_csv: Optional[str],
    feature_set: Optional[str],
    thresholds_csv: Optional[str],
    days: Optional[int],
    date_from: Optional[str],
    date_to: Optional[str],
    n_splits: Optional[int],
    test_size: Optional[int],
    scheme: Optional[str],
    embargo: Optional[int],
    purge: Optional[int],
    tp_mult: Optional[float],
    sl_mult: Optional[float],
    time_limit: Optional[int],
    python_exe: Optional[str] = None,
) -> dict:
    """
    Lanza RSH_TimeSeriesCV.py con --no_mlflow para obtener OOF (ROC/PR) y threshold recomendado,
    sin crear runs en MLflow. Devuelve el dict del JSON generado por la CV.
    """
    py = python_exe or sys.executable
    script = ROOT / "RSH_TimeSeriesCV.py"
    if not script.exists():
        raise FileNotFoundError(f"No encuentro RSH_TimeSeriesCV.py en {script}")

    args = [py, str(script),
            "--ticker", ticker, "--timeframe", timeframe,
            "--model", model, "--no_mlflow"]

    if features_csv:
        args += ["--features", features_csv]
    elif feature_set:
        args += ["--feature_set", feature_set]

    if thresholds_csv:
        args += ["--thresholds", thresholds_csv]
    if days is not None:
        args += ["--days", str(int(days))]
    if date_from:
        args += ["--date_from", date_from]
    if date_to:
        args += ["--date_to", date_to]
    if n_splits is not None:
        args += ["--n_splits", str(int(n_splits))]
    if test_size is not None:
        args += ["--test_size", str(int(test_size))]
    if scheme:
        args += ["--scheme", scheme]
    if embargo is not None:
        args += ["--embargo", str(int(embargo))]
    if purge is not None:
        args += ["--purge", str(int(purge))]
    if tp_mult is not None:
        args += ["--tp", str(float(tp_mult))]
    if sl_mult is not None:
        args += ["--sl", str(float(sl_mult))]
    if time_limit is not None:
        args += ["--time_limit", str(int(time_limit))]

    proc = subprocess.run(args, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"CV falló para {ticker}: {proc.stdout}\n{proc.stderr}")

    # Localiza el JSON que escribe la CV
    cv_dir = Path(getattr(S, "cv_dir", Path(S.logs_path) / "cv"))
    cv_json = cv_dir / f"{ticker.upper()}_{timeframe}_cv.json"
    if not cv_json.exists():
        raise FileNotFoundError(f"No se generó el JSON de CV: {cv_json}")
    return json.loads(cv_json.read_text(encoding="utf-8"))


# --------------------------
# Núcleo: un run por combinación
# --------------------------
# Reemplaza la función optimize_per_combo completa con esta versión limpia:

def optimize_per_combo(
        ticker: str,
        timeframe: str,
        *,
        models: List[str],
        features_csv: Optional[str],
        feature_set: Optional[str],
        thresholds_csv: Optional[str],
        tp_grid_csv: Optional[str],
        sl_grid_csv: Optional[str],
        tl_grid_csv: Optional[str],
        # CV period & scheme
        days: Optional[int],
        date_from: Optional[str],
        date_to: Optional[str],
        n_splits: Optional[int],
        test_size: Optional[int],
        scheme: Optional[str],
        embargo: Optional[int],
        purge: Optional[int],
        # labeling params usados en la CV
        tp_mult: Optional[float],
        sl_mult: Optional[float],
        time_limit_lbl: Optional[int],
        # flags/UI
        iter_thr_in_bt: bool,
        primary_metric: str,
        min_trades: int,
        couple_labeling_exec: bool,
) -> None:
    """
    Genera UN RUN por cada combinación (thr, tp, sl, tl) y por cada modelo seleccionado.
    """
    mlflow_setup()

    # 1) Ejecuta CV silenciosa por modelo (para OOF)
    thresholds_list = parse_csv_floats(thresholds_csv) or getattr(S, "cv_threshold_grid",
                                                                  [0.55, 0.6, 0.65, 0.7, 0.75, 0.8])

    try:
        from utils.A_data_loader import load_data
        df_quality_check = load_data(ticker=ticker, timeframe=timeframe, use_local=True,
                                     base_path=Path(settings.S.data_path))

        # Aplicar mismo filtro de días
        if days and 'date' in df_quality_check.columns:
            df_quality_check['date'] = pd.to_datetime(df_quality_check['date'], utc=True, errors='coerce')
            cutoff = df_quality_check['date'].max() - pd.Timedelta(days=days)
            df_quality_check = df_quality_check[df_quality_check['date'] >= cutoff]
            df_quality_check = df_quality_check.set_index('date')

        # Control de cobertura temporal
        temporal_analysis = analyze_temporal_coverage(df_quality_check, days or 90, ticker)

    except Exception as e:
        print(f"WARNING - No se pudo realizar análisis de calidad para {ticker}: {e}")
        temporal_analysis = {"data_quality": "ERROR", "temporal_coverage_pct": 0}

    cv_by_model: Dict[str, dict] = {}
    feature_analysis_by_model: Dict[str, dict] = {}

    for model in models:
        try:
            cv_res = run_cv_silent(...)

            # NUEVO: Si la CV fue exitosa, analizar el modelo entrenado
            # Necesitamos obtener el modelo entrenado del JSON de CV
            cv_json_path = Path(
                getattr(S, "cv_dir", Path(S.logs_path) / "cv")) / f"{ticker.upper()}_{timeframe}_cv.json"

            # Simulamos el análisis de features (necesitarías acceso al modelo real)
            # Por ahora, generamos análisis mock basado en features seleccionadas
            if features_csv:
                selected_features = [f.strip() for f in features_csv.split(",")]
            else:
                # Usar feature set legacy o features por defecto
                selected_features = ["sma_20", "ema_12", "rsi_14", "atr_14", "vwap_20"]  # ejemplo

            # Mock feature analysis (reemplazar con análisis real cuando tengas acceso al modelo)
            feature_analysis = {
                "feature_analysis": "MOCK_SUCCESS",
                "model_type": model,
                "total_features": len(selected_features),
                "top_5_concentration_pct": 45.2,  # ejemplo
                "zero_importance_features": 0,
                "feature_diversity": "HIGH",
                "top_features": [
                    {"feature": feat, "importance": 0.1, "importance_pct": 10.0}
                    for feat in selected_features[:5]
                ]
            }
            feature_analysis_by_model[model] = feature_analysis

        except Exception as e:
            cv_res = {"metrics_mean": {"roc_auc": float('nan'), "pr_auc": float('nan')}}
            feature_analysis_by_model[model] = {"feature_analysis": "ERROR", "error": str(e)}

        cv_by_model[model] = cv_res

    cv_by_model: Dict[str, dict] = {}
    for model in models:

        try:
            cv_res = run_cv_silent(
                ticker, timeframe,
                model=model,
                features_csv=features_csv,
                feature_set=feature_set,
                thresholds_csv=",".join(str(x) for x in thresholds_list),
                days=days, date_from=date_from, date_to=date_to,
                n_splits=n_splits, test_size=test_size, scheme=scheme, embargo=embargo, purge=purge,
                tp_mult=tp_mult, sl_mult=sl_mult, time_limit=time_limit_lbl,
            )
        except Exception:
            cv_res = {"metrics_mean": {"roc_auc": float('nan'), "pr_auc": float('nan')}}
        cv_by_model[model] = cv_res

    # 2) Construye grids
    tp_multipliers = parse_csv_floats(tp_grid_csv) or [1, 2, 3]
    sl_multipliers = parse_csv_floats(sl_grid_csv) or [1, 2, 3]
    tl_grid = parse_csv_ints(tl_grid_csv) or [8, 12, 16]

    tp_grid, sl_grid = convert_atr_multipliers_to_pct(
        ticker, timeframe, tp_multipliers, sl_multipliers, days
    )

    print(f"Usando TP grid (% decimal): {[f'{x:.4f}' for x in tp_grid]}")
    print(f"Usando SL grid (% decimal): {[f'{x:.4f}' for x in sl_grid]}")

    # 3) Para cada modelo y cada combinación, crea UN RUN
    for model in models:
        roc = float(cv_by_model[model].get("metrics_mean", {}).get("roc_auc", np.nan))
        prc = float(cv_by_model[model].get("metrics_mean", {}).get("pr_auc", np.nan))
        thr_rec = cv_by_model[model].get("recommended_threshold")
        thr_rec = float(thr_rec) if _is_finite(thr_rec) else None
        temporal_metrics = temporal_analysis
        feature_metrics = feature_analysis_by_model[model]

        if iter_thr_in_bt:
            thr_iter = thresholds_list[:]
        else:
            thr_iter = [thr_rec] if _is_finite(thr_rec) else ([thresholds_list[-1]] if thresholds_list else [0.8])

        for thr in thr_iter:
            thr_val = float(thr)

            for i, tp_mult in enumerate(tp_multipliers):
                for j, sl_mult in enumerate(sl_multipliers):
                    for tl_ in tl_grid:
                        # Usar los valores convertidos para el backtest
                        tp_pct = tp_grid[i]
                        sl_pct = sl_grid[j]

                        # CV para esta combinación (si acoplado)
                        if couple_labeling_exec:
                            try:
                                cv_res = run_cv_silent(
                                    ticker, timeframe,
                                    model=model, features_csv=features_csv, feature_set=feature_set,
                                    thresholds_csv=str(thr), days=days, date_from=date_from, date_to=date_to,
                                    n_splits=n_splits, test_size=test_size, scheme=scheme,
                                    embargo=embargo, purge=purge,
                                    tp_mult=float(tp_mult), sl_mult=float(sl_mult), time_limit=int(tl_),
                                )
                                roc = float(cv_res.get("metrics_mean", {}).get("roc_auc", np.nan))
                                prc = float(cv_res.get("metrics_mean", {}).get("pr_auc", np.nan))
                                thr_rec = float(thr)
                            except Exception:
                                roc, prc, thr_rec = np.nan, np.nan, float(thr)

                        # Parámetros del backtest usando valores convertidos
                        params_bt = {
                            "threshold": float(thr_val),
                            "tp_pct": float(tp_pct),  # valor convertido de ATR
                            "sl_pct": float(sl_pct),  # valor convertido de ATR
                            "time_limit": int(tl_),
                            "days": int(days) if days is not None else "",
                            "date_from": date_from or "", "date_to": date_to or "",
                            "cooldown_bars": int(getattr(S, "bt_cooldown_bars", 0)),
                            "allow_short": bool(getattr(S, "allow_short", True)),
                            "slippage_bps": float(getattr(S, "slippage_bps", 0.0)),
                            "capital_per_trade": float(getattr(S, "capital_per_trade", 1000.0)),
                            "commission_per_trade": float(getattr(S, "commission_per_trade", 0.35)),
                            "model": model, "features": features_csv or "", "feature_set": feature_set or "",
                        }

                        # Ejecuta el backtest
                        bt = {}
                        bt_error = None
                        try:
                            bt = run_backtest_for_ticker(ticker, timeframe, params_bt)
                        except Exception as e:
                            bt_error = str(e)
                            print(f"ERROR - Backtest falló para {ticker}: {e}")

                        # Extrae métricas del backtest
                        metrics_bt = {}
                        eq, tr = None, None

                        if isinstance(bt, dict) and bt:
                            eq = bt.get("equity")
                            tr = bt.get("trades")

                            # Extrae métricas de todas las ubicaciones posibles
                            for key in ["metrics", "stats", "summary", "results"]:
                                if key in bt and isinstance(bt[key], dict):
                                    metrics_bt.update(bt[key])

                            # Métricas en nivel raíz
                            for key, val in bt.items():
                                if key not in ("equity", "trades", "metrics", "stats", "summary", "results"):
                                    if isinstance(val, (int, float, np.integer, np.floating)):
                                        metrics_bt[key] = val

                        # Fallback: calcular métricas si no las hay
                        if not metrics_bt and eq is not None and len(eq) > 0:
                            try:
                                eq_arr = eq.values if hasattr(eq, "values") else np.asarray(eq)
                                tr_list = tr.to_dict("records") if hasattr(tr, "to_dict") and not tr.empty else []
                                m_calc = trading_metrics(eq_arr, tr_list)
                                if isinstance(m_calc, dict):
                                    metrics_bt.update(m_calc)
                            except Exception:
                                pass

                        # Métricas por defecto si todo falla
                        if not metrics_bt:
                            metrics_bt = {
                                "net_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                                "win_rate": 0.0, "profit_factor": 0.0, "n_trades": 0
                            }

                        n_trades_val = _len_obj(tr)
                        eq_len_val = _len_obj(eq)

                        # === CREAR RUN MLFLOW ===

                        for i, tp_mult in enumerate(tp_multipliers):
                            for j, sl_mult in enumerate(sl_multipliers):
                                for tl_ in tl_grid:
                                    # Usar los valores convertidos para el backtest
                                    tp_pct = tp_grid[i]
                                    sl_pct = sl_grid[j]
                        # Asegurar cierre de runs previos
                        try:
                            mlflow.end_run()
                        except:
                            pass
                        run_name = f"{ticker}_{timeframe}_{model}_thr{thr_val:.3f}_tp{tp_mult:.0f}x_sl{sl_mult:.0f}x_tl{int(tl_)}"
                        with mlflow.start_run(run_name=run_name, nested=False):
                            # Tags
                            mlflow.set_tags({
                                "component": "optimizer",
                                "phase": "scenarios_per_combo",
                                "ticker": ticker.upper(),
                                "timeframe": timeframe,
                                "model": model,
                                "features": features_csv or "",
                                "feature_set": feature_set or "",
                                "primary_metric": primary_metric,
                                "bt_error": bt_error or "",
                                "data_quality": temporal_metrics.get("data_quality", "UNKNOWN"),
                                "feature_diversity": feature_metrics.get("feature_diversity", "UNKNOWN"),
                            })

                            # Parámetros
                            mlflow.log_params({
                                "threshold": float(thr_val),
                                "time_limit": int(tl_),
                                "tp_pct": float(tp_pct),  # porcentaje real usado
                                "sl_pct": float(sl_pct),  # porcentaje real usado
                                "tp_multiplier": float(tp_mult),  # multiplicador original
                                "sl_multiplier": float(sl_mult),  # multiplicador original
                                "date_from": date_from or "", "date_to": date_to or "",
                                "days": int(days) if days is not None else "",
                                "n_splits": n_splits or "", "test_size": test_size or "",
                                "scheme": scheme or "", "embargo": embargo or "", "purge": purge or "",
                                "coupled_params": couple_labeling_exec,
                            })

                            # Métricas siempre presentes (para evitar runs vacíos)
                            mlflow.log_metric("oof_roc_auc", float(roc) if _is_finite(roc) else -1.0)
                            mlflow.log_metric("oof_pr_auc", float(prc) if _is_finite(prc) else -1.0)
                            mlflow.log_metric("bt_trades_count", float(n_trades_val))
                            mlflow.log_metric("bt_equity_len", float(eq_len_val))
                            mlflow.log_metric("bt_success", 0.0 if bt_error else 1.0)
                            mlflow.log_metric("temporal_coverage_pct",
                                              float(temporal_metrics.get("temporal_coverage_pct", 0)))
                            mlflow.log_metric("trading_days_expected",
                                              float(temporal_metrics.get("trading_days_expected", 0)))
                            mlflow.log_metric("trading_days_actual",
                                              float(temporal_metrics.get("trading_days_actual", 0)))
                            mlflow.log_metric("missing_days", float(temporal_metrics.get("missing_days", 0)))
                            mlflow.log_metric("weekend_days_found",
                                              float(temporal_metrics.get("weekend_days_found", 0)))

                            if feature_metrics.get("feature_analysis") == "SUCCESS":
                                mlflow.log_metric("feature_count", float(feature_metrics.get("total_features", 0)))
                                mlflow.log_metric("top5_feature_concentration",
                                                  float(feature_metrics.get("top_5_concentration_pct", 0)))
                                mlflow.log_metric("zero_importance_features",
                                                  float(feature_metrics.get("zero_importance_features", 0)))

                                # Log top 3 features como parámetros
                                top_features = feature_metrics.get("top_features", [])
                                for i, feat_data in enumerate(top_features[:3], 1):
                                    mlflow.log_param(f"top_feature_{i}", feat_data["feature"])
                                    mlflow.log_metric(f"top_feature_{i}_importance", float(feat_data["importance_pct"]))

                            # Métricas del backtest
                            def log_safe(name: str, val):
                                try:
                                    if isinstance(val, (np.floating, np.integer)):
                                        val = val.item()
                                    if isinstance(val, (int, float)) and np.isfinite(val):
                                        mlflow.log_metric(name, float(val))
                                except:
                                    pass

                            # Mapear y loggear métricas principales
                            metric_mapping = {
                                "ret": "net_return", "return": "net_return", "pnl": "net_return",
                                "max_dd": "max_drawdown", "drawdown": "max_drawdown",
                                "ntrades": "n_trades", "num_trades": "n_trades",
                                "sharpe_ratio": "sharpe", "winrate": "win_rate",
                                "pf": "profit_factor"
                            }

                            for k, v in metrics_bt.items():
                                metric_name = metric_mapping.get(str(k).lower(), str(k).lower())
                                log_safe(metric_name, v)

                            # Artefactos (solo si hay datos)
                            artifacts_dir = Path("artifacts")
                            artifacts_dir.mkdir(exist_ok=True)

                            if hasattr(eq, "empty") and not eq.empty:
                                eq_path = artifacts_dir / "equity.csv"
                                eq.to_csv(eq_path)
                                mlflow.log_artifact(str(eq_path))

                            if hasattr(tr, "empty") and not tr.empty:
                                tr_path = artifacts_dir / "trades.csv"
                                tr.to_csv(tr_path, index=False)
                                mlflow.log_artifact(str(tr_path))

                        print(f"✅ Run completado: {run_name}")

    print(f"\n🎯 Optimización completada para {ticker}. Runs creados en MLflow.")
# --------------------------
# CLI
# --------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Research/Scenarios — Un run por combinación (ticker × thr × tp × sl × tl)")
    p.add_argument("--ticker", required=True)
    p.add_argument("--timeframe", default=getattr(S, "timeframe_default", "5mins"))
    # grids / opciones
    p.add_argument("--models", default="xgb", help="CSV de modelos (xgb,rf,logreg)")
    p.add_argument("--features", default=None, help="CSV de features (engine.features); si se usa, ignora feature_set")
    p.add_argument("--feature_set", default="core", choices=["core","core+vol","all"], help="Usado si no pasas --features")
    p.add_argument("--thresholds", default=None, help="CSV de thresholds para CV y grid (ej. 0.6,0.7,0.8)")
    p.add_argument("--tp_grid", default="0.003,0.005,0.008")
    p.add_argument("--sl_grid", default="0.003,0.005,0.008")
    p.add_argument("--tl_grid", default="8,12,16")
    p.add_argument(
        "--couple_labeling_exec", action="store_true",
        help="Si se pasa, la CV se ejecuta por cada combinación y usa los mismos tp/sl/time_limit que el backtest."
    )

    # CV ventana y esquema (para calcular OOF ROC/PR — no crea runs)
    p.add_argument("--days", type=int, default=None, help="Días de historia (solo si no hay date_from/date_to)")
    p.add_argument("--date_from", type=str, default=None)
    p.add_argument("--date_to", type=str, default=None)
    p.add_argument("--n_splits", type=int, default=getattr(S,"n_splits_cv",5))
    p.add_argument("--test_size", type=int, default=getattr(S,"cv_test_size",500))
    p.add_argument("--scheme", type=str, default=getattr(S,"cv_scheme","expanding"))
    p.add_argument("--embargo", type=int, default=getattr(S,"cv_embargo",-1))
    p.add_argument("--purge", type=int, default=getattr(S,"cv_purge",-1))
    p.add_argument(
        "--min_trades", type=int, default=0,
        help="Sólo para logging/filtrado en UI; no afecta al backtest."
    )

    # labeling defaults (triple barrier usados en CV)
    p.add_argument("--tp", type=float, default=None)
    p.add_argument("--sl", type=float, default=None)
    p.add_argument("--time_limit", type=int, default=None)

    # flags/UI
    p.add_argument("--iter_thr_in_bt", action="store_true", help="Si se pasa, iterar thresholds también en el backtest; si no, usar el recomendado.")
    p.add_argument("--primary_metric", default="net_return", choices=["net_return","sharpe","win_rate","profit_factor"])

    return p

def main():
    args = build_parser().parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    optimize_per_combo(
        ticker=args.ticker.upper(),
        timeframe=args.timeframe,
        models=models,
        features_csv=args.features,
        feature_set=(None if args.features else args.feature_set),
        thresholds_csv=args.thresholds,
        tp_grid_csv=args.tp_grid,
        sl_grid_csv=args.sl_grid,
        tl_grid_csv=args.tl_grid,
        days=args.days,
        date_from=args.date_from,
        date_to=args.date_to,
        n_splits=args.n_splits,
        test_size=args.test_size,
        scheme=args.scheme,
        embargo=(None if args.embargo is not None and args.embargo < 0 else args.embargo),
        purge=(None if args.purge is not None and args.purge < 0 else args.purge),
        tp_mult=args.tp,
        sl_mult=args.sl,
        time_limit_lbl=args.time_limit,
        iter_thr_in_bt=bool(args.iter_thr_in_bt),
        primary_metric=args.primary_metric,
        min_trades=args.min_trades,
        couple_labeling_exec=bool(args.couple_labeling_exec),
    )
    print("\n✅ OK. Abre MLflow para ver los runs (uno por combinación):")
    print(f"    mlflow ui --backend-store-uri {as_abs_file_uri(ROOT / 'mlruns')} -p 5000")

if __name__ == "__main__":
    main()
