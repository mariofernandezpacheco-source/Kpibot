# engine/backtest_runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from utils.B_feature_engineering import add_technical_indicators, load_context_data

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Settings opcionales (coge de settings.S si existe; si no, defaults seguros)
# -----------------------------------------------------------------------------
try:
    import settings as _settings  # type: ignore

    S = _settings.S
    _CAPITAL = float(getattr(S, "capital_per_trade", 1000.0))
    _COMMISSION = float(getattr(S, "commission_per_trade", 0.35))
    _ALLOW_SHORT = bool(getattr(S, "allow_short", True))
except Exception:
    _CAPITAL = 1000.0
    _COMMISSION = 0.35
    _ALLOW_SHORT = True


@dataclass
class BTParams:
    threshold: float = 0.80
    tp_multiplier: float = 2.0  # Nuevo
    sl_multiplier: float = 1.5  # Nuevo
    use_atr_multipliers: bool = True  # Nuevo
    time_limit: int = 16  # Nuevo
    tp_pct: float = 0.005  # Mantener para compatibilidad
    sl_pct: float = 0.005
    cooldown_bars: int = 0
    allow_short: bool = _ALLOW_SHORT
    slippage_bps: float = 0.0
    capital_per_trade: float = _CAPITAL
    commission_per_trade: float = _COMMISSION

def _compute_atr14(df: pd.DataFrame) -> pd.Series:
    """Calcula ATR(14) si no existe"""
    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/14, adjust=False).mean()
# -----------------------------------------------------------------------------
# Carga de datos/señales (ADÁPTALO A TU PIPELINE)
# -----------------------------------------------------------------------------
def load_ohlc_and_signals(ticker: str, timeframe: str, params: dict[str, Any]) -> pd.DataFrame:
    """
    Devuelve un DataFrame con índice datetime y columnas OHLC + señales
    """
    print(f"DEBUG LOAD - ticker: {ticker}, timeframe: {timeframe}")
    print(f"DEBUG LOAD - params keys: {list(params.keys())}")

    try:
        from pathlib import Path
        import settings

        parquet_base = Path("DAT_data/parquet")
        ticker_dir = parquet_base / "ohlcv" / f"ticker={ticker}"
        print(f"DEBUG LOAD - Buscando en: {ticker_dir}")

        if not ticker_dir.exists():
            print(f"DEBUG LOAD - No existe directorio: {ticker_dir}")
            return pd.DataFrame()

        # Buscar archivos parquet más recientes
        parquet_files = list(ticker_dir.rglob("*.parquet"))
        if not parquet_files:
            print("DEBUG LOAD - No hay archivos parquet")
            return pd.DataFrame()

        print(f"DEBUG LOAD - Encontrados {len(parquet_files)} archivos parquet")

        # NUEVO: Filtrar archivos por fecha si hay parámetros de fecha
        filtered_files = []
        if params.get('date_from') or params.get('date_to'):
            date_from = pd.to_datetime(params.get('date_from')) if params.get('date_from') else None
            date_to = pd.to_datetime(params.get('date_to')) if params.get('date_to') else None
            print(f"DEBUG LOAD - Filtrando archivos entre {date_from} y {date_to}")

            for pf in parquet_files:
                try:
                    # Extraer fecha del path: .../date=2025-07-16/...
                    date_str = pf.parent.name.replace('date=', '')
                    file_date = pd.to_datetime(date_str)

                    if date_from and file_date < date_from:
                        continue
                    if date_to and file_date > date_to:
                        continue

                    filtered_files.append(pf)
                except:
                    continue

            parquet_files = filtered_files
            print(f"DEBUG LOAD - {len(parquet_files)} archivos después del filtro de fechas")
        else:
            # Sin filtro de fechas, usar últimos archivos por modificación
            parquet_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            parquet_files = parquet_files[:10]
            print(f"DEBUG LOAD - Usando últimos 10 archivos por fecha de modificación")

        if not parquet_files:
            print("DEBUG LOAD - No hay archivos válidos después del filtro")
            return pd.DataFrame()

        # Cargar archivos filtrados
        dfs = []
        for pf in parquet_files:
            try:
                print(f"DEBUG LOAD - Leyendo archivo: {pf}")
                df_part = pd.read_parquet(pf)
                print(f"DEBUG LOAD - Archivo leído: {len(df_part)} filas, columnas: {list(df_part.columns)}")
                dfs.append(df_part)
            except Exception as e:
                print(f"DEBUG LOAD - Error leyendo {pf}: {e}")

        if not dfs:
            print("DEBUG LOAD - No se cargaron archivos válidos")
            return pd.DataFrame()

        # Concatenar y procesar
        print(f"DEBUG LOAD - Concatenando {len(dfs)} DataFrames...")
        df = pd.concat(dfs, ignore_index=True)
        print(f"DEBUG LOAD - Concatenación exitosa: {len(df)} filas")
        print(f"DEBUG LOAD - Columnas disponibles: {list(df.columns)}")

        # Convertir timestamp a date
        if 'timestamp' in df.columns and 'date' not in df.columns:
            print("DEBUG LOAD - Convirtiendo 'timestamp' a 'date'")
            df['date'] = df['timestamp']

        # Procesar fechas
        if 'date' not in df.columns:
            print("DEBUG LOAD - ERROR: No hay columna 'date'")
            return pd.DataFrame()

        print(f"DEBUG LOAD - Procesando fechas...")
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date').drop_duplicates()
        print(f"DEBUG LOAD - Después de ordenar: {len(df)} filas")

        # Filtro de días
        days = params.get('days')
        if days and days > 0:
            print(f"DEBUG LOAD - Aplicando filtro de {days} días...")
            cutoff = df['date'].max() - pd.Timedelta(days=days)
            df = df[df['date'] >= cutoff]
            print(f"DEBUG LOAD - Después del filtro: {len(df)} filas")

        # Establecer índice

        print(f"DEBUG LOAD - Índice establecido. Rango: {df.index[0]} a {df.index[-1]}")


        print(f"DEBUG LOAD - DataFrame final: {len(df)} filas, {len(df.columns)} columnas")

        context_data = load_context_data(timeframe, Path("DAT_data"))
        df = add_technical_indicators(df, context_data)
        try:
            print("DEBUG LOAD - Cargando contexto (VIX, SPY)...")
            context_data = load_context_data(timeframe, Path("DAT_data"))
            print(f"DEBUG LOAD - Contexto cargado: {list(context_data.keys())}")

            print("DEBUG LOAD - Aplicando ingeniería de features...")
            df = add_technical_indicators(df, context_data)
            print(f"DEBUG LOAD - Features aplicados. Nuevas columnas: {len(df.columns)}")

        except Exception as e:
            print(f"DEBUG LOAD - Error con contexto/features: {e}")
        df = df.set_index('date').sort_index()
        print(f"DEBUG LOAD - Índice establecido. Rango: {df.index[0]} a {df.index[-1]}")

        # En load_ohlc_and_signals(), reemplazar la sección de señales dummy:

        # Generar señales: cargar modelo entrenado o usar dummy como fallback
        signal_cols = ['signal', 'pred', 'proba_up', 'proba']
        if not any(col in df.columns for col in signal_cols):
            model_loaded = False

            try:
                import joblib
                import json

                model_path = Path(f"02_models/{ticker.upper()}/pipeline.pkl")
                meta_path = Path(f"02_models/{ticker.upper()}/pipeline_meta.json")

                if model_path.exists() and meta_path.exists():
                    print(f"DEBUG LOAD - Cargando modelo: {model_path}")

                    # Cargar modelo y metadata
                    pipeline = joblib.load(model_path)
                    meta = json.loads(meta_path.read_text())
                    expected_features = meta.get('features', [])

                    print(f"DEBUG LOAD - Features esperadas: {len(expected_features)}")

                    # Verificar features disponibles
                    available_features = [f for f in expected_features if f in df.columns]
                    coverage = len(available_features) / len(expected_features) if expected_features else 0

                    print(f"DEBUG LOAD - Cobertura de features: {coverage:.2%}")

                    if coverage >= 0.8:  # 80% mínimo
                        # Preparar datos para predicción
                        X = df[available_features].fillna(0)

                        # Generar predicciones reales
                        proba = pipeline.predict_proba(X)
                        df['proba_up'] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

                        print(f"DEBUG LOAD - Modelo aplicado exitosamente")
                        print(
                            f"DEBUG LOAD - Rango probabilidades: {df['proba_up'].min():.3f} - {df['proba_up'].max():.3f}")
                        model_loaded = True
                    else:
                        print(f"DEBUG LOAD - Cobertura insuficiente ({coverage:.1%}), usando dummy")
                else:
                    print(f"DEBUG LOAD - No existe modelo para {ticker}")

            except Exception as e:
                print(f"DEBUG LOAD - Error cargando modelo: {e}")

            # Fallback a señales dummy solo si no se cargó modelo
            if not model_loaded:
                print(f"DEBUG LOAD - ERROR: No hay modelo entrenado para {ticker}, saltando")
                return pd.DataFrame()  # DataFrame vacío - forzar skip



        return df


    except Exception as e:
        print(f"DEBUG LOAD - Error específico: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def _generate_signals(df: pd.DataFrame, p: BTParams) -> pd.Series:
    print(f"DEBUG SIGNALS - Columnas proba: {[col for col in df.columns if 'proba' in col]}")
    print(f"DEBUG SIGNALS - allow_short: {p.allow_short}, threshold: {p.threshold}")
    if "signal" in df.columns:
        return df["signal"].clip(-1, 1).fillna(0).astype(int)

    if "pred" in df.columns:
        return df["pred"].clip(-1, 1).fillna(0).astype(int)

    thr = float(p.threshold)

    # NUEVO: Manejo de 3 probabilidades
    if all(col in df.columns for col in ["proba_up", "proba_down", "proba_hold"]):
        # Caso ideal: 3 probabilidades explícitas
        long_signal = (df["proba_up"] >= thr).astype(int)
        short_signal = (df["proba_down"] >= thr).astype(int) * -1 if p.allow_short else 0
        # proba_hold resulta en signal = 0 (no acción)

        sig = long_signal + short_signal
        return sig.clip(-1, 1)

    elif "proba_up" in df.columns:
        # Caso actual: solo proba_up
        # Interpretar como: alto = long, bajo = short, medio = hold
        up = df["proba_up"].astype(float)

        long = (up >= thr).astype(int)
        short = (up <= (1.0 - thr)).astype(int) * -1 if p.allow_short else 0
        # Zona media (1-thr < proba_up < thr) = hold (signal = 0)

        sig = long + short
        return sig.clip(-1, 1)

    # Fallback: sin señales
    return pd.Series(0, index=df.index, name="signal")


# -----------------------------------------------------------------------------
# Simulador simple: 1 posición a la vez, TP/SL ±pct, cierre EOD, sin solapes
# -----------------------------------------------------------------------------
def _simulate(df: pd.DataFrame, sig: pd.Series, p: BTParams) -> tuple[pd.Series, pd.DataFrame]:
    """
    Ejecuta backtest discreto a cierre de barra...
    """
    if 'atr_14' not in df.columns:
        df = df.copy()
        prev_close = df["close"].shift(1)
        tr1 = (df["high"] - df["low"]).abs()
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        return pd.Series(dtype="float64"), pd.DataFrame()

    # NUEVO: Añadir warmup period aquí
    warmup_bars = 20  # Saltar primeras 20 barras para estabilizar indicadores
    if len(df) <= warmup_bars:
        print(f"DEBUG - Dataset muy pequeño ({len(df)} barras), no se puede aplicar warmup")
        df_warmup = df
        sig_warmup = sig
    else:
        print(f"DEBUG - Aplicando warmup de {warmup_bars} barras, dataset: {len(df)} -> {len(df) - warmup_bars}")
        df_warmup = df.iloc[warmup_bars:]
        sig_warmup = sig.iloc[warmup_bars:]

    # Continuar con df_warmup y sig_warmup en lugar de df y sig
    df = df_warmup.copy()
    df = df.sort_index()
    sig = sig_warmup.reindex(df.index).fillna(0).astype(int)


    equity = []
    trades = []

    position = 0  # -1 short, 0 flat, 1 long
    entry_price = None
    entry_time = None
    shares = 0
    cooldown = 0

    capital = float(p.capital_per_trade)
    slip = float(p.slippage_bps) / 10000.0  # bps → pct


    last_day = None
    eq_val = capital  # equity local del trade; exportaremos curva acumulada normalizada

    for i, (ts, row) in enumerate(df.iterrows()):
        day = ts.date()
        px_open = float(row.get("open", row["close"]))
        px_hi = float(row.get("high", row["close"]))
        px_lo = float(row.get("low", row["close"]))
        px_close = float(row["close"])

        # Cierre EOD si cambia el día y hay posición abierta (cerramos al close anterior)
        if last_day is not None and day != last_day and position != 0:
            # cerramos al close de la barra anterior (aprox EOD)
            exit_price = prev_close * (1 - slip) if position == 1 else prev_close * (1 + slip)
            pnl_per_share = (
                (exit_price - entry_price) if position == 1 else (entry_price - exit_price)
            )
            gross = pnl_per_share * shares
            net = gross - 2 * p.commission_per_trade
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": prev_ts,
                    "side": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "shares": shares,
                    "pnl": net,
                    "pnl_pct": net / capital if capital > 0 else np.nan,
                    "reason": "EOD",
                }
            )
            position = 0
            entry_price = None
            entry_time = None
            shares = 0
            cooldown = int(p.cooldown_bars)

        # Si no hay posición, podemos abrir (si cooldown == 0)
        if position == 0:
            if cooldown > 0:
                cooldown -= 1
            else:
                s = int(sig.iloc[i])
                if s == 1 or (s == -1 and p.allow_short):
                    # Abrimos en el close actual ± slippage
                    px_entry = px_close * (1 + slip) if s == 1 else px_close * (1 - slip)
                    sh = int(np.floor(capital / px_entry)) if px_entry > 0 else 0
                    if sh > 0:
                        position = s
                        entry_price = px_entry
                        entry_time = ts
                        shares = sh
                        # Definimos barreras
                        if p.use_atr_multipliers:
                            atr_at_entry = df.loc[ts, 'atr_14'] if ts in df.index else df['atr_14'].iloc[-1]
                            if position == 1:
                                tp_level = entry_price + (atr_at_entry * p.tp_multiplier)
                                sl_level = entry_price - (atr_at_entry * p.sl_multiplier)
                            else:
                                tp_level = entry_price - (atr_at_entry * p.tp_multiplier)
                                sl_level = entry_price + (atr_at_entry * p.sl_multiplier)
                        else:
                            # Fallback a porcentajes fijos
                            if position == 1:
                                tp_level = entry_price * (1 + p.tp_pct)
                                sl_level = entry_price * (1 - p.sl_pct)
                # si no abre, seguimos

        else:
            # Gestionamos TP/SL en esta barra
            exit_reason = None
            exit_price = None

            if position == 1:
                # LONG: SL si low <= sl, TP si high >= tp
                hit_sl = px_lo <= sl_level
                hit_tp = px_hi >= tp_level
                if hit_sl and hit_tp:
                    # priorizamos TP si ambos (conservador), o usa midpoint
                    exit_price = tp_level
                    exit_reason = "TP"
                elif hit_tp:
                    exit_price = tp_level
                    exit_reason = "TP"
                elif hit_sl:
                    exit_price = sl_level
                    exit_reason = "SL"
            else:
                # SHORT
                hit_sl = px_hi >= sl_level
                hit_tp = px_lo <= tp_level
                if hit_sl and hit_tp:
                    exit_price = tp_level
                    exit_reason = "TP"
                elif hit_tp:
                    exit_price = tp_level
                    exit_reason = "TP"
                elif hit_sl:
                    exit_price = sl_level
                    exit_reason = "SL"

            if exit_price is not None:
                # Aplicamos slippage de salida
                exit_price = exit_price * (1 - slip) if position == 1 else exit_price * (1 + slip)
                pnl_per_share = (
                    (exit_price - entry_price) if position == 1 else (entry_price - exit_price)
                )
                gross = pnl_per_share * shares
                net = gross - 2 * p.commission_per_trade
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "side": "LONG" if position == 1 else "SHORT",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "shares": shares,
                        "pnl": net,
                        "pnl_pct": net / capital if capital > 0 else np.nan,
                        "reason": exit_reason,
                    }
                )
                position = 0
                entry_price = None
                entry_time = None
                shares = 0
                cooldown = int(p.cooldown_bars)

        # Equity “marcada” al close
        # Para simplicidad, si hay posición, marcamos PnL no realizado sobre capital
        if position != 0 and entry_price is not None and shares > 0:
            pnl_unrealized = (
                (px_close - entry_price) if position == 1 else (entry_price - px_close)
            ) * shares
            eq_val = capital + pnl_unrealized
        else:
            # equity base
            eq_val = capital

        equity.append((ts, eq_val))
        prev_ts, prev_close = ts, px_close
        last_day = day

    # Cierre si queda posición al final (última barra)
    if position != 0 and entry_price is not None and shares > 0:
        exit_price = prev_close * (1 - slip) if position == 1 else prev_close * (1 + slip)
        pnl_per_share = (exit_price - entry_price) if position == 1 else (entry_price - exit_price)
        gross = pnl_per_share * shares
        net = gross - 2 * p.commission_per_trade
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": prev_ts,
                "side": "LONG" if position == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "pnl": net,
                "pnl_pct": net / capital if capital > 0 else np.nan,
                "reason": "END",
            }
        )

    eq = pd.Series([v for _, v in equity], index=[t for t, _ in equity], name="equity")
    tr = pd.DataFrame(trades)
    return eq, tr


# -----------------------------------------------------------------------------
# Métricas de trading
# -----------------------------------------------------------------------------
def _trading_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {}

    eq = equity.dropna()
    if eq.empty or eq.shape[0] < 2:
        return {}

    r = eq.pct_change().dropna()
    # Agregamos a diario si es intradía
    daily = r.resample("1D").sum(min_count=1).dropna()

    net_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    sharpe = (
        float(((daily.mean()) / (daily.std() + 1e-12)) * np.sqrt(252)) if len(daily) > 1 else np.nan
    )
    max_dd = float((eq / eq.cummax() - 1).min())

    if trades is None or trades.empty:
        win_rate = np.nan
        profit_factor = np.nan
        n_trades = 0
    else:
        wins = trades["pnl"] > 0
        win_rate = float(wins.mean()) if len(trades) else np.nan
        gains = trades.loc[trades["pnl"] > 0, "pnl"].sum()
        losses = trades.loc[trades["pnl"] < 0, "pnl"].sum()
        profit_factor = float(gains / abs(losses)) if losses < 0 else np.nan
        n_trades = int(len(trades))

    return {
        "net_return": net_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
    }


# -----------------------------------------------------------------------------
# API pública
# -----------------------------------------------------------------------------
# En backtest_runner.py, reemplaza la función run_backtest_for_ticker con:

def run_backtest_for_ticker(
        ticker: str,
        timeframe: str,
        params: dict[str, Any],
        df_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Ejecuta backtest simple para (ticker,timeframe) con 'params'
    """
    print(f"BACKTEST DEBUG - Iniciando para {ticker} {timeframe}")

    try:
        bt_params = BTParams(
            threshold=float(params.get("threshold", 0.80)),
            tp_multiplier=float(params.get("tp_multiplier", 2.0)),  # Nuevo
            sl_multiplier=float(params.get("sl_multiplier", 1.5)),
            use_atr_multipliers=bool(params.get("use_atr_multipliers", True)),
            cooldown_bars=int(params.get("cooldown_bars", 0)),
            allow_short=bool(params.get("allow_short", _ALLOW_SHORT)),
            slippage_bps=float(params.get("slippage_bps", 0.0)),
            capital_per_trade=float(params.get("capital_per_trade", _CAPITAL)),
            commission_per_trade=float(params.get("commission_per_trade", _COMMISSION)),
        )
        print(
            f"BACKTEST DEBUG - Parámetros BT: threshold={bt_params.threshold}, tp_mult={bt_params.tp_multiplier}, sl_mult={bt_params.sl_multiplier}")

        df = (
            df_override if df_override is not None
            else load_ohlc_and_signals(ticker, timeframe, params)
        )

        print(f"BACKTEST DEBUG - DataFrame cargado: {len(df)} filas")
        if df.empty:
            print("BACKTEST DEBUG - DataFrame vacío, retornando resultado vacío")
            return {"metrics": {}, "equity": pd.Series(dtype="float64"), "trades": pd.DataFrame()}

        print(f"BACKTEST DEBUG - Columnas disponibles: {list(df.columns)}")

        # Verificar columnas requeridas
        required = {"close"}
        missing = required - set(df.columns)
        if missing:
            print(f"BACKTEST DEBUG - Faltan columnas requeridas: {missing}")
            return {"metrics": {}, "equity": pd.Series(dtype="float64"), "trades": pd.DataFrame()}

        sig = _generate_signals(df, bt_params)


        print(f"BACKTEST DEBUG - Señales generadas: {len(sig)} valores")
        print(f"BACKTEST DEBUG - Distribución de señales: {sig.value_counts().to_dict()}")

        if sig.abs().sum() == 0:
            print("BACKTEST DEBUG - No hay señales activas (todas cero)")

        equity, trades = _simulate(df, sig, bt_params)
        print(f"BACKTEST DEBUG - Simulación completada")
        print(f"BACKTEST DEBUG - Equity: {len(equity)} puntos")
        print(f"BACKTEST DEBUG - Trades: {len(trades)} operaciones")

        if not trades.empty:
            print(f"BACKTEST DEBUG - Ejemplo de trades:\n{trades.head()}")

        metrics = _trading_metrics(equity, trades)
        print(f"BACKTEST DEBUG - Métricas calculadas: {list(metrics.keys())}")
        print(f"BACKTEST DEBUG - Métricas: {metrics}")

        result = {"metrics": metrics, "equity": equity, "trades": trades}
        print(f"BACKTEST DEBUG - Resultado final: claves={list(result.keys())}")
        return result

    except Exception as e:
        print(f"BACKTEST DEBUG - ERROR en backtest: {e}")
        import traceback
        traceback.print_exc()
        # Retornar estructura vacía pero válida en caso de error
        return {"metrics": {}, "equity": pd.Series(dtype="float64"), "trades": pd.DataFrame()}
