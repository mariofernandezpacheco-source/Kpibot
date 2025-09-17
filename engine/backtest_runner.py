# engine/backtest_runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    tp_pct: float = 0.005  # 0.5% → 0.005
    sl_pct: float = 0.005
    cooldown_bars: int = 0
    allow_short: bool = _ALLOW_SHORT
    slippage_bps: float = 0.0  # 1bp=0.01%; 10bps=0.1%
    capital_per_trade: float = _CAPITAL
    commission_per_trade: float = _COMMISSION


# -----------------------------------------------------------------------------
# Carga de datos/señales (ADÁPTALO A TU PIPELINE)
# -----------------------------------------------------------------------------
def load_ohlc_and_signals(ticker: str, timeframe: str, params: dict[str, Any]) -> pd.DataFrame:
    """
    Devuelve un DataFrame con:
      index: datetime (ordenado)
      columnas: 'open','high','low','close'  (requeridas)
                y al menos una de: 'signal' (-1/0/1) | 'pred' (-1/0/1) |
                                   'proba_up' (0..1) | 'proba' (0..1)


    """

    print(f"DEBUG LOAD - ticker: {ticker}, timeframe: {timeframe}")
    print(f"DEBUG LOAD - params keys: {list(params.keys())}")

    # Intenta cargar desde tu sistema de datos

    # OPCIÓN 2: Cargar desde parquet directo
    try:
        from pathlib import Path
        import settings

        parquet_base = Path(getattr(settings.S, "parquet_base_path",
                                    Path(settings.S.data_path) / "parquet"))

        # Esquema: parquet/ohlcv/ticker=AAPL_5MINS/date=2024-XX-XX/data.parquet
        tf_up = timeframe.replace(" ", "").upper()
        ticker_dir = parquet_base / "ohlcv" / f"ticker={ticker}_{tf_up}"

        print(f"DEBUG LOAD - Buscando en: {ticker_dir}")

        if ticker_dir.exists():
            # Busca archivos parquet más recientes
            parquet_files = list(ticker_dir.rglob("*.parquet"))
            if parquet_files:
                print(f"DEBUG LOAD - Encontrados {len(parquet_files)} archivos parquet")

                # Carga los últimos N días de archivos
                parquet_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                dfs = []
                for pf in parquet_files[:5]:  # Últimos 5 archivos
                    try:
                        df_part = pd.read_parquet(pf)
                        dfs.append(df_part)
                    except Exception as e:
                        print(f"DEBUG LOAD - Error leyendo {pf}: {e}")

                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
                    df = df.sort_values('date').drop_duplicates()

                    # Aplicar filtro de días si viene en params
                    days = params.get('days')
                    if days and 'date' in df.columns:
                        cutoff = pd.to_datetime(df['date']).max() - pd.Timedelta(days=days)
                        df = df[df['date'] >= cutoff]

                    print(f"DEBUG LOAD - Parquet cargado: {len(df)} filas")

                    # Generar señales dummy si no existen
                    signal_cols = ['signal', 'pred', 'proba_up', 'proba']
                    if not any(col in df.columns for col in signal_cols):
                        np.random.seed(42)
                        df['proba_up'] = np.random.uniform(0.4, 0.7, len(df))
                        print("DEBUG LOAD - Señales dummy añadidas")
                        # En backtest_runner.py, en la función load_ohlc_and_signals,
                        # reemplaza la sección de manejo de fechas con:

                        # CRÍTICO: Manejar timestamps correctamente
                        if 'date' in df.columns:
                            print(f"DEBUG LOAD - Tipo original de 'date': {df['date'].dtype}")
                            print(f"DEBUG LOAD - Muestra de valores: {df['date'].head()}")

                            # Si ya es timestamp, convertir a datetime explícitamente
                            if df['date'].dtype == 'int64':
                                # Timestamp Unix (segundos o nanosegundos)
                                try:
                                    # Probar primero nanosegundos (más común en pandas)
                                    df['date'] = pd.to_datetime(df['date'], unit='ns', utc=True)
                                except:
                                    try:
                                        # Si falla, probar segundos
                                        df['date'] = pd.to_datetime(df['date'], unit='s', utc=True)
                                    except:
                                        # Fallback: conversión estándar
                                        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
                            else:
                                # Ya es datetime-like, solo asegurar que es UTC
                                df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')

                            print(f"DEBUG LOAD - Tipo después de conversión: {df['date'].dtype}")

                            # Eliminar filas con fechas inválidas
                            original_len = len(df)
                            df = df.dropna(subset=['date'])
                            if len(df) < original_len:
                                print(f"DEBUG LOAD - Eliminadas {original_len - len(df)} filas con fechas inválidas")

                            # ESTABLECER COMO ÍNDICE
                            df = df.set_index('date')
                            df = df.sort_index()

                            print(f"DEBUG LOAD - Índice final: {type(df.index)}")
                            print(f"DEBUG LOAD - Tipo de índice: {df.index.dtype}")
                            if len(df) > 0:
                                print(f"DEBUG LOAD - Rango de fechas: {df.index[0]} a {df.index[-1]}")
                        else:
                            print("DEBUG LOAD - ERROR: No hay columna 'date' en los datos")
                            return pd.DataFrame()  # Sin fechas no podemos continua

                    return df




        else:
            print(f"DEBUG LOAD - No existe directorio: {ticker_dir}")

    except Exception as e:
        print(f"DEBUG LOAD - Error con parquet: {e}")

    print("DEBUG LOAD - RETORNANDO DATAFRAME VACÍO")
    return pd.DataFrame()  # DataFrame vacío si todo falla


def _generate_signals(df: pd.DataFrame, p: BTParams) -> pd.Series:
    """
    Deriva señales -1/0/1 desde columnas comunes:
      - 'signal' → la usamos tal cual
      - 'pred'   → la usamos tal cual
      - 'proba_up' o 'proba' → long si >= threshold; short si <= 1-thr (si allow_short)
      - fallback: sin señales → todo 0
    """
    if "signal" in df.columns:
        sig = df["signal"].clip(-1, 1).fillna(0)
        return sig.astype(int)

    if "pred" in df.columns:
        sig = df["pred"].clip(-1, 1).fillna(0)
        return sig.astype(int)

    thr = float(p.threshold)
    if "proba_up" in df.columns:
        up = df["proba_up"].astype(float)
        long = (up >= thr).astype(int)
        short = ((up <= 1.0 - thr).astype(int) * -1) if p.allow_short else 0
        sig = long + short
        sig = sig.clip(-1, 1)
        return sig

    if "proba" in df.columns:
        up = df["proba"].astype(float)
        long = (up >= thr).astype(int)
        short = ((up <= 1.0 - thr).astype(int) * -1) if p.allow_short else 0
        sig = long + short
        sig = sig.clip(-1, 1)
        return sig

    return pd.Series(0, index=df.index, name="signal")


# -----------------------------------------------------------------------------
# Simulador simple: 1 posición a la vez, TP/SL ±pct, cierre EOD, sin solapes
# -----------------------------------------------------------------------------
def _simulate(df: pd.DataFrame, sig: pd.Series, p: BTParams) -> tuple[pd.Series, pd.DataFrame]:
    """
    Ejecuta backtest discreto a cierre de barra:
      - abre en el 'close' de la barra donde aparece la señal y no hay posición,
      - aplica TP/SL con 'high/low' intra-bar en las barras siguientes,
      - cierra al final del día si sigue abierta,
      - respeta cooldown_bars antes de aceptar nueva entrada,
      - 1 posición a la vez, sin solapes.
    """
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        # No hay OHLC → equity vacía, trades vacíos
        return pd.Series(dtype="float64"), pd.DataFrame()

    # Orden y limpieza
    df = df.copy()
    df = df.sort_index()
    sig = sig.reindex(df.index).fillna(0).astype(int)

    equity = []
    trades = []

    position = 0  # -1 short, 0 flat, 1 long
    entry_price = None
    entry_time = None
    shares = 0
    cooldown = 0

    capital = float(p.capital_per_trade)
    slip = float(p.slippage_bps) / 10000.0  # bps → pct
    tp_pct = float(p.tp_pct)
    sl_pct = float(p.sl_pct)

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
                        if position == 1:
                            tp_level = entry_price * (1 + tp_pct)
                            sl_level = entry_price * (1 - sl_pct)
                        else:
                            tp_level = entry_price * (1 - tp_pct)
                            sl_level = entry_price * (1 + sl_pct)
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
            tp_pct=float(params.get("tp_pct", 0.005)),
            sl_pct=float(params.get("sl_pct", 0.005)),
            cooldown_bars=int(params.get("cooldown_bars", 0)),
            allow_short=bool(params.get("allow_short", _ALLOW_SHORT)),
            slippage_bps=float(params.get("slippage_bps", 0.0)),
            capital_per_trade=float(params.get("capital_per_trade", _CAPITAL)),
            commission_per_trade=float(params.get("commission_per_trade", _COMMISSION)),
        )
        print(
            f"BACKTEST DEBUG - Parámetros BT: threshold={bt_params.threshold}, tp={bt_params.tp_pct}, sl={bt_params.sl_pct}")

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
