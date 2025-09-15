# utils/data_update.py
from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
import settings as settings
S = settings.S

# -----------------------------------------------------------------------------
# ib_insync: import perezoso con diagnóstico
# -----------------------------------------------------------------------------
IB = Index = Stock = util = None  # se rellenan en _ensure_ib()
_last_ib_import_error: Optional[str] = None


def _ensure_ib() -> bool:
    """Intenta importar ib_insync on-demand. Guarda el último error para diagnosticar."""
    global IB, Index, Stock, util, _last_ib_import_error
    if IB is not None:
        return True
    try:
        mod = importlib.import_module("ib_insync")
        IB = getattr(mod, "IB")
        Index = getattr(mod, "Index", None)
        Stock = getattr(mod, "Stock")
        util = getattr(mod, "util")
        _last_ib_import_error = None
        return True
    except Exception as e:
        _last_ib_import_error = repr(e)
        return False


def ib_import_status() -> str:
    """'ok' | 'missing:<detalle>' — útil para mostrar en la UI."""
    return "ok" if _ensure_ib() else f"missing:{_last_ib_import_error}"


# -----------------------------------------------------------------------------
# Helpers de rutas y formatos
# -----------------------------------------------------------------------------
def normalize_tf(tf: str) -> str:
    return str(tf).lower().replace(" ", "")


def tf_token_upper(tf: str) -> str:
    """Devuelve el token de timeframe en MAYÚSCULAS sin espacios (ej: '5mins' -> '5MINS')."""
    return normalize_tf(tf).upper()


def bar_size_for(tf: str) -> str:
    # mapea tu config: S.bar_size_by_tf = {"5mins": "5 mins", ...}
    return getattr(S, "bar_size_by_tf", {}).get(tf, "5 mins")


def get_tickers_from_file(file_path: Path) -> List[str]:
    """Lee un .txt con tickers (uno por línea)."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Tickers file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip()]


def _parquet_base() -> Path:
    data_root = Path(S.data_path)
    return Path(
        getattr(S, "parquet_base_path", getattr(S, "parquet", {}).get("base_path", data_root / "parquet"))
    )


def _ohlcv_dir_for(sym: str, tf: str) -> Path:
    """Esquema nuevo: [DAT]_data/parquet/ohlcv/ticker=<SYM>_<TFUP>"""
    parquet_root = _parquet_base()
    return parquet_root / "ohlcv" / f"ticker={sym}_{tf_token_upper(tf)}"


def _list_recent_day_files(sym: str, tf: str, max_days: int = 8) -> List[Path]:
    """
    Lista los ficheros `data.parquet` de las particiones de fecha más recientes
    bajo el nuevo esquema (date=YYYY-MM-DD), ordenado desc por fecha.
    """
    base = _ohlcv_dir_for(sym, tf)
    if not base.exists():
        return []
    # subdirs tipo date=YYYY-MM-DD
    parts = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("date=")]
    # ordena por fecha asc, luego coge los últimos max_days
    def _key(p: Path):
        try:
            return pd.to_datetime(p.name.replace("date=", ""), format="%Y-%m-%d")
        except Exception:
            return pd.Timestamp.min

    parts_sorted = sorted(parts, key=_key)
    recent = parts_sorted[-max_days:]
    out: List[Path] = []
    for d in reversed(recent):  # más recientes primero
        fp = d / "data.parquet"
        if fp.exists():
            out.append(fp)
        else:
            # si hay múltiples ficheros en la partición, recoge todos
            out.extend(sorted(d.rglob("*.parquet"), key=lambda x: x.stat().st_mtime, reverse=True))
    return out


def _parquet_candidates(base: Path, tf: str, sym: str) -> List[Path]:
    """
    Busca parquet siguiendo varias convenciones, con PRIORIDAD al nuevo esquema:
      1) base/ohlcv/ticker=<sym>_<TFUP>/date=YYYY-MM-DD/data.parquet (recientes)
      2) base/<tf>/<sym>.parquet (antiguo)
      3) base/<tf>/<sym>/*.parquet
      4) base/<tf>/ticker=<sym>/*.parquet
      5) base/ticker=<sym>/timeframe=<tf>/*.parquet
    """
    base = Path(base)
    tf_norm = normalize_tf(tf)
    # 1) nuevo esquema
    cands = _list_recent_day_files(sym, tf_norm, max_days=8)

    # fallback antiguos:
    tf_dir = base / tf_norm
    if (tf_dir / f"{sym}.parquet").is_file():
        cands.append(tf_dir / f"{sym}.parquet")
    if (tf_dir / sym).is_dir():
        cands.extend(sorted((tf_dir / sym).rglob("*.parquet")))
    if (tf_dir / f"ticker={sym}").is_dir():
        cands.extend(sorted((tf_dir / f"ticker={sym}").rglob("*.parquet")))
    if (base / f"ticker={sym}" / f"timeframe={tf_norm}").is_dir():
        cands.extend(sorted((base / f"ticker={sym}" / f"timeframe={tf_norm}").rglob("*.parquet")))

    # quitar duplicados preservando orden
    seen, ordered = set(), []
    for p in cands:
        if p not in seen:
            ordered.append(p)
            seen.add(p)
    return ordered


def last_dt_from_parquet(base: Path, tf: str, sym: str) -> Optional[pd.Timestamp]:
    """Devuelve el máximo 'date' encontrado en parquet (prioridad al esquema nuevo particionado por día)."""
    try:
        files = _parquet_candidates(base, tf, sym)
        if not files:
            return None
        last: Optional[pd.Timestamp] = None
        for fp in files[:8]:  # ya viene ordenado por recientes primero
            try:
                df = pd.read_parquet(fp)
                if "date" not in df.columns and "time" in df.columns:
                    df = df.rename(columns={"time": "date"})
                if "date" in df.columns:
                    dt = pd.to_datetime(df["date"], utc=True, errors="coerce").max()
                    if pd.notna(dt):
                        last = dt if last is None or dt > last else last
            except Exception:
                continue
        return last
    except Exception:
        return None


def last_dt_from_csv(base: Path, tf: str, sym: str) -> Optional[pd.Timestamp]:
    """
    Fallback por si aún tienes CSV históricos:
      base/<tf>/<sym>.csv, base/<sym>.csv, base/csv/<sym>.csv
    """
    base = Path(base)
    tf_dir = base / normalize_tf(tf)
    candidates = [tf_dir / f"{sym}.csv", base / f"{sym}.csv", base / "csv" / f"{sym}.csv"]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                col = "date" if "date" in df.columns else "timestamp" if "timestamp" in df.columns else None
                if col:
                    dt = pd.to_datetime(df[col], utc=True, errors="coerce").max()
                    if pd.notna(dt):
                        return dt
            except Exception:
                continue
    return None


def discover_last_dt(sym: str, tf: str, *, parquet_base: Path, csv_base: Path) -> Optional[pd.Timestamp]:
    dt = last_dt_from_parquet(parquet_base, tf, sym)
    if dt is not None:
        return dt
    return last_dt_from_csv(csv_base, tf, sym)


# -----------------------------------------------------------------------------
# IBKR fetch (async)
# -----------------------------------------------------------------------------
async def _fetch_one_async(ib, symbol: str, duration_days: int, bar_size: str) -> pd.DataFrame:
    # asegurar import
    if not _ensure_ib():
        raise RuntimeError(f"ib_insync no está disponible. Detalle: {_last_ib_import_error}")

    # contrato + feeds
    sym = symbol.upper().strip()
    if sym in {"VIX", "^VIX"} and Index is not None:
        contract = Index("VIX", "CBOE")
        feeds = ["TRADES", "BID_ASK", "MIDPOINT"]
    else:
        if Stock is None:
            raise RuntimeError("ib_insync no está instalado; no se pueden crear contratos")
        contract = Stock(sym, "SMART", "USD")
        feeds = ["TRADES"]

    try:
        await ib.qualifyContractsAsync(contract)
    except Exception:
        pass

    duration = f"{int(max(1, duration_days))} D"
    last_err = None
    for what in feeds:
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what,
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
                timeout=float(getattr(S, "ib_timeout_s", 30.0)),
            )
            if not bars:
                raise RuntimeError(f"Respuesta vacía ({what})")
            df = util.df(bars)
            if "date" not in df.columns and "time" in df.columns:
                df = df.rename(columns={"time": "date"})
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

            # si BID_ASK/MIDPOINT: rellena OHLC con close y volumen 0 si falta
            if what in {"BID_ASK", "MIDPOINT"}:
                if "close" not in df.columns and {"bid", "ask"}.issubset(df.columns):
                    df["close"] = (pd.to_numeric(df["bid"], errors="coerce") +
                                   pd.to_numeric(df["ask"], errors="coerce")) / 2
                for c in ["open", "high", "low"]:
                    if c not in df.columns or df[c].isna().all():
                        df[c] = df.get("close")
                if "volume" not in df.columns:
                    df["volume"] = 0

            for c in ["open", "high", "low", "close", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["date"]).sort_values("date")
            if df.empty:
                raise RuntimeError(f"DF vacío tras limpieza ({what})")
            return df.reset_index(drop=True)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"{symbol}: sin datos válidos tras probar {feeds}. Último error: {last_err}")


# -----------------------------------------------------------------------------
# API pública: update_many / last_available_table
# -----------------------------------------------------------------------------
@dataclass
class UpdateResult:
    ticker: str
    added_rows: int
    last_dt_after: Optional[pd.Timestamp]
    error: Optional[str] = None


async def update_many_async(tickers: Iterable[str], timeframe: str) -> List[UpdateResult]:
    """Descarga incremental IBKR y persiste en:
    [DAT]_data/parquet/ohlcv/ticker=<SYM>_<TFUP>/date=<YYYY-MM-DD>/data.parquet
    """
    if not _ensure_ib():
        raise RuntimeError(f"ib_insync no está disponible. Detalle: {_last_ib_import_error}")

    tf_norm = normalize_tf(timeframe)
    bar_size = bar_size_for(tf_norm)

    data_root = Path(S.data_path)
    parquet_root = _parquet_base()
    (parquet_root / "ohlcv").mkdir(parents=True, exist_ok=True)

    # días a pedir por símbolo según última fecha disponible
    now_utc = pd.Timestamp.now(tz="UTC")
    per_symbol_days: Dict[str, int] = {}
    for sym in tickers:
        last_dt = discover_last_dt(sym, tf_norm, parquet_base=parquet_root, csv_base=data_root)
        if last_dt is None:
            base_need = max(int(getattr(S, "days_of_data", 90)), 90)
            per_symbol_days[sym] = min(int(getattr(S, "ib_initial_days_cap", 120)), base_need)
        else:
            per_symbol_days[sym] = max(1, (now_utc - last_dt).days + 1)

    # Conexión IB
    ib = IB()
    await ib.connectAsync(S.ib_host, S.ib_port, clientId=S.ib_client_id, readonly=True)
    if not ib.isConnected():
        raise RuntimeError("No se pudo conectar a IBKR")

    sem = asyncio.Semaphore(int(getattr(S, "ib_max_concurrent", 2)))
    tasks = [_bounded_fetch(ib, sem, sym, per_symbol_days[sym], bar_size) for sym in tickers]

    results: List[UpdateResult] = []
    try:
        for fut in asyncio.as_completed(tasks):
            sym, df_new, err = await fut
            if err is not None:
                results.append(UpdateResult(ticker=sym, added_rows=0, last_dt_after=None, error=str(err)))
                continue

            # incremental contra parquet existente (o csv si aplica)
            last_dt_prev = discover_last_dt(sym, tf_norm, parquet_base=parquet_root, csv_base=data_root)
            if last_dt_prev is not None:
                df_new = df_new[df_new["date"] > last_dt_prev]

            added = int(len(df_new))
            last_dt_after = last_dt_prev

            if added > 0:
                # particiona por día (UTC)
                dser = df_new["date"].dt.tz_convert("UTC") if df_new["date"].dt.tz is not None else df_new["date"]
                df_new = df_new.copy()
                df_new["__day__"] = dser.dt.date

                for day, df_day in df_new.groupby("__day__", sort=True):
                    out_dir = _ohlcv_dir_for(sym, tf_norm) / f"date={pd.to_datetime(day).date().isoformat()}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    outp = out_dir / "data.parquet"

                    try:
                        if outp.exists():
                            old = pd.read_parquet(outp)
                            comb = pd.concat([old, df_day.drop(columns="__day__", errors="ignore")], ignore_index=True)
                            comb = comb.drop_duplicates(subset="date").sort_values("date")
                            comb.to_parquet(outp, index=False)
                        else:
                            df_day.drop(columns="__day__", errors="ignore").to_parquet(outp, index=False)
                    except Exception as e:
                        results.append(UpdateResult(ticker=sym, added_rows=0, last_dt_after=last_dt_prev, error=f"Parquet error ({day}): {e}"))
                        continue

                last_dt_after = discover_last_dt(sym, tf_norm, parquet_base=parquet_root, csv_base=data_root)

            results.append(UpdateResult(ticker=sym, added_rows=added, last_dt_after=last_dt_after, error=None))
    finally:
        if ib.isConnected():
            ib.disconnect()
    return results


async def _bounded_fetch(ib, sem: asyncio.Semaphore, symbol: str, duration_days: int, bar_size: str) -> Tuple[str, pd.DataFrame, Optional[Exception]]:
    async with sem:
        try:
            df = await _fetch_one_async(ib, symbol, duration_days, bar_size)
            return symbol, df, None
        except Exception as e:
            return symbol, pd.DataFrame(), e


def update_many(tickers: Iterable[str], timeframe: str) -> List[UpdateResult]:
    """Wrapper síncrono para apps/CLI."""
    return asyncio.run(update_many_async(tickers, timeframe))


def last_available_table(tickers: Iterable[str], timeframe: str) -> pd.DataFrame:
    """DataFrame con última fecha por ticker/timeframe (Parquet/CSV)."""
    tf_norm = normalize_tf(timeframe)
    data_root = Path(S.data_path)
    parquet_root = _parquet_base()
    rows = []
    for sym in tickers:
        last_dt = discover_last_dt(sym, tf_norm, parquet_base=parquet_root, csv_base=data_root)
        rows.append({"ticker": sym, "timeframe": tf_norm, "last_dt": last_dt})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["last_dt"] = pd.to_datetime(df["last_dt"])
        df = df.sort_values(["last_dt", "ticker"], ascending=[False, True])
    return df


__all__ = [
    "get_tickers_from_file",
    "last_available_table",
    "update_many",
    "ib_import_status",
    "UpdateResult",
]
