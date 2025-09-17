# apps/APP_cuadro_mando.py
from __future__ import annotations

from pathlib import Path
import sys
import io
import platform
import subprocess
import importlib
from typing import List, Dict
import json
from datetime import datetime
import numpy as np
import mlflow

from mlflow.tracking import MlflowClient
from engine.backtest_runner import run_backtest_for_ticker  # usado en Datos (no en research lanzado)
from engine.features import FEATURES as FEATURE_REG

import pandas as pd
import streamlit as st

# --- asegura imports desde la raíz del repo (porque este archivo vive en apps/) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------- settings & backend --------
import settings as settings
S = settings.S  # snapshot inicial

# Importa funciones y también el módulo para poder actualizar su S dinámicamente
import utils.data_update as data_update_mod
from utils.data_update import (
    get_tickers_from_file,
    last_available_table,
    update_many,
    ib_import_status,
)

# opcional: mitigar conflictos de loops en algunos entornos
try:
    import nest_asyncio  # type: ignore
    nest_asyncio.apply()
except Exception:
    pass


def _mlflow_setup_for_app():
    """Configura tracking URI/experimento para que todo caiga en el mismo sitio."""
    abs_uri = (ROOT / "mlruns").resolve().as_uri()
    tracking_uri_cfg = getattr(S, "mlflow_tracking_uri", None)
    try:
        if tracking_uri_cfg:
            if tracking_uri_cfg.startswith("file:"):
                if tracking_uri_cfg in ("file:./mlruns", "file:mlruns") or tracking_uri_cfg.startswith("file:./"):
                    mlflow.set_tracking_uri(abs_uri)
                else:
                    mlflow.set_tracking_uri(tracking_uri_cfg)
            else:
                mlflow.set_tracking_uri(tracking_uri_cfg)
        else:
            mlflow.set_tracking_uri(abs_uri)
    except Exception:
        mlflow.set_tracking_uri(abs_uri)

    exp_name = getattr(S, "mlflow_experiment", "PHIBOT")
    mlflow.set_experiment(exp_name)
    return exp_name


def _cv_json_path(ticker: str, timeframe: str) -> Path:
    cv_dir = Path(getattr(S, "cv_dir", Path(S.logs_path) / "cv"))
    cv_dir.mkdir(parents=True, exist_ok=True)
    return cv_dir / f"{ticker.upper()}_{timeframe}_cv.json"


def _run_cv_subprocess(
    ticker: str,
    timeframe: str,
    *,
    date_from: str | None,
    date_to: str | None,
    thresholds: list[float] | None,
    model: str | None,
    feature_set: str | None,
    days: int | None,
    tp: float | None,
    sl: float | None,
    time_limit: int | None,
    features_csv: str | None = None,
    n_splits: int | None = None,
    test_size: int | None = None,
    capture_output: bool = True,
) -> tuple[int, str]:
    """Lanza RSH_TimeSeriesCV.py con overrides via CLI. Devuelve (rc, salida)."""
    script = ROOT / "RSH_TimeSeriesCV.py"
    if not script.exists():
        return (1, f"No encuentro RSH_TimeSeriesCV.py en {script}")

    args = [sys.executable, str(script), "--ticker", ticker, "--timeframe", timeframe]

    if date_from:
        args += ["--date_from", date_from]
    if date_to:
        args += ["--date_to", date_to]
    if thresholds and len(thresholds) > 0:
        args += ["--thresholds", ",".join(str(float(x)) for x in thresholds)]
    if model:
        args += ["--model", model]
    if feature_set:
        args += ["--feature_set", feature_set]
    if days is not None and not (date_from or date_to):
        args += ["--days", str(int(days))]
    if tp is not None:
        args += ["--tp", str(float(tp))]
    if sl is not None:
        args += ["--sl", str(float(sl))]
    if time_limit is not None:
        args += ["--time_limit", str(int(time_limit))]
    if features_csv:
        args += ["--features", features_csv]
    if n_splits is not None:
        args += ["--n_splits", str(int(n_splits))]
    if test_size is not None:
        args += ["--test_size", str(int(test_size))]

    try:
        proc = subprocess.run(args, check=False, capture_output=capture_output, text=True)
        out = (proc.stdout or "") + ("\n" + (proc.stderr or "") if proc.stderr else "")
        return (proc.returncode, out)
    except Exception as e:
        return (1, f"Error lanzando CV: {e}")


def _read_cv_json(ticker: str, timeframe: str) -> dict | None:
    p = _cv_json_path(ticker, timeframe)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_scenarios_subprocess(
    ticker: str,
    timeframe: str,
    *,
    models_csv: str,
    features_csv: str | None,
    feature_set: str | None,
    thresholds_csv: str | None,
    tp_grid: str | None,
    sl_grid: str | None,
    tl_grid: str | None,
    iter_thr_bt: bool,
    date_from: str | None,
    date_to: str | None,
    days: int | None,
    n_splits: int | None,
    test_size: int | None,
    scheme: str | None,
    embargo: int | None,
    purge: int | None,
    tp: float | None,
    sl: float | None,
    couple_labeling_exec: bool,
    time_limit: int | None,
    primary_metric: str,
    min_trades: int,
) -> tuple[int, str]:
    """Lanza RSH_Scenarios.py para un único run MLflow por ticker."""
    script = ROOT / "RSH_Scenarios.py"
    if not script.exists():
        return (1, f"No encuentro {script}")
    args = [sys.executable, str(script),
            "--ticker", ticker, "--timeframe", timeframe,
            "--models", models_csv,
            "--primary_metric", primary_metric,
            "--min_trades", str(int(min_trades))]
    if features_csv:
        args += ["--features", features_csv]
    else:
        args += ["--feature_set", feature_set or "core"]
    if thresholds_csv:
        args += ["--thresholds", thresholds_csv]
    if tp_grid:
        args += ["--tp_grid", tp_grid]
    if sl_grid:
        args += ["--sl_grid", sl_grid]
    if tl_grid:
        args += ["--tl_grid", tl_grid]
    if iter_thr_bt:
        args += ["--iter_thr_in_bt"]
    if date_from:
        args += ["--date_from", date_from]
    if date_to:
        args += ["--date_to", date_to]
    if days is not None:
        args += ["--days", str(int(days))]
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
    if tp is not None:
        args += ["--tp", str(float(tp))]
    if sl is not None:
        args += ["--sl", str(float(sl))]
    if time_limit is not None:
        args += ["--time_limit", str(int(time_limit))]
    if couple_labeling_exec:
        args += ["--couple_labeling_exec"]
    else:
        args += ["--no_couple_labeling_exec"]

    proc = subprocess.run(args, check=False, capture_output=True, text=True)
    out = (proc.stdout or "") + ("\n" + (proc.stderr or "") if proc.stderr else "")
    return (proc.returncode, out)


# ---------------------- UI base ----------------------
st.set_page_config(page_title="πBOT • Cuadro de mando", layout="wide")
st.title("📊 πBOT — Cuadro de mando")

# ---------------------- Sidebar: diagnóstico & control de rutas ----------------------
st.sidebar.header("Diagnóstico entorno")
st.sidebar.write("Python:", sys.executable)
st.sidebar.write("Versión:", platform.python_version())
st.sidebar.write("ib_insync:", ib_import_status())

with st.sidebar.expander("sys.path", expanded=False):
    st.code("\n".join(sys.path), language="text")

# Recargar settings en caliente
if st.sidebar.button("🔁 Recargar settings"):
    importlib.reload(settings)
    S = settings.S
    # Alinear el backend con los settings recargados
    data_update_mod.S = S
    st.sidebar.success("Settings recargados y backend sincronizado.")

# Ruta parquet actual desde settings
_current_parquet_base = Path(
    getattr(S, "parquet_base_path", getattr(S, "parquet", {}).get("base_path", Path(S.data_path) / "parquet"))
)
st.sidebar.write("Parquet (settings):", str(_current_parquet_base))

# Permite sobrescribir la ruta de parquet solo para esta sesión
_override = st.sidebar.text_input(
    "Override parquet base (opcional)",
    value=str(_current_parquet_base),
    help="Usa esto si settings apunta a 'dataset' y quieres forzar [DAT]_data/parquet para esta sesión."
)
_use_override = st.sidebar.checkbox("Usar override en esta sesión", value=True)

# Aplica override: settings.S y backend.S
if _use_override:
    try:
        setattr(S, "parquet_base_path", _override)
        setattr(settings, "S", S)      # actualizar settings global
        data_update_mod.S = S          # asegurar que el backend usa la misma S
        st.sidebar.success("Override aplicado en settings.S y backend.")
    except Exception as e:
        st.sidebar.warning(f"No pude fijar override en settings/backend: {e}")

# Si falta ib_insync, ofrece instalarlo desde la UI
if ib_import_status() != "ok":
    if st.sidebar.button("📦 Instalar ib-insync en este entorno"):
        with st.sidebar:
            st.write("Ejecutando: pip install ib-insync …")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ib-insync"])
                st.success("Instalado. Pulsa ‘Rerun’ para recargar la app.")
            except subprocess.CalledProcessError as e:
                st.error(f"Fallo instalando ib-insync: {e}")

# ------------------ NAV ------------------
tab = st.sidebar.radio("Módulo", ["Datos", "Research", "Entrenamiento", "Live"], index=0)

# ================== TAB: DATOS ==================
if tab == "Datos":
    st.header("📦 Datos (IBKR → Parquet incremental)")

    timeframes = ["1min", "5mins", "15mins", "1hour", "1day"]
    default_tf = getattr(S, "timeframe_default", "5mins")
    idx = timeframes.index(default_tf) if default_tf in timeframes else 1
    timeframe = st.selectbox("Timeframe", timeframes, index=idx)

    # Info de destino (esquema particionado por día)
    data_root = Path(S.data_path)
    parquet_root = Path(getattr(S, "parquet_base_path",
                                getattr(S, "parquet", {}).get("base_path", data_root / "parquet")))
    st.info(f"📂 Parquet base en uso: `{parquet_root}`")
    st.caption("Esquema: `parquet/ohlcv/ticker=<TICKER>_<TFUP>/date=<YYYY-MM-DD>/data.parquet`")

    exists = parquet_root.exists()
    st.write("✅ Existe" if exists else "❌ **No existe**")
    if not exists:
        st.stop()

    # Carga de tickers
    cfg_dir = Path(getattr(S, "config_path"))
    default_txt = cfg_dir / "sp500_tickers.txt"

    src = st.radio("Origen de tickers", ["Archivo (.txt)", "Pegar lista"], index=0, horizontal=True)
    tickers: List[str] = []
    if src == "Archivo (.txt)":
        st.caption(f"Archivo por defecto: `{default_txt}`")
        up = st.file_uploader("…o sube otro .txt", type=["txt"])
        if up is None:
            try:
                tickers = get_tickers_from_file(default_txt)
            except Exception as e:
                st.error(f"No pude leer {default_txt.name}: {e}")
                tickers = []
        else:
            content = up.read().decode("utf-8", errors="ignore")
            tickers = [t.strip().upper() for t in content.splitlines() if t.strip()]
    else:
        pasted = st.text_area("Tickers (uno por línea)", height=160, placeholder="AAPL\nMSFT\nGOOGL")
        tickers = [t.strip().upper() for t in pasted.splitlines() if t.strip()]

    tickers = sorted(set(tickers))
    st.success(f"{len(tickers)} tickers cargados")

    # Explorador del esquema nuevo (ohlcv particionado por fecha)
    with st.expander("📄 Archivos parquet (ohlcv particionado por día)", expanded=False):
        ohlcv_dir = parquet_root / "ohlcv"
        files = []
        if ohlcv_dir.exists():
            for p in sorted(ohlcv_dir.rglob("*.parquet"), key=lambda x: x.stat().st_mtime, reverse=True)[:200]:
                files.append({
                    "file": str(p.relative_to(parquet_root)),
                    "size_MB": round(p.stat().st_size / 1_000_000, 3),
                    "modified": pd.to_datetime(p.stat().st_mtime, unit="s"),
                })
        df_files = pd.DataFrame(files)
        if df_files.empty:
            st.write("No hay ficheros aún en `parquet/ohlcv`.")
        else:
            st.dataframe(df_files, use_container_width=True, hide_index=True)

        # Previsualización de un ticker + fecha
        if tickers:
            sym_prev = st.selectbox("🔎 Previsualizar ticker", options=tickers)
            tf_up = timeframe.replace(" ", "").upper()
            tdir = ohlcv_dir / f"ticker={sym_prev}_{tf_up}"
            if tdir.exists():
                dates = [p.name.replace("date=", "") for p in tdir.iterdir() if p.is_dir() and p.name.startswith("date=")]
                dates = sorted(dates)
                if dates:
                    dsel = st.selectbox("Fecha (partición)", options=list(reversed(dates)))
                    cand = tdir / f"date={dsel}" / "data.parquet"
                    if cand.exists():
                        try:
                            df_prev = pd.read_parquet(cand).tail(10)
                            st.caption(f"Vista de `{cand}` (últimas 10 filas)")
                            st.dataframe(df_prev, use_container_width=True, hide_index=True)
                        except Exception as e:
                            st.warning(f"No se pudo leer `{cand}`: {e}")
                    else:
                        st.info("No hay fichero `data.parquet` para esa fecha.")
                else:
                    st.info("Ese ticker aún no tiene particiones de fecha.")
            else:
                st.info("Ese ticker aún no tiene parquet en este timeframe.")

    # ---- Actividad de escritura: nº de parquet por día (para este timeframe) ----
    st.subheader("📈 Actividad de escritura (parquets por día)")
    tf_up = timeframe.replace(" ", "").upper()
    ohlcv_dir = parquet_root / "ohlcv"

    counts = {}
    if ohlcv_dir.exists():
        for tdir in ohlcv_dir.glob(f"ticker=*_{tf_up}"):
            if not tdir.is_dir():
                continue
            for ddir in tdir.glob("date=*"):
                if not ddir.is_dir():
                    continue
                date_str = ddir.name.replace("date=", "")
                data_pq = ddir / "data.parquet"
                has_pq = data_pq.exists() or any(ddir.rglob("*.parquet"))
                if not has_pq:
                    continue
                counts[date_str] = counts.get(date_str, 0) + 1

    if not counts:
        st.info("No se han encontrado particiones en este timeframe todavía.")
        df_counts = pd.DataFrame(columns=["date", "n_parquets"])
    else:
        df_counts = (
            pd.DataFrame({"date": pd.to_datetime(list(counts.keys())), "n_parquets": list(counts.values())})
            .sort_values("date")
            .reset_index(drop=True)
        )
        max_days = len(df_counts)
        default_days = 60 if max_days >= 60 else max_days
        window = st.slider("Ventana (días visibles)", min_value=7, max_value=max(7, max_days),
                           value=default_days, step=1, key="counts_window")
        df_view = df_counts.tail(window).copy()
        df_view.set_index("date", inplace=True)
        st.bar_chart(df_view["n_parquets"])
        with st.expander("Ver tabla resumen (últimos N días)"):
            st.dataframe(df_view.reset_index().rename(columns={"date": "día", "n_parquets": "#parquets"}),
                         use_container_width=True, hide_index=True)

    # Último dato disponible
    st.subheader("🕒 Último dato disponible")
    if tickers:
        df_last = last_available_table(tickers, timeframe)
        c1, c2 = st.columns([3, 1])
        with c1:
            q = st.text_input("Filtrar por ticker…").strip().upper()
        with c2:
            n_show = st.slider("Top N más antiguos", 10, 200, 50, 10)

        view = df_last.copy()
        if q:
            view = view[view["ticker"].str.contains(q)]
        st.caption("🔧 Revisa primero los más antiguos (data freshness)")
        st.dataframe(view.sort_values("last_dt", ascending=True).head(n_show), use_container_width=True, hide_index=True)
        with st.expander("Ver todos (descendente)", expanded=False):
            st.dataframe(df_last.sort_values("last_dt", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Carga tickers para ver el estado y poder actualizar.")

    # Actualización (descarga incremental)
    st.divider()
    st.subheader("⬇️ Actualizar desde IBKR → Parquet (incremental)")

    if ib_import_status() != "ok":
        st.info("Falta `ib_insync` en este proceso. Instálalo (botón en la barra lateral) o ejecuta `poetry add ib-insync`.")
    elif not tickers:
        st.info("Carga tickers para activar la actualización.")
    else:
        sel = st.multiselect("Selecciona tickers (vacío = todos)", options=tickers, default=[])
        to_update = sel if sel else tickers

        # Progreso por TICKER: tablero de estados + barra
        batch_size = st.number_input("Tamaño de lote (descarga paralela interna)", min_value=5, max_value=200, value=25, step=5)
        status_map: Dict[str, str] = {t: "Pending" for t in to_update}
        status_placeholder = st.empty()
        progress = st.progress(0, text="Pendiente…")

        def render_status():
            df_status = pd.DataFrame(
                [{"ticker": t, "status": status_map[t]} for t in sorted(status_map.keys())]
            )
            order = {"Running": 0, "Pending": 1, "OK": 2, "ERR": 3}
            df_status["ord"] = df_status["status"].map(order).fillna(9)
            df_status = df_status.sort_values(["ord", "ticker"]).drop(columns="ord")
            status_placeholder.dataframe(df_status, use_container_width=True, hide_index=True)

        render_status()

        if st.button(f"🚀 Actualizar {len(to_update)} ticker(s)", type="primary", use_container_width=True):
            all_rows = []
            total = len(to_update)
            done = 0
            batches = [to_update[i:i+batch_size] for i in range(0, len(to_update), batch_size)]

            try:
                for bi, batch in enumerate(batches, start=1):
                    for t in batch:
                        status_map[t] = "Running"
                    render_status()
                    progress.progress(done / total, text=f"Lote {bi}/{len(batches)}: ejecutando {len(batch)} tickers…")

                    res = update_many(batch, timeframe)

                    for r in res:
                        all_rows.append({
                            "ticker": r.ticker,
                            "added_rows": r.added_rows,
                            "last_dt_after": r.last_dt_after,
                            "status": "OK" if r.error is None else "ERR",
                        })
                        status_map[r.ticker] = "OK" if r.error is None else "ERR"

                    done += len(batch)
                    render_status()
                    progress.progress(min(done / total, 1.0), text=f"Procesados {done}/{total} tickers")

                st.success("Actualización completada.")
                df_res = pd.DataFrame(all_rows).sort_values(["status", "ticker"])
                st.dataframe(df_res, use_container_width=True, hide_index=True)

                buf = io.StringIO()
                df_res.to_csv(buf, index=False)
                st.download_button(
                    "⬇️ Descargar resumen (.csv)",
                    buf.getvalue(),
                    file_name=f"update_summary_{timeframe}.csv",
                    mime="text/csv",
                )

                st.subheader("🕒 Último dato tras la actualización")
                st.dataframe(last_available_table(to_update, timeframe), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Error en la actualización: {e}")

# ================== TAB: RESEARCH ==================
elif tab == "Research":
    st.header("🔬 Research (CV multi-modelo + grid en un único run por ticker)")

    # ---------- Lanzador ----------
    st.subheader("🚀 Lanzador de escenarios (single-run)")

    # ✅ Nuevo: acoplar labeling y backtest (mismo grid TP/SL/TL/threshold)
    couple_labeling_exec = st.checkbox(
        "Acoplar labeling y backtest (usar el mismo grid para TP/SL/TL/threshold)",
        value=True,
        help=(
            "Si está activado, la CV (labeling) se ejecuta por cada combinación del grid "
            "usando exactamente los mismos TP/SL/TimeLimit/Threshold que el backtest. "
            "Así cada run corresponde a una única combinación coherente."
        ),
        key="rsh_coupled",
    )

    # timeframe
    timeframes = ["1min", "5mins", "15mins", "1hour", "1day"]
    default_tf = getattr(S, "timeframe_default", "5mins")
    idx = timeframes.index(default_tf) if default_tf in timeframes else 1
    timeframe = st.selectbox("Timeframe", timeframes, index=idx, key="rsh_tf")

    # tickers (archivo o pegar)
    cfg_dir = Path(getattr(S, "config_path"))
    default_txt = cfg_dir / "sp500_tickers.txt"
    src = st.radio("Origen de tickers", ["Archivo (.txt)", "Pegar lista"], horizontal=True, key="rsh_src")
    tickers: List[str] = []
    if src == "Archivo (.txt)":
        st.caption(f"Archivo por defecto: `{default_txt}`")
        up = st.file_uploader("…o sube otro .txt", type=["txt"], key="rsh_up")
        if up is None:
            try:
                tickers = get_tickers_from_file(default_txt)
            except Exception as e:
                st.error(f"No pude leer {default_txt.name}: {e}")
                tickers = []
        else:
            content = up.read().decode("utf-8", errors="ignore")
            tickers = [t.strip().upper() for t in content.splitlines() if t.strip()]
    else:
        pasted = st.text_area("Tickers (uno por línea)", height=120, placeholder="AAPL\nMSFT\nGOOGL", key="rsh_paste")
        tickers = [t.strip().upper() for t in pasted.splitlines() if t.strip()]
    tickers = sorted(set(tickers))
    st.success(f"{len(tickers)} tickers cargados")

    # fechas / ventana
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        date_from = st.date_input("Fecha FROM (UTC)", value=None, key="rsh_from")
        date_from_str = date_from.strftime("%Y-%m-%d") if date_from else None
    with c2:
        date_to = st.date_input("Fecha TO (UTC)", value=None, key="rsh_to")
        date_to_str = date_to.strftime("%Y-%m-%d") if date_to else None
    with c3:
        days_hist = st.number_input("Últimos N días (solo si no hay fechas)", min_value=0, value=0, step=5, key="rsh_days", help="Se usa solo si date_from/date_to están vacíos")

    # CV config básica
    c_ns, c_ts = st.columns([1, 1])
    with c_ns:
        n_splits = st.number_input("n_splits", min_value=2, max_value=10, value=int(getattr(S, "n_splits_cv", 5)), step=1, key="rsh_ns")
    with c_ts:
        test_size = st.number_input("test_size (barras)", min_value=20, max_value=2000, value=int(getattr(S, "cv_test_size", 500)), step=10, key="rsh_ts")

    # modelos & features
    c4, c5, c6 = st.columns([1, 1, 1])
    with c4:
        models_sel = st.multiselect("Modelos", ["xgb", "rf", "logreg"], default=["xgb"], key="rsh_models")
        models_csv = ",".join(models_sel) if models_sel else "xgb"
    with c5:
        feature_set = st.selectbox("Feature set (legacy)", options=["core", "core+vol", "all"],
                                   index=["core","core+vol","all"].index(getattr(S, "feature_set", "core")),
                                   key="rsh_featset")
    with c6:
        st.caption("Label/triple-barrier (si *no* acoplas)")
        if not couple_labeling_exec:
            tp_lbl = st.number_input("TP× (labeling)", min_value=0.1, max_value=10.0, value=3.0, step=0.1, key="rsh_tp_lbl")
            sl_lbl = st.number_input("SL× (labeling)", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="rsh_sl_lbl")
            tl_lbl = st.number_input("Time limit (labeling, barras)", min_value=1, max_value=500, value=16, step=1, key="rsh_tl_lbl")
        else:
            st.caption("🔗 Acoplado: la CV usará el grid de TP/SL/TL del backtest.")
            tp_lbl = sl_lbl = tl_lbl = None

    # Features seleccionables
    st.markdown("**Features seleccionables (engine.features):**")
    feat_keys = sorted(list(FEATURE_REG.keys()))
    selected_feats = st.multiselect(
        "Selecciona features (opcional; vacío = usar 'feature_set' legacy)",
        options=feat_keys,
        default=["sma_20","ema_12","rsi_14","atr_14","gk_14","vwap_20","volz_20"],
        key="rsh_feat_sel"
    )
    features_csv = ",".join(selected_feats) if selected_feats else None

    # thresholds & grids
    default_thr = getattr(S, "cv_threshold_grid", [0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    thr_text = st.text_input("Thresholds (CSV, 0–1)", value=",".join(str(x) for x in default_thr), key="rsh_thr")
    tp_grid_txt = st.text_input("TP grid (CSV, multiplicadores absolutos)", value="1,2,3", key="rsh_tpgrid")
    sl_grid_txt = st.text_input("SL grid (CSV, multiplicadores absolutos)", value="1,2,3", key="rsh_slgrid")
    tl_grid_txt = st.text_input("Time limit grid (CSV, barras)", value="8,12,16", key="rsh_tlgrid")
    iter_thr_in_bt = st.checkbox("Iterar también los thresholds en el backtest", value=True, key="iter_thr_bt")

    # selección del “mejor”
    cmet, cmtr = st.columns([1, 1])
    with cmet:
        primary_metric = st.selectbox("Métrica primaria de selección", options=["net_return","sharpe","win_rate","profit_factor"], index=0)
    with cmtr:
        min_trades = st.number_input("Mínimo de trades para ser candidato", min_value=0, max_value=10000, value=20, step=5)

    # botones
    colb1, colb2 = st.columns([1, 1])
    run_scen_btn = colb1.button("🏁 Ejecutar Research (run único por ticker)", type="primary", use_container_width=True)
    run_cv_btn = colb2.button("▶️ Solo CV (debug)", use_container_width=True)

    # trackers
    log_area = st.empty()
    progress = st.progress(0.0, text="Esperando…")

    if (run_scen_btn or run_cv_btn) and tickers:
        exp_name = _mlflow_setup_for_app()
        logs = []
        total = len(tickers)
        done = 0

        for tk in tickers:
            progress.progress(done / total, text=f"Procesando {tk}…")

            if run_cv_btn and not run_scen_btn:
                # Solo CV “suelta” (útil para comprobar datos/fechas/features)
                rc, out = _run_cv_subprocess(
                    tk, timeframe,
                    date_from=date_from_str, date_to=date_to_str,
                    thresholds=[float(x) for x in thr_text.split(",") if str(x).strip() != ""],
                    model=models_sel[0] if models_sel else "xgb",
                    feature_set=feature_set if not selected_feats else None,
                    days=int(days_hist) if days_hist else None,
                    # estos tres solo se aplican si NO acoplas; si acoplas se ignoran por el script
                    tp=float(tp_lbl) if (tp_lbl is not None) else None,
                    sl=float(sl_lbl) if (sl_lbl is not None) else None,
                    time_limit=int(tl_lbl) if (tl_lbl is not None) else None,
                    features_csv=features_csv,
                    n_splits=int(n_splits) if n_splits else None,
                    test_size=int(test_size) if test_size else None,
                )
            else:
                # Research: un run por combinación coherente
                rc, out = _run_scenarios_subprocess(
                    ticker=tk, timeframe=timeframe,
                    models_csv=",".join(models_sel) if models_sel else "xgb",
                    features_csv=features_csv,
                    feature_set=feature_set,
                    thresholds_csv=thr_text if thr_text.strip() else None,
                    tp_grid=tp_grid_txt, sl_grid=sl_grid_txt, tl_grid=tl_grid_txt,
                    iter_thr_bt=bool(iter_thr_in_bt),
                    date_from=date_from_str, date_to=date_to_str, days=int(days_hist) if days_hist else None,
                    n_splits=int(n_splits) if n_splits else None,
                    test_size=int(test_size) if test_size else None,
                    scheme="expanding", embargo=16, purge=16,
                    # labeling manual SOLO si está des-acoplado:
                    tp=float(tp_lbl) if (not couple_labeling_exec and tp_lbl is not None) else None,
                    sl=float(sl_lbl) if (not couple_labeling_exec and sl_lbl is not None) else None,
                    time_limit=int(tl_lbl) if (not couple_labeling_exec and tl_lbl is not None) else None,
                    primary_metric=primary_metric, min_trades=int(min_trades),
                    # ✅ Flag de acoplamiento hacia el script
                    couple_labeling_exec=bool(couple_labeling_exec),
                )

            returncode_msg = "OK" if rc == 0 else "ERROR"
            logs.append(f"=== {tk} / {timeframe} ===\n{returncode_msg}:\n{out}\n")
            log_area.code("\n".join(logs[-8:]), language="bash")

            done += 1
            progress.progress(done / total, text=f"Procesados {done}/{total}")

        st.success("Lanzamiento finalizado. Revisa la sección de visualización y/o la UI de MLflow.")
    # En APP_cuadro_mando.py, añadir una nueva sección después del lanzador de scenarios:

    # ---------- Panel de Calidad de Modelos ----------
    st.divider()
    st.subheader("📊 Panel de Calidad de Modelos")

    if st.button("🔍 Analizar Calidad de Runs Recientes"):
        try:
            exp_name = _mlflow_setup_for_app()
            client = MlflowClient()
            experiment = client.get_experiment_by_name(exp_name)

            if experiment:
                # Obtener runs recientes
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=50,
                    order_by=["start_time DESC"]
                )

                if runs:
                    # Extraer métricas de calidad
                    quality_data = []
                    for run in runs:
                        metrics = run.data.metrics
                        params = run.data.params

                        quality_data.append({
                            "run_name": run.info.run_name,
                            "ticker": params.get("ticker", "N/A"),
                            "model": params.get("model", "N/A"),
                            "temporal_coverage": metrics.get("temporal_coverage_pct", 0),
                            "trading_days_actual": metrics.get("trading_days_actual", 0),
                            "missing_days": metrics.get("missing_days", 0),
                            "feature_count": metrics.get("feature_count", 0),
                            "top5_concentration": metrics.get("top5_feature_concentration", 0),
                            "zero_features": metrics.get("zero_importance_features", 0),
                            "net_return": metrics.get("net_return", 0),
                            "sharpe": metrics.get("sharpe", 0),
                            "n_trades": metrics.get("n_trades", 0),
                            "data_quality": run.data.tags.get("data_quality", "UNKNOWN"),
                            "feature_diversity": run.data.tags.get("feature_diversity", "UNKNOWN")
                        })

                    df_quality = pd.DataFrame(quality_data)

                    # Filtros
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        min_coverage = st.slider("Cobertura temporal mínima (%)", 0, 100, 80)
                        df_filtered = df_quality[df_quality["temporal_coverage"] >= min_coverage]
                    with col2:
                        max_concentration = st.slider("Concentración máxima top 5 (%)", 0, 100, 80)
                        df_filtered = df_filtered[df_filtered["top5_concentration"] <= max_concentration]
                    with col3:
                        min_trades = st.number_input("Trades mínimos", 0, 1000, 10)
                        df_filtered = df_filtered[df_filtered["n_trades"] >= min_trades]

                    st.write(f"Mostrando {len(df_filtered)}/{len(df_quality)} runs que pasan los filtros de calidad")

                    # Panel de calidad
                    col_temp, col_feat = st.columns([1, 1])

                    with col_temp:
                        st.markdown("**📅 Calidad Temporal**")
                        quality_counts = df_filtered["data_quality"].value_counts()

                        for quality, count in quality_counts.items():
                            color = {
                                "EXCELLENT": "🟢", "GOOD": "🟡",
                                "ACCEPTABLE": "🟠", "POOR": "🔴",
                                "CRITICAL": "❌"
                            }.get(quality, "⚪")
                            st.write(f"{color} {quality}: {count} runs")

                        # Gráfico de cobertura temporal
                        if len(df_filtered) > 0:
                            st.bar_chart(df_filtered.set_index("ticker")["temporal_coverage"])

                    with col_feat:
                        st.markdown("**🎯 Diversidad de Features**")
                        diversity_counts = df_filtered["feature_diversity"].value_counts()

                        for diversity, count in diversity_counts.items():
                            color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(diversity, "⚪")
                            st.write(f"{color} {diversity}: {count} runs")

                        # Gráfico de concentración
                        if len(df_filtered) > 0:
                            st.bar_chart(df_filtered.set_index("ticker")["top5_concentration"])

                    # Tabla detallada
                    st.markdown("**📋 Detalle de Runs**")
                    display_cols = [
                        "ticker", "model", "temporal_coverage", "missing_days",
                        "feature_count", "top5_concentration", "zero_features",
                        "net_return", "sharpe", "n_trades"
                    ]

                    st.dataframe(
                        df_filtered[display_cols].round(2),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Alertas automáticas
                    st.markdown("**⚠️ Alertas de Calidad**")
                    alerts = []

                    low_coverage = df_filtered[df_filtered["temporal_coverage"] < 80]
                    if len(low_coverage) > 0:
                        alerts.append(f"🔴 {len(low_coverage)} runs con cobertura temporal < 80%")

                    high_concentration = df_filtered[df_filtered["top5_concentration"] > 85]
                    if len(high_concentration) > 0:
                        alerts.append(f"🟡 {len(high_concentration)} runs con alta concentración de features (>85%)")

                    few_trades = df_filtered[df_filtered["n_trades"] < 20]
                    if len(few_trades) > 0:
                        alerts.append(f"🟠 {len(few_trades)} runs con pocos trades (<20)")

                    if alerts:
                        for alert in alerts:
                            st.warning(alert)
                    else:
                        st.success("✅ Todos los runs pasan los controles de calidad básicos")

                else:
                    st.info("No se encontraron runs recientes para analizar")
            else:
                st.error("No se encontró el experimento MLflow")

        except Exception as e:
            st.error(f"Error analizando calidad: {e}")

    # ---------- Visualización de Features Top ----------
    st.divider()
    st.subheader("🎯 Features Más Importantes")

    ticker_analysis = st.selectbox("Selecciona ticker para análisis de features",
                                   options=tickers if tickers else ["AAPL"],
                                   key="feature_analysis_ticker")

    if st.button("📊 Analizar Features"):
        # Esto requeriría acceso a los modelos entrenados
        # Por ahora, mostrar placeholder
        st.info("Funcionalidad en desarrollo - requiere acceso a modelos entrenados")

        # Mock data para demostrar la interfaz
        mock_features = [
            {"feature": "rsi_14", "importance": 18.5},
            {"feature": "atr_14", "importance": 15.2},
            {"feature": "sma_20", "importance": 12.8},
            {"feature": "vwap_20", "importance": 11.1},
            {"feature": "ema_12", "importance": 9.7},
        ]

        col1, col2 = st.columns([2, 1])
        with col1:
            chart_data = pd.DataFrame(mock_features)
            st.bar_chart(chart_data.set_index("feature")["importance"])

        with col2:
            st.write("**Top Features:**")
            for feat in mock_features:
                st.write(f"• {feat['feature']}: {feat['importance']:.1f}%")

elif tab == "Entrenamiento":
    st.header("🏋️ Entrenamiento")
    st.info("Próximamente: entrenamiento y registro del mejor modelo por ticker.")

elif tab == "Live":
    st.header("🟢 Live (paper)")
    st.info("Próximamente: ejecución en paper/live y monitor de operaciones.")
