# apps/APP_cuadro_mando.py
from __future__ import annotations

from pathlib import Path
import sys
import io
import os
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
from engine.runs_extractor import extract_mlflow_runs

import pandas as pd
import streamlit as st

RSH_SCENARIOS_PATH   = r"C:\Users\Mario_user\Documents\BOTRADING\RSH_Scenarios.py"
RSH_TIMESERIESCV_PATH = r"C:\Users\Mario_user\Documents\BOTRADING\RSH_TimeseriesCV.py"

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

    exp_name = getattr(S, "mlflow_experiment", "PHIBOT_TRAINING")
    mlflow.set_experiment(exp_name)
    return exp_name


def _cv_json_path(ticker: str, timeframe: str) -> Path:
    cv_dir = Path(getattr(S, "cv_dir", Path(S.logs_path) / "RSH_TimeSeriesCV"))
    cv_dir.mkdir(parents=True, exist_ok=True)
    return cv_dir / f"{ticker.upper()}_{timeframe}_cv.json"


def _run_cv_subprocess(
        ticker: str,
        timeframe: str,
        *,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        thresholds: Optional[List[float]] = None,
        model: Optional[str] = None,
        feature_set: Optional[str] = None,
        days: Optional[int] = None,
        tp: Optional[float] = None,
        sl: Optional[float] = None,
        time_limit: Optional[int] = None,
        features_csv: Optional[str] = None,
        n_splits: Optional[int] = None,
        test_size: Optional[int] = None,
) -> Tuple[int, str]:
    """Lanza RSH_TimeseriesCV.py (sin subcomandos)."""
    py = sys.executable
    script = (ROOT / "RSH_TimeseriesCV.py").as_posix()  # ✅ script correcto

    args = [py, script, "--ticker", ticker, "--timeframe", timeframe]
    if date_from:    args += ["--date_from", date_from]
    if date_to:      args += ["--date_to", date_to]
    if thresholds:   args += ["--thresholds", ",".join(str(x) for x in thresholds)]
    if model:        args += ["--model", model]
    if feature_set:  args += ["--feature_set", feature_set]
    if days is not None and not (date_from or date_to): args += ["--days", str(int(days))]
    if tp is not None:         args += ["--tp", str(float(tp))]
    if sl is not None:         args += ["--sl", str(float(sl))]
    if time_limit is not None: args += ["--time_limit", str(int(time_limit))]  # o --candle_limit si así lo usa tu CLI
    if features_csv:           args += ["--features", features_csv]
    if n_splits is not None:   args += ["--n_splits", str(int(n_splits))]
    if test_size is not None:  args += ["--test_size", str(int(test_size))]

    pr = subprocess.run(args, capture_output=True, text=True)
    out = pr.stdout + ("\nSTDERR:\n" + pr.stderr if pr.stderr else "")
    return pr.returncode, out


def _read_cv_json(ticker: str, timeframe: str) -> dict | None:
    p = _cv_json_path(ticker, timeframe)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_scenarios_subprocess(
        *,
        ticker: str,
        timeframe: str,
        models_csv: str,
        features_csv: Optional[str],
        feature_set: Optional[str],
        thresholds_csv: Optional[str],
        tp_grid: str,
        sl_grid: str,
        tl_grid: str,
        iter_thr_bt: bool,
        date_from: Optional[str],
        date_to: Optional[str],
        days: Optional[int],
        n_splits: Optional[int],
        test_size: Optional[int],
        scheme: str,
        embargo: int,
        purge: int,
        primary_metric: str,
        min_trades: int,
        couple_labeling_exec: bool,
) -> Tuple[int, str]:
    """Lanza RSH_Scenarios.py (sin subcomandos)."""
    py = sys.executable
    script = (ROOT / "RSH_Scenarios.py").as_posix()  # ✅ script correcto

    # si tu script usa --candle_limit en vez de --tl_grid, cambia el nombre aquí:
    FLAG_TL_GRID = "--tl_grid"

    args = [
        py, script,
        "--ticker", ticker,
        "--timeframe", timeframe,
        "--models", models_csv,
        "--scheme", scheme,
        "--embargo", str(int(embargo)),
        "--purge", str(int(purge)),
        "--primary_metric", primary_metric,
        "--min_trades", str(int(min_trades)),
    ]
    if features_csv:   args += ["--features", features_csv]
    if feature_set:    args += ["--feature_set", feature_set]
    if thresholds_csv: args += ["--thresholds", thresholds_csv]
    if tp_grid:        args += ["--tp_grid", tp_grid]
    if sl_grid:        args += ["--sl_grid", sl_grid]
    if tl_grid:        args += [FLAG_TL_GRID, tl_grid]
    if iter_thr_bt:    args += ["--iter_thr_in_bt"]   # ✅ nombre real del flag
    if date_from:      args += ["--date_from", date_from]
    if date_to:        args += ["--date_to", date_to]
    if days is not None and not (date_from or date_to):
                       args += ["--days", str(int(days))]
    if n_splits is not None:
                       args += ["--n_splits", str(int(n_splits))]
    if test_size is not None:
                       args += ["--test_size", str(int(test_size))]
    if couple_labeling_exec:
                       args += ["--couple_labeling_exec"]

    pr = subprocess.run(args, capture_output=True, text=True)
    out = pr.stdout + ("\nSTDERR:\n" + pr.stderr if pr.stderr else "")
    return pr.returncode, out




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
tab = st.sidebar.radio("Módulo", ["Datos", "Research", "Entrenamiento", "Análisis","Live"], index=0)

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
    # ============== TAB: RESEARCH (limpia y eficiente) ==============

    from pathlib import Path
    import sys, platform, subprocess, contextlib
    from typing import List, Optional, Tuple
    import datetime as dt

    import streamlit as st


    # ---- Helpers de UI seguros + control de ejecución ----
    def safe_ui(fn):
        """Evita romper la app si el WebSocket ya se cerró."""
        with contextlib.suppress(Exception):
            fn()


    if "running" not in st.session_state:
        st.session_state.running = False
    if "cancel" not in st.session_state:
        st.session_state.cancel = False


    def start_run():
        st.session_state.running = True
        st.session_state.cancel = False


    def stop_run():
        st.session_state.cancel = True


    # ---- Rutas e imports de tu repo ----
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import settings as settings

    S = settings.S
    from utils.data_update import get_tickers_from_file

    # Parche Win para stdout UTF-8 (no imprescindible)
    try:
        if platform.system() == "Windows":
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass

    # ---- Render de la pestaña Research ----
    def render_tab_research():
        st.header("🔬 Research (grid coherente o CV)")

        with st.expander("⚙️ Configuración y selección de datos", expanded=True):
            couple_labeling_exec = st.checkbox(
                "Acoplar labeling + backtest", value=True,
                help=(
                    "Si está activo, el grid de TP/SL/TL del backtest también se usa en la generación de labels y la CV.\n"
                    "Así cada run corresponde a una única combinación coherente."),
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
                    content_txt = up.read().decode("utf-8", errors="ignore")
                    tickers = [t.strip().upper() for t in content_txt.splitlines() if t.strip()]
            else:
                pasted = st.text_area("Tickers (uno por línea)", height=120, placeholder="AAPL\nMSFT\nGOOGL",
                                      key="rsh_paste")
                tickers = [t.strip().upper() for t in (pasted or "").splitlines() if t.strip()]

            # ======== Valores por defecto que pediste ========
            default_from = dt.date(2025, 1, 1)
            default_to = dt.date(2025, 6, 30)
            # ================================================

            c1, c2, c3 = st.columns(3)
            with c1:
                date_from = st.date_input("Desde", value=default_from, key="rsh_from")
            with c2:
                date_to = st.date_input("Hasta", value=default_to, key="rsh_to")
            with c3:
                days_hist = st.number_input("Días de histórico (si no fechas)", min_value=0, max_value=2000, value=0,
                                            step=20, key="rsh_days")

            c4, c5, c6 = st.columns(3)
            with c4:
                models_sel = st.multiselect("Modelos", options=["xgb", "rf", "logreg"], default=["xgb"],
                                            key="rsh_models")
            with c5:
                feature_set = st.text_input("Feature set (legacy, opcional)", value="", key="rsh_featset")
            with c6:
                st.caption("Label/triple-barrier (si *no* acoplas)")
                if not couple_labeling_exec:
                    tp_lbl = st.number_input("TP× (labeling)", min_value=0.1, max_value=10.0, value=3.0, step=0.1,
                                             key="rsh_tp_lbl")
                    sl_lbl = st.number_input("SL× (labeling)", min_value=0.1, max_value=10.0, value=2.0, step=0.1,
                                             key="rsh_sl_lbl")
                    tl_lbl = st.number_input("Time limit (labeling, barras)", min_value=1, max_value=500, value=16,
                                             step=1, key="rsh_tl_lbl")
                else:
                    st.caption("🔗 Acoplado: la CV usará el grid de TP/SL/TL del backtest.")
                    tp_lbl = sl_lbl = tl_lbl = None

            # Features seleccionables — por defecto TODAS seleccionadas
            try:
                feat_keys = sorted(list(FEATURE_REG.keys()))
            except Exception:
                feat_keys = []
            st.markdown("**Features seleccionables (engine.features):**")
            selected_feats = st.multiselect(
                "Selecciona features (vacío = usar 'feature_set' legacy)",
                options=feat_keys,
                default=feat_keys,  # ✅ todas por defecto
                key="rsh_feat_sel"
            )
            features_csv = ",".join(selected_feats) if selected_feats else None

            # thresholds & grids
            default_thr = getattr(S, "cv_threshold_grid", [0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
            thr_text = st.text_input("Thresholds (CSV, 0–1)", value=",".join(str(x) for x in default_thr),
                                     key="rsh_thr")

            # ✅ TP/SL grid por defecto 2,3,4 — Candle limit 16
            tp_grid_txt = st.text_input("TP grid (multiplicadores ATR)", value="2,3,4", key="rsh_tpgrid")
            sl_grid_txt = st.text_input("SL grid (multiplicadores ATR)", value="2,3,4", key="rsh_slgrid")
            tl_grid_txt = st.text_input("Time limit grid (barras)", value="16", key="rsh_tlgrid")

            iter_thr_in_bt = st.checkbox("Iterar thresholds dentro del backtest (si acoplado)", value=True,
                                         key="rsh_iter_thr_bt")

            c7, c8 = st.columns(2)
            with c7:
                n_splits = st.number_input("CV - n_splits", min_value=3, max_value=15, value=5, step=1,
                                           key="rsh_nsplits")
            with c8:
                test_size = st.number_input("CV - test_size (barras)", min_value=50, max_value=3000, value=400, step=50,
                                            key="rsh_testsize")

            c9, c10 = st.columns(2)
            with c9:
                primary_metric = st.selectbox("Métrica principal", options=["sharpe", "sortino", "cagr", "winrate"],
                                              index=0, key="rsh_metric")
            with c10:
                min_trades = st.number_input("Mínimo nº de trades", min_value=5, max_value=200, value=20, step=1,
                                             key="rsh_mintrades")

        colb1, colb2 = st.columns(2)
        run_scen_btn = colb1.button("▶️ Ejecutar Research (combinaciones coherentes)", use_container_width=True,
                                    disabled=st.session_state.running)
        run_cv_btn = colb2.button("▶️ Solo CV (debug)", use_container_width=True, disabled=st.session_state.running)
        safe_ui(lambda: st.button("⛔ Cancelar", on_click=stop_run, disabled=not st.session_state.running))

        # trackers
        log_area = st.empty()
        # ======== Lanzamiento simplificado: solo progreso ========
        progress = st.progress(0.0, text="Esperando…")

        if (run_scen_btn or run_cv_btn) and tickers:
            start_run()
            try:
                total = len(tickers)
                for i, tk in enumerate(tickers, start=1):
                    if st.session_state.cancel:
                        break

                    # actualiza progreso
                    pct = i / total
                    safe_ui(lambda: progress.progress(
                        pct,
                        text=f"Procesando {tk} ({i}/{total})…"
                    ))

                    # === Llamada al runner correcto ===
                    if run_cv_btn and not run_scen_btn:
                        rc, out = _run_cv_subprocess(
                            tk, timeframe,
                            date_from=str(date_from) if date_from else None,
                            date_to=str(date_to) if date_to else None,
                            thresholds=[float(x) for x in thr_text.split(",") if str(x).strip() != ""],
                            model=models_sel[0] if models_sel else "xgb",
                            feature_set=feature_set if (feature_set and not selected_feats) else None,
                            days=None if (date_from and date_to) else 0,
                            tp=float(tp_lbl) if (tp_lbl is not None) else None,
                            sl=float(sl_lbl) if (sl_lbl is not None) else None,
                            time_limit=int(tl_lbl) if (tl_lbl is not None) else None,
                            features_csv=features_csv,
                            n_splits=int(n_splits) if n_splits else None,
                            test_size=int(test_size) if test_size else None,
                        )
                    else:
                        rc, out = _run_scenarios_subprocess(
                            ticker=tk, timeframe=timeframe,
                            models_csv=",".join(models_sel) if models_sel else "xgb",
                            features_csv=features_csv,
                            feature_set=feature_set if (feature_set and not selected_feats) else None,
                            thresholds_csv=thr_text if thr_text.strip() else None,
                            tp_grid=tp_grid_txt, sl_grid=sl_grid_txt, tl_grid=tl_grid_txt,
                            iter_thr_bt=bool(iter_thr_in_bt),
                            date_from=str(date_from) if date_from else None,
                            date_to=str(date_to) if date_to else None,
                            days=None if (date_from and date_to) else 0,
                            n_splits=int(n_splits) if n_splits else None,
                            test_size=int(test_size) if test_size else None,
                            scheme="expanding", embargo=16, purge=16,
                            primary_metric=primary_metric, min_trades=int(min_trades),
                            couple_labeling_exec=bool(couple_labeling_exec),
                        )

                    # (no pintamos logs, solo seguimos)

                safe_ui(lambda: progress.progress(1.0, text=f"Procesados {total}/{total}"))
                safe_ui(lambda: st.success("✅ Lanzamiento finalizado."))
            finally:
                st.session_state.running = False


    render_tab_research()
    # ---- Ejemplo de uso:
    # if tab == "Research":
    #     render_tab_research()
    # ============ FIN TAB: RESEARCH =============


elif tab == "Entrenamiento":
    st.header("🏋️ Entrenamiento de Modelos ML")

    # ---------- Configuración de entrenamiento ----------
    st.subheader("⚙️ Configuración")

    # Selección de tickers
    cfg_dir = Path(getattr(S, "config_path"))
    default_txt = cfg_dir / "sp500_tickers.txt"
    src = st.radio("Origen de tickers", ["Archivo (.txt)", "Pegar lista", "Ticker único"], horizontal=True,
                   key="trn_src")

    tickers = []
    if src == "Archivo (.txt)":
        st.caption(f"Archivo por defecto: `{default_txt}`")
        up = st.file_uploader("...o sube otro .txt", type=["txt"], key="trn_up")
        if up is None:
            try:
                tickers = get_tickers_from_file(default_txt)[:20]  # Limitar para UI
            except Exception as e:
                st.error(f"No pude leer {default_txt.name}: {e}")
                tickers = []
        else:
            content = up.read().decode("utf-8", errors="ignore")
            tickers = [t.strip().upper() for t in content.splitlines() if t.strip()][:20]
    elif src == "Pegar lista":
        pasted = st.text_area("Tickers (uno por línea)", height=120, placeholder="AAPL\nMSFT\nGOOGL", key="trn_paste")
        tickers = [t.strip().upper() for t in pasted.splitlines() if t.strip()][:20]
    else:
        single_ticker = st.text_input("Ticker único", placeholder="AAPL", key="trn_single").strip().upper()
        tickers = [single_ticker] if single_ticker else []

    st.info(f"📊 {len(tickers)} tickers seleccionados para entrenamiento")

    # Parámetros de entrenamiento
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        timeframe = st.selectbox("Timeframe", ["1min", "5mins", "15mins", "1hour", "1day"],
                                 index=1, key="trn_tf")
        model_type = st.selectbox("Modelo ML", ["xgb", "rf", "logreg"], index=0, key="trn_model")
        feature_set = st.selectbox("Feature Set", ["core", "core+vol", "all"], index=1, key="trn_featset")

    with col2:
        days = st.number_input("Días de historia", min_value=30, max_value=365, value=90, step=10, key="trn_days")
        tp_mult = st.number_input("TP Multiplier (ATR)", min_value=0.5, max_value=10.0, value=2.0, step=0.5,
                                  key="trn_tp")
        sl_mult = st.number_input("SL Multiplier (ATR)", min_value=0.5, max_value=10.0, value=1.5, step=0.5,
                                  key="trn_sl")

    with col3:
        time_limit = st.number_input("Time Limit (barras)", min_value=4, max_value=100, value=16, step=2, key="trn_tl")
        force_relabel = st.checkbox("Forzar re-etiquetado", value=False, key="trn_relabel")
        overwrite = st.checkbox("Sobrescribir modelos existentes", value=False, key="trn_overwrite")

    # Opciones avanzadas
    with st.expander("⚙️ Configuración Avanzada"):
        col_adv1, col_adv2 = st.columns([1, 1])

        with col_adv1:
            date_from = st.date_input("Fecha FROM (opcional)", value=None, key="trn_from")
            date_to = st.date_input("Fecha TO (opcional)", value=None, key="trn_to")

        with col_adv2:
            inference_threshold = st.number_input("Threshold de inferencia",
                                                  min_value=0.1, max_value=0.9, value=0.7, step=0.05, key="trn_thresh")
            hparams_json = st.text_area("Hiperparámetros (JSON)",
                                        placeholder='{"max_depth": 6, "n_estimators": 200}', key="trn_hparams")

    # ---------- Estado de modelos existentes ----------
    st.divider()
    st.subheader("📁 Estado de Modelos Existentes")

    if st.button("🔍 Revisar modelos existentes"):
        models_dir = Path(getattr(S, "models_path", "02_models"))
        if models_dir.exists():
            model_status = []
            for ticker in tickers[:10]:  # Limitar para performance
                ticker_dir = models_dir / ticker.upper()
                pipeline_path = ticker_dir / "pipeline.pkl"
                meta_path = ticker_dir / "pipeline_meta.json"

                if pipeline_path.exists():
                    try:
                        # Leer metadata
                        if meta_path.exists():
                            meta = json.loads(meta_path.read_text())
                            model_info = {
                                "ticker": ticker,
                                "status": "✅ Existe",
                                "model_type": meta.get("model_type", "N/A"),
                                "feature_set": meta.get("feature_set", "N/A"),
                                "features_count": len(meta.get("features", [])),
                                "tp_mult": meta.get("tp_multiplier", "N/A"),
                                "sl_mult": meta.get("sl_multiplier", "N/A"),
                            }
                        else:
                            model_info = {
                                "ticker": ticker,
                                "status": "⚠️ Sin metadata",
                                "model_type": "N/A", "feature_set": "N/A",
                                "features_count": "N/A", "tp_mult": "N/A", "sl_mult": "N/A"
                            }
                    except Exception:
                        model_info = {
                            "ticker": ticker,
                            "status": "❌ Error",
                            "model_type": "N/A", "feature_set": "N/A",
                            "features_count": "N/A", "tp_mult": "N/A", "sl_mult": "N/A"
                        }
                else:
                    model_info = {
                        "ticker": ticker,
                        "status": "❌ No existe",
                        "model_type": "N/A", "feature_set": "N/A",
                        "features_count": "N/A", "tp_mult": "N/A", "sl_mult": "N/A"
                    }
                model_status.append(model_info)

            if model_status:
                df_models = pd.DataFrame(model_status)
                st.dataframe(df_models, use_container_width=True, hide_index=True)
        else:
            st.info("📁 Directorio de modelos no existe aún")

    # ---------- Lanzador de entrenamiento ----------
    st.divider()
    st.subheader("🚀 Entrenamiento")

    if not tickers:
        st.warning("⚠️ Selecciona al menos un ticker para entrenar")
    else:
        progress_placeholder = st.empty()
        log_placeholder = st.empty()

        if st.button(f"🏋️ Entrenar {len(tickers)} modelo(s)", type="primary", use_container_width=True):
            # Preparar parámetros
            overrides = {
                "tp_multiplier": float(tp_mult),
                "sl_multiplier": float(sl_mult),
                "time_limit_candles": int(time_limit),
                "model": model_type,
                "feature_set": feature_set,
                "days": int(days),
                "date_from": date_from.strftime("%Y-%m-%d") if date_from else None,
                "date_to": date_to.strftime("%Y-%m-%d") if date_to else None,
                "inference_threshold": float(inference_threshold),
            }

            # Parsear hiperparámetros JSON
            if hparams_json.strip():
                try:
                    hparams = json.loads(hparams_json)
                    overrides["hparams"] = hparams
                except json.JSONDecodeError as e:
                    st.error(f"Error en JSON de hiperparámetros: {e}")
                    st.stop()

            # Importar función de entrenamiento
            try:
                from TRN_Train import train_ticker

                exp_name = _mlflow_setup_for_app()

                results = []
                total = len(tickers)

                for i, ticker in enumerate(tickers):
                    progress_placeholder.progress((i) / total, text=f"Entrenando {ticker}...")

                    try:
                        # Entrenar modelo
                        model_path = train_ticker(
                            ticker=ticker,
                            timeframe=timeframe,
                            overrides=overrides,
                            force_relabel=force_relabel,
                            clean=overwrite,
                            print_stats=False
                        )

                        results.append({
                            "ticker": ticker,
                            "status": "✅ Éxito",
                            "model_path": str(model_path),
                            "error": ""
                        })

                        log_placeholder.success(f"✅ {ticker}: Modelo entrenado correctamente")

                    except Exception as e:
                        results.append({
                            "ticker": ticker,
                            "status": "❌ Error",
                            "model_path": "",
                            "error": str(e)
                        })
                        log_placeholder.error(f"❌ {ticker}: {str(e)}")

                progress_placeholder.progress(1.0, text="Entrenamiento completado")

                # Mostrar resumen
                st.success("🎯 Entrenamiento finalizado")
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)

                # Estadísticas
                success_count = len([r for r in results if "Éxito" in r["status"]])
                error_count = len([r for r in results if "Error" in r["status"]])

                col_s1, col_s2 = st.columns([1, 1])
                col_s1.metric("✅ Modelos entrenados", success_count)
                col_s2.metric("❌ Errores", error_count)

                if success_count > 0:
                    st.info(f"🎉 {success_count} modelos listos para usar en Research y Live Trading")

            except ImportError as e:
                st.error(f"Error importando TRN_Train: {e}")
            except Exception as e:
                st.error(f"Error durante el entrenamiento: {e}")

    # ---------- Información útil ----------
    st.divider()
    st.subheader("ℹ️ Información")

    with st.expander("📖 Guía de uso"):
        st.markdown("""
        **Flujo recomendado:**
        1. **Entrenar modelos** aquí con parámetros optimizados
        2. **Verificar en MLflow** que el entrenamiento fue exitoso
        3. **Usar en Research** para backtesting con modelos reales
        4. **Deplotar en Live** para trading automático

        **Recomendaciones:**
        - Usar `core+vol` para features balanceadas
        - TP/SL basados en análisis previo de volatilidad
        - Time limit entre 8-24 barras según timeframe
        - Entrenar con al menos 60-90 días de datos
        """)

    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Errores comunes:**
        - **Sin datos**: Verifica que el ticker tiene datos en el periodo seleccionado
        - **Features insuficientes**: Aumenta el periodo de datos o cambia feature_set
        - **Pocas clases**: Ajusta TP/SL/Time_limit para generar más variedad de labels
        - **Memoria insuficiente**: Reduce el número de días o entrena de uno en uno
        """)

elif tab == "Análisis":
    st.header("📊 Análisis de Performance MLflow")

    data_path = Path("data/mlflow_runs.parquet")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Extraer datos de MLflow"):
            with st.spinner("Extrayendo runs..."):
                df = extract_mlflow_runs()
            st.success(f"Extraídos {len(df)} runs")

    with col2:
        if st.button("📁 Recargar datos (forzar actualización)"):
            if os.path.exists(data_path):
                os.remove(data_path)  # Eliminar archivo existente
            df = extract_mlflow_runs()  # Forzar nueva extracción
            st.success(f"Datos actualizados: {len(df)} runs")

    if 'df' in locals() and not df.empty:
        # Extraer ticker y model del run_name
        # Formato esperado: "MMM_5mins_xgb_thr0.550_tp1x_sl1x_tl8"
        potential_tags = [col for col in df.columns if
                          col.isalpha() and col not in ['run_id', 'run_name', 'start_time', 'status']]
        st.write("Posibles tags:", potential_tags)
        df['ticker'] = df['run_name'].str.split('_').str[0]
        df['timeframe'] = df['run_name'].str.split('_').str[1]
        df['model'] = df['run_name'].str.split('_').str[2]

        # Verificar que funcionó
        tickers_clean = df['ticker'].dropna().astype(str).unique()
        st.write("Tickers extraídos:", sorted(tickers_clean))
        models_clean = df['model'].dropna().astype(str).unique()
        st.write("Modelos extraídos:", sorted(models_clean))

        # NUEVO: Verificar columnas disponibles
        st.write(f"Columnas disponibles: {list(df.columns)}")

        # Verificar columnas necesarias
        required_cols = ['ticker', 'sharpe', 'n_trades', 'model', 'win_rate']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Columnas faltantes: {missing_cols}")
            st.write("Verifica que los runs de MLflow tengan estos parámetros registrados")
            st.stop()

        # Filtros (solo si las columnas existen)
        # Reemplazar la sección de filtros existente:
        st.subheader("🔍 Filtros")
        col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1, 1])

        with col_f1:
            min_trades = st.slider("Trades mínimos", 0, int(df['n_trades'].max()), 10)
            if 'ticker' in df.columns:
                tickers = st.multiselect("Tickers", sorted(df['ticker'].dropna().astype(str).unique()), default=[])

        with col_f2:
            sharpe_range = st.slider("Rango Sharpe",
                                     float(df['sharpe'].min()),
                                     float(df['sharpe'].max()),
                                     (0.0, float(df['sharpe'].max())))

        with col_f3:
            # Filtros de parámetros
            if 'threshold' in df.columns:
                threshold_values = sorted(df['threshold'].dropna().unique())
                selected_thresholds = st.multiselect("Thresholds", threshold_values, default=threshold_values)
            else:
                selected_thresholds = []

            if 'tp_multiplier' in df.columns:
                tp_values = sorted(df['tp_multiplier'].dropna().unique())
                selected_tp = st.multiselect("TP Multipliers", tp_values, default=tp_values)
            else:
                selected_tp = []

        with col_f4:
            if 'sl_multiplier' in df.columns:
                sl_values = sorted(df['sl_multiplier'].dropna().unique())
                selected_sl = st.multiselect("SL Multipliers", sl_values, default=sl_values)
            else:
                selected_sl = []

            models = st.multiselect("Modelos", df['model'].dropna().astype(str).unique(),
                                    default=df['model'].dropna().astype(str).unique())

        # Aplicar todos los filtros
        filtered = df[
            (df['n_trades'] >= min_trades) &
            (df['sharpe'].between(sharpe_range[0], sharpe_range[1])) &
            (df['model'].isin(models))
            ]

        if tickers:
            filtered = filtered[filtered['ticker'].isin(tickers)]
        if selected_thresholds:
            filtered = filtered[filtered['threshold'].isin(selected_thresholds)]
        if selected_tp:
            filtered = filtered[filtered['tp_multiplier'].isin(selected_tp)]
        if selected_sl:
            filtered = filtered[filtered['sl_multiplier'].isin(selected_sl)]

        # Dashboard de métricas
        st.subheader("📈 Resumen")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        col_m1.metric("Total Runs", len(filtered))
        col_m2.metric("Sharpe Promedio", f"{filtered['sharpe'].mean():.2f}")
        col_m3.metric("Mejor Sharpe", f"{filtered['sharpe'].max():.2f}")
        col_m4.metric("Win Rate Promedio", f"{filtered['win_rate'].mean():.1%}")

        # Top performers
        st.subheader("🏆 Top Performers")
        top_runs = filtered.nlargest(20, 'sharpe')[
            ['ticker', 'model', 'threshold', 'tp_multiplier', 'sl_multiplier',
             'sharpe', 'net_return', 'win_rate', 'n_trades']
        ]
        st.dataframe(top_runs, use_container_width=True, hide_index=True)

        # Análisis por parámetros
        # Reemplazar la sección "Análisis por Parámetros":
        st.subheader("⚙️ Análisis por Parámetros")

        # Selector de parámetros para agrupar
        available_params = ['tp_multiplier', 'sl_multiplier', 'threshold', 'model', 'ticker']
        available_params = [p for p in available_params if p in df.columns]

        group_params = st.multiselect(
            "Agrupar por parámetros:",
            available_params,
            default=['tp_multiplier', 'sl_multiplier', 'threshold'][:len(available_params)]
        )

        # Selector de métricas para mostrar
        available_metrics = ['sharpe', 'net_return', 'win_rate', 'profit_factor', 'max_drawdown']
        available_metrics = [m for m in available_metrics if m in df.columns]

        show_metrics = st.multiselect(
            "Métricas a mostrar:",
            available_metrics,
            default=['sharpe', 'net_return', 'win_rate']
        )

        if group_params and show_metrics:
            # Crear análisis agrupado
            agg_dict = {}
            for metric in show_metrics:
                agg_dict[metric] = ['mean', 'count'] if metric == show_metrics[0] else 'mean'

            param_analysis = filtered.groupby(group_params).agg(agg_dict).round(3)

            # Ordenar por la primera métrica (descendente)
            sort_col = (show_metrics[0], 'mean') if len(show_metrics) > 0 else param_analysis.columns[0]
            param_analysis = param_analysis.sort_values(sort_col, ascending=False)

            st.dataframe(param_analysis, use_container_width=True)

            # Mostrar estadísticas del agrupamiento
            st.caption(f"Agrupado por: {', '.join(group_params)} | Mostrando: {', '.join(show_metrics)}")
        else:
            st.warning("Selecciona al menos un parámetro de agrupación y una métrica")

elif tab == "Live":
    st.header("🟢 Live (paper)")
    st.info("Próximamente: ejecución en paper/live y monitor de operaciones.")
