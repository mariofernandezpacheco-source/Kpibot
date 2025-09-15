# 06_dashboard.py — Dashboard en vivo + Histórico + pestaña MLflow (experimentos, runs y artefactos)
import glob
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import plotly.express as px
import streamlit as st

from settings import S
from utils.io_utils import safe_read_csv
from utils.reproducibility import set_global_determinism, write_env_versions

# ========= Ajustes reproducibilidad y página =========
set_global_determinism(S.seed, set_pythonhash=S.pythonhashseed)

if S.record_versions:
    write_env_versions(S.env_versions_path)

st.set_page_config(page_title="πbot Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- RUTAS Y CONFIG DESDE YAML ---
LOGS_DIR = S.logs_path
LIVE_DATA_DIR = LOGS_DIR / "data_live"
today_str = datetime.utcnow().strftime("%d_%m_%y")
TRADES_LOG_PATH = LOGS_DIR / f"paper_trades_{today_str}.csv"
OPEN_POSITIONS_LOG_PATH = LIVE_DATA_DIR / "live_open_positions.csv"
PROBS_LOG_PATH = LIVE_DATA_DIR / "live_probabilities.csv"
CHARTS_DATA_DIR = LIVE_DATA_DIR / "live_charts_data"

CAPITAL_PER_TRADE = float(S.capital_per_trade)

# --- Calendario de mercado (apertura real de hoy en UTC) ---
_nyse = mcal.get_calendar(S.calendar)  # por defecto 'XNYS'


def today_open_utc() -> pd.Timestamp | None:
    now_utc = pd.Timestamp.now(tz="UTC")
    d_local = now_utc.tz_convert("America/New_York").date()
    sched = _nyse.schedule(start_date=d_local, end_date=d_local)
    if sched.empty:
        return None
    return sched.iloc[0]["market_open"].tz_convert("UTC")


# --- Funciones para leer datos ---
@st.cache_data(ttl=10)
def load_csv(file_path: Path):
    try:
        if not file_path.exists():
            return pd.DataFrame()

        header_df = safe_read_csv(file_path, nrows=0)
        cols = list(header_df.columns)
        date_cols = [
            c for c in ["entry_time", "exit_time", "time_limit", "date", "timestamp"] if c in cols
        ]

        df = safe_read_csv(file_path, parse_dates=date_cols)

        for c in date_cols:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

        return df
    except Exception as e:
        st.error(f"Error al leer {file_path.name}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_full_history(logs_path: Path):
    files = glob.glob(str(logs_path / "paper_trades_*.csv"))
    if not files:
        return pd.DataFrame()
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=["entry_time", "exit_time"])
            for c in ["entry_time", "exit_time"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
            df_list.append(df)
        except Exception:
            continue
    if not df_list:
        return pd.DataFrame()
    full_history_df = pd.concat(df_list, ignore_index=True)
    full_history_df.sort_values(by="exit_time", inplace=True)
    full_history_df.reset_index(drop=True, inplace=True)
    return full_history_df


# ================== INTERFAZ PRINCIPAL ==================
st.title("📊 πbot - Dashboard de Operaciones")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Dashboard en Vivo",
        "Análisis Histórico",
        "MLflow (Modelos)",
        "Optimizer",
        "Event-Driven BT",
        "Lanzar Optimizer",
    ]
)

# -------- TAB 1: EN VIVO --------
with tab1:
    st.sidebar.title("Controles del Dashboard")
    if st.sidebar.button("🔄 Refrescar Datos"):
        st.cache_data.clear()
    st.sidebar.caption(f"Datos cargados a las: {datetime.now().strftime('%H:%M:%S')}")

    closed_trades_df = load_csv(TRADES_LOG_PATH)
    open_positions_df = load_csv(OPEN_POSITIONS_LOG_PATH)
    probs_df = load_csv(PROBS_LOG_PATH)

    # Normaliza P&L a la columna 'pnl'
    if (
        not closed_trades_df.empty
        and "net_profit" in closed_trades_df.columns
        and "pnl" not in closed_trades_df.columns
    ):
        closed_trades_df["pnl"] = pd.to_numeric(closed_trades_df["net_profit"], errors="coerce")
    if not closed_trades_df.empty and "pnl" in closed_trades_df.columns:
        closed_trades_df["pnl"] = pd.to_numeric(closed_trades_df["pnl"], errors="coerce").fillna(
            0.0
        )

    st.subheader("Resultados de la Sesión de Hoy")
    if not closed_trades_df.empty and "pnl" in closed_trades_df.columns:
        pnl_total = float(closed_trades_df["pnl"].sum())
        win_rate = float((closed_trades_df["pnl"] > 0).mean() * 100)
        num_trades = int(len(closed_trades_df))
        c1, c2, c3 = st.columns(3)
        c1.metric("P/L Neto (Hoy)", f"€{pnl_total:.2f}")
        c2.metric("Win Rate (Hoy)", f"{win_rate:.1f}%")
        c3.metric("# Trades (Hoy)", num_trades)
    else:
        st.info("Aún no hay trades cerrados en la sesión de hoy.")

    capital_in_use = len(open_positions_df) * CAPITAL_PER_TRADE
    st.subheader("💥 Exposición de Capital")
    st.metric("Capital Invertido (Ahora)", f"€{capital_in_use:,.2f}")

    st.subheader("🔄 Operaciones Abiertas")
    if not open_positions_df.empty:
        if "Unnamed: 0" in open_positions_df.columns:
            open_positions_df = open_positions_df.rename(columns={"Unnamed: 0": "ticker"})

        for _, pos_data in open_positions_df.iterrows():
            ticker_val = pos_data.get("ticker", None)
            if isinstance(ticker_val, pd.Series):
                ticker = ticker_val.iloc[0]
            else:
                ticker = ticker_val

            with st.expander(
                f"**{ticker}** | Señal: {'Compra' if pos_data['signal'] == 1 else 'Venta'} | Entrada: {pos_data['entry_price']:.2f}"
            ):
                chart_data_path = CHARTS_DATA_DIR / f"{ticker}_chart_data.csv"
                chart_df = load_csv(chart_data_path)

                if chart_df is not None and not chart_df.empty:
                    if "date" in chart_df.columns:
                        chart_df["date"] = pd.to_datetime(
                            chart_df["date"], errors="coerce", utc=True
                        )

                    open_utc = today_open_utc()
                    if open_utc is not None and "date" in chart_df.columns:
                        chart_df_today = chart_df[chart_df["date"] >= open_utc]
                    else:
                        chart_df_today = chart_df

                    if not chart_df_today.empty:
                        fig = px.line(
                            chart_df_today,
                            x="date",
                            y="close",
                            title=f"Evolución de {ticker} (Hoy)",
                        )
                        fig.add_hline(
                            y=pos_data["tp_price"],
                            line_dash="dash",
                            line_color="green",
                            annotation_text=f"Take Profit ({pos_data['tp_price']:.2f})",
                        )
                        fig.add_hline(
                            y=pos_data["entry_price"],
                            line_dash="solid",
                            line_color="blue",
                            annotation_text=f"Entrada ({pos_data['entry_price']:.2f})",
                        )
                        fig.add_hline(
                            y=pos_data["sl_price"],
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Stop Loss ({pos_data['sl_price']:.2f})",
                        )

                        entry_time = pd.to_datetime(
                            pos_data["entry_time"], errors="coerce", utc=True
                        )
                        fig.add_vline(x=entry_time, line_dash="dot", line_color="purple")
                        fig.add_annotation(
                            x=entry_time,
                            y=(
                                chart_df_today["high"].max()
                                if "high" in chart_df_today.columns
                                else chart_df_today["close"].max()
                            ),
                            text=f"Entrada ({pd.to_datetime(entry_time).strftime('%H:%M')})",
                            showarrow=True,
                            arrowhead=1,
                        )
                        fig.update_layout(yaxis_title="Precio", xaxis_title="Fecha")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Esperando datos de precios para la jornada de hoy.")
                else:
                    st.info("Esperando datos del gráfico para esta operación...")
    else:
        st.info("No hay operaciones activas.")

    st.subheader("🧠 Probabilidades del Último Ciclo")
    if not probs_df.empty:
        probs_df_display = probs_df.copy()
        if "ticker" in probs_df_display.columns:
            probs_df_display.set_index("ticker", inplace=True)
        for col in ["prob_down", "prob_hold", "prob_up"]:
            if col in probs_df_display.columns:
                probs_df_display[col] = pd.to_numeric(probs_df_display[col], errors="coerce")
        st.dataframe(
            probs_df_display.style.format(
                "{:.2%}", na_rep="-", subset=["prob_down", "prob_hold", "prob_up"]
            ).background_gradient(cmap="RdYlGn", subset=["prob_down", "prob_up"])
        )
    else:
        st.info("Sin fichero de probabilidades todavía.")

# -------- TAB 2: HISTÓRICO --------
with tab2:
    st.header("📈 Rendimiento Histórico Acumulado")
    full_history_df = load_full_history(LOGS_DIR)

    if not full_history_df.empty:
        if "net_profit" in full_history_df.columns and "pnl" not in full_history_df.columns:
            full_history_df["pnl"] = pd.to_numeric(full_history_df["net_profit"], errors="coerce")
        if "pnl" in full_history_df.columns:
            full_history_df["pnl"] = pd.to_numeric(full_history_df["pnl"], errors="coerce").fillna(
                0.0
            )

    if not full_history_df.empty and "pnl" in full_history_df.columns:
        pnl_acumulado = float(full_history_df["pnl"].sum())
        win_rate_acumulado = float((full_history_df["pnl"] > 0).mean() * 100)
        num_trades_acumulado = int(len(full_history_df))

        c1, c2, c3 = st.columns(3)
        c1.metric("P/L Total Neto (Histórico)", f"€{pnl_acumulado:.2f}")
        c2.metric("Win Rate (Histórico)", f"{win_rate_acumulado:.1f}%")
        c3.metric("# Trades (Histórico)", num_trades_acumulado)

        st.subheader("Curva de Capital")
        full_history_df["cumulative_pnl"] = full_history_df["pnl"].cumsum()
        idx_col = "exit_time" if "exit_time" in full_history_df.columns else full_history_df.index
        st.line_chart(full_history_df.set_index(idx_col)["cumulative_pnl"])

        with st.expander("Ver Historial de Todos los Trades"):
            st.dataframe(full_history_df)
    else:
        st.info(f"No se encontraron ficheros de trades en {LOGS_DIR}.")

# -------- TAB 3: MLFLOW (Lazy + alias métricas + sin bloqueos) --------
with tab3:
    st.header("🧪 MLflow – Experimentos y Modelos")
    mlflow_ok = True
    try:
        import mlflow
        from mlflow import MlflowClient
    except Exception:
        mlflow_ok = False
        st.error("MLflow no está disponible. Instala `mlflow` e intenta de nuevo.")

    if mlflow_ok:
        import tempfile
        from datetime import datetime
        from pathlib import Path

        import pandas as pd
        from PIL import Image

        # ---------- Compatibilidad listar experimentos ----------
        def setup_mlflow():
            uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
            mlflow.set_tracking_uri(uri)
            client = MlflowClient()
            # intentar varios métodos según versión
            try:
                exp_objs = client.list_experiments()
            except Exception:
                try:
                    from mlflow.entities import ViewType

                    exp_objs = mlflow.search_experiments(view_type=ViewType.ACTIVE_ONLY)
                except Exception:
                    try:
                        exp_objs = mlflow.search_experiments()
                    except Exception:
                        exp_objs = []
            exp_names = sorted(
                [
                    getattr(e, "name", None)
                    for e in exp_objs
                    if getattr(e, "name", None)
                    and (not hasattr(e, "lifecycle_stage") or e.lifecycle_stage == "active")
                ]
            )
            default_name = "PHIBOT"
            if default_name not in exp_names:
                exp_names = [default_name] + exp_names

            exp_name = st.sidebar.selectbox(
                "Experimento",
                options=exp_names,
                index=exp_names.index(default_name) if default_name in exp_names else 0,
                key="mlflow_exp_name",
            )
            exp = client.get_experiment_by_name(exp_name)
            if exp is None:
                try:
                    mlflow.set_experiment(exp_name)
                    exp = client.get_experiment_by_name(exp_name)
                except Exception:
                    st.warning(
                        f"No se pudo crear/leer el experimento '{exp_name}'. Revisa MLFLOW_TRACKING_URI."
                    )
                    return client, None
            return client, exp

        client, exp = setup_mlflow()

        # ---------- Alias de métricas: detecta el primer nombre disponible ----------
        METRIC_ALIASES = {
            # Concepto -> posibles nombres en tus runs
            "pr_auc_mean": ["pr_auc_mean", "mean_pr_auc", "cv_pr_auc_mean", "oof_pr_auc_mean"],
            "roc_auc_mean": ["roc_auc_mean", "mean_roc_auc", "cv_roc_auc_mean", "oof_roc_auc_mean"],
            "pr_auc": ["pr_auc", "oof_pr_auc"],
            "roc_auc": ["roc_auc", "oof_roc_auc"],
            "ev_mean_at_thr": ["ev_mean_at_thr", "ev_oof_at_thr", "ev_at_thr_mean"],
            "optimizer_score": ["optimizer_score"],
            "train_pr_auc": ["train_pr_auc"],
            "train_roc_auc": ["train_roc_auc"],
        }

        def pick_metric(mdict: dict, concept: str):
            """Devuelve el valor de la primera métrica disponible para un 'concepto' (o None)."""
            for k in METRIC_ALIASES.get(concept, [concept]):
                if k in mdict:
                    return mdict[k]
            return None

        @st.cache_resource(show_spinner=False)
        def get_recent_tags(exp_id: str, limit: int = 200):
            """Devuelve sets de tickers/timeframes a partir de los últimos 'limit' runs (rápido)."""
            try:
                rs = client.search_runs(
                    [exp_id], "", order_by=["attributes.start_time DESC"], max_results=int(limit)
                )
            except Exception:
                rs = []
            tickers = sorted({r.data.tags.get("ticker") for r in rs if r.data.tags.get("ticker")})
            timeframes = sorted(
                {r.data.tags.get("timeframe") for r in rs if r.data.tags.get("timeframe")}
            )
            return tickers, timeframes

        @st.cache_resource(show_spinner=False)
        def search_runs_cached(exp_id: str, query: str, limit: int):
            """Búsqueda de runs cacheada (sin order_by del servidor)."""
            try:
                rs = client.search_runs([exp_id], query, max_results=int(limit))
            except Exception:
                rs = []
            return rs

        def sort_runs_client_side(runs, concept_key: str):
            """Ordena runs de mayor a menor según el 'concepto' de métrica (con alias)."""

            def key_fn(r):
                v = pick_metric(r.data.metrics, concept_key)
                try:
                    return float(v) if v is not None else float("-inf")
                except Exception:
                    return float("-inf")

            return sorted(runs, key=key_fn, reverse=True)

        # ---------- UI lateral: carga perezosa ----------
        if exp is None:
            st.info("Selecciona o crea el experimento 'PHIBOT' en la barra lateral.")
            runs = []
        else:
            st.sidebar.markdown("---")
            lazy_load = st.sidebar.checkbox(
                "Cargar filtros MLflow",
                value=False,
                help="Actívalo para listar tickers/timeframes desde los últimos runs.",
            )
            tickers, timeframes = ([], [])
            if lazy_load:
                with st.sidebar.spinner("Leyendo runs recientes…"):
                    tickers, timeframes = get_recent_tags(exp.experiment_id, limit=200)

            ticker_sel = st.sidebar.selectbox(
                "Ticker", options=["(cualquiera)"] + tickers, index=0, key="mlflow_ticker"
            )
            timeframe_sel = st.sidebar.selectbox(
                "Timeframe", options=["(cualquiera)"] + timeframes, index=0, key="mlflow_timeframe"
            )
            phase_sel = st.sidebar.selectbox(
                "Fase",
                options=["cualquiera", "training", "cv", "optimizer", "opt_trial"],
                index=0,
                key="mlflow_phase",
            )

            # Importante: ‘ordenar por’ usa conceptos (no nombres concretos)
            order_options = [
                "pr_auc_mean",
                "roc_auc_mean",
                "ev_mean_at_thr",
                "optimizer_score",
                "train_pr_auc",
                "train_roc_auc",
                "pr_auc",
                "roc_auc",
            ]
            metric_pref = st.sidebar.selectbox(
                "Ordenar por métrica (concepto)", options=order_options, index=0
            )

            max_results = st.sidebar.number_input(
                "Máx. runs a mostrar", min_value=20, max_value=5000, value=200, step=10
            )

            do_search = st.sidebar.button("🔎 Buscar runs")

            runs = st.session_state.get("_runs_cache", [])
            if do_search:
                # Construir query
                filters = []
                if phase_sel and phase_sel != "cualquiera":
                    filters.append(f"tags.phase = '{phase_sel}'")
                if ticker_sel and ticker_sel != "(cualquiera)":
                    filters.append(f"tags.ticker = '{ticker_sel}'")
                if timeframe_sel and timeframe_sel != "(cualquiera)":
                    filters.append(f"tags.timeframe = '{timeframe_sel}'")
                query = " and ".join(filters) if filters else ""
                with st.spinner("Buscando runs…"):
                    rs = search_runs_cached(exp.experiment_id, query, int(max_results))
                    runs = sort_runs_client_side(rs, metric_pref)
                st.session_state["_runs_cache"] = runs

        # ---------- Tabla y detalles ----------
        if not exp:
            st.info("Sin experimento seleccionado.")
        elif not runs:
            st.info(
                "No se encontraron runs. Activa «Cargar filtros MLflow», ajusta filtros y pulsa «🔎 Buscar runs»."
            )
        else:

            def _row(r):
                m = r.data.metrics
                d = {
                    "run_id": r.info.run_id[:8] + "…",
                    "full_run_id": r.info.run_id,
                    "start_time": datetime.fromtimestamp(r.info.start_time / 1000.0).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "phase": r.data.tags.get("phase"),
                    "ticker": r.data.tags.get("ticker"),
                    "timeframe": r.data.tags.get("timeframe"),
                    "strategy": r.data.tags.get("strategy"),
                    # Mostramos SIEMPRE las versiones 'conceptuales'
                    "pr_auc_mean": pick_metric(m, "pr_auc_mean"),
                    "roc_auc_mean": pick_metric(m, "roc_auc_mean"),
                    "ev_mean_at_thr": pick_metric(m, "ev_mean_at_thr"),
                    "optimizer_score": pick_metric(m, "optimizer_score"),
                    "train_pr_auc": pick_metric(m, "train_pr_auc"),
                    "train_roc_auc": pick_metric(m, "train_roc_auc"),
                }
                # Redondeo amable
                for k in [
                    "pr_auc_mean",
                    "roc_auc_mean",
                    "ev_mean_at_thr",
                    "optimizer_score",
                    "train_pr_auc",
                    "train_roc_auc",
                ]:
                    if d[k] is not None:
                        try:
                            d[k] = round(float(d[k]), 4)
                        except Exception:
                            pass
                return d

            df_runs = pd.DataFrame([_row(r) for r in runs])
            st.subheader("📋 Runs encontrados")
            st.dataframe(df_runs, use_container_width=True)

            options = list(range(len(runs)))
            run_idx = st.selectbox(
                "Selecciona un run para ver detalles",
                options=options,
                index=0 if options else None,
                format_func=lambda i: f"{df_runs.iloc[i]['run_id']} | {df_runs.iloc[i]['ticker']} | {df_runs.iloc[i]['timeframe']} | {df_runs.iloc[i]['phase']}",
                key="mlflow_run_idx",
            )

            if run_idx is None or not (0 <= run_idx < len(runs)):
                st.warning("Selecciona un run válido en la tabla superior.")
            else:
                sel_run = runs[run_idx]
                sel_id = sel_run.info.run_id
                st.markdown(f"**Run ID completo:** `{sel_id}`")

                with st.expander("🔧 Parámetros"):
                    st.json(sel_run.data.params)
                with st.expander("🏷️ Tags"):
                    st.json(sel_run.data.tags)

                st.subheader("📏 Métricas")
                m = sel_run.data.metrics

                # Lista completa de conceptos a mostrar en detalle
                conceptos_detalle = [
                    "train_roc_auc",
                    "train_pr_auc",
                    "train_log_loss",
                    "train_brier",
                    "train_ks",
                    "train_accuracy_at_thr",
                    "train_precision_at_thr",
                    "train_recall_at_thr",
                    "train_f1_at_thr",
                    "train_balanced_acc_at_thr",
                    "train_specificity_at_thr",
                    "train_fit_seconds",
                    "pr_auc",
                    "roc_auc",
                    "pr_auc_mean",
                    "roc_auc_mean",
                    "ev_mean_at_thr",
                    "folds_covered",
                    "optimizer_score",
                ]

                # Expandimos conceptos a claves reales mediante alias
                filas = []
                ya_mostradas = set()
                for concept in conceptos_detalle:
                    if concept in METRIC_ALIASES:
                        # mostrar solo una fila por concepto (primer alias presente)
                        val = pick_metric(m, concept)
                        if val is not None:
                            filas.append({"métrica": concept, "valor": val})
                            ya_mostradas.add(concept)
                    else:
                        # métrica literal
                        if concept in m:
                            filas.append({"métrica": concept, "valor": m[concept]})

                if filas:
                    dfm = pd.DataFrame(filas)
                    # redondeo
                    for col in ["valor"]:
                        dfm[col] = pd.to_numeric(dfm[col], errors="ignore")
                    st.dataframe(dfm, use_container_width=True)
                else:
                    st.caption(
                        "Este run no tiene métricas reconocidas por el dashboard (revisa los nombres en MLflow)."
                    )

                # Artefactos (previews)
                st.subheader("🗂️ Artefactos")
                st.caption("Mostrando imágenes y textos comunes (plots, importancia, meta).")
                tmpdir = Path(tempfile.mkdtemp())

                def _try_show_png(artifact_subpath, label):
                    from mlflow import MlflowClient

                    try:
                        p = MlflowClient().download_artifacts(
                            sel_id, artifact_subpath, tmpdir.as_posix()
                        )
                        st.image(Image.open(p), caption=label, use_column_width=True)
                        return True
                    except Exception:
                        return False

                def _try_show_text(artifact_subpath, label, max_chars=4000):
                    from mlflow import MlflowClient

                    try:
                        p = MlflowClient().download_artifacts(
                            sel_id, artifact_subpath, tmpdir.as_posix()
                        )
                        txt = Path(p).read_text(encoding="utf-8")
                        if len(txt) > max_chars:
                            txt = txt[:max_chars] + "\n...\n(truncado)"
                        st.code(txt, language="text")
                        return True
                    except Exception:
                        return False

                ok1 = _try_show_png("plots/roc.png", "ROC curve")
                ok2 = _try_show_png("plots/pr.png", "PR curve")
                if not (ok1 or ok2):
                    st.caption("No encontré plots guardados ('plots/roc.png', 'plots/pr.png').")

                try:
                    from mlflow import MlflowClient

                    p = MlflowClient().download_artifacts(
                        sel_id, "importance/feature_importance.csv", tmpdir.as_posix()
                    )
                    st.subheader("🔎 Feature importance / coeficientes")
                    st.dataframe(pd.read_csv(p).head(30), use_container_width=True)
                except Exception:
                    st.caption("No se encontró 'importance/feature_importance.csv'.")

                st.subheader("🧾 Meta")
                shown_meta = _try_show_text(
                    "meta/pipeline_meta.json", "pipeline_meta.json"
                ) or _try_show_text("meta/features.txt", "features.txt")
                if not shown_meta:
                    st.caption("No se encontraron artefactos de meta.")


# -------- TAB 4: OPTIMIZER (02c) --------
with tab4:
    st.header("🧪 Optimizer (02c)")

    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    exp_name = os.getenv("MLFLOW_EXPERIMENT", "PHIBOT")
    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)

    if exp is None:
        st.info("Aún no hay experimento. Lanza el 02c o usa la pestaña 'Lanzar Optimizer'.")
    else:
        # Escenarios (runs padre)
        parents = client.search_runs(
            [exp.experiment_id],
            "tags.phase = 'optimizer'",
            order_by=["metrics.optimizer_score DESC"],
            max_results=1000,
        )

        def row_parent(r):
            p, m, t = r.data.params, r.data.metrics, r.data.tags

            def _flt(x, default="nan"):
                try:
                    return float(x)
                except Exception:
                    return float("nan")

            return {
                "run_id": r.info.run_id,
                "periodo": t.get("period_name"),
                "tp": _flt(p.get("tp", p.get("tp_multiplier", "nan"))),
                "sl": _flt(p.get("sl", p.get("sl_multiplier", "nan"))),
                "tl": _flt(p.get("time_limit", p.get("time_limit_candles", "nan"))),
                "model": p.get("model"),
                "feature_set": p.get("feature_set"),
                "n_tickers": int(float(p.get("n_tickers", "0") or 0)),
                "mean_pr_auc": _flt(m.get("mean_pr_auc", "nan")),
                "mean_roc_auc": _flt(m.get("mean_roc_auc", "nan")),
                "std_pr_auc": _flt(m.get("std_pr_auc", "nan")),
                "std_roc_auc": _flt(m.get("std_roc_auc", "nan")),
                "pr_auc_mean_avg": _flt(m.get("pr_auc_mean_avg", "nan")),
                "roc_auc_mean_avg": _flt(m.get("roc_auc_mean_avg", "nan")),
                "ev_mean_avg": _flt(m.get("ev_mean_avg", "nan")),
                "folds_covered_avg": _flt(m.get("folds_covered_avg", "nan")),
                "optimizer_score": _flt(m.get("optimizer_score", "nan")),
            }

        df_par = pd.DataFrame([row_parent(r) for r in parents])
        if df_par.empty:
            st.info(
                "No hay runs del Optimizer. Ejecuta 02c primero (o usa la pestaña 'Lanzar Optimizer')."
            )
        else:
            st.subheader("🏁 Escenarios")
            st.dataframe(
                df_par.sort_values("optimizer_score", ascending=False), use_container_width=True
            )

            # Heatmap (TP vs SL) por score
            st.subheader("🔥 Heatmap TP vs SL (score)")
            try:
                heat = df_par.pivot_table(
                    index="sl", columns="tp", values="optimizer_score", aggfunc="mean"
                ).sort_index()
                fig = px.imshow(
                    heat, origin="lower", aspect="auto", labels=dict(x="TP", y="SL", color="Score")
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.caption("No se pudo construir el heatmap (faltan datos).")

            # Selección de escenario para ver tickers (runs hijos)
            st.subheader("🔎 Detalle por ticker")
            idx = st.selectbox(
                "Elige un escenario",
                options=list(range(len(df_par))),
                format_func=lambda i: f"[{df_par.iloc[i]['run_id'][:8]}] {df_par.iloc[i]['periodo']} | TP={df_par.iloc[i]['tp']} SL={df_par.iloc[i]['sl']} TL={df_par.iloc[i]['tl']} | Score={df_par.iloc[i]['optimizer_score']:.3f}",
            )
            parent_id = df_par.iloc[idx]["run_id"]

            children = client.search_runs(
                [exp.experiment_id], f"tags.mlflow.parentRunId = '{parent_id}'", max_results=5000
            )

            def row_child(r):
                m, p, t = r.data.metrics, r.data.params, r.data.tags

                def _flt(x):
                    try:
                        return float(x)
                    except Exception:
                        return float("nan")

                return {
                    "ticker": t.get("ticker"),
                    "pr_auc_mean": _flt(m.get("pr_auc_mean", "nan")),
                    "roc_auc_mean": _flt(m.get("roc_auc_mean", "nan")),
                    "ev_mean_at_thr": _flt(m.get("ev_mean_at_thr", "nan")),
                    "folds_covered": int(float(m.get("folds_covered", "0") or 0)),
                    "recommended_threshold": (
                        _flt(p.get("recommended_threshold", "nan"))
                        if p.get("recommended_threshold")
                        else None
                    ),
                    "recommended_by": p.get("recommended_by", ""),
                }

            df_ch = pd.DataFrame([row_child(r) for r in children]).sort_values(
                "ev_mean_at_thr", ascending=False
            )
            st.dataframe(df_ch, use_container_width=True)

            # Botón para guardar parámetros del escenario
            st.subheader("💾 Guardar parámetros del escenario")
            c1, c2, c3 = st.columns(3)
            tp_sel = c1.number_input("TP", value=float(df_par.iloc[idx]["tp"]), step=0.1)
            sl_sel = c2.number_input("SL", value=float(df_par.iloc[idx]["sl"]), step=0.1)
            tl_sel = c3.number_input(
                "Time limit (velas)", value=int(df_par.iloc[idx]["tl"]), step=1
            )
            if st.button("Guardar como selección del Optimizer"):
                out = Path(S.config_path) / "optimizer_selected.json"
                payload = {
                    "timeframe": getattr(S, "timeframe_default", "5mins"),
                    "tp_multiplier": float(tp_sel),
                    "sl_multiplier": float(sl_sel),
                    "time_limit_candles": int(tl_sel),
                }
                out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                st.success(f"Guardado en {out}")

    # ======= NUEVO: Selección automática por ticker (por roc_auc_mean) =======
    st.subheader("🤖 Selección automática por ticker (por roc_auc_mean)")

    # Alias de métricas por si tus runs usan nombres distintos
    METRIC_ALIASES_OPT = {
        "roc_auc_mean": ["roc_auc_mean", "mean_roc_auc", "cv_roc_auc_mean", "oof_roc_auc_mean"],
        "pr_auc_mean": ["pr_auc_mean", "mean_pr_auc", "cv_pr_auc_mean", "oof_pr_auc_mean"],
    }

    def pick_metric_opt(mdict: dict, concept: str):
        for k in METRIC_ALIASES_OPT.get(concept, [concept]):
            if k in mdict:
                try:
                    return float(mdict[k])
                except Exception:
                    pass
        return None

    if exp is not None:
        if st.button("Calcular mejores por ticker (roc_auc_mean)"):
            with st.spinner("Buscando runs 'opt_trial' en MLflow y agregando por ticker…"):
                # 1) Traer TODOS los hijos (opt_trial) de todos los escenarios
                children_all = client.search_runs(
                    [exp.experiment_id],
                    "tags.phase = 'opt_trial'",
                    max_results=50000,  # si tu servidor limita, traerá el máximo permitido
                )

                if not children_all:
                    st.info("No encontré runs con phase='opt_trial'. Lanza 02c primero.")
                else:
                    # 2) Cache de runs padre (escenarios) para extraer TP/SL/TL/model/features
                    parent_ids = sorted(
                        {
                            r.data.tags.get("mlflow.parentRunId")
                            for r in children_all
                            if r.data.tags.get("mlflow.parentRunId")
                        }
                    )
                    parent_cache = {}
                    for pid in parent_ids:
                        try:
                            prun = client.get_run(pid)
                            p, t = prun.data.params, prun.data.tags

                            def _flt(x):
                                try:
                                    return float(x)
                                except Exception:
                                    return None

                            parent_cache[pid] = {
                                "tp": _flt(p.get("tp", p.get("tp_multiplier"))),
                                "sl": _flt(p.get("sl", p.get("sl_multiplier"))),
                                "tl": (
                                    int(
                                        float(p.get("time_limit", p.get("time_limit_candles", "0")))
                                    )
                                    if p.get("time_limit") or p.get("time_limit_candles")
                                    else None
                                ),
                                "timeframe": t.get("timeframe") or p.get("timeframe"),
                                "model": p.get("model"),
                                "feature_set": p.get("feature_set"),
                            }
                        except Exception:
                            continue

                    # 3) Construir tabla con (ticker, métrica, y params del escenario padre)
                    rows = []
                    for r in children_all:
                        tkr = r.data.tags.get("ticker")
                        pid = r.data.tags.get("mlflow.parentRunId")
                        mval = pick_metric_opt(r.data.metrics, "roc_auc_mean")
                        if tkr and (mval is not None) and pid in parent_cache:
                            rows.append(
                                {
                                    "ticker": tkr,
                                    "roc_auc_mean": mval,
                                    "parent_run_id": pid,
                                    "child_run_id": r.info.run_id,
                                    "tp": parent_cache[pid]["tp"],
                                    "sl": parent_cache[pid]["sl"],
                                    "tl": parent_cache[pid]["tl"],
                                    "timeframe": parent_cache[pid]["timeframe"],
                                    "model": parent_cache[pid]["model"],
                                    "feature_set": parent_cache[pid]["feature_set"],
                                    "recommended_threshold": r.data.params.get(
                                        "recommended_threshold"
                                    ),
                                }
                            )

                    import pandas as pd

                    df_all = pd.DataFrame(rows)
                    if df_all.empty:
                        st.warning(
                            "No hay métricas 'roc_auc_mean' disponibles en los runs 'opt_trial'."
                        )
                    else:
                        # 4) Elegir el mejor por ticker (máximo roc_auc_mean)
                        df_best = (
                            df_all.sort_values(["ticker", "roc_auc_mean"], ascending=[True, False])
                            .drop_duplicates(subset=["ticker"], keep="first")
                            .reset_index(drop=True)
                        )

                        st.success(f"Seleccionados {len(df_best)} tickers (mejor 'roc_auc_mean').")
                        st.dataframe(
                            df_best[
                                [
                                    "ticker",
                                    "roc_auc_mean",
                                    "tp",
                                    "sl",
                                    "tl",
                                    "timeframe",
                                    "model",
                                    "feature_set",
                                    "parent_run_id",
                                    "child_run_id",
                                    "recommended_threshold",
                                ]
                            ],
                            use_container_width=True,
                        )

                        # 5) Guardar selección por-ticker a JSON
                        save_col1, save_col2 = st.columns([1, 1])
                        out_json = Path(S.config_path) / "optimizer_selected_by_ticker.json"
                        if save_col1.button(
                            "💾 Guardar selección por ticker → optimizer_selected_by_ticker.json"
                        ):
                            payload = {
                                "metric": "roc_auc_mean",
                                "timeframe_default": getattr(S, "timeframe_default", "5mins"),
                                "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "per_ticker": {},
                            }
                            for _, row in df_best.iterrows():
                                thr = row.get("recommended_threshold")
                                try:
                                    thr = float(thr) if thr is not None else None
                                except Exception:
                                    thr = None
                                payload["per_ticker"][row["ticker"]] = {
                                    "tp_multiplier": (
                                        float(row["tp"]) if row["tp"] is not None else None
                                    ),
                                    "sl_multiplier": (
                                        float(row["sl"]) if row["sl"] is not None else None
                                    ),
                                    "time_limit_candles": (
                                        int(row["tl"]) if row["tl"] is not None else None
                                    ),
                                    "timeframe": row["timeframe"]
                                    or getattr(S, "timeframe_default", "5mins"),
                                    "model": row["model"],
                                    "feature_set": row["feature_set"],
                                    "recommended_threshold": thr,
                                    "parent_run_id": row["parent_run_id"],
                                    "child_run_id": row["child_run_id"],
                                }
                            out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                            st.success(f"Guardado en {out_json}")

                        # 6) (Opcional) Exportar CSV rápido
                        if save_col2.button("⬇️ Exportar CSV"):
                            csv_path = Path(S.logs_path) / "optimizer_best_by_ticker.csv"
                            df_best.to_csv(csv_path, index=False)
                            st.success(f"CSV exportado a {csv_path}")

# -------- TAB 5: EVENT-DRIVEN BT --------
with tab5:
    st.header("🧪 Event-Driven Backtest (C.1)")
    default_trades_path = S.logs_path / "backtests" / "trades.csv"
    trades_path_str = st.text_input(
        "Ruta del trades.csv (generado por `engine/backtest_runner.py`)",
        value=str(default_trades_path),
    )

    def _load_trades(path: str) -> pd.DataFrame | None:
        p = Path(path)
        if not p.exists():
            st.info(
                "Aún no encontré `trades.csv`. Lanza primero el backtest.sh:\n\n"
                "`python engine/backtest_runner.py --tickers_file all_tickers.txt --timeframe 5mins`"
            )
            return None
        df = pd.read_csv(p)
        for c in ["entry_time", "exit_time", "ts"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        if "ts" not in df.columns:
            df["ts"] = df["exit_time"].fillna(df["entry_time"])
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str).str.upper()
        if "fee" in df.columns:
            df["fee"] = pd.to_numeric(df["fee"], errors="coerce").fillna(0.0)
        return df

    df_raw = _load_trades(trades_path_str)

    st.subheader("Filtros")
    tickers = (
        sorted(df_raw["ticker"].dropna().unique().tolist())
        if (df_raw is not None and "ticker" in df_raw.columns)
        else []
    )
    sel_tickers = st.multiselect("Filtra tickers", options=tickers, default=tickers)

    if df_raw is None or df_raw.empty:
        st.info(
            "Aún no hay datos para mostrar aquí. Genera `trades.csv` con el backtest.sh event-driven."
        )
    else:
        min_ts = pd.to_datetime(df_raw["ts"].min()).date()
        max_ts = pd.to_datetime(df_raw["ts"].max()).date()
        dr = st.date_input("Rango de fechas (UTC)", value=(min_ts, max_ts))
        if isinstance(dr, tuple):
            date_from, date_to = dr
        else:
            date_from, date_to = min_ts, max_ts

        df = df_raw.copy()
        if sel_tickers:
            df = df[df["ticker"].isin(sel_tickers)]
        df = df[
            (df["ts"].dt.date >= pd.to_datetime(date_from).date())
            & (df["ts"].dt.date <= pd.to_datetime(date_to).date())
        ]
        df = df.sort_values("ts").reset_index(drop=True)

        if df.empty:
            st.warning("No hay operaciones tras aplicar los filtros.")
        else:
            # Reconstruir operaciones cerradas (ENTRY->EXIT/EOD_CLOSE)
            import numpy as np

            records, open_pos = [], {}
            for _, r in df.sort_values("ts").iterrows():
                typ = str(r.get("type", "")).upper()
                tkr = r["ticker"]
                fee = float(r.get("fee", 0.0))
                side = int(r.get("side", 0))
                qty = int(r.get("qty", 0))
                price = float(r.get("price", np.nan))
                ts = pd.to_datetime(r["ts"])

                if typ == "ENTRY":
                    open_pos[tkr] = {
                        "ticker": tkr,
                        "entry_ts": ts,
                        "side": side,
                        "qty": qty,
                        "entry_price": price,
                        "fees_entry": fee,
                    }
                elif typ in ("EXIT", "EOD_CLOSE"):
                    pos = open_pos.get(tkr)
                    if pos is None:
                        continue
                    pnl_gross = (price - pos["entry_price"]) * pos["qty"] * pos["side"]
                    fees = pos["fees_entry"] + fee
                    pnl = pnl_gross - fees
                    duration_min = (ts - pos["entry_ts"]).total_seconds() / 60.0
                    records.append(
                        {
                            "ticker": tkr,
                            "entry_time": pos["entry_ts"],
                            "exit_time": ts,
                            "side": pos["side"],
                            "qty": pos["qty"],
                            "entry_price": pos["entry_price"],
                            "exit_price": price,
                            "fees": fees,
                            "pnl": pnl,
                            "exit_type": typ,
                            "duration_min": duration_min,
                        }
                    )
                    open_pos.pop(tkr, None)

            df_trades = pd.DataFrame(records)
            if df_trades.empty:
                st.warning(
                    "No se pudieron emparejar entradas y salidas. Revisa el contenido de `trades.csv`."
                )
            else:
                # KPIs
                wins = (df_trades["pnl"] > 0).sum()
                losses = (df_trades["pnl"] <= 0).sum()
                n_tr = len(df_trades)
                win_rate = wins / n_tr if n_tr else float("nan")
                avg_pnl = df_trades["pnl"].mean() if n_tr else float("nan")
                total_pnl = df_trades["pnl"].sum() if n_tr else 0.0

                notional = (df_trades["entry_price"] * df_trades["qty"]).replace(0, pd.NA)
                df_trades["ret"] = df_trades["pnl"] / notional
                mean_ret = df_trades["ret"].mean()
                std_ret = df_trades["ret"].std(ddof=1)
                sharpe_tr = (
                    (mean_ret / std_ret * (n_tr**0.5)) if std_ret and std_ret > 0 else float("nan")
                )

                df_trades = df_trades.sort_values("exit_time")
                df_trades["equity"] = df_trades["pnl"].cumsum()

                eq = df_trades["equity"].values
                if len(eq) > 0:
                    roll_max = (pd.Series(eq).cummax()).values
                    dd = roll_max - eq
                    max_dd = float(dd.max()) if len(dd) else 0.0
                else:
                    max_dd = 0.0

                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Trades", n_tr)
                c2.metric("Win rate", f"{win_rate*100:.1f}%")
                c3.metric("PnL total", f"{total_pnl:,.2f}")
                c4.metric("PnL medio", f"{avg_pnl:,.2f}")
                c5.metric("Sharpe (por trade)", f"{sharpe_tr:.2f}" if pd.notna(sharpe_tr) else "—")
                c6.metric("Max Drawdown", f"{max_dd:,.2f}")

                st.subheader("📈 Equity Curve (PnL acumulado)")
                fig_eq = px.line(df_trades, x="exit_time", y="equity", markers=False)
                st.plotly_chart(fig_eq, use_container_width=True)

                st.subheader("📊 PnL por ticker")
                pnl_by_ticker = (
                    df_trades.groupby("ticker", as_index=False)["pnl"]
                    .sum()
                    .sort_values("pnl", ascending=False)
                )
                fig_bar = px.bar(pnl_by_ticker, x="ticker", y="pnl")
                st.plotly_chart(fig_bar, use_container_width=True)

                st.subheader("Detalle de trades")
                st.dataframe(
                    df_trades[
                        [
                            "ticker",
                            "entry_time",
                            "exit_time",
                            "side",
                            "qty",
                            "entry_price",
                            "exit_price",
                            "fees",
                            "pnl",
                            "exit_type",
                            "duration_min",
                        ]
                    ],
                    use_container_width=True,
                )

# -------- TAB 6: LANZAR OPTIMIZER (02c) --------
with tab6:
    import shlex
    import subprocess
    import time
    from itertools import product

    st.header("🚀 Lanzar Optimizer (02c)")

    tickers_file = st.text_input(
        "Archivo de tickers (en 04_config o ruta completa)", value="sp500_tickers.txt"
    )
    timeframe = st.selectbox("Timeframe", options=["5mins", "15mins", "1hour"], index=0)

    # Multi-valores para hacer grid
    c1, c2 = st.columns(2)
    tp_vals = c1.text_input("TP multipliers (CSV)", value="2.0, 1.5")
    sl_vals = c2.text_input("SL multipliers (CSV)", value="2.0, 1.0")

    c3, c4 = st.columns(2)
    tl_vals = c3.text_input("Time limit (velas, CSV)", value="16, 32")
    thr_vals = c4.text_input("Thresholds (CSV o JSON)", value="0.6, 0.65, 0.7, 0.75")

    days_grid = st.text_input("Days grid (CSV o JSON)", value="90")
    model = st.selectbox("Modelo", options=["xgb", "rf", "logreg"], index=0)
    feature_set = st.selectbox("Feature set", options=["core", "core+vol", "all"], index=1)
    hparams = st.text_area("Hiperparámetros (JSON opcional)", value="")
    train_after_cv = st.checkbox("Entrenar tras CV (04)", value=True)
    max_tickers = st.number_input("Máx. tickers (debug)", value=0, step=1, help="0 = todos")

    def _parse_num_csv(s, cast=float):
        xs = []
        for p in s.replace("[", "").replace("]", "").split(","):
            p = p.strip()
            if p:
                xs.append(cast(p))
        return xs

    tp_list = _parse_num_csv(tp_vals, float)
    sl_list = _parse_num_csv(sl_vals, float)
    tl_list = _parse_num_csv(tl_vals, int)

    thr_str = thr_vals.strip()
    days_str = days_grid.strip()

    escenarios = list(product(tp_list, sl_list, tl_list))
    st.write(f"Se crearán **{len(escenarios)}** escenarios × tickers de `{tickers_file}`.")

    # Comando base (plantilla)
    base = [
        "python",
        "RSH_Scenarios.py",
        "--tickers_file",
        tickers_file,
        "--timeframe",
        timeframe,
        "--threshold_grid",
        thr_str,
        "--model",
        model,
        "--feature_set",
        feature_set,
    ]
    if hparams.strip():
        base += ["--hparams", hparams.strip()]
    if train_after_cv:
        base += ["--train_after_cv"]
    if max_tickers and int(max_tickers) > 0:
        base += ["--max_tickers", str(int(max_tickers))]
    if days_str:
        base += ["--days_grid", days_str]

    st.subheader("Comando (plantilla)")
    st.code(
        " ".join(
            shlex.quote(x)
            for x in base + ["--tp_grid", "[TP]", "--sl_grid", "[SL]", "--time_limit_grid", "[TL]"]
        ),
        language="bash",
    )

    status = st.empty()

    # Lanzar (sin barra, mostrando "escenario i/total")
    if st.button("Lanzar todos los escenarios"):
        logs_dir = Path(S.logs_path) / "optimizer_runs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        total = len(escenarios)
        for i, (tp, sl, tl) in enumerate(escenarios, start=1):
            status.write(f"Ejecutando escenario {i}/{total} | TP={tp} SL={sl} TL={tl}")
            cmd = base + [
                "--tp_grid",
                json.dumps([float(tp)]),
                "--sl_grid",
                json.dumps([float(sl)]),
                "--time_limit_grid",
                json.dumps([int(tl)]),
            ]
            log_file = logs_dir / f"run_tp{tp}_sl{sl}_tl{tl}_{int(time.time())}.log"
            with open(log_file, "w", encoding="utf-8") as lf:
                subprocess.run(
                    cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parent
                )

        status.write(
            "✅ Todos los escenarios terminados. Revisa la pestaña Optimizer y la UI de MLflow."
        )
