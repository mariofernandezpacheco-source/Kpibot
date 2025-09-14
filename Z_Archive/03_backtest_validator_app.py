# 03_backtest_validator_app.py  — añade pestaña "📑 Informe CV"
import json  # <-- añadido
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, train_test_split

# Modelos
from xgboost import XGBClassifier

# Config (settings.py en la raíz)
from settings import S

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# Utils del proyecto
from utils.A_data_loader import load_data
from utils.B_feature_engineering import add_technical_indicators, load_context_data
from utils.C_label_generator import generate_triple_barrier_labels
from utils.D_backtest_utils import backtest_signals
from utils.reproducibility import set_global_determinism, write_env_versions
from utils.time_utils import to_ui_tz

# Fija semillas globales
set_global_determinism(S.seed, set_pythonhash=S.pythonhashseed)

# (Opcional pero recomendado) Registra versiones del entorno
if S.record_versions:
    write_env_versions(S.env_versions_path)

st.set_page_config(
    page_title="Validador de Backtests", layout="wide", initial_sidebar_state="collapsed"
)
st.title("🔬 Validador de Estrategias y Modelos")

TOP_FEATURES = [
    "atr_14",
    "vix_roc_5",
    "spy_rsi_14",
    "rsi_14",
    "macd_diff",
    "relative_strength",
    "stoch_d",
    "willr_14",
    "trend_cci_20",
    "roc_10",
    "mfi_14",
    "stoch_k",
    "bb_bb_low",
    "bb_bb_high",
    "trend_adx_14",
]


def get_tickers_from_file(file_path: Path) -> list:
    if not file_path.exists():
        st.error(f"Error: El fichero de tickers no se encontró en {file_path}")
        return []
    with open(file_path) as f:
        tickers = [line.strip() for line in f if line.strip()]
    return sorted(tickers)


# --- BARRA LATERAL DE CONFIGURACIÓN ---
st.sidebar.header("1. Configuración General")
project_root = Path(__file__).resolve().parent
full_tickers_filepath = project_root / "04_config" / "sp500_tickers.txt"
robust_tickers_filepath = project_root / "04_config" / "top_100_robustos.txt"
ALL_TICKERS = get_tickers_from_file(full_tickers_filepath)
ROBUST_TICKERS = get_tickers_from_file(robust_tickers_filepath)
ticker_options = ["TODOS", "ROBUSTOS"] + ALL_TICKERS
ticker = st.sidebar.selectbox("Selecciona el Ticker o Grupo", ticker_options)

# Timeframe
timeframe_options = ["5mins", "10mins"]
tf_default_index = (
    timeframe_options.index(S.timeframe_default) if S.timeframe_default in timeframe_options else 0
)
timeframe_input = st.sidebar.selectbox(
    "Selecciona el Timeframe", options=timeframe_options, index=tf_default_index
)

MODEL_OPTIONS = {
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )
}
model_name = st.sidebar.selectbox("Selecciona el Modelo", list(MODEL_OPTIONS.keys()))

st.sidebar.header("2. Período del Backtest")
end_date_default = datetime.now().date()
start_date_default = end_date_default - timedelta(days=S.days_of_data)
start_date_input = st.sidebar.date_input("Fecha de Inicio", value=start_date_default)
end_date_input = st.sidebar.date_input("Fecha de Fin", value=end_date_default)

st.sidebar.header("3. Parámetros de la Estrategia")
tp_mult = st.sidebar.number_input(
    "Multiplicador Take Profit (x ATR)", 0.5, 10.0, float(S.tp_multiplier), 0.1
)
sl_mult = st.sidebar.number_input(
    "Multiplicador Stop Loss (x ATR)", 0.5, 10.0, float(S.sl_multiplier), 0.1
)
time_limit = st.sidebar.slider("Límite de Tiempo (Velas)", 1, 60, int(S.time_limit_candles), 1)
batch_threshold = st.sidebar.number_input(
    "Threshold para 'TODOS'/'ROBUSTOS'", 0.4, 0.99, float(S.threshold_default), 0.01
)

st.sidebar.header("4. Simulación Realista")
capital_input = st.sidebar.number_input("Capital por Operación (€)", 100, 10000, 1000, 100)
commission_input = st.sidebar.number_input("Comisión por Trade (€)", 0.0, 5.0, 0.35, 0.01)

st.sidebar.header("5. Selección de Features")
feature_selection_mode = st.sidebar.selectbox(
    "Modo de Features:", options=["Todos los Features", "Top Features (Definidos en código)"]
)

if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None


# --- FUNCIONES ---
@st.cache_data(show_spinner="Cargando y procesando datos...")
def load_and_prepare_data(
    ticker, timeframe, start_dt, end_dt, tp_multiplier, sl_multiplier, time_limit_candles
):
    df = load_data(ticker=ticker, timeframe=timeframe, use_local=True, base_path=S.data_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    mask = (df["date"].dt.date >= start_dt) & (df["date"].dt.date <= end_dt)
    df = df.loc[mask].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()
    context_data = load_context_data(timeframe, S.data_path)
    df = add_technical_indicators(df, context_data=context_data)
    df = generate_triple_barrier_labels(
        data=df,
        volatility_col="atr_14",
        tp_multiplier=tp_multiplier,
        sl_multiplier=sl_multiplier,
        time_limit_candles=time_limit_candles,
    )
    df = df.dropna()
    return df


def run_evaluation(df, model_instance, features, capital, commission, n_splits=None):
    n_splits = n_splits or S.n_splits_cv
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_metrics, all_trades_list, all_events_list = [], [], []
    st.write(f"Realizando validación cruzada con {n_splits} splits...")
    progress_bar = st.progress(0)
    X, y = df[features], df["label"]

    last_model = None

    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        mdl = clone(model_instance)
        y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
        if y_train_mapped.nunique() < 2:
            progress_bar.progress((i + 1) / n_splits)
            continue

        mdl.fit(X_train, y_train_mapped)
        probs = mdl.predict_proba(X_val)

        thresholds = np.arange(0.4, 0.95, 0.05)
        for th in thresholds:
            class_map = {c: i for i, c in enumerate(mdl.classes_)}
            prob_buy = probs[:, class_map.get(2, -1)] if 2 in class_map else np.zeros(len(probs))
            prob_sell = probs[:, class_map.get(0, -1)] if 0 in class_map else np.zeros(len(probs))
            signals = np.select([prob_buy > th, prob_sell > th], [1, -1], default=0)

            df_val_signals = pd.DataFrame(
                {
                    "timestamp": df["date"].iloc[val_idx],
                    "close": df["close"].iloc[val_idx],
                    "signal": signals,
                }
            )
            metrics_bt, trades_df, events_df = backtest_signals(
                df_val_signals, capital_per_trade=capital, commission_per_trade=commission
            )
            metrics_bt.update({"threshold": th, "fold": i + 1})
            all_metrics.append(metrics_bt)

            if not trades_df.empty:
                trades_df["threshold"], trades_df["fold"] = th, i + 1
                all_trades_list.append(trades_df)
                events_df["threshold"] = th
                all_events_list.append(events_df)

        progress_bar.progress((i + 1) / n_splits)
        last_model = mdl

    if not all_metrics:
        return pd.DataFrame(), None, pd.DataFrame(), pd.DataFrame()

    results_df = pd.DataFrame(all_metrics)
    avg_results = (
        results_df.groupby("threshold")
        .mean(numeric_only=True)
        .drop(columns="fold")
        .sort_values("sharpe_ratio", ascending=False)
    )
    all_trades_df = (
        pd.concat(all_trades_list, ignore_index=True) if all_trades_list else pd.DataFrame()
    )
    all_events_df = (
        pd.concat(all_events_list, ignore_index=True) if all_events_list else pd.DataFrame()
    )
    return avg_results, last_model, all_trades_df, all_events_df


def run_batch_backtest(
    tickers,
    model_instance,
    timeframe,
    start_dt,
    end_dt,
    tp_mult,
    sl_mult,
    time_limit,
    threshold,
    features_to_use_mode,
    capital,
    commission,
):
    st.info(f"Ejecutando backtest.sh en {len(tickers)} tickers...")
    all_results, all_importances_list, all_trades_list = [], [], []

    debug_log = []  # diagnóstico de rango de fechas por ticker

    progress_bar = st.progress(0)
    context_data = load_context_data(timeframe, S.data_path)

    for i, current_ticker in enumerate(tickers):
        try:
            df = load_data(
                ticker=current_ticker, timeframe=timeframe, use_local=True, base_path=S.data_path
            )
            df["date"] = pd.to_datetime(df["date"], utc=True)

            if not df.empty:
                debug_log.append(
                    f"Ticker: {current_ticker} | Datos: {df['date'].min().date()} → {df['date'].max().date()}"
                )
            else:
                debug_log.append(f"Ticker: {current_ticker} | Fichero vacío.")

            mask = (df["date"].dt.date >= start_dt) & (df["date"].dt.date <= end_dt)
            df = df.loc[mask].reset_index(drop=True)
            if not df.empty:
                df = add_technical_indicators(df, context_data=context_data)
                df = generate_triple_barrier_labels(
                    data=df,
                    volatility_col="atr_14",
                    tp_multiplier=tp_mult,
                    sl_multiplier=sl_mult,
                    time_limit_candles=time_limit,
                )
                df = df.dropna()
        except FileNotFoundError:
            debug_log.append(f"Ticker: {current_ticker} | Fichero no encontrado.")
            progress_bar.progress((i + 1) / len(tickers))
            continue

        if df.empty or df["label"].nunique() < 2:
            progress_bar.progress((i + 1) / len(tickers))
            continue

        cols_to_exclude = ["date", "label", "open", "high", "low", "close", "volume", "index"]
        all_features_list = [col for col in df.columns if col not in cols_to_exclude]
        features = (
            TOP_FEATURES
            if features_to_use_mode == "Top Features (Definidos en código)"
            else all_features_list
        )
        features = [f for f in features if f in df.columns]

        X, y = df[features], df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
        if y_train.nunique() < 2:
            progress_bar.progress((i + 1) / len(tickers))
            continue

        mdl = clone(model_instance)
        mdl.fit(X_train, y_train.map({-1: 0, 0: 1, 1: 2}))

        if hasattr(mdl, "feature_importances_"):
            importance_df = pd.DataFrame(
                {"feature": features, "importance": mdl.feature_importances_}
            )
            all_importances_list.append(importance_df)

        probs = mdl.predict_proba(X_test)
        class_map = {c: i for i, c in enumerate(mdl.classes_)}
        prob_buy = probs[:, class_map.get(2, -1)] if 2 in class_map else np.zeros(len(probs))
        prob_sell = probs[:, class_map.get(0, -1)] if 0 in class_map else np.zeros(len(probs))
        signals = np.select([prob_buy > threshold, prob_sell > threshold], [1, -1], default=0)

        df_test_signals = pd.DataFrame(
            {
                "timestamp": df["date"].loc[X_test.index],
                "close": df["close"].loc[X_test.index],
                "signal": signals,
            }
        )
        metrics_bt, trades_df, _ = backtest_signals(
            df_test_signals, capital_per_trade=capital, commission_per_trade=commission
        )
        metrics_bt["ticker"] = current_ticker
        all_results.append(metrics_bt)

        if not trades_df.empty:
            all_trades_list.append(trades_df)

        progress_bar.progress((i + 1) / len(tickers))

    if not all_trades_list:
        portfolio_max_exposure = 0
    else:
        full_trades_df = pd.concat(all_trades_list, ignore_index=True)
        opens = full_trades_df[["entry_time"]].rename(columns={"entry_time": "timestamp"})
        opens["capital_change"] = capital
        closes = full_trades_df[["exit_time"]].rename(columns={"exit_time": "timestamp"})
        closes["capital_change"] = -capital
        events_df = pd.concat([opens, closes]).sort_values("timestamp").reset_index(drop=True)
        events_df["capital_in_use"] = events_df["capital_change"].cumsum()
        portfolio_max_exposure = events_df["capital_in_use"].max()

    aggregated_importances = (
        pd.concat(all_importances_list)
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        if all_importances_list
        else pd.DataFrame()
    )

    return pd.DataFrame(all_results), aggregated_importances, portfolio_max_exposure, debug_log


def make_equity_curve(trades_df: pd.DataFrame, start_capital: float = 0.0) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    if "exit_time" in df.columns:
        df = df.sort_values("exit_time")
        timeline = df[["exit_time", "pnl"]].rename(columns={"exit_time": "timestamp"})
    else:
        df = df.sort_values("entry_time")
        timeline = df[["entry_time", "pnl"]].rename(columns={"entry_time": "timestamp"})
    timeline["equity"] = start_capital + timeline["pnl"].cumsum()
    return timeline


# --- LÓGICA DE EJECUCIÓN ---
col1, col2 = st.sidebar.columns(2)
if col1.button("🚀 Ejecutar Backtest", use_container_width=True):
    if start_date_input > end_date_input:
        st.error("Error: La fecha de inicio no puede ser posterior a la fecha de fin.")
    else:
        tickers_to_run = []
        if ticker == "TODOS":
            tickers_to_run = ALL_TICKERS
        elif ticker == "ROBUSTOS":
            tickers_to_run = ROBUST_TICKERS

        if ticker in ["TODOS", "ROBUSTOS"]:
            results_df, agg_importances, portfolio_max_exposure, debug_log = run_batch_backtest(
                tickers_to_run,
                MODEL_OPTIONS[model_name],
                timeframe_input,
                start_date_input,
                end_date_input,
                tp_mult,
                sl_mult,
                time_limit,
                batch_threshold,
                feature_selection_mode,
                capital=capital_input,
                commission=commission_input,
            )
            st.session_state.backtest_results = {
                "type": "batch",
                "results": results_df,
                "importances": agg_importances,
                "max_exposure": portfolio_max_exposure,
                "debug_log": debug_log,
            }
        else:
            df_full = load_and_prepare_data(
                ticker,
                timeframe_input,
                start_date_input,
                end_date_input,
                tp_mult,
                sl_mult,
                time_limit,
            )
            if df_full.empty:
                st.warning("No se encontraron datos para el período seleccionado.")
                st.session_state.backtest_results = None
            else:
                cols_to_exclude = [
                    "date",
                    "label",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "index",
                ]
                all_features_list = [col for col in df_full.columns if col not in cols_to_exclude]
                features_to_use = (
                    TOP_FEATURES
                    if feature_selection_mode == "Top Features (Definidos en código)"
                    else all_features_list
                )
                features_to_use = [f for f in features_to_use if f in df_full.columns]

                st.info(f"Usando {len(features_to_use)} features ('{feature_selection_mode}').")

                results, last_model, all_trades, all_events = run_evaluation(
                    df_full,
                    MODEL_OPTIONS[model_name],
                    features_to_use,
                    capital=capital_input,
                    commission=commission_input,
                    n_splits=S.n_splits_cv,
                )
                st.session_state.backtest_results = {
                    "type": "single",
                    "ticker": ticker,
                    "df_full": df_full,
                    "results": results,
                    "last_model": last_model,
                    "all_trades": all_trades,
                    "all_events": all_events,
                    "features": features_to_use,
                }

if col2.button("🧹 Limpiar", use_container_width=True):
    st.session_state.backtest_results = None

# --- LÓGICA DE VISUALIZACIÓN (igual que tenías)
if st.session_state.backtest_results is None:
    st.info(
        "Ajusta los parámetros en la barra lateral y haz clic en 'Ejecutar Backtest' para comenzar."
    )
else:
    results_data = st.session_state.backtest_results

    if results_data["type"] == "batch":
        results_df, agg_importances, portfolio_max_exposure, debug_log = (
            results_data["results"],
            results_data["importances"],
            results_data["max_exposure"],
            results_data["debug_log"],
        )
        if results_df.empty:
            st.warning("No se pudo completar el backtest.sh para ningún ticker.")
        else:
            st.subheader("📊 Resumen Agregado del Backtest Masivo")
            avg_metrics = results_df.mean(numeric_only=True)
            total_profit = results_df["profit"].sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Profit Total Agregado", f"€{total_profit:,.2f}")
            c2.metric("Sharpe Ratio Promedio", f"{avg_metrics['sharpe_ratio']:.2f}")
            c3.metric("💥 Exposición Máxima Simultánea", f"€{portfolio_max_exposure:,.2f}")
            c4, c5, c6 = st.columns(3)
            c4.metric("Win Rate Promedio", f"{avg_metrics['win_rate']:.1f}%")
            c5.metric("# Trades Promedio", f"{avg_metrics['num_trades']:.1f}")
            c6.metric("Profit Promedio / Ticker", f"€{avg_metrics['profit']:.2f}")

            st.subheader("Gráfico de Dispersión: Riesgo vs. Retorno")
            plot_df = results_df[results_df["sharpe_ratio"] > 0].copy()
            if not plot_df.empty:
                fig = px.scatter(
                    plot_df,
                    x="sharpe_ratio",
                    y="win_rate",
                    hover_data=["ticker"],
                    color="profit",
                    color_continuous_scale=px.colors.sequential.Viridis,
                    size="num_trades",
                    title="Rendimiento de Tickers Individuales",
                )
                fig.update_layout(xaxis_title="Sharpe Ratio", yaxis_title="Win Rate (%)")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("🏆 Top 100 Mejores Tickers (por Profit Total)")
            st.dataframe(
                results_df.sort_values("profit", ascending=False).head(100)[
                    ["ticker", "win_rate", "sharpe_ratio", "profit"]
                ]
            )

            st.subheader("📉 Top 10 Peores Tickers (por Profit Total)")
            st.dataframe(results_df.sort_values("profit", ascending=True).head(10))

            st.subheader("🧠 Importancia de Features Promedio (Todos los Tickers)")
            if not agg_importances.empty:
                fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
                top_features = agg_importances.head(15)
                ax_imp.barh(top_features["feature"], top_features["importance"])
                ax_imp.set_title(f"Top 15 Features (Promedio sobre {len(results_df)} tickers)")
                ax_imp.invert_yaxis()
                st.pyplot(fig_imp)
            else:
                st.info("No se pudo generar el análisis de importancia de features.")

            with st.expander("Ver Log de Fechas de Datos"):
                st.code("\n".join(debug_log))

    elif results_data["type"] == "single":
        st.subheader(f"📈 Backtest individual — {results_data['ticker']}")
        results, last_model, all_trades, all_events = (
            results_data["results"],
            results_data["last_model"],
            results_data["all_trades"],
            results_data["all_events"],
        )
        df_full = results_data["df_full"]
        features_used = results_data["features"]

        if results is None or results.empty:
            st.warning("Sin resultados. Ajusta parámetros o período.")
        else:
            st.write("Resultados promedio por threshold (CV temporal):")
            st.dataframe(results[["sharpe_ratio", "win_rate", "num_trades", "profit"]].round(3))

            # Selección del threshold óptimo
            best_threshold = float(results.sort_values("sharpe_ratio", ascending=False).index[0])
            th_col1, th_col2 = st.columns([1, 2])
            selected_threshold = th_col1.number_input(
                "Threshold seleccionado", 0.4, 0.99, best_threshold, 0.01
            )
            th_col2.info(f"Threshold óptimo (Sharpe máx): **{best_threshold:.2f}**")

            # Filtrar trades y eventos del threshold elegido
            trades_sel = (
                all_trades[all_trades["threshold"].round(3) == round(selected_threshold, 3)]
                if not all_trades.empty
                else pd.DataFrame()
            )
            events_sel = (
                all_events[all_events["threshold"].round(3) == round(selected_threshold, 3)]
                if not all_events.empty
                else pd.DataFrame()
            )

            # Métricas del threshold elegido
            if not results.empty and selected_threshold in results.index:
                sel_metrics = results.loc[selected_threshold].to_dict()
            else:
                sel_metrics = {}

            st.markdown("### 📊 Métricas del threshold seleccionado")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe", f"{sel_metrics.get('sharpe_ratio', 0):.2f}")
            c2.metric("Win Rate", f"{sel_metrics.get('win_rate', 0):.1f}%")
            c3.metric("# Trades", f"{sel_metrics.get('num_trades', 0):.0f}")
            c4.metric("Profit (€)", f"€{sel_metrics.get('profit', 0):.2f}")

            # Equity curve
            st.markdown("### 📈 Equity Curve")
            eq_df = make_equity_curve(trades_sel, start_capital=0.0)
            if eq_df.empty:
                st.info("No hay trades para dibujar equity con el threshold seleccionado.")
            else:
                eq_df["timestamp_ui"] = to_ui_tz(eq_df["timestamp"])
                fig_eq = px.line(
                    eq_df, x="timestamp_ui", y="equity", title="Evolución del Equity (Hora local)"
                )
                fig_eq.update_layout(xaxis_title="Tiempo", yaxis_title="€")
                st.plotly_chart(fig_eq, use_container_width=True)

            # Tablas
            col_trades, col_events = st.columns(2)
            with col_trades:
                st.markdown("### 🧾 Trades")
                if trades_sel.empty:
                    st.info("No hay trades para este threshold.")
                else:
                    show_cols = [c for c in trades_sel.columns if c not in ["threshold", "fold"]]
                    st.dataframe(trades_sel[show_cols])
            with col_events:
                st.markdown("### ⏱️ Eventos")
                if events_sel.empty:
                    st.info("No hay eventos para este threshold.")
                else:
                    show_cols_e = [c for c in events_sel.columns if c not in ["threshold", "fold"]]
                    st.dataframe(events_sel[show_cols_e])

            with st.expander("Ver muestra de datos e indicadores"):
                st.dataframe(
                    df_full[
                        ["date", "open", "high", "low", "close", "volume"]
                        + [f for f in features_used if f in df_full.columns]
                    ].head(50)
                )

# =========================
# 📑 Informe CV (3ª pestaña)
# =========================
st.markdown("---")
st.subheader("📑 Informe de CV (medias / dispersiones)")


def _load_cv_jsons(cv_dir: Path):
    items = []
    if not cv_dir.exists():
        return items
    for p in cv_dir.glob("*_cv.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            items.append(d)
        except Exception:
            continue
    return items


def _cv_items_to_df(items: list[dict]) -> pd.DataFrame:
    rows = []
    for d in items:
        mm = d.get("metrics_mean", {}) or {}
        ms = d.get("metrics_std", {}) or {}
        evs = d.get("ev_at_recommended", {}) or {}
        oof = d.get("oof", {}) or {}
        rows.append(
            {
                "ticker": d.get("ticker"),
                "timeframe": d.get("timeframe"),
                "recommended_threshold": d.get("recommended_threshold"),
                "recommended_by": d.get("recommended_by"),
                "roc_auc_mean": mm.get("roc_auc"),
                "roc_auc_std": ms.get("roc_auc"),
                "pr_auc_mean": mm.get("pr_auc"),
                "pr_auc_std": ms.get("pr_auc"),
                "ev_mean_at_thr": evs.get("mean"),
                "ev_std_at_thr": evs.get("std"),
                "folds_ev_covered": evs.get("folds_covered"),
                "oof_samples": oof.get("n_samples"),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["timeframe", "ticker"]).reset_index(drop=True)
    return df


cv_dir = Path(getattr(S, "cv_dir", Path(S.logs_path) / "cv"))
cv_items = _load_cv_jsons(cv_dir)
cv_df = _cv_items_to_df(cv_items)

if cv_df.empty:
    st.info(
        f"No hay resultados de CV en {cv_dir}. Lanza la TSCV (worker o script) para generarlos."
    )
else:
    st.dataframe(cv_df)
    csv_bytes = cv_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV del informe",
        data=csv_bytes,
        file_name=f"cv_summary_{S.timeframe_default}.csv",
        mime="text/csv",
    )
