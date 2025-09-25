#!/usr/bin/env python3
"""
Ejemplos de migración manual de print() a logging estructurado
Muestra cómo migrar patrones comunes de tu código
"""

import time
from datetime import datetime
import numpy as np
from utils.logging_enhanced import get_logger, with_correlation_id, with_component, log_function_call


# ============================================================================
# EJEMPLO 1: Migración básica de prints en módulo de datos
# ============================================================================

# ANTES - DAT_Data_download.py (ejemplo)
def download_data_old():
    print("Iniciando descarga de datos...")
    print(f"Procesando {len(tickers)} tickers")
    print("✅ Descarga completada exitosamente")


# DESPUÉS - Con logging estructurado
logger = get_logger(__name__)


def download_data_new():
    logger.info("download_start", message="Iniciando descarga de datos")
    logger.info("processing_tickers", count=len(tickers), message=f"Procesando {len(tickers)} tickers")
    logger.info("download_success", message="Descarga completada exitosamente")


# ============================================================================
# EJEMPLO 2: Migración de logging por ticker (LIV_PaperWorker.py)
# ============================================================================

# ANTES - Prints dispersos
def process_ticker_old(ticker):
    print(f"[{ticker}] Cargando modelo...")
    print(f"[{ticker}] Generando señal: BUY")
    print(f"[{ticker}] ERROR: No se pudo cargar datos")


# DESPUÉS - Con contexto por ticker
def process_ticker_new(ticker):
    # Logger con contexto bindeado
    ticker_logger = logger.bind(ticker=ticker, timeframe="5mins")

    ticker_logger.info("model_loading", message="Cargando modelo")
    ticker_logger.info("signal_generated", signal="BUY", message="Generando señal")
    ticker_logger.error("data_load_error", message="No se pudo cargar datos")


# ============================================================================
# EJEMPLO 3: Logging con correlation IDs (para tracking de flujos)
# ============================================================================

# Ejemplo de tracking de una operación completa
def execute_trade_with_tracking(ticker, signal):
    # Cada trade tiene un correlation ID único
    with with_correlation_id() as corr_id:
        logger.info("trade_start",
                    ticker=ticker,
                    signal=signal,
                    message="Iniciando ejecución de trade")

        try:
            # Todo el logging dentro mantiene el mismo correlation ID
            model_prediction = get_model_prediction(ticker)
            logger.info("model_prediction_received",
                        prediction=model_prediction,
                        confidence=0.85)

            trade_result = place_order(ticker, signal)
            logger.info("order_placed",
                        order_id=trade_result['order_id'],
                        price=trade_result['price'])

            logger.info("trade_success",
                        pnl=trade_result.get('pnl', 0),
                        message="Trade ejecutado exitosamente")

        except Exception as e:
            logger.error("trade_error",
                         error_type=type(e).__name__,
                         error_message=str(e),
                         exc_info=True)
            raise


# ============================================================================
# EJEMPLO 4: Logging de componentes (DAT, RSH, TRN, OPS)
# ============================================================================

# Para módulos grandes, usar contexto de componente
def research_module_example():
    with with_component("RSH"):
        logger.info("cv_start",
                    n_splits=5,
                    test_size=500,
                    message="Iniciando Cross Validation")

        for fold in range(5):
            fold_logger = logger.bind(fold=fold)
            fold_logger.info("fold_training", message="Entrenando fold")
            # Todos los logs de este fold tendrán fold=X en el contexto


# ============================================================================
# EJEMPLO 5: Decorator automático para funciones críticas
# ============================================================================

# Para funciones importantes, usar el decorator
@log_function_call(log_args=True, log_result=True)
def calculate_portfolio_metrics(positions, prices):
    """Función crítica que queremos trackear automáticamente"""
    total_value = sum(pos['quantity'] * prices[pos['ticker']]
                      for pos in positions)
    return {
        'total_value': total_value,
        'num_positions': len(positions)
    }


# ============================================================================
# EJEMPLO 6: Migración de error handling
# ============================================================================

# ANTES - Error handling básico
def load_model_old(ticker):
    try:
        model = joblib.load(f"models/{ticker}.pkl")
        print(f"Modelo {ticker} cargado exitosamente")
        return model
    except FileNotFoundError:
        print(f"ERROR: Modelo {ticker} no encontrado")
        return None
    except Exception as e:
        print(f"ERROR inesperado cargando {ticker}: {e}")
        return None


# DESPUÉS - Error handling estructurado
def load_model_new(ticker):
    model_logger = logger.bind(ticker=ticker)

    try:
        import joblib  # Import aquí para el ejemplo
        model_logger.info("model_load_start", message="Iniciando carga de modelo")
        model = joblib.load(f"models/{ticker}.pkl")

        model_logger.info("model_load_success",
                          model_type=type(model).__name__,
                          message="Modelo cargado exitosamente")
        return model

    except ImportError:
        model_logger.error("joblib_not_available", message="joblib no está instalado")
        return None
    except FileNotFoundError:
        model_logger.warning("model_not_found",
                             path=f"models/{ticker}.pkl",
                             message="Archivo de modelo no encontrado")
        return None

    except Exception as e:
        model_logger.error("model_load_error",
                           error_type=type(e).__name__,
                           error_message=str(e),
                           exc_info=True,
                           message="Error inesperado cargando modelo")
        return None


# ============================================================================
# EJEMPLO 7: Logging de métricas y performance
# ============================================================================

def backtest_with_metrics():
    perf_logger = logger.bind(component="backtest")

    start_time = time.time()

    # Log métricas de performance
    perf_logger.info("backtest_start",
                     tickers_count=len(TICKERS),
                     timeframe="5mins",
                     days=90)

    results = run_backtest()

    duration = time.time() - start_time

    # Log resultados estructurados
    perf_logger.info("backtest_complete",
                     duration_seconds=duration,
                     total_trades=len(results['trades']),
                     win_rate=results['win_rate'],
                     sharpe_ratio=results['sharpe'],
                     max_drawdown=results['max_drawdown'],
                     final_pnl=results['total_pnl'])


# ============================================================================
# EJEMPLO 8: Logging para debugging de ML
# ============================================================================

def train_model_with_debug_logging(ticker, X, y):
    ml_logger = logger.bind(ticker=ticker, component="TRN")

    # Mock XGBClassifier para el ejemplo
    try:
        from xgboost import XGBClassifier
    except ImportError:
        # Mock class para que el ejemplo funcione sin XGBoost
        class XGBClassifier:
            def __init__(self, **kwargs):
                self.params = kwargs

            def fit(self, X, y):
                pass

            def score(self, X, y):
                return 0.85

    # Log información del dataset
    ml_logger.info("dataset_info",
                   n_samples=len(X),
                   n_features=X.shape[1] if hasattr(X, 'shape') else len(X[0]),
                   class_distribution=dict(np.unique(y, return_counts=True)[1]) if hasattr(y, '__iter__') else {},
                   message="Dataset preparado para entrenamiento")

    # Log hiperparámetros
    hyperparams = {'max_depth': 5, 'n_estimators': 100}
    ml_logger.info("hyperparams_set", **hyperparams)

    # Entrenar con timing
    start_time = time.time()
    model = XGBClassifier(**hyperparams)

    # Simular fit para el ejemplo
    if hasattr(X, '__len__'):
        model.fit(X, y)

    training_time = time.time() - start_time

    # Log resultados de entrenamiento
    train_score = model.score(X, y) if hasattr(X, '__len__') else 0.85
    ml_logger.info("training_complete",
                   training_time_seconds=training_time,
                   train_accuracy=train_score,
                   model_type="XGBoost")

    return model


# ============================================================================
# EJEMPLO 9: Logging para monitoreo en vivo
# ============================================================================

def live_trading_monitoring():
    live_logger = logger.bind(component="OPS", session_id="live_session_2024")

    # Métricas que se actualizan continuamente
    portfolio_value = calculate_portfolio_value()
    open_positions = get_open_positions()

    live_logger.info("portfolio_update",
                     portfolio_value=portfolio_value,
                     num_open_positions=len(open_positions),
                     timestamp=datetime.utcnow().isoformat())

    # Alertas basadas en condiciones
    if portfolio_value < ALERT_THRESHOLD:
        live_logger.warning("portfolio_alert",
                            current_value=portfolio_value,
                            threshold=ALERT_THRESHOLD,
                            message="Portfolio por debajo del umbral de alerta")


# ============================================================================
# EJEMPLO 10: Migración de logging complejo con f-strings
# ============================================================================

# ANTES - F-strings complejos
def complex_logging_old(ticker, price, volume, signal):
    print(f"[{datetime.now()}] {ticker}: Precio={price:.2f}, Vol={volume:,}, Señal={signal}")
    print(f"Procesando {ticker} - RSI={get_rsi(ticker):.1f}, ATR={get_atr(ticker):.3f}")


# DESPUÉS - Structured logging con toda la información
def complex_logging_new(ticker, price, volume, signal):
    market_logger = logger.bind(ticker=ticker)

    market_logger.info("market_data_update",
                       price=round(price, 2),
                       volume=volume,
                       signal=signal,
                       timestamp=datetime.utcnow().isoformat(),
                       message="Actualización de datos de mercado")

    # Indicadores técnicos como evento separado
    rsi_value = get_rsi(ticker)
    atr_value = get_atr(ticker)

    market_logger.info("technical_indicators_update",
                       rsi=round(rsi_value, 1),
                       atr=round(atr_value, 3),
                       message="Indicadores técnicos actualizados")


if __name__ == "__main__":
    # Test de los ejemplos
    import time
    from datetime import datetime

    print("Ejecutando ejemplos de logging estructurado...")

    # Simular algunos datos para los ejemplos
    tickers = ["AAPL", "MSFT", "GOOGL"]
    ALERT_THRESHOLD = 10000


    # Funciones mock para que los ejemplos funcionen
    def get_model_prediction(ticker): return 0.75


    def place_order(ticker, signal): return {'order_id': 123, 'price': 150.0, 'pnl': 50}


    def get_rsi(ticker): return 65.5


    def get_atr(ticker): return 2.341


    def calculate_portfolio_value(): return 9500


    def get_open_positions(): return [{'ticker': 'AAPL', 'quantity': 100}]


    def run_backtest(): return {
        'trades': [1, 2, 3], 'win_rate': 0.65, 'sharpe': 1.2,
        'max_drawdown': -0.05, 'total_pnl': 150
    }


    # Ejecutar ejemplos
    download_data_new()
    process_ticker_new("AAPL")
    execute_trade_with_tracking("AAPL", "BUY")

    print("Ejemplos ejecutados - revisa los logs generados")