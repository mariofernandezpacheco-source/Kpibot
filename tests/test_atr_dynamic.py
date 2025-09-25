# test_atr_dynamic.py
import pandas as pd
import numpy as np
from engine.backtest_runner import run_backtest_for_ticker


def test_atr_calculation():
    print("=== TEST ATR DINÁMICO ===")

    # Test con AAPL usando multiplicadores conocidos
    params = {
        "threshold": 0.7,  # Más bajo para generar señales
        "tp_multiplier": 2.0,  # 2x ATR
        "sl_multiplier": 1.0,  # 1x ATR
        "use_atr_multipliers": True,
        "time_limit": 16,
        "days": 30,  # Solo últimos 30 días para test rápido
    }

    result = run_backtest_for_ticker("AAPL", "5mins", params)

    print(f"Trades generados: {len(result.get('trades', []))}")

    if not result.get('trades', pd.DataFrame()).empty:
        trades = result['trades']

        # Mostrar ejemplos de trades para verificar TP/SL
        print("\nPrimeros 3 trades:")
        for i, trade in trades.head(3).iterrows():
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']

            print(f"\nTrade {i + 1}:")
            print(f"  Entry: ${entry_price:.2f}")
            print(f"  Exit: ${exit_price:.2f}")
            print(f"  Side: {trade['side']}")
            print(f"  Reason: {trade['reason']}")

            # Calcular qué múltiplo de ATR representó el movimiento
            if trade['reason'] in ['TP', 'SL']:
                price_diff = abs(exit_price - entry_price)
                print(f"  Diferencia precio: ${price_diff:.2f}")

        return True
    else:
        print("No se generaron trades para analizar")
        return False


if __name__ == "__main__":
    test_atr_calculation()


def compare_atr_methods():
    print("=== COMPARACIÓN ATR FIJO vs DINÁMICO ===")

    # Test con ATR dinámico
    params_dynamic = {
        "threshold": 0.7,
        "tp_multiplier": 2.0,
        "sl_multiplier": 1.0,
        "use_atr_multipliers": True,
        "days": 30,
    }

    # Test con porcentajes fijos (método anterior)
    params_fixed = {
        "threshold": 0.7,
        "tp_pct": 0.01,  # 1% fijo
        "sl_pct": 0.005,  # 0.5% fijo
        "use_atr_multipliers": False,
        "days": 30,
    }

    result_dynamic = run_backtest_for_ticker("AAPL", "5mins", params_dynamic)
    result_fixed = run_backtest_for_ticker("AAPL", "5mins", params_fixed)

    print(f"ATR Dinámico - Trades: {len(result_dynamic.get('trades', []))}")
    print(f"Porcentaje Fijo - Trades: {len(result_fixed.get('trades', []))}")

    print(f"ATR Dinámico - Net Return: {result_dynamic.get('metrics', {}).get('net_return', 0):.4f}")
    print(f"Porcentaje Fijo - Net Return: {result_fixed.get('metrics', {}).get('net_return', 0):.4f}")


def test_mlflow_params():
    print("=== TEST PARÁMETROS MLFLOW ===")

    # Ejecutar scenario simple
    import subprocess
    import sys

    cmd = [
        sys.executable, "RSH_Scenarios.py",
        "--ticker", "AAPL",
        "--timeframe", "5mins",
        "--models", "xgb",
        "--days", "30",
        "--tp_grid", "1.5,2.0",  # Multiplicadores ATR
        "--sl_grid", "1.0,1.5",
        "--thresholds", "0.7"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("Return code:", result.returncode)
    if result.returncode == 0:
        print("✅ Scenarios ejecutado correctamente")
        print("Revisa MLflow para verificar que tp_multiplier y sl_multiplier aparecen como 1.5, 2.0, etc.")
    else:
        print("❌ Error en scenarios:")
        print(result.stderr)