#!/usr/bin/env python3
"""
Script de migración para convertir settings.py a sistema unificado
Ejecuta este script para migrar automáticamente tu configuración existente
"""

import sys
from pathlib import Path
import yaml
import importlib.util
from typing import Dict, Any


def import_old_settings(settings_path: str = "settings.py"):
    """Importa settings.py existente"""
    try:
        spec = importlib.util.spec_from_file_location("old_settings", settings_path)
        if spec and spec.loader:
            old_settings = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(old_settings)
            return old_settings.S
        return None
    except Exception as e:
        print(f"Error importando settings.py: {e}")
        return None


def extract_config_from_old_settings(old_s) -> Dict[str, Any]:
    """Extrae configuración del objeto S original"""

    config = {
        "data": {},
        "models": {},
        "trading": {},
        "mlflow": {},
        "logging": {},
        "backtest": {},
        "system": {}
    }

    # Mapeo de atributos del S original a la nueva estructura
    mappings = {
        # Data section
        "data": {
            "storage_backend": getattr(old_s, "storage_backend", "parquet"),
            "base_path": getattr(old_s, "data_path", "[DAT]_data"),
            "parquet_base_path": getattr(old_s, "parquet_base_path", None),
            "timeframe_default": getattr(old_s, "timeframe_default", "5mins"),
            "bar_size_by_tf": getattr(old_s, "bar_size_by_tf", {}),
        },

        # Models section
        "models": {
            "tp_multiplier": getattr(old_s, "tp_multiplier", 3.0),
            "sl_multiplier": getattr(old_s, "sl_multiplier", 2.0),
            "time_limit_candles": getattr(old_s, "time_limit_candles", 16),
            "label_window": getattr(old_s, "label_window", 5),
            "model_type": getattr(old_s, "model_type", "xgb"),
            "feature_set": getattr(old_s, "feature_set", "core"),
            "cv_splits": getattr(old_s, "n_splits_cv", 5),
            "cv_test_size": getattr(old_s, "cv_test_size", 500),
            "cv_scheme": getattr(old_s, "cv_scheme", "expanding"),
            "cv_embargo": getattr(old_s, "cv_embargo", -1),
            "cv_purge": getattr(old_s, "cv_purge", -1),
            "threshold_default": getattr(old_s, "threshold_default", 0.80),
            "threshold_min": getattr(old_s, "threshold_min", 0.50),
            "threshold_max": getattr(old_s, "threshold_max", 0.95),
            "cv_threshold_grid": getattr(old_s, "cv_threshold_grid", [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]),
            "models_path": getattr(old_s, "models_path", "02_models"),
            "cv_dir": getattr(old_s, "cv_dir", "logs/cv"),
        },

        # Trading section
        "trading": {
            "capital_per_trade": getattr(old_s, "capital_per_trade", 1000.0),
            "commission_per_trade": getattr(old_s, "commission_per_trade", 0.35),
            "max_positions": getattr(old_s, "max_positions", 10),
            "max_daily_loss": getattr(old_s, "max_daily_loss", 500.0),
            "max_drawdown": getattr(old_s, "max_drawdown", 0.05),
            "allow_short": getattr(old_s, "allow_short", True),
            "slippage_bps": getattr(old_s, "slippage_bps", 0.0),
            "market_close_buffer_min": getattr(old_s, "market_close_buffer_min", 30),
            "ib_host": getattr(old_s, "ib_host", "127.0.0.1"),
            "ib_port": getattr(old_s, "ib_port", 7497),
            "ib_client_id": getattr(old_s, "ib_client_id", 1),
        },

        # MLflow section
        "mlflow": {
            "tracking_uri": getattr(old_s, "mlflow_tracking_uri", "file:./mlruns"),
            "experiment": getattr(old_s, "mlflow_experiment", "PHIBOT"),
            "auto_log": getattr(old_s, "mlflow_auto_log", True),
            "log_models": getattr(old_s, "mlflow_log_models", True),
        },

        # Logging section
        "logging": {
            "level": getattr(old_s, "log_level", "INFO"),
            "structured": getattr(old_s, "structured_logging", True),
            "correlation_id": getattr(old_s, "correlation_id", True),
            "console_output": getattr(old_s, "console_output", True),
            "file_output": getattr(old_s, "file_output", True),
            "log_dir": getattr(old_s, "logs_path", "logs"),
        },

        # Backtest section
        "backtest": {
            "cooldown_bars": getattr(old_s, "bt_cooldown_bars", 0),
            "bt_allow_short": getattr(old_s, "bt_allow_short", True),
        },

        # System section
        "system": {
            "data_path": getattr(old_s, "data_path", "[DAT]_data"),
            "config_path": getattr(old_s, "config_path", "04_config"),
            "logs_path": getattr(old_s, "logs_path", "logs"),
            "parquet_enabled": getattr(old_s, "parquet_enabled", True),
            "downloader_on_start": getattr(old_s, "downloader_on_start", False),
            "training_on_start": getattr(old_s, "training_on_start", False),
            "cv_update_on_start": getattr(old_s, "cv_update_on_start", False),
            "seed": getattr(old_s, "seed", 42),
            "calendar": getattr(old_s, "calendar", "XNYS"),
        }
    }

    # Aplicar mappings
    for section, attrs in mappings.items():
        for attr, default_value in attrs.items():
            config[section][attr] = default_value

    return config


def backup_old_settings(settings_path: str = "settings.py"):
    """Crea backup del settings.py original"""
    settings_file = Path(settings_path)
    if settings_file.exists():
        backup_path = settings_file.with_suffix(".py.backup")
        # Leer con UTF-8 y escribir con UTF-8
        with open(settings_file, 'r', encoding='utf-8') as original:
            content = original.read()
        with open(backup_path, 'w', encoding='utf-8') as backup:
            backup.write(content)
        print(f"Backup creado: {backup_path}")
        return True
    return False


def create_new_settings_py():
    """Crea nuevo settings.py que usa el sistema unificado"""
    new_settings_content = '''# settings.py - Configuracion unificada para PhiBot
"""
Sistema de configuracion modernizado que usa YAML centralizado.
Este archivo mantiene compatibilidad con el codigo existente.
"""

from config.config_manager import S, get_config, get_config_manager

# Mantiene compatibilidad total con código existente
# Ejemplo de uso:
# from settings import S
# print(S.tp_multiplier)  # Funciona igual que antes

# Funciones adicionales para acceso avanzado
def reload_config():
    """Recarga configuración desde archivo YAML"""
    return get_config_manager().load_config(force_reload=True)

def get_current_config():
    """Obtiene configuración actual completa"""
    return get_config()

def validate_current_config():
    """Valida configuración actual"""
    config_mgr = get_config_manager()
    config_ = config_mgr.get_config()
    errors = config_mgr.validate_config(config_)

    if errors:
        print("⚠️ Errores de validación encontrados:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ Configuración válida")
        return True

if __name__ == "__main__":
    print("🧪 Test de configuración:")
    print(f"Data path: {S.data_path}")
    print(f"TP multiplier: {S.tp_multiplier}")
    print(f"Capital per trade: {S.capital_per_trade}")
    print(f"MLflow experiment: {S.mlflow_experiment}")

    validate_current_config()
'''

    with open("settings.py", "w", encoding='utf-8') as f:
        f.write(new_settings_content)

    print("✅ Nuevo settings.py creado")


def main():
    """Script principal de migracion"""
    print("Iniciando migracion a sistema de configuracion unificado")
    print("=" * 60)

    # 1. Crear directorios necesarios
    Path("config").mkdir(exist_ok=True)

    # 2. Importar configuración actual
    print("Importando configuracion actual...")
    old_s = import_old_settings()

    if old_s is None:
        print("No se encontro settings.py o no se pudo importar")
        print("Creando configuracion por defecto...")
        config_data = {}
    else:
        print("Configuracion actual importada")
        config_data = extract_config_from_old_settings(old_s)

    # 3. Crear archivo YAML
    yaml_path = "config/phibot.yaml"
    print(f"Creando archivo de configuracion: {yaml_path}")

    with open(yaml_path, "w", encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)

    print(f"Configuracion YAML creada: {yaml_path}")

    # 4. Backup del settings.py original
    if Path("settings.py").exists():
        print("Creando backup del settings.py original...")
        backup_old_settings()

    # 5. Crear nuevo settings.py
    print("Creando nuevo settings.py...")
    create_new_settings_py()

    # 6. Test del nuevo sistema
    print("\nProbando nuevo sistema...")
    try:
        # Import the new config_ system
        sys.path.insert(0, ".")
        from config.config_manager import get_config, S as new_S

        config = get_config()
        print(f"Configuracion cargada correctamente")
        print(f"   - Data path: {new_S.data_path}")
        print(f"   - TP multiplier: {new_S.tp_multiplier}")
        print(f"   - Capital per trade: {new_S.capital_per_trade}")

    except Exception as e:
        print(f"Error probando nuevo sistema: {e}")
        return False

    print("\nMigracion completada exitosamente!")
    print("\nProximos pasos:")
    print("1. Revisa el archivo config_/phibot.yaml y ajusta segun necesites")
    print("2. Ejecuta: python -c 'from settings import validate_current_config; validate_current_config()'")
    print("3. Prueba tus scripts existentes para verificar compatibilidad")
    print("4. Si todo funciona bien, puedes eliminar settings.py.backup")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)