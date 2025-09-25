# config_/config_manager.py
"""
Sistema de configuración unificado para π-Bot
Centraliza toda la configuración en archivos YAML con validación y hot-reload
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import logging
from dataclasses import dataclass, field
from datetime import datetime


# Configuración usando dataclasses para validación automática
@dataclass
class DataConfig:
    storage_backend: str = "parquet"  # "csv" o "parquet"
    base_path: str = "DAT_data"
    parquet_base_path: Optional[str] = None
    csv_base_path: Optional[str] = None

    # Quality checks
    min_coverage_pct: float = 80.0
    max_missing_days: int = 5
    max_gap_hours: int = 24

    # Timeframes disponibles
    timeframes_available: list[str] = field(default_factory=lambda: ["1min", "5mins", "15mins", "1hour", "1day"])
    timeframe_default: str = "5mins"

    # Bar size mapping para IBKR
    bar_size_by_tf: Dict[str, str] = field(default_factory=lambda: {
        "1min": "1 min",
        "5mins": "5 mins",
        "15mins": "15 mins",
        "1hour": "1 hour",
        "1day": "1 day"
    })

    def __post_init__(self):
        # Auto-configurar paths si no están definidos
        base = Path(self.base_path)
        if self.parquet_base_path is None:
            self.parquet_base_path = str(base / "parquet").replace("\\", "/")
        if self.csv_base_path is None:
            self.csv_base_path = str(base / "csv").replace("\\", "/")


@dataclass
class ModelConfig:
    # Parámetros por defecto del modelo
    tp_multiplier: float = 3.0
    sl_multiplier: float = 2.0
    time_limit_candles: int = 16
    label_window: int = 5

    # Modelo por defecto
    model_type: str = "xgb"
    feature_set: str = "core"

    # Optimización
    cv_splits: int = 5
    cv_test_size: int = 500
    cv_scheme: str = "expanding"
    cv_embargo: int = -1
    cv_purge: int = -1

    # Thresholds
    threshold_default: float = 0.80
    threshold_min: float = 0.50
    threshold_max: float = 0.95
    cv_threshold_grid: list[float] = field(default_factory=lambda: [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])

    # Paths de modelos
    models_path: str = "02_models"
    cv_dir: str = "logs/cv"


@dataclass
class TradingConfig:
    # Capital y comisiones
    capital_per_trade: float = 1000.0
    commission_per_trade: float = 0.35
    max_positions: int = 10

    # Risk management
    max_daily_loss: float = 500.0
    max_drawdown: float = 0.05
    allow_short: bool = True

    # Slippage y timing
    slippage_bps: float = 0.0
    market_close_buffer_min: int = 30

    # IBKR Connection
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1


@dataclass
class MLflowConfig:
    tracking_uri: str = "file:./mlruns"
    experiment: str = "PHIBOT"
    auto_log: bool = True
    log_models: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    structured: bool = True
    correlation_id: bool = True
    console_output: bool = True
    file_output: bool = True
    log_dir: str = "LOG_logs"

    # Formatos
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class BacktestConfig:
    cooldown_bars: int = 0
    bt_allow_short: bool = True


@dataclass
class SystemConfig:
    # Paths principales
    data_path: str = "DAT_data"
    config_path: str = "config"
    logs_path: str = "logs"

    # Flags de sistema
    parquet_enabled: bool = True
    downloader_on_start: bool = False
    training_on_start: bool = False
    cv_update_on_start: bool = False

    # Seeds y otros
    seed: int = 42
    calendar: str = "XNYS"  # NYSE calendar


@dataclass
class UnifiedConfig:
    """Configuración unificada del sistema π-Bot"""
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    _config_file: Optional[str] = None
    _last_modified: Optional[datetime] = None


class ConfigManager:
    """Manager para carga, validación y hot-reload de configuración"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._find_config_file()
        self._config: Optional[UnifiedConfig] = None
        self._watchers = []

    def _find_config_file(self) -> str:
        """Busca archivo de configuración en ubicaciones estándar"""
        candidates = [
            "config_/phibot.yaml",
            "config_/settings.yaml",
            "phibot.yaml",
            "settings.yaml",
            "04_config/phibot.yaml"
        ]

        for candidate in candidates:
            if Path(candidate).exists():
                return candidate

        # Si no existe, crear uno por defecto
        default_path = "config/phibot.yaml"
        self._create_default_config(default_path)
        return default_path

    def _create_default_config(self, config_path: str):
        """Crea archivo de configuración por defecto"""
        config_path_obj = Path(config_path)
        config_path_obj.parent.mkdir(parents=True, exist_ok=True)

        default_config = UnifiedConfig()
        config_dict = self._dataclass_to_dict(default_config)

        with open(config_path_obj, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        print(f"Creado archivo de configuracion por defecto: {config_path}")
        return config_path_obj

    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convierte dataclass a dict para YAML"""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name, field_def in obj.__dataclass_fields__.items():
                value = getattr(obj, field_name)
                if hasattr(value, '__dataclass_fields__'):
                    result[field_name] = self._dataclass_to_dict(value)
                else:
                    result[field_name] = value
            return result
        return obj

    def load_config(self, force_reload: bool = False) -> UnifiedConfig:
        """Carga configuración desde archivo YAML"""
        config_path = Path(self.config_file)

        if not force_reload and self._config is not None:
            # Check si el archivo cambió
            if config_path.exists():
                mtime = datetime.fromtimestamp(config_path.stat().st_mtime)
                if self._last_modified and mtime <= self._last_modified:
                    return self._config

        if not config_path.exists():
            print(f"Archivo de configuracion no encontrado: {config_path}")
            print("Creando configuracion por defecto...")
            self._create_default_config(str(config_path))
            # Ahora el archivo debería existir

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)

            # Crear configuración desde YAML
            config = UnifiedConfig()

            if yaml_data:
                # Actualizar cada sección
                for section_name, section_data in yaml_data.items():
                    if hasattr(config, section_name) and isinstance(section_data, dict):
                        section_obj = getattr(config, section_name)
                        for key, value in section_data.items():
                            if hasattr(section_obj, key):
                                setattr(section_obj, key, value)

            config._config_file = str(config_path)
            config._last_modified = datetime.fromtimestamp(config_path.stat().st_mtime)

            self._config = config
            self._last_modified = config._last_modified

            print(f"Configuracion cargada desde: {config_path}")
            return config

        except Exception as e:
            print(f"Error cargando configuracion: {e}")
            print("Usando configuracion por defecto")
            self._config = UnifiedConfig()
            return self._config

    def save_config(self, config: UnifiedConfig):
        """Guarda configuración a archivo YAML"""
        try:
            config_dict = self._dataclass_to_dict(config)
            # Remove private fields
            config_dict.pop('_config_file', None)
            config_dict.pop('_last_modified', None)

            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            print(f"✅ Configuración guardada en: {self.config_file}")

        except Exception as e:
            print(f"❌ Error guardando configuración: {e}")

    def get_config(self) -> UnifiedConfig:
        """Obtiene configuración actual (con auto-reload si cambió)"""
        return self.load_config()

    def validate_config(self, config: UnifiedConfig) -> list[str]:
        """Valida configuración y retorna lista de errores"""
        errors = []

        # Validar paths existen
        for path in [config.system.data_path, config.system.config_path, config.system.logs_path]:
            if not Path(path).exists():
                errors.append(f"Path no existe: {path}")

        # Validar rangos de valores
        if not 0.0 <= config.models.threshold_default <= 1.0:
            errors.append(f"threshold_default debe estar entre 0 y 1, recibido: {config.models.threshold_default}")

        if config.trading.capital_per_trade <= 0:
            errors.append(f"capital_per_trade debe ser positivo, recibido: {config.trading.capital_per_trade}")

        if config.models.cv_splits < 2:
            errors.append(f"cv_splits debe ser >= 2, recibido: {config.models.cv_splits}")

        return errors


# Singleton global para acceso fácil
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Obtiene instancia singleton del ConfigManager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> UnifiedConfig:
    """Obtiene configuración actual (función de conveniencia)"""
    return get_config_manager().get_config()


# Backward compatibility - crea objeto S compatible
class CompatibilitySettings:
    """Wrapper para mantener compatibilidad con el settings.S existente"""

    def __init__(self):
        self._config = get_config()

    def __getattr__(self, name: str) -> Any:
        config = get_config()  # Always get fresh config_

        # Mapeo de atributos del settings.py original
        attr_mapping = {
            # Data paths - convertir a string para compatibilidad
            'data_path': str(config.system.data_path),
            'config_path': str(config.system.config_path),
            'logs_path': str(config.system.logs_path),
            'models_path': str(config.models.models_path),
            'cv_dir': str(config.models.cv_dir),
            'parquet_base_path': str(config.data.parquet_base_path),

            # Trading
            'capital_per_trade': config.trading.capital_per_trade,
            'commission_per_trade': config.trading.commission_per_trade,
            'allow_short': config.trading.allow_short,
            'ib_host': config.trading.ib_host,
            'ib_port': config.trading.ib_port,

            # Models
            'tp_multiplier': config.models.tp_multiplier,
            'sl_multiplier': config.models.sl_multiplier,
            'time_limit_candles': config.models.time_limit_candles,
            'model_type': config.models.model_type,
            'feature_set': config.models.feature_set,
            'threshold_default': config.models.threshold_default,
            'threshold_min': config.models.threshold_min,
            'threshold_max': config.models.threshold_max,

            # CV
            'n_splits_cv': config.models.cv_splits,
            'cv_test_size': config.models.cv_test_size,
            'cv_scheme': config.models.cv_scheme,
            'cv_threshold_grid': config.models.cv_threshold_grid,

            # MLflow
            'mlflow_tracking_uri': config.mlflow.tracking_uri,
            'mlflow_experiment': config.mlflow.experiment,

            # Logging - NUEVO
            'log_level': config.logging.level,
            'logging_level': config.logging.level,
            'structured_logging': config.logging.structured,
            'log_dir': str(config.logging.log_dir),

            # System
            'seed': config.system.seed,
            'calendar': config.system.calendar,
            'timeframe_default': config.data.timeframe_default,
            'bar_size_by_tf': config.data.bar_size_by_tf,

            # Flags
            'parquet_enabled': config.system.parquet_enabled,
            'downloader_on_start': config.system.downloader_on_start,
            'training_on_start': config.system.training_on_start,
        }

        if name in attr_mapping:
            return attr_mapping[name]

        # Si no está en el mapeo, buscar en las secciones de config_
        for section_name in ['data', 'models', 'trading', 'mlflow', 'logging', 'backtest', 'system']:
            section = getattr(config, section_name)
            if hasattr(section, name):
                value = getattr(section, name)
                # Convertir Path objects a strings para compatibilidad
                if hasattr(value, '__fspath__'):  # Es un Path object
                    return str(value)
                return value

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# Para uso inmediato - reemplaza el settings.S original
S = CompatibilitySettings()

if __name__ == "__main__":
    # Test del sistema
    config_mgr = get_config_manager()
    config = config_mgr.get_config()

    print("🧪 Testing sistema de configuración:")
    print(f"Data path: {config.system.data_path}")
    print(f"TP multiplier: {config.models.tp_multiplier}")
    print(f"Capital per trade: {config.trading.capital_per_trade}")
    print(f"MLflow experiment: {config.mlflow.experiment}")

    # Test compatibilidad
    print(f"\n🔄 Test compatibilidad S.tp_multiplier: {S.tp_multiplier}")
    print(f"🔄 Test compatibilidad S.data_path: {S.data_path}")

    # Test validación
    errors = config_mgr.validate_config(config)
    if errors:
        print(f"\n⚠️ Errores de validación: {errors}")
    else:
        print("\n✅ Configuración válida")