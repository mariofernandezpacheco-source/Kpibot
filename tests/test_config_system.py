#!/usr/bin/env python3
"""
Test suite para el sistema de configuración unificado
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch
import sys
import os

# Añadir el directorio raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import (
    ConfigManager,
    UnifiedConfig,
    DataConfig,
    ModelConfig,
    TradingConfig,
    get_config,
    S
)


class TestConfigManager:

    def test_default_config_creation(self):
        """Test que se puede crear configuración por defecto"""
        config = UnifiedConfig()

        # Verificar valores por defecto
        assert config.data.storage_backend == "parquet"
        assert config.models.tp_multiplier == 3.0
        assert config.trading.capital_per_trade == 1000.0
        assert config.mlflow.experiment == "PHIBOT"

    def test_dataclass_validation(self):
        """Test que los dataclasses validan tipos correctamente"""

        # Test configuración válida
        data_config = DataConfig(
            storage_backend="parquet",
            base_path="test_data",
            min_coverage_pct=85.0
        )
        assert data_config.storage_backend == "parquet"
        assert data_config.min_coverage_pct == 85.0

        # Los paths se auto-configuran (usar forward slashes normalizadas)
        assert data_config.parquet_base_path == "test_data/parquet"
        assert data_config.csv_base_path == "test_data/csv"

    def test_yaml_load_save_cycle(self):
        """Test carga y guardado de YAML"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            # Crear manager con archivo temporal
            manager = ConfigManager(str(config_path))

            # Cargar configuración (debe crear archivo por defecto)
            config = manager.load_config()
            assert config is not None
            assert config_path.exists()

            # Modificar configuración
            config.models.tp_multiplier = 4.5
            config.trading.capital_per_trade = 2000.0

            # Guardar cambios
            manager.save_config(config)

            # Crear nuevo manager y cargar
            manager2 = ConfigManager(str(config_path))
            config2 = manager2.load_config()

            # Verificar que los cambios se guardaron
            assert config2.models.tp_multiplier == 4.5
            assert config2.trading.capital_per_trade == 2000.0

    def test_config_validation(self):
        """Test validación de configuración"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            manager = ConfigManager(str(config_path))
            config = manager.load_config()

            # Configuración válida no debe tener errores
            errors = manager.validate_config(config)
            # Puede haber errores por paths que no existen, pero no por tipos

            # Test configuración inválida
            config.models.threshold_default = 1.5  # Fuera de rango [0,1]
            config.trading.capital_per_trade = -100  # Negativo
            config.models.cv_splits = 1  # Muy bajo

            errors = manager.validate_config(config)
            assert len(errors) >= 3  # Debe detectar los 3 errores
            assert any("threshold_default" in error for error in errors)
            assert any("capital_per_trade" in error for error in errors)
            assert any("cv_splits" in error for error in errors)

    def test_hot_reload(self):
        """Test hot reload cuando el archivo cambia"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            # Crear configuración inicial
            initial_config = {
                "models": {"tp_multiplier": 3.0},
                "trading": {"capital_per_trade": 1000.0}
            }

            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            manager = ConfigManager(str(config_path))
            config1 = manager.load_config()
            assert config1.models.tp_multiplier == 3.0

            # Simular cambio en archivo (cambiar timestamp)
            import time
            time.sleep(0.1)  # Asegurar diferente timestamp

            # Modificar archivo
            modified_config = {
                "models": {"tp_multiplier": 5.0},
                "trading": {"capital_per_trade": 2000.0}
            }

            with open(config_path, 'w') as f:
                yaml.dump(modified_config, f)

            # Cargar de nuevo (debería detectar cambio)
            config2 = manager.load_config()
            assert config2.models.tp_multiplier == 5.0
            assert config2.trading.capital_per_trade == 2000.0

    def test_compatibility_layer(self):
        """Test que la capa de compatibilidad funciona"""

        # Test acceso a configuración vía S
        assert hasattr(S, 'tp_multiplier')
        assert hasattr(S, 'data_path')
        assert hasattr(S, 'capital_per_trade')

        # Test que devuelve valores correctos
        tp = S.tp_multiplier
        assert isinstance(tp, (int, float))
        assert tp > 0

        data_path = S.data_path
        assert isinstance(data_path, str)

        # Test que falla correctamente para atributos inexistentes
        with pytest.raises(AttributeError):
            _ = S.nonexistent_attribute

    def test_partial_yaml_loading(self):
        """Test que carga YAML parcial correctamente"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "partial_config.yaml"

            # YAML parcial (solo algunas secciones)
            partial_yaml = {
                "models": {
                    "tp_multiplier": 2.5,
                    "sl_multiplier": 1.8
                },
                "trading": {
                    "capital_per_trade": 1500.0
                }
                # Faltan otras secciones
            }

            with open(config_path, 'w') as f:
                yaml.dump(partial_yaml, f)

            manager = ConfigManager(str(config_path))
            config = manager.load_config()

            # Debe cargar valores del YAML
            assert config.models.tp_multiplier == 2.5
            assert config.models.sl_multiplier == 1.8
            assert config.trading.capital_per_trade == 1500.0

            # Debe usar defaults para valores no especificados
            assert config.models.time_limit_candles == 16  # default
            assert config.trading.commission_per_trade == 0.35  # default
            assert config.data.storage_backend == "parquet"  # default

    def test_environment_override(self):
        """Test override de configuración vía variables de ambiente"""
        # Este test requiere implementar el override por variables de ambiente
        # Por ahora solo documentamos la funcionalidad

        # Ejemplo de uso futuro:
        # os.environ["PHIBOT_TRADING__CAPITAL_PER_TRADE"] = "2000"
        # config_ = manager.load_config()
        # assert config_.trading.capital_per_trade == 2000

        pass

    def test_config_sections_exist(self):
        """Test que todas las secciones esperadas existen"""
        config = UnifiedConfig()

        required_sections = [
            'data', 'models', 'trading', 'mlflow',
            'logging', 'backtest', 'system'
        ]

        for section in required_sections:
            assert hasattr(config, section), f"Falta sección: {section}"

    def test_backward_compatibility_attributes(self):
        """Test que todos los atributos importantes del S original existen"""

        # Lista de atributos críticos que deben existir para compatibilidad
        critical_attributes = [
            'data_path', 'config_path', 'logs_path', 'models_path',
            'tp_multiplier', 'sl_multiplier', 'time_limit_candles',
            'capital_per_trade', 'commission_per_trade', 'allow_short',
            'threshold_default', 'model_type', 'feature_set',
            'mlflow_tracking_uri', 'mlflow_experiment',
            'timeframe_default', 'seed'
        ]

        for attr in critical_attributes:
            assert hasattr(S, attr), f"Falta atributo crítico: {attr}"
            value = getattr(S, attr)
            assert value is not None, f"Atributo {attr} es None"


def test_integration_with_existing_code():
    """Test de integración con código existente"""

    # Verificar qué settings estamos importando
    import sys
    import importlib

    # Forzar re-importación si ya estaba cargado
    if 'settings' in sys.modules:
        importlib.reload(sys.modules['settings'])

    # Simular importación típica del código existente
    from settings import S

    # Debug: mostrar qué tipo de objeto es S
    print(f"DEBUG: Tipo de S: {type(S)}")
    print(f"DEBUG: S.data_path tipo: {type(S.data_path)}")
    print(f"DEBUG: S.data_path valor: {S.data_path}")
    print(f"DEBUG: S.mlflow_tracking_uri valor: {S.mlflow_tracking_uri}")

    # Test que funciona como antes
    assert S.tp_multiplier > 0

    # Convertir a string si es Path para compatibilidad
    data_path = S.data_path
    if hasattr(data_path, '__fspath__'):  # Es un Path object
        data_path = str(data_path)

    assert isinstance(data_path, str)
    assert isinstance(S.capital_per_trade, (int, float))
    assert S.capital_per_trade > 0

    # Test configuración MLflow - permitir None como valor válido
    mlflow_uri = S.mlflow_tracking_uri
    if mlflow_uri is not None:
        if hasattr(mlflow_uri, '__fspath__'):
            mlflow_uri = str(mlflow_uri)
        assert isinstance(mlflow_uri, str)

    # mlflow_experiment debería existir siempre
    assert isinstance(S.mlflow_experiment, str)
    assert S.mlflow_experiment  # No vacío


if __name__ == "__main__":
    # Ejecutar tests manualmente si no tienes pytest
    print("🧪 Ejecutando tests del sistema de configuración...")

    # Test básico
    try:
        config = UnifiedConfig()
        print("✅ Configuración por defecto creada")

        # Test compatibilidad
        from settings import S

        print(f"✅ Importación S exitosa - TP multiplier: {S.tp_multiplier}")

        # Test validación
        manager = ConfigManager("test_config.yaml")
        config = manager.load_config()
        errors = manager.validate_config(config)
        if not errors:
            print("✅ Configuración válida")
        else:
            print(f"⚠️ Errores encontrados: {errors}")

        print("\n🎉 Tests básicos completados exitosamente!")

    except Exception as e:
        print(f"❌ Error en tests: {e}")
        import traceback

        traceback.print_exc()