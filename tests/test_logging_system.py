#!/usr/bin/env python3
"""
Test suite para el sistema de logging estructurado
"""

import json
import tempfile
import logging
from pathlib import Path
from io import StringIO
import sys
import pytest

# Añadir el directorio raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_enhanced import (
    get_logger,
    setup_logging,
    StructuredFormatter,
    with_correlation_id,
    with_session_id,
    with_component,
    log_function_call,
    shutdown_logging
)


class TestStructuredLogging:

    def setup_method(self):
        """Setup para cada test"""
        # Limpiar handlers existentes
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

    def teardown_method(self):
        """Cleanup después de cada test"""
        shutdown_logging()

    def test_structured_formatter(self):
        """Test que el formatter genera JSON válido"""
        formatter = StructuredFormatter()

        # Crear un log record
        logger = logging.getLogger("test")
        record = logger.makeRecord(
            name="test_logger",
            level=logging.INFO,
            fn="test.py",
            lno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )

        # Añadir campos extra
        record.ticker = "AAPL"
        record.price = 150.0

        # Formatear
        formatted = formatter.format(record)

        # Debe ser JSON válido
        log_data = json.loads(formatted)

        # Verificar campos obligatorios
        assert "timestamp" in log_data
        assert "level" in log_data
        assert "message" in log_data
        assert log_data["level"] == "INFO"
        assert log_data["message"] == "Test message"

        # Verificar campos extra
        assert log_data["ticker"] == "AAPL"
        assert log_data["price"] == 150.0

    def test_structured_logger_basic(self):
        """Test funcionalidad básica del StructuredLogger"""
        # Capturar output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        logger_impl = logging.getLogger("test")
        logger_impl.addHandler(handler)
        logger_impl.setLevel(logging.DEBUG)

        # Crear structured logger
        logger = get_logger("test")

        # Test diferentes niveles
        logger.debug("debug_event", message="Debug message", extra_field="debug_value")
        logger.info("info_event", message="Info message", ticker="AAPL")
        logger.warning("warning_event", message="Warning message")
        logger.error("error_event", message="Error message")

        # Obtener output
        output = stream.getvalue()
        lines = [line for line in output.strip().split('\n') if line]

        assert len(lines) == 4

        # Verificar que cada línea es JSON válido
        for line in lines:
            log_data = json.loads(line)
            assert "event" in log_data
            assert "timestamp" in log_data
            assert "message" in log_data

    def test_correlation_id_context(self):
        """Test context manager para correlation ID"""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        logger_impl = logging.getLogger("test_corr")
        logger_impl.addHandler(handler)
        logger_impl.setLevel(logging.INFO)

        logger = get_logger("test_corr")

        # Test sin correlation ID
        logger.info("no_correlation", message="Without correlation")

        # Test con correlation ID
        with with_correlation_id("test-correlation-123") as corr_id:
            logger.info("with_correlation", message="With correlation")

            # Nested logging debe mantener el mismo ID
            logger.info("nested_log", message="Nested message")

        # Después del context, no debe tener correlation ID
        logger.info("after_correlation", message="After correlation")

        # Verificar output
        output = stream.getvalue()
        lines = [json.loads(line) for line in output.strip().split('\n') if line]

        assert len(lines) == 4

        # Primera línea no debe tener correlation_id
        assert "correlation_id" not in lines[0]

        # Segunda y tercera línea deben tener el mismo correlation_id
        assert lines[1]["correlation_id"] == "test-correlation-123"
        assert lines[2]["correlation_id"] == "test-correlation-123"

        # Cuarta línea no debe tener correlation_id
        assert "correlation_id" not in lines[3]

    def test_bound_logger(self):
        """Test logger con contexto bindeado"""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        logger_impl = logging.getLogger("test_bound")
        logger_impl.addHandler(handler)
        logger_impl.setLevel(logging.INFO)

        logger = get_logger("test_bound")

        # Logger con contexto bindeado
        ticker_logger = logger.bind(ticker="AAPL", timeframe="5mins")

        ticker_logger.info("bound_event", message="Bound message", price=150.0)

        # Verificar que el contexto se incluye
        output = stream.getvalue()
        log_data = json.loads(output.strip())

        assert log_data["ticker"] == "AAPL"
        assert log_data["timeframe"] == "5mins"
        assert log_data["price"] == 150.0
        assert log_data["event"] == "bound_event"

    def test_multiple_context_managers(self):
        """Test múltiples context managers anidados"""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        logger_impl = logging.getLogger("test_multi")
        logger_impl.addHandler(handler)
        logger_impl.setLevel(logging.INFO)

        logger = get_logger("test_multi")

        with with_correlation_id("corr-123"):
            with with_session_id("session-456"):
                with with_component("TEST_COMPONENT"):
                    logger.info("multi_context", message="Multiple contexts")

        output = stream.getvalue()
        log_data = json.loads(output.strip())

        assert log_data["correlation_id"] == "corr-123"
        assert log_data["session_id"] == "session-456"
        assert log_data["component"] == "TEST_COMPONENT"

    def test_function_call_decorator(self):
        """Test decorator de logging automático"""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        logger_impl = logging.getLogger("test_decorator")
        logger_impl.addHandler(handler)
        logger_impl.setLevel(logging.DEBUG)

        @log_function_call(log_args=True, log_result=True)
        def test_function(x: int, y: str = "default") -> str:
            return f"Result: {x} - {y}"

        # Ejecutar función
        result = test_function(42, "test")

        assert result == "Result: 42 - test"

        # Verificar logs generados
        output = stream.getvalue()
        lines = [json.loads(line) for line in output.strip().split('\n') if line]

        # Debe haber al menos 2 logs: start y success
        assert len(lines) >= 2

        start_log = lines[0]
        success_log = lines[-1]

        assert start_log["event"] == "function_call_start"
        assert start_log["function"] == "test_function"
        assert "arguments" in start_log

        assert success_log["event"] == "function_call_success"
        assert success_log["status"] == "success"
        assert "duration_ms" in success_log

    def test_function_call_decorator_with_error(self):
        """Test decorator con función que falla"""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        logger_impl = logging.getLogger("test_decorator_error")
        logger_impl.addHandler(handler)
        logger_impl.setLevel(logging.DEBUG)

        @log_function_call(log_args=True)
        def failing_function(x: int) -> int:
            if x == 0:
                raise ValueError("Cannot be zero")
            return x * 2

        # Ejecutar función que falla
        with pytest.raises(ValueError):
            failing_function(0)

        # Verificar logs
        output = stream.getvalue()
        lines = [json.loads(line) for line in output.strip().split('\n') if line]

        assert len(lines) >= 2

        error_log = lines[-1]
        assert error_log["event"] == "function_call_error"
        assert error_log["status"] == "error"
        assert error_log["error_type"] == "ValueError"
        assert "Cannot be zero" in error_log["error_message"]

    def test_exception_logging(self):
        """Test logging de excepciones"""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        logger_impl = logging.getLogger("test_exception")
        logger_impl.addHandler(handler)
        logger_impl.setLevel(logging.ERROR)

        logger = get_logger("test_exception")

        try:
            raise ValueError("Test exception")
        except Exception:
            logger.error("test_error", message="Error occurred", exc_info=True)

        output = stream.getvalue()
        log_data = json.loads(output.strip())

        assert log_data["event"] == "test_error"
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test exception"
        assert isinstance(log_data["exception"]["traceback"], list)

    def test_non_serializable_objects(self):
        """Test logging de objetos no serializables"""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        logger_impl = logging.getLogger("test_nonserial")
        logger_impl.addHandler(handler)
        logger_impl.setLevel(logging.INFO)

        logger = get_logger("test_nonserial")

        # Objeto no serializable
        class NonSerializable:
            def __str__(self):
                return "NonSerializableObject"

        obj = NonSerializable()

        logger.info("non_serializable_test",
                    normal_field="normal",
                    non_serializable=obj,
                    message="Test with non-serializable object")

        output = stream.getvalue()
        log_data = json.loads(output.strip())

        assert log_data["normal_field"] == "normal"
        assert log_data["non_serializable"] == "NonSerializableObject"

    def test_file_logging(self):
        """Test logging a archivo"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            # Configurar handler de archivo
            file_handler = logging.FileHandler(
                log_dir / "test.log",
                encoding='utf-8'
            )
            file_handler.setFormatter(StructuredFormatter())

            logger_impl = logging.getLogger("test_file")
            logger_impl.addHandler(file_handler)
            logger_impl.setLevel(logging.INFO)

            logger = get_logger("test_file")

            # Generar logs
            logger.info("file_test_1", message="First message")
            logger.info("file_test_2", message="Second message", ticker="AAPL")

            # Cerrar handler para asegurar que se escriba
            file_handler.close()

            # Verificar archivo
            log_file = log_dir / "test.log"
            assert log_file.exists()

            content = log_file.read_text(encoding='utf-8')
            lines = [json.loads(line) for line in content.strip().split('\n') if line]

            assert len(lines) == 2
            assert lines[0]["event"] == "file_test_1"
            assert lines[1]["event"] == "file_test_2"
            assert lines[1]["ticker"] == "AAPL"


def test_integration_example():
    """Test de integración que simula uso real"""

    # Setup logging capturado
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # Simular procesamiento de ticker
    def process_ticker(ticker: str):
        with with_correlation_id() as corr_id:
            with with_component("PROCESSING"):
                logger = get_logger(__name__).bind(ticker=ticker)

                logger.info("processing_start", message="Iniciando procesamiento")

                # Simular carga de datos
                logger.info("data_loading", rows=1000, timeframe="5mins")

                # Simular predicción
                prediction = 0.75
                logger.info("prediction_generated",
                            prediction=prediction,
                            model_type="XGBoost")

                # Simular trade
                if prediction > 0.7:
                    logger.info("trade_signal", signal="BUY", confidence=prediction)

                logger.info("processing_complete", message="Procesamiento completado")

                return {"signal": "BUY", "confidence": prediction}

    # Ejecutar
    result = process_ticker("AAPL")

    # Verificar resultado
    assert result["signal"] == "BUY"
    assert result["confidence"] == 0.75

    # Verificar logs
    output = stream.getvalue()
    lines = [json.loads(line) for line in output.strip().split('\n') if line]

    assert len(lines) >= 5

    # Todos los logs deben tener el mismo correlation_id
    correlation_ids = [line.get("correlation_id") for line in lines]
    assert all(cid == correlation_ids[0] for cid in correlation_ids)

    # Todos deben tener component
    assert all(line.get("component") == "PROCESSING" for line in lines)

    # Todos deben tener ticker
    assert all(line.get("ticker") == "AAPL" for line in lines)


if __name__ == "__main__":
    # Ejecutar tests manualmente
    print("Ejecutando tests del sistema de logging...")

    test_obj = TestStructuredLogging()

    try:
        # Tests básicos
        test_obj.setup_method()
        test_obj.test_structured_formatter()
        test_obj.teardown_method()
        print("✓ Test structured formatter")

        test_obj.setup_method()
        test_obj.test_structured_logger_basic()
        test_obj.teardown_method()
        print("✓ Test structured logger básico")

        test_obj.setup_method()
        test_obj.test_correlation_id_context()
        test_obj.teardown_method()
        print("✓ Test correlation ID")

        test_obj.setup_method()
        test_obj.test_bound_logger()
        test_obj.teardown_method()
        print("✓ Test bound logger")

        # Test integración
        test_integration_example()
        print("✓ Test integración")

        print("\nTodos los tests básicos pasaron!")
        print("Ejecuta: python -m pytest tests/test_logging_system.py -v para tests completos")

    except Exception as e:
        print(f"✗ Test falló: {e}")
        import traceback

        traceback.print_exc()