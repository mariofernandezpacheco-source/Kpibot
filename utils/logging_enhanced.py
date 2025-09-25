#!/usr/bin/env python3
"""
Sistema de logging estructurado para π-Bot
Proporciona logging consistente, trazabilidad y correlation IDs
"""

import json
import logging
import logging.handlers
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import traceback
import functools
import inspect

from config.config_manager import get_config

# Context variables para correlation tracking
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
session_id: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
component: ContextVar[Optional[str]] = ContextVar('component', default=None)


class StructuredFormatter(logging.Formatter):
    """
    Formatter que genera logs en formato JSON estructurado
    """

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Formatea el log record como JSON estructurado"""

        # Datos base del log
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Añadir context variables si existen
        if correlation_id.get():
            log_data["correlation_id"] = correlation_id.get()
        if session_id.get():
            log_data["session_id"] = session_id.get()
        if component.get():
            log_data["component"] = component.get()

        # Añadir extra fields del record
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'message', 'getMessage'
                }:
                    # Serializar objetos complejos
                    try:
                        if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                            log_data[key] = value
                        else:
                            log_data[key] = str(value)
                    except Exception:
                        log_data[key] = f"<non-serializable: {type(value).__name__}>"

        # Añadir traceback si hay excepción
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }

        return json.dumps(log_data, ensure_ascii=False)


class StructuredLogger:
    """
    Logger wrapper que facilita el logging estructurado
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _log_with_context(self, level: int, event: str, message: str = "", **kwargs):
        """Log con contexto estructurado"""
        extra = {
            "event": event,
            **kwargs
        }

        # Si no hay mensaje, usar el event como mensaje
        if not message:
            message = event.replace("_", " ").title()

        self._logger.log(level, message, extra=extra)

    def debug(self, event: str, message: str = "", **kwargs):
        """Log debug con contexto"""
        self._log_with_context(logging.DEBUG, event, message, **kwargs)

    def info(self, event: str, message: str = "", **kwargs):
        """Log info con contexto"""
        self._log_with_context(logging.INFO, event, message, **kwargs)

    def warning(self, event: str, message: str = "", **kwargs):
        """Log warning con contexto"""
        self._log_with_context(logging.WARNING, event, message, **kwargs)

    def error(self, event: str, message: str = "", exc_info: bool = False, **kwargs):
        """Log error con contexto y opcionalmente traceback"""
        self._log_with_context(logging.ERROR, event, message, **kwargs)
        if exc_info:
            self._logger.error("", exc_info=True, extra={"event": f"{event}_traceback"})

    def critical(self, event: str, message: str = "", exc_info: bool = False, **kwargs):
        """Log critical con contexto"""
        self._log_with_context(logging.CRITICAL, event, message, **kwargs)
        if exc_info:
            self._logger.critical("", exc_info=True, extra={"event": f"{event}_traceback"})

    def bind(self, **kwargs) -> 'StructuredLogger':
        """Crea un logger con contexto adicional bindeado"""
        return BoundStructuredLogger(self._logger, **kwargs)


class BoundStructuredLogger(StructuredLogger):
    """Logger con contexto adicional pre-bindeado"""

    def __init__(self, logger: logging.Logger, **bound_context):
        super().__init__(logger)
        self.bound_context = bound_context

    def _log_with_context(self, level: int, event: str, message: str = "", **kwargs):
        """Log con contexto estructurado + contexto bindeado"""
        combined_kwargs = {**self.bound_context, **kwargs}
        super()._log_with_context(level, event, message, **combined_kwargs)


def setup_logging() -> None:
    """
    Configura el sistema de logging estructurado del proyecto
    """
    config = get_config()

    # Crear directorio de logs
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configuración del root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.logging.level.upper()))

    # Limpiar handlers existentes
    root_logger.handlers.clear()

    # Handler para consola
    if config.logging.console_output:
        console_handler = logging.StreamHandler(sys.stdout)

        if config.logging.structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    fmt=config.logging.log_format,
                    datefmt=config.logging.date_format
                )
            )

        root_logger.addHandler(console_handler)

    # Handler para archivo
    if config.logging.file_output:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "phibot.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

        # Handler separado para errores
        error_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "phibot_errors.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(error_handler)


def get_logger(name: str, **context) -> StructuredLogger:
    """
    Obtiene un logger estructurado con contexto opcional

    Args:
        name: Nombre del logger (normalmente __name__ o identificador del módulo)
        **context: Contexto adicional a bindear (ticker, timeframe, etc.)

    Returns:
        StructuredLogger configurado
    """
    logger = logging.getLogger(name)
    structured = StructuredLogger(logger)

    if context:
        return structured.bind(**context)

    return structured


def with_correlation_id(correlation_id_value: Optional[str] = None):
    """
    Context manager para establecer correlation ID
    """

    class CorrelationContextManager:
        def __init__(self, corr_id: Optional[str]):
            self.corr_id = corr_id or str(uuid.uuid4())[:8]
            self.token = None

        def __enter__(self):
            self.token = correlation_id.set(self.corr_id)
            return self.corr_id

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.token:
                correlation_id.reset(self.token)

    return CorrelationContextManager(correlation_id_value)


def with_session_id(session_id_value: str):
    """Context manager para establecer session ID"""

    class SessionContextManager:
        def __init__(self, sess_id: str):
            self.sess_id = sess_id
            self.token = None

        def __enter__(self):
            self.token = session_id.set(self.sess_id)
            return self.sess_id

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.token:
                session_id.reset(self.token)

    return SessionContextManager(session_id_value)


def with_component(component_name: str):
    """Context manager para establecer componente"""

    class ComponentContextManager:
        def __init__(self, comp_name: str):
            self.comp_name = comp_name
            self.token = None

        def __enter__(self):
            self.token = component.set(self.comp_name)
            return self.comp_name

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.token:
                component.reset(self.token)

    return ComponentContextManager(component_name)


def log_function_call(logger: Optional[StructuredLogger] = None,
                      log_args: bool = False,
                      log_result: bool = False):
    """
    Decorator para loggear calls a funciones automáticamente

    Args:
        logger: Logger a usar (si None, crea uno basado en el módulo de la función)
        log_args: Si loggear los argumentos de la función
        log_result: Si loggear el resultado de la función
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Obtener o crear logger
            func_logger = logger or get_logger(func.__module__)

            # Información de la función
            func_name = func.__name__
            module_name = func.__module__

            # Datos base del log
            log_data = {
                "function": func_name,
                "module": module_name,
            }

            # Añadir argumentos si se solicita
            if log_args:
                try:
                    # Obtener nombres de parámetros
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    # Serializar argumentos de forma segura
                    serialized_args = {}
                    for param_name, param_value in bound_args.arguments.items():
                        try:
                            if isinstance(param_value, (str, int, float, bool)) or param_value is None:
                                serialized_args[param_name] = param_value
                            elif isinstance(param_value, (list, dict)):
                                # Limitar tamaño para evitar logs enormes
                                serialized_args[param_name] = str(param_value)[:200]
                            else:
                                serialized_args[param_name] = str(type(param_value).__name__)
                        except Exception:
                            serialized_args[param_name] = "<non-serializable>"

                    log_data["arguments"] = serialized_args

                except Exception as e:
                    log_data["arguments_error"] = str(e)

            # Log inicio de función
            func_logger.debug("function_call_start", **log_data)

            try:
                # Ejecutar función
                start_time = datetime.utcnow()
                result = func(*args, **kwargs)
                end_time = datetime.utcnow()

                # Log éxito
                success_data = {
                    **log_data,
                    "duration_ms": int((end_time - start_time).total_seconds() * 1000),
                    "status": "success"
                }

                if log_result and result is not None:
                    try:
                        if isinstance(result, (str, int, float, bool)):
                            success_data["result"] = result
                        elif isinstance(result, (list, dict)):
                            success_data["result"] = str(result)[:200]
                        else:
                            success_data["result_type"] = type(result).__name__
                    except Exception:
                        success_data["result"] = "<non-serializable>"

                func_logger.debug("function_call_success", **success_data)
                return result

            except Exception as e:
                # Log error
                end_time = datetime.utcnow()
                error_data = {
                    **log_data,
                    "duration_ms": int((end_time - start_time).total_seconds() * 1000),
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }

                func_logger.error("function_call_error", exc_info=True, **error_data)
                raise

        return wrapper

    return decorator


def shutdown_logging():
    """Cierra todos los handlers de logging limpiamente"""
    logging.shutdown()


# Configuración automática del logging cuando se importa el módulo
try:
    setup_logging()
except Exception as e:
    # Fallback a logging básico si falla la configuración
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.error(f"Error configurando logging estructurado, usando fallback: {e}")

if __name__ == "__main__":
    # Test del sistema de logging
    print("Testing sistema de logging estructurado...")

    # Test logger básico
    logger = get_logger("test_module")
    logger.info("test_event", message="Test message", extra_field="extra_value")

    # Test con correlation ID
    with with_correlation_id("test-correlation-123"):
        logger.info("test_with_correlation", ticker="AAPL", action="buy")

        # Test logger bindeado
        ticker_logger = logger.bind(ticker="AAPL", timeframe="5mins")
        ticker_logger.info("trade_executed", price=150.0, quantity=100)


    # Test función decorada
    @log_function_call(log_args=True, log_result=True)
    def test_function(x: int, y: str = "default") -> str:
        return f"Result: {x} - {y}"


    result = test_function(42, "test")
    print(f"Function result: {result}")

    # Test error
    try:
        @log_function_call(log_args=True)
        def failing_function(x: int) -> int:
            if x == 0:
                raise ValueError("Cannot be zero")
            return x * 2


        failing_function(0)
    except ValueError:
        logger.info("test_error_handled", message="Error was properly logged")

    print("Test completado - revisa los logs generados")