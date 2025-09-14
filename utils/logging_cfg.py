# utils/logging_cfg.py
from __future__ import annotations

import json
import logging
import queue
import sys
from datetime import UTC, datetime
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Any

try:
    from settings import S
except Exception:
    # fallback mínimo si settings no está todavía listo
    class _S:  # noqa
        logging_enabled = True
        logging_level = "INFO"
        logging_console_pretty = True
        logging_dir = "03_logs/structured"
        logging_rotate_bytes = 50 * 1024 * 1024
        logging_backup_count = 20
        logging_include_tracebacks = True

    S = _S()

_STD_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
}


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "ts": _utc_iso(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = {k: v for k, v in record.__dict__.items() if k not in _STD_KEYS}
        if extra:
            obj.update(extra)
        if record.exc_info:
            exc_type = str(record.exc_info[0].__name__) if record.exc_info[0] else "Exception"
            exc_msg = str(record.exc_info[1]) if record.exc_info[1] else ""
            obj["exc_type"] = exc_type
            obj["exc_msg"] = exc_msg
            if getattr(S, "logging_include_tracebacks", True):
                obj["stack"] = self.formatException(record.exc_info)
        return json.dumps(obj, ensure_ascii=False)


class PrettyFormatter(logging.Formatter):
    # consola legible para humanos (sin JSON)
    def format(self, record: logging.LogRecord) -> str:
        ts = _utc_iso()
        base = f"{ts} | {record.levelname} | {record.getMessage()}"
        extra = {k: v for k, v in record.__dict__.items() if k not in _STD_KEYS and k != "event"}
        if "event" in record.__dict__:
            base += f" [{record.__dict__['event']}]"
        if extra:
            base += " | " + ", ".join(f"{k}={v}" for k, v in extra.items())
        return base


class BoundLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # recoge extra explícito (si lo pasan)
        extra = kwargs.pop("extra", {}) or {}

        # Mueve cualquier kw no estándar (p.ej. event=, ticker=, universe=) a extra
        for k in list(kwargs.keys()):
            if k not in ("exc_info", "stack_info") and k not in _STD_KEYS:
                extra[k] = kwargs.pop(k)

        # Fusiona el contexto enlazado con el extra puntual
        merged = {**(self.extra or {}), **extra}
        kwargs["extra"] = merged
        return msg, kwargs

    def bind(self, **ctx) -> BoundLogger:
        merged = {**(self.extra or {}), **ctx}
        return BoundLogger(self.logger, merged)


_listener: QueueListener | None = None
_configured = False


def _make_file_handler(path: Path, level: int) -> RotatingFileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = RotatingFileHandler(
        filename=str(path),
        maxBytes=int(getattr(S, "logging_rotate_bytes", 50 * 1024 * 1024)),
        backupCount=int(getattr(S, "logging_backup_count", 20)),
        encoding="utf-8",
    )
    h.setLevel(level)
    h.setFormatter(JsonFormatter())
    return h


def _make_console_handler(level: int) -> logging.Handler:
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(level)
    fmt = PrettyFormatter() if getattr(S, "logging_console_pretty", True) else JsonFormatter()
    h.setFormatter(fmt)
    return h


def setup_logging(component: str = "app", **base_ctx):
    """Configura logging raíz una sola vez con cola no bloqueante y dos ficheros JSONL (app y errors)."""
    global _configured, _listener
    if _configured or not getattr(S, "logging_enabled", True):
        return

    level = getattr(logging, str(getattr(S, "logging_level", "INFO")).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    log_dir = Path(getattr(S, "logging_dir", "03_logs/structured"))
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    app_path = log_dir / day / "app.jsonl"
    err_path = log_dir / day / "errors.jsonl"

    # handlers “reales” (procesados por el listener)
    app_fh = _make_file_handler(app_path, level)
    err_fh = _make_file_handler(err_path, logging.WARNING)  # solo WARNING+

    # cola no bloqueante
    q: queue.Queue[logging.LogRecord] = queue.Queue(-1)
    qh = QueueHandler(q)
    root.addHandler(qh)

    # consola directa (fuera del listener, para ver en tiempo real)
    root.addHandler(_make_console_handler(level))

    _listener = QueueListener(q, app_fh, err_fh, respect_handler_level=True)
    _listener.daemon = True
    _listener.start()

    _configured = True


def get_logger(component: str, **ctx) -> BoundLogger:
    """Devuelve un logger con contexto enlazado y asegura setup."""
    setup_logging(component, **ctx)
    base = {"component": component}
    base.update(ctx or {})
    return BoundLogger(logging.getLogger(component), base)


def shutdown_logging():
    global _listener
    try:
        if _listener:
            _listener.stop()
    except Exception:
        pass
