import warnings
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent  # carpeta donde está settings.py


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")

    # Paths
    data_path: Path = BASE_DIR / "01_data"
    models_path: Path = BASE_DIR / "02_models"
    logs_path: Path = BASE_DIR / "03_logs"
    config_path: Path = BASE_DIR / "04_config"

    # Market / time
    market_timezone: str = "America/New_York"
    local_timezone: str = "Europe/Madrid"
    market_close_buffer_min: int = 10
    tz_internal: str = "UTC"
    calendar: str = "XNYS"
    timeframe_default: str = "5mins"
    tz_ui: str = "Europe/Madrid"

    # Estrategia
    tp_multiplier: float = 3.0
    sl_multiplier: float = 2.0
    threshold_default: float = 0.8
    threshold_min: float = 0.55
    threshold_max: float = 0.95
    close_eod: bool = True
    bar_size_by_tf: dict = {"5mins": "5 mins", "10mins": "10 mins"}
    candle_size_min_by_tf: dict = {"5mins": 5, "10mins": 10}
    capital_per_trade: float = 1000.0
    commission_per_trade: float = 0.35
    label_map_mode: str = "multiclass_3way"  # "multiclass_3way" | "binary_up_vs_rest"

    # IB
    ib_host: str = "127.0.0.1"
    ib_port: int = 4004
    ib_live_batch_size: int = 10
    ib_max_rps: float = 2.0
    ib_request_pause_ms: int = 0
    ib_batch_pause_s: float = 2.0
    ib_timeout_s: int = 300
    ib_market_data_type: int = 1
    ib_duration_live: str = "3 D"
    ib_client_id: int = 11
    ib_max_retries: int = 5
    ib_backoff_min_s: float = 0.5
    ib_backoff_max_s: float = 8.0
    ib_max_concurrent: int = 1
    ib_circuit_fail_threshold: int = 10
    ib_circuit_open_seconds: int = 30
    ib_retry_attempts: int = 3
    ib_retry_backoff_s: float = 2.0
    ib_probe_head_on_fail: bool = True
    ib_alt_what_to_show: list[str] = ["TRADES", "MIDPOINT"]

    # Labels / research
    label_window: int = 5
    time_limit_candles: int = 16
    n_splits_cv: int = 5
    days_of_data: int = 90

    # CV thresholds/criterios
    use_cv_threshold_first: bool = True
    cv_dir: Path = BASE_DIR / "03_logs" / "cv"
    cv_update_on_start: bool = True
    cv_test_size: int = 500
    cv_scheme: str = "expanding"
    cv_embargo: int = -1
    cv_purge: int = -1
    cv_threshold_grid: list[float] = Field(
        default_factory=lambda: [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    )
    cv_min_trades_per_fold: int = 5
    cv_min_folds_covered: int = 3
    cv_min_total_trades: int = 100

    # MLflow
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str | None = None  # None → local ./mlruns
    mlflow_experiment: str = "PHIBOT"
    mlflow_nested: bool = False
    mlflow_tags: dict = Field(default_factory=lambda: {"project": "phibot", "owner": "research"})

    # Logging
    log_level: str = "INFO"
    log_json: bool = True

    # Arranques
    training_on_start: bool = False
    downloader_on_start: bool = True

    # Reproducibilidad
    seed: int = 42
    pythonhashseed: bool = True
    deterministic: bool = True
    n_jobs_train: int = 1
    n_jobs_cv: int = 1
    record_versions: bool = True
    env_versions_path: Path = BASE_DIR / "03_logs/env_versions.json"

    parquet_enabled: bool = True
    parquet_base_path: str = "dataset"
    parquet_compression: str = "zstd"
    parquet_engine: str = "pyarrow"
    parquet_files_per_partition: int = 1
    storage_write_csv: bool = True
    storage_write_parquet: bool = True
    csv_keep_last_days: int = 0

    logging_enabled: bool = True
    logging_level: str = "INFO"
    logging_console_pretty: bool = True
    logging_dir: str = "03_logs/structured"
    logging_rotate_bytes: int = 50 * 1024 * 1024
    logging_backup_count: int = 20
    logging_include_tracebacks: bool = True

    allow_short: bool = True
    prefer_stronger_side: bool = True

    model_config = SettingsConfigDict(env_prefix="PHIBOT_", env_file=None)

    @classmethod
    def from_yaml(cls, path: Path = BASE_DIR / "config.yaml"):
        data = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
            p = y.get("paths", {})
            m = y.get("market", {})
            l = y.get("logging", {})
            s = y.get("strategy", {})
            ib = y.get("ib", {})
            lab = y.get("labels", {})
            res = y.get("research", {})
            t = y.get("trading", {})
            rep = y.get("reproducibility", {})
            ml = y.get("mlflow", {})
            pq = y.get("parquet", {})
            st = y.get("storage", {})
            lg = y.get("logging", {})
            tr = y.get("trading", {})

            data = dict(
                data_path=BASE_DIR / p.get("data", "01_data"),
                models_path=BASE_DIR / p.get("models", "02_models"),
                logs_path=BASE_DIR / p.get("logs", "03_logs"),
                config_path=BASE_DIR / p.get("config", "04_config"),
                ib_live_batch_size=int(ib.get("live_batch_size", 10)),
                ib_max_rps=float(ib.get("max_rps", 2.0)),
                ib_request_pause_ms=int(ib.get("request_pause_ms", 0)),
                ib_batch_pause_s=float(ib.get("batch_pause_s", 2.0)),
                ib_timeout_s=int(ib.get("timeout_s", 300)),
                ib_market_data_type=int(ib.get("market_data_type", 1)),
                ib_duration_live=str(ib.get("duration_live", "3 D")),
                tz_ui=m.get("tz_ui", "Europe/Madrid"),
                tz_internal=m.get("tz_internal", "UTC"),
                calendar=m.get("calendar", "XNYS"),
                bar_size_by_tf=m.get("bar_size_by_tf", {"5mins": "5 mins", "10mins": "10 mins"}),
                candle_size_min_by_tf=m.get("candle_size_min_by_tf", {"5mins": 5, "10mins": 10}),
                timeframe_default=m.get("timeframe_default", "5mins"),
                tp_multiplier=float(s.get("tp_multiplier", 3.0)),
                sl_multiplier=float(s.get("sl_multiplier", 2.0)),
                threshold_default=float(s.get("threshold_default", 0.8)),
                threshold_min=float(s.get("threshold_min", 0.55)),
                threshold_max=float(s.get("threshold_max", 0.95)),
                close_eod=bool(s.get("close_eod", True)),
                ib_host=ib.get("host", "127.0.0.1"),
                ib_port=int(ib.get("port", 4004)),
                ib_client_id=int(ib.get("client_id", 11)),
                ib_max_retries=int(ib.get("max_retries", 5)),
                ib_backoff_min_s=float(ib.get("backoff_min_s", 0.5)),
                ib_backoff_max_s=float(ib.get("backoff_max_s", 8.0)),
                ib_max_concurrent=int(ib.get("max_concurrent", 8)),
                ib_circuit_fail_threshold=int(ib.get("circuit_fail_threshold", 10)),
                ib_circuit_open_seconds=int(ib.get("circuit_open_seconds", 30)),
                ib_retry_attempts=int(ib.get("retry_attempts", 3)),
                ib_retry_backoff_s=float(ib.get("retry_backoff_s", 2.0)),
                ib_probe_head_on_fail=bool(ib.get("probe_head_on_fail", True)),
                ib_alt_what_to_show=list(ib.get("alt_what_to_show", ["TRADES", "MIDPOINT"])),
                label_window=int(lab.get("label_window", 5)),
                time_limit_candles=int(lab.get("time_limit_candles", 16)),
                n_splits_cv=int(res.get("n_splits_cv", 5)),
                days_of_data=int(res.get("days_of_data", 90)),
                use_cv_threshold_first=bool(res.get("use_cv_threshold_first", True)),
                cv_dir=BASE_DIR / res.get("cv_dir", "03_logs/cv"),
                cv_test_size=int(res.get("cv_test_size", 500)),
                cv_scheme=str(res.get("cv_scheme", "expanding")),
                cv_embargo=int(res.get("cv_embargo", -1)),
                cv_purge=int(res.get("cv_purge", -1)),
                cv_threshold_grid=list(
                    res.get("cv_threshold_grid", [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
                ),
                # ← Añadidos para poder configurarlos por YAML:
                cv_min_trades_per_fold=int(res.get("cv_min_trades_per_fold", 5)),
                cv_min_folds_covered=int(res.get("cv_min_folds_covered", 3)),
                cv_min_total_trades=int(res.get("cv_min_total_trades", 100)),
                capital_per_trade=float(t.get("capital_per_trade", 1000.0)),
                commission_per_trade=float(t.get("commission_per_trade", 0.35)),
                log_level=l.get("level", "INFO"),
                log_json=bool(l.get("json", True)),
                seed=int(rep.get("seed", 42)),
                pythonhashseed=bool(rep.get("pythonhashseed", True)),
                deterministic=bool(rep.get("deterministic", True)),
                n_jobs_train=int(rep.get("n_jobs_train", 1)),
                n_jobs_cv=int(rep.get("n_jobs_cv", 1)),
                record_versions=bool(rep.get("record_versions", True)),
                env_versions_path=BASE_DIR / rep.get("record_to", "03_logs/env_versions.json"),
                mlflow_enabled=bool(ml.get("enabled", True)),
                mlflow_tracking_uri=ml.get("tracking_uri", None),
                mlflow_experiment=str(ml.get("experiment", "PHIBOT")),
                mlflow_nested=bool(ml.get("nested", False)),
                mlflow_tags=dict(ml.get("tags", {"project": "phibot", "owner": "research"})),
                parquet_enabled=bool(pq.get("enabled", True)),
                parquet_base_path=str(pq.get("base_path", "dataset")),
                parquet_compression=str(pq.get("compression", "zstd")),
                parquet_engine=str(pq.get("engine", "pyarrow")),
                parquet_files_per_partition=int(pq.get("files_per_partition", 1)),
                storage_write_csv=bool(st.get("write_csv", True)),
                storage_write_parquet=bool(st.get("write_parquet", True)),
                csv_keep_last_days=int(st.get("csv_keep_last_days", 0)),
                logging_level=str(lg.get("level", "INFO")),
                logging_console_pretty=bool(lg.get("console_pretty", True)),
                logging_dir=str(BASE_DIR / lg.get("dir", "03_logs/structured")),
                logging_rotate_bytes=int(lg.get("rotate_bytes", 50 * 1024 * 1024)),
                logging_backup_count=int(lg.get("backup_count", 20)),
                logging_include_tracebacks=bool(lg.get("include_tracebacks", True)),
                allow_short=bool(tr.get("allow_short", True)),
                prefer_stronger_side=bool(tr.get("prefer_stronger_side", True)),
                label_map_mode=str(lab.get("label_map_mode", "multiclass_3way")),
            )
        return cls(**data)


S = Settings.from_yaml()

for d in [S.data_path, S.models_path, S.logs_path, S.config_path, (S.logs_path / "cv")]:
    Path(d).mkdir(parents=True, exist_ok=True)


@property
def ib_max_concurrency(self) -> int:  # alias legacy
    warnings.warn(
        "ib_max_concurrency está deprecado; usa ib_max_concurrent", DeprecationWarning, stacklevel=2
    )
    return self.ib_max_concurrent
