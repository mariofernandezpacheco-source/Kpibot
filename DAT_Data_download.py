# DAT_Data_download.py - MIGRADO A LOGGING ESTRUCTURADO
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

import settings as settings

S = settings.S

from utils.data_update import (
    get_tickers_from_file,
    update_many,
    last_available_table,
)

# NUEVO: Sistema de logging estructurado
from utils.logging_enhanced import get_logger, with_component, with_correlation_id

# Crear logger para este módulo
logger = get_logger(__name__)


def main():
    # Usar contexto de componente para identificar logs del módulo DAT
    with with_component("DAT"):
        with with_correlation_id() as corr_id:
            ap = argparse.ArgumentParser("[DAT] Data Download (IBKR → Parquet incremental)")
            ap.add_argument("--timeframe", default=getattr(S, "timeframe_default", "5mins"))
            ap.add_argument("--tickers-file", default=str(Path(S.config_path) / "sp500_tickers.txt"))
            ap.add_argument("--only-show-last", action="store_true",
                            help="Solo mostrar última fecha por ticker (no descarga)")
            args = ap.parse_args()

            # Log inicio con parámetros
            logger.info("data_download_start",
                        timeframe=args.timeframe,
                        tickers_file=args.tickers_file,
                        only_show_last=args.only_show_last,
                        correlation_id=corr_id,
                        message="Iniciando descarga de datos")

            tickers_file = Path(args.tickers_file)

            try:
                tickers = get_tickers_from_file(tickers_file)
                logger.info("tickers_loaded",
                            count=len(tickers),
                            source_file=str(tickers_file),
                            message=f"Cargados {len(tickers)} tickers desde archivo")

            except Exception as e:
                logger.error("tickers_load_error",
                             file_path=str(tickers_file),
                             error_type=type(e).__name__,
                             error_message=str(e),
                             exc_info=True,
                             message="Error cargando archivo de tickers")
                return

            if args.only_show_last:
                logger.info("showing_last_dates_only",
                            message="Modo solo mostrar - no se descargarán datos")
                try:
                    df = last_available_table(tickers, args.timeframe)
                    logger.info("last_dates_retrieved",
                                tickers_processed=len(df),
                                message="Última fecha obtenida por ticker")
                    print(df.to_string(index=False))

                except Exception as e:
                    logger.error("last_dates_error",
                                 error_type=type(e).__name__,
                                 error_message=str(e),
                                 exc_info=True,
                                 message="Error obteniendo últimas fechas")
                return

            # Proceso de descarga principal
            logger.info("download_process_start",
                        tickers_count=len(tickers),
                        timeframe=args.timeframe,
                        storage_backend=S.storage_backend if hasattr(S, 'storage_backend') else 'unknown',
                        message=f"Descargando {len(tickers)} tickers en {args.timeframe}")

            try:
                # Ejecutar descarga con timing
                import time
                start_time = time.time()

                res = update_many(tickers, args.timeframe)

                download_duration = time.time() - start_time

                # Procesar y loggear resultados
                rows = []
                successful_downloads = 0
                failed_downloads = 0
                total_rows_added = 0

                for r in res:
                    status = "OK" if r.error is None else "ERROR"
                    if r.error is None:
                        successful_downloads += 1
                        total_rows_added += r.added_rows or 0
                    else:
                        failed_downloads += 1

                    rows.append({
                        "ticker": r.ticker,
                        "added_rows": r.added_rows,
                        "last_dt_after": r.last_dt_after,
                        "status": status,
                        "error": str(r.error) if r.error else None
                    })

                    # Log individual por ticker
                    ticker_logger = logger.bind(ticker=r.ticker)
                    if r.error is None:
                        ticker_logger.info("ticker_download_success",
                                           rows_added=r.added_rows or 0,
                                           last_datetime=str(r.last_dt_after) if r.last_dt_after else None,
                                           message="Ticker descargado exitosamente")
                    else:
                        ticker_logger.error("ticker_download_error",
                                            error_message=str(r.error),
                                            message="Error descargando ticker")

                # Log resumen final
                logger.info("download_process_complete",
                            duration_seconds=round(download_duration, 2),
                            successful_downloads=successful_downloads,
                            failed_downloads=failed_downloads,
                            total_rows_added=total_rows_added,
                            success_rate=round(successful_downloads / len(tickers) * 100, 1) if tickers else 0,
                            message="Proceso de descarga completado")

                # Mostrar tabla de resultados
                df_res = pd.DataFrame(rows).sort_values(["status", "ticker"])
                print(df_res.to_string(index=False))

                # Mostrar resumen de última fecha tras la actualización
                logger.info("generating_last_dates_summary",
                            message="Generando resumen de última fecha tras actualización")

                try:
                    logger.info("last_dates_summary_start", message="Última fecha tras la actualización:")
                    df_last = last_available_table(tickers, args.timeframe)
                    print(df_last.to_string(index=False))

                    logger.info("last_dates_summary_complete",
                                tickers_summarized=len(df_last),
                                message="Resumen de fechas completado")

                except Exception as e:
                    logger.error("last_dates_summary_error",
                                 error_type=type(e).__name__,
                                 error_message=str(e),
                                 exc_info=True,
                                 message="Error generando resumen de fechas")

            except Exception as e:
                logger.error("download_process_error",
                             error_type=type(e).__name__,
                             error_message=str(e),
                             exc_info=True,
                             message="Error en proceso principal de descarga")
                raise


if __name__ == "__main__":
    main()