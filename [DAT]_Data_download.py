# [DAT]_Data_download.py

from pathlib import Path

from settings import S
from utils.A_data_loader import load_data  # Ajustado a tu estructura real


def get_tickers_from_file(file_path: Path) -> list:
    """Carga la lista de tickers desde un fichero de texto."""
    if not file_path.exists():
        raise FileNotFoundError(f"El fichero de tickers no se encontró en: {file_path}")

    with open(file_path) as f:
        tickers = [line.strip() for line in f if line.strip()]

    print(f"Encontrados {len(tickers)} tickers en {file_path.name}.")
    return tickers


def check_data_for_tickers(tickers_list: list, timeframe: str, data_folder: Path):
    """
    Verifica que los datos históricos existen para una lista de tickers y un timeframe dados.
    """
    print(f"\nIniciando verificación de datos para timeframe: '{timeframe}'...")

    for ticker in tickers_list:
        try:
            # Pasamos la ruta de la carpeta de datos a la función de carga
            df = load_data(
                ticker=ticker, timeframe=timeframe, use_local=True, base_path=data_folder
            )
            print(f"  ✅ {ticker} ({timeframe}) | OK ({len(df)} velas)")
        except FileNotFoundError:
            print(f"  ❌ {ticker} ({timeframe}) | No encontrado. Requiere descarga.")
        except Exception as e:
            print(f"  🔥 {ticker} ({timeframe}) | Error: {e}")


if __name__ == "__main__":
    # --- CONFIGURACIÓN DESDE config.yaml ---
    tickers_filepath = (
        Path("utils") / "sp500_tickers.txt"
    )  # Podrías mover esto también a config.yaml si lo quieres 100% parametrizable
    data_folder_path = S.data_path
    timeframe = S.timeframe_default

    # --- Ejecución del Proceso ---
    try:
        tickers = get_tickers_from_file(tickers_filepath)
        check_data_for_tickers(tickers, timeframe, data_folder_path)
        print("\nVerificación completada.")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nOcurrió un error inesperado: {e}")
