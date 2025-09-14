# utils/file_utils.py

import sys

import pandas as pd


def create_ticker_txt_from_csv(input_csv_path: str, output_txt_path: str):
    """
    Lee un fichero CSV, extrae la columna 'ticker' y la guarda en un fichero TXT.
    """
    try:
        df = pd.read_csv(input_csv_path)
        if "ticker" not in df.columns:
            print(f"Error: La columna 'ticker' no se encontró en {input_csv_path}")
            return

        tickers = df["ticker"].dropna().unique().tolist()

        with open(output_txt_path, "w") as f:
            for ticker in tickers:
                f.write(f"{ticker}\n")

        print(f"✅ Se ha creado '{output_txt_path}' con {len(tickers)} tickers.")
    except Exception as e:
        print(f"Error al procesar los ficheros: {e}")


if __name__ == "__main__":
    # Esto permite llamar al script desde la línea de comandos
    if len(sys.argv) != 3:
        print("Uso: python file_utils.py <ruta_csv_entrada> <ruta_txt_salida>")
    else:
        create_ticker_txt_from_csv(sys.argv[1], sys.argv[2])
