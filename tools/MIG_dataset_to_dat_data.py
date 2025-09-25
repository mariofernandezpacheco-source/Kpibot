# tools/MIG_dataset_to_dat_data.py
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys

def main():
    ap = argparse.ArgumentParser("Migra parquet de dataset -> [DAT]_data (manteniendo estructura ohlcv)")
    ap.add_argument("--old-root", required=True, help=r"Ej: C:\...\BOTRADING\dataset\ohlcv")
    ap.add_argument("--new-root", required=True, help=r"Ej: C:\...\BOTRADING\DAT_data\parquet\ohlcv")
    ap.add_argument("--mode", choices=["move","copy"], default="move", help="Mover (por defecto) o copiar")
    ap.add_argument("--dry-run", action="store_true", help="No escribe; solo muestra lo que haría")
    args = ap.parse_args()

    old_root = Path(args.old_root).resolve()
    new_root = Path(args.new_root).resolve()

    if not old_root.exists():
        print(f"[ERR] No existe old-root: {old_root}")
        sys.exit(1)
    new_root.mkdir(parents=True, exist_ok=True)

    moved = 0
    examined = 0

    # Estructura esperada: old_root / ticker=<SYM>_<TFUP> / date=YYYY-MM-DD / data.parquet
    for tdir in old_root.glob("ticker=*"):
        if not tdir.is_dir():
            continue
        for ddir in tdir.glob("date=*"):
            if not ddir.is_dir():
                continue
            examined += 1
            src = ddir / "data.parquet"
            if not src.exists():
                # si no existe data.parquet, mueve todos .parquet de esa partición
                for src2 in ddir.rglob("*.parquet"):
                    rel = src2.relative_to(old_root)
                    dest = new_root / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    print(f"[{args.mode.upper()}] {src2} -> {dest}")
                    if not args.dry_run:
                        if args.mode == "move":
                            shutil.move(str(src2), str(dest))
                        else:
                            shutil.copy2(str(src2), str(dest))
                        moved += 1
                continue

            # caso normal: hay data.parquet
            rel = src.relative_to(old_root)
            dest = new_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            print(f"[{args.mode.upper()}] {src} -> {dest}")
            if not args.dry_run:
                if args.mode == "move":
                    shutil.move(str(src), str(dest))
                else:
                    shutil.copy2(str(src), str(dest))
                moved += 1

    print(f"\nResumen: particiones examinadas={examined}, ficheros {args.mode}d={moved}")
    print(f"Destino: {new_root}")

if __name__ == "__main__":
    main()
