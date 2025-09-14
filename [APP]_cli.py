# pibot_cli.py  (o [APP]_cli.py)
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd: list[str], cwd: Path | None = None):
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd or ROOT)


# ---------- DAT ----------
def dat_update(args):
    script = ROOT / "[DAT]_Data_download.py"
    if not script.exists():
        print(
            "⚠️  No encuentro [DAT]_Data_download.py. Ajusta el nombre en pibot_cli.py si tu script se llama distinto."
        )
        return
    opts = []
    if args.timeframe:
        opts += ["--timeframe", args.timeframe]
    if args.days:
        opts += ["--days", str(args.days)]
    run([PY, str(script), *opts])


# ---------- RSH ----------
def rsh_cv(args):
    script = ROOT / "[RSH]_TimeSeriesCV.py"
    if not script.exists():
        print("⚠️  No encuentro [RSH]_TimeSeriesCV.py")
        return
    if args.ticker:
        run([PY, str(script), "--ticker", args.ticker, "--timeframe", args.timeframe])
    elif args.tickers_file:
        # iteramos archivo (uno por línea)
        for tk in Path(args.tickers_file).read_text(encoding="utf-8").splitlines():
            tk = tk.strip()
            if tk:
                run([PY, str(script), "--ticker", tk, "--timeframe", args.timeframe])
    else:
        print("⚠️  Debes pasar --ticker o --tickers-file")


def rsh_select(args):
    script = ROOT / "[RSH]_Scenarios.py"
    if not script.exists():
        print("⚠️  No encuentro [RSH]_Scenarios.py")
        return
    if args.ticker:
        cmd = [PY, str(script), "--ticker", args.ticker, "--timeframe", args.timeframe]
        if args.skip_cv:
            cmd += ["--skip-cv"]
        run(cmd)
    elif args.tickers_file:
        cmd = [PY, str(script), "--tickers-file", args.tickers_file, "--timeframe", args.timeframe]
        if args.skip_cv:
            cmd += ["--skip-cv"]
        run(cmd)
    else:
        print("⚠️  Debes pasar --ticker o --tickers-file")


# ---------- TRN ----------
def trn_fit(args):
    script = ROOT / "[TRN]_Train.py"
    if not script.exists():
        print("⚠️  No encuentro [TRN]_Train.py")
        return
    opts = []
    if args.ticker:
        opts += ["--ticker", args.ticker]
    if args.timeframe:
        opts += ["--timeframe", args.timeframe]
    if args.params_file:
        opts += ["--params-file", args.params_file]
    run([PY, str(script), *opts])


# ---------- LIV ----------
def liv_paper(args):
    script = ROOT / "[LIV]_PaperWorker.py"
    if not script.exists():
        print("⚠️  No encuentro [LIV]_PaperWorker.py")
        return
    opts = ["--mode", "paper"]
    if args.tickers_file:
        opts += ["--tickers-file", args.tickers_file]
    if args.ticker:
        opts += ["--ticker", args.ticker]
    if args.timeframe:
        opts += ["--timeframe", args.timeframe]
    run([PY, str(script), *opts])


# ---------- CLI ----------
def build_cli():
    p = argparse.ArgumentParser(prog="pibot", description="Orquestador E2E de BOTRADING")
    sub = p.add_subparsers(dest="cmd")

    # DAT
    p_dat = sub.add_parser("dat", help="[DAT] Datos")
    sub_dat = p_dat.add_subparsers(dest="subcmd")
    p_dat_upd = sub_dat.add_parser("update", help="Descargar/actualizar datos")
    p_dat_upd.add_argument("--timeframe", default="5mins")
    p_dat_upd.add_argument("--days", type=int, default=5, help="Días a actualizar")
    p_dat_upd.set_defaults(func=dat_update)

    # RSH
    p_rsh = sub.add_parser("rsh", help="[RSH] Research")
    sub_rsh = p_rsh.add_subparsers(dest="subcmd")

    p_cv = sub_rsh.add_parser("cv", help="Cross-Validation por ticker")
    p_cv.add_argument("--ticker")
    p_cv.add_argument("--tickers-file")
    p_cv.add_argument("--timeframe", default="5mins")
    p_cv.set_defaults(func=rsh_cv)

    p_sel = sub_rsh.add_parser("select", help="Seleccionar threshold/escenario por ticker")
    p_sel.add_argument("--ticker")
    p_sel.add_argument("--tickers-file")
    p_sel.add_argument("--timeframe", default="5mins")
    p_sel.add_argument("--skip-cv", action="store_true")
    p_sel.set_defaults(func=rsh_select)

    # TRN
    p_trn = sub.add_parser("trn", help="[TRN] Entrenamiento")
    sub_trn = p_trn.add_subparsers(dest="subcmd")
    p_fit = sub_trn.add_parser("fit", help="Entrenar y registrar modelo(s)")
    p_fit.add_argument("--ticker")
    p_fit.add_argument("--timeframe", default="5mins")
    p_fit.add_argument("--params-file", help="YAML/JSON con parámetros por ticker")
    p_fit.set_defaults(func=trn_fit)

    # LIV
    p_liv = sub.add_parser("liv", help="[LIV] Live (paper)")
    sub_liv = p_liv.add_subparsers(dest="subcmd")
    p_paper = sub_liv.add_parser("paper", help="Ejecutar paper trading")
    p_paper.add_argument("--ticker")
    p_paper.add_argument("--tickers-file")
    p_paper.add_argument("--timeframe", default="5mins")
    p_paper.set_defaults(func=liv_paper)

    return p


def main():
    cli = build_cli()
    args = cli.parse_args()
    if not hasattr(args, "func"):
        cli.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
