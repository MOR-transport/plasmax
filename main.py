"""Entry point: run from the `python/` directory (or after `pip install -e .`)."""

import argparse
from pathlib import Path

from src.config import load_config
from src.sim import simulate
from src.adjoint_method import optimize
from src.convergence import plot_errors

# Default YAML next to project root `python/`, sibling of `src/` and `params/`.
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "params" / "landau_damping.yaml"


def main():
    parser = argparse.ArgumentParser(description="Vlasov–Poisson driver (predcorr / NuFI stub).")
    parser.add_argument(
        "task", 
        choices=["sim", "opt", "err"], 
        help="La partie du programme à exécuter"
    )
    parser.add_argument(
        "--params",
        type=Path,
        nargs="?",
        default=_DEFAULT_CONFIG,
        help="Path to YAML (default: params/landau_damping.yaml next to src/)",
    )
    args = parser.parse_args()

    cfg = load_config(args.params)

    if args.task == "sim":
        simulate(cfg)
    elif args.task == "opt":
        optimize(cfg)
    elif args.task == "err":
        plot_errors(cfg)

if __name__ == "__main__":
    main()
