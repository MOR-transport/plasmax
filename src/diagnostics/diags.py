from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from ..config import load_config

if TYPE_CHECKING:
    import matplotlib as mpl


def export2tikz(fig: "mpl.figure.Figure", out_tex_path: Path) -> bool:
    """Best-effort tikzplotlib export with compatibility shims."""
    try:
        from matplotlib.legend import Legend
        import matplotlib.backends.backend_pgf as backend_pgf
        import numpy as np
        import webcolors

        if not hasattr(backend_pgf, "common_texification") and hasattr(backend_pgf, "_tex_escape"):
            backend_pgf.common_texification = backend_pgf._tex_escape
        if not hasattr(np, "float_"):
            np.float_ = np.float64
        if not hasattr(Legend, "legendHandles"):
            Legend.legendHandles = property(
                lambda self: getattr(self, "legend_handles", self.get_lines())
            )
        if not hasattr(Legend, "_ncol"):
            Legend._ncol = property(lambda self: getattr(self, "_ncols", 1))
        if not hasattr(webcolors, "CSS3_NAMES_TO_HEX"):
            css3_names = list(webcolors.names("css3"))
            webcolors.CSS3_NAMES_TO_HEX = {
                name: webcolors.name_to_hex(name, spec="css3") for name in css3_names
            }
        if not hasattr(webcolors, "CSS3_HEX_TO_NAMES"):
            webcolors.CSS3_HEX_TO_NAMES = {
                hex_value: name for name, hex_value in webcolors.CSS3_NAMES_TO_HEX.items()
            }

        tikzplotlib = importlib.import_module("tikzplotlib")
        tikzplotlib.save(
            out_tex_path,
            figure=fig,
            extra_axis_parameters=[
                "width=1\\figurewidth",
                "height=1\\figureheight",
            ],
        )
        print(f"Saved TikZ plot to '{out_tex_path}'")
        return True
    except Exception as exc:
        print(f"Warning: TikZ export skipped ({exc})")
        return False


def load_diagnostics(csv_path: Path) -> tuple[list[float], dict[str, list[float]]]:
    """Load diagnostics csv and return times and data dictionary."""
    times = []
    data = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            for key, value in row.items():
                if key != "iter" and key != "time":
                    if key not in data:
                        data[key] = []
                    data[key].append(float(value))

    return times, data


def save_figure(fig: plt.Figure, output_path: Path, fmt: str = "both"):
    """save figure in png and/or tikz format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt in ["png", "both"]:
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"Saved png plot to '{png_path}'")

    if fmt in ["tex", "both"]:
        tex_path = output_path.with_suffix(".tex")
        export2tikz(fig, tex_path)


def main():
    parser = argparse.ArgumentParser(description="Generate plots from diagnostics.csv")
    parser.add_argument("--params", type=str, nargs="+", required=True, help="Path(s) to YAML params file(s)")

    parser.add_argument("--epot", action="store_true", help="Plot Potential Energy")
    parser.add_argument("--ekin", action="store_true", help="Plot Kinetic Energy")
    parser.add_argument("--etot", action="store_true", help="Plot Total Energy")
    parser.add_argument("--mass", action="store_true", help="Plot Mass")
    parser.add_argument("--momentum", action="store_true", help="Plot Momentum")
    parser.add_argument("--l2norm", action="store_true", help="Plot L2 Norm")
    parser.add_argument("--frob", action="store_true", help="Plot Frobenius Error")
    parser.add_argument("--all", action="store_true", help="Generate all plots (default if no specific flags)")

    parser.add_argument("--format", choices=["png", "tex", "both"], default="both", help="Output format (default: both)")

    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")

    args = parser.parse_args()

    if not any((args.epot, args.ekin, args.etot, args.mass, args.momentum, args.l2norm)):
        args.all = True

    quantities = []
    if args.all or args.epot:
        quantities.append("epot")
    if args.all or args.ekin:
        quantities.append("ekin")
    if args.all or args.etot:
        quantities.append("etot")
    if args.all or args.mass:
        quantities.append("mass")
    if args.all or args.momentum:
        quantities.append("momentum")
    if args.all or args.l2norm:
        quantities.append("l2norm")

    name_mapping = {
        "epot": "epot",
        "ekin": "ekin",
        "etot": "etot",
        "mass": "mass",
        "momentum": "momentum",
        "l2norm": "l2norm",
    }

    cases_data = []

    if args.output_dir:
        figure_dir = Path(args.output_dir)
    elif len(args.params) > 1:
        figure_dir = Path("results/comparisons")
    else:
        figure_dir = None

    for params_path_str in args.params:
        params_path = Path(params_path_str)
        if not params_path.exists():
            print(f"Warning: Params file '{params_path}' not found. Skipping.")
            continue

        if params_path.suffix == ".csv":
            case_name = params_path.parent.parent.name
            csv_path = params_path
            if figure_dir is None:
                figure_dir = params_path.parent.parent / "plots"
        else:
            cfg = load_config(params_path)
            case_name = cfg.inicond.case
            csv_path = cfg.paths.data_dir / "diagnostics.csv"
            if figure_dir is None:
                figure_dir = cfg.paths.plot_dir

        if not csv_path.exists():
            print(f"Warning: {csv_path} not found for {case_name}. Run simulation first.")
            continue

        times, data = load_diagnostics(csv_path)
        cases_data.append({
            "name": case_name,
            "times": times,
            "data": data,
            "csv_path": csv_path,
        })

    if not cases_data:
        print("Error: No valid data found.")
        return

    assert figure_dir is not None
    figure_dir.mkdir(parents=True, exist_ok=True)

    for quantity in quantities:
        csv_key = name_mapping[quantity]

        fig, ax = plt.subplots(figsize=(8, 5))

        for case in cases_data:
            if csv_key not in case["data"]:
                print(f"Warning: {csv_key} not found in {case['name']}")
                continue

            ax.semilogy(
                case["times"],
                case["data"][csv_key],
                label=case["name"],
                linewidth=2,
            )

        ax.set_xlabel("Time", fontsize=12)

        labels = {
            "epot": ("Electric Potential Energy", r"$E_{pot}$"),
            "ekin": ("Kinetic Energy", r"$E_{kin}$"),
            "etot": ("Total Energy", r"$E_{tot}$"),
            "mass": ("Mass", "Mass"),
            "momentum": ("Momentum", "Momentum"),
            "l2norm": ("L2 Norm", r"$L_2$ norm"),
        }

        title, ylabel = labels[quantity]
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if len(cases_data) == 1:
            filename = f"{quantity}_{cases_data[0]['name']}"
        else:
            filename = f"{quantity}_comparaison"

        output_path = figure_dir / filename
        save_figure(fig, output_path, args.format)
        plt.close(fig)

    # Frobenius error plot
    if args.frob or args.all:
        import numpy as np
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted_frob = False

        for case in cases_data:
            frob_path = case["csv_path"].parent / "pod_frobenius_errors.csv"

            if frob_path.exists():
                try:
                    frob_data = np.genfromtxt(frob_path, delimiter=',', skip_header=1)

                    if frob_data.ndim == 1:
                        frob_data = frob_data.reshape(1, -1)

                    if frob_data.shape[1] >= 2:
                        times_frob = frob_data[:, 0]
                        errors_frob = frob_data[:, 1]

                        # Logarithmic scale for better plotting of the errors
                        ax.semilogy(
                            times_frob, errors_frob,
                            marker='o', linestyle='-', linewidth=2,
                            markersize=6, alpha=0.8,
                            label=case["name"]
                        )
                        plotted_frob = True
                except Exception as e:
                    print(f"Warning: Could not read {frob_path}: {e}")

        if plotted_frob:
            ax.set_xlabel("Time", fontsize=14, labelpad=10)
            ax.set_ylabel("Relative Frobenius Error", fontsize=14, labelpad=10)
            ax.set_title("POD Truncation Error Over Time", fontsize=16, pad=15)
            ax.legend(loc="best", fontsize=12, frameon=True, edgecolor='black')

            ax.grid(True, which='major', linestyle='-', alpha=0.5)
            ax.grid(True, which='minor', linestyle=':', alpha=0.2)
            fig.tight_layout()

            filename = "frob_error_comparison" if len(cases_data) > 1 else f"frob_error_{cases_data[0]['name']}"
            save_figure(fig, figure_dir / filename, args.format)
        plt.close(fig)

    print(f"\nPlots saved to {figure_dir}.")


if __name__ == "__main__":
    main()
