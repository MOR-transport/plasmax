from __future__ import annotations

import argparse
import csv
import numpy as np
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

#Palette fixe par architecture INR 
INR_ARCH_STYLES: dict[str, dict] = {
    # mlp
    "mlp_16": {"color": "#89CFF0", "linestyle": "--", "marker": "s", "label": "MLP 16×3"},
    "mlp_64": {"color": "#4A90E2", "linestyle": "-",  "marker": "o", "label": "MLP 64×3"},
    "mlp_128": {"color": "#1E3A8A", "linestyle": "-",  "marker": "D", "label": "MLP 128×3"},
    "deep_128": {"color": "#0F2B5C", "linestyle": "-.", "marker": "^", "label": "MLP DEEP 128×5"},
    # siren
    "siren": {"color": "#F4A261", "linestyle": "-",   "marker": "*", "label": "SIREN 64×3"},
    "siren_128": {"color": "#E76F51", "linestyle": "-.", "marker": "v", "label": "SIREN 128×3"},
    "siren_deep_128": {"color": "#C1121F", "linestyle": "-",   "marker": "X", "label": "SIREN DEEP 128×5"},
    # fourier mlp
    "fourier_mlp": {"color": "#2A9D8F", "linestyle": "-",   "marker": "P", "label": "Fourier MLP 64×3"},
    "fourier_mlp_128": {"color": "#1B6B5E", "linestyle": "-.", "marker": "P", "label": "Fourier MLP 128×3"},
    "fourier_mlp_deep_128": {"color": "#0D3B33", "linestyle": "-",   "marker": "P", "label": "Fourier MLP DEEP 128×5"},
    # ----- periodic versions (same colors as non‑periodic) -----
    # mlp
    "periodic_mlp_16": {"color": "#89CFF0", "linestyle": ":", "marker": "s", "label": "Periodic MLP 16×3"},
    "periodic_mlp_64": {"color": "#4A90E2", "linestyle": ":", "marker": "o", "label": "Periodic MLP 64×3"},
    "periodic_mlp_128": {"color": "#1E3A8A", "linestyle": ":", "marker": "D", "label": "Periodic MLP 128×3"},
    #siren
    "periodic_siren": {"color": "#F4A261", "linestyle": ":", "marker": "*", "label": "Periodic SIREN 64×3"},
    "periodic_siren_128": {"color": "#E76F51", "linestyle": ":", "marker": "v", "label": "Periodic SIREN 128×3"},
    "periodic_siren_deep_128": {"color": "#C1121F", "linestyle": ":", "marker": "X", "label": "Periodic SIREN DEEP 128×5"},
    #fourier mlp
    "periodic_fourier_mlp": {"color": "#2A9D8F", "linestyle": ":", "marker": "P", "label": "Periodic Fourier MLP 64×3"},
    "periodic_fourier_mlp_128": {"color": "#1B6B5E", "linestyle": ":", "marker": "P", "label": "Periodic Fourier MLP 128×3"},
    "periodic_fourier_mlp_deep_128": {"color": "#0D3B33", "linestyle": ":", "marker": "P", "label": "Periodic Fourier MLP DEEP 128×5"},
}

def arch_style(arch: str) -> dict:
    """
    Returns the matplotlib style for a given architecture.
    Generates a fallback style for any unknown architecture.
    """
    if arch in INR_ARCH_STYLES:
        return INR_ARCH_STYLES[arch]
    #Fallback: couleur dérivée du nom
    import hashlib
    h = int(hashlib.md5(arch.encode()).hexdigest()[:6], 16)
    r = ((h >> 16) & 0xFF) / 255
    g = ((h >> 8)  & 0xFF) / 255
    b = (h         & 0xFF) / 255
    return {"color": (r, g, b), "linestyle": ":", "marker": "x", "label": arch}

def load_inr_errors(data_dir: Path) -> dict[str, tuple[list, list]]:
    inr_path = data_dir / "inr_errors.csv"
    if not inr_path.exists():
        return {}
    
    arch_data: dict[str, tuple[list, list]] = {}
    try:
        with open(inr_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                arch = row["arch"].strip()
                t = float(row["time"])
                err = float(row["frobenius_error"])
                if arch not in arch_data:
                    arch_data[arch] = ([], [])
                arch_data[arch][0].append(t)
                arch_data[arch][1].append(err)
    except Exception as e:
        print(f"Warning: could not read {inr_path}: {e}")
        
    return arch_data
    
def plot_frob_errors(cases_data: list[dict], figure_dir: Path,fmt: str, show_inr: bool = True, show_pod: bool = True):
    """ 
    Generate frobenius error plot
    """
    fig, ax = plt.subplots(figsize=(17, 10))
    plotted_any = False

    #POD
    if show_pod:
        for case in cases_data:
            frob_path = case["csv_path"].parent / "pod_frobenius_errors.csv"
            if not frob_path.exists():
                continue
            try:
                frob_data = np.genfromtxt(frob_path, delimiter=",", skip_header=1)
                if frob_data.ndim == 1:
                    frob_data = frob_data.reshape(1, -1)
                if frob_data.shape[1] >= 2:
                    times_frob = frob_data[:, 0]
                    errors_frob = frob_data[:, 1]
                    ax.semilogy(
                        times_frob, errors_frob,
                        linestyle="-", linewidth=2.5,
                        marker="o", markersize=5, alpha=0.85,
                        label=f"POD — {case['name']}",
                    )
                    plotted_any = True
            except Exception as e:
                print(f"Warning: Could not read {frob_path}: {e}")

    # INR
    if show_inr:
        already_plotted_archs = set()  # évite les doublons si plusieurs csv du même arch

        for case in cases_data:
            arch_data = load_inr_errors(case["csv_path"].parent)
            
            if not arch_data:
                data_dir = case["csv_path"].parent
                folder_name = data_dir.parent.name          
                if folder_name.startswith("inr_"):
                    guessed_arch = folder_name[4:]         
                    print(f"  Info: inr_errors.csv missing for :'{case['name']}', "
                          f"architecture deduced from the file : '{guessed_arch}'")
                continue

            for arch, (times, errors) in arch_data.items():
                if arch in already_plotted_archs or not times:
                    continue
                style = arch_style(arch)
                ax.semilogy(
                    times, errors,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2,
                    marker=style["marker"],
                    markersize=6,
                    alpha=0.9,
                    label=style["label"],
                )
                already_plotted_archs.add(arch)
                plotted_any = True

    if not plotted_any:
        print("Warning: No Frobenius error data found (POD or INR).")
        plt.close(fig)
        return

    ax.set_xlabel("Time", fontsize=14, labelpad=10)
    ax.set_ylabel("Relative Frobenius Error", fontsize=14, labelpad=10)

    if show_pod and show_inr:
        title = "Compression Error: POD vs INR"
    elif show_inr:
        title = "INR Compression Error — Architecture Comparison"
    else:
        title = "POD Truncation Error"
    ax.set_title(title, fontsize=15, pad=14)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11, frameon=True, edgecolor="#cbd5e1", framealpha=0.9)
    ax.grid(True, which="major", linestyle="-",  alpha=0.4)
    ax.grid(True, which="minor", linestyle=":",  alpha=0.15)
    fig.tight_layout()

    if show_pod and show_inr:
        filename = "frob_error_pod_vs_inr"
    elif show_inr:
        filename = "frob_error_inr_archs"
    else:
        filename = (
            "frob_error_comparison" if len(cases_data) > 1
            else f"frob_error_{cases_data[0]['name']}"
        )

    save_figure(fig, figure_dir / filename, fmt)
    plt.close(fig)
        
def plot_inr_loss_curves(cases_data: list[dict], figure_dir: Path,fmt: str):
    """ 
    Final loss per segment for each INR architecture for diagnosing convergence indepenently of physical error
    """    
    inr_loss_path_found = False 
    fig, ax = plt.subplots(figsize=(17, 10))
    
    for case in cases_data:
        inr_path = case["csv_path"].parent / "inr_errors.csv"
        if not inr_path.exists():
            continue 
        
        inr_loss_path_found = True
        arch_loss: dict[str, tuple[list, list]] = {}
        try:
            with open(inr_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    arch = row["arch"].strip()
                    t = float(row["time"])
                    loss = float(row["final_loss"])
                    if arch not in arch_loss:
                        arch_loss[arch] = ([], [])
                    arch_loss[arch][0].append(t)
                    arch_loss[arch][1].append(loss)           
        except Exception as e:
            print(f"Warning: could not read {inr_path}: {e}")
            continue
        
        for arch, (times, losses) in arch_loss.items():
            style = arch_style(arch)
            ax.semilogy(
                times, losses,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                marker=style["marker"],
                markersize=5,
                alpha=0.9,
                label=style["label"],
            )
    
    if not inr_loss_path_found:
        plt.close(fig)
        return 
    
    ax.set_xlabel("Time", fontsize=14, labelpad=10)
    ax.set_ylabel("Final loss (MSE)", fontsize=14, labelpad=10)
    ax.set_title("INR convergence per segment - Comparison of architectures", fontsize=15, pad=14)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11, frameon=True, edgecolor="#cbd5e1", framealpha=0.9)
    ax.grid(True, which="major", linestyle="-", alpha=0.4)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)
    fig.tight_layout()
    
    save_figure(fig, figure_dir / "inr_loss_convergence", fmt)
    plt.close(fig)        
    
def plot_cpu_time(cases_Data: list[dict], figure_dir: Path, fmt: str):
    """ Generate stacked bar chart for CPU times (Simulation + Compression)"""
    labels = []
    sim_times = []
    comp_times = []
    
    for case in cases_Data:
        data_dir = case["csv_path"].parent
        folder_name = data_dir.parent.name
        
        inr_path = data_dir / "inr_errors.csv"
        pod_path = data_dir / "pod_frobenius_errors.csv"
        
        sim_t = 0.0
        comp_t = 0.0
        label = folder_name
        
        #INR
        if inr_path.exists():
            with open(inr_path, "r") as f:
                reader = list(csv.DictReader(f))
                if not reader: continue #if the file is empty, skip
                if "sim_time" in reader[0]:
                    sim_t = sum(float(row["sim_time"]) for row in reader) 
                    comp_t = sum(float(row["comp_time"]) for row in reader)
                label = f"INR {reader[-1].get('arch', folder_name)}"
        #POD
        elif pod_path.exists():
            with open(pod_path, "r") as f:
                reader = list(csv.DictReader(f))
                if not reader: continue
                if "sim_time" in reader[0]:
                    sim_t = sum(float(row["sim_time"]) for row in reader) 
                    comp_t = sum(float(row["comp_time"]) for row in reader)
                label = f"POD {folder_name}"
        else:
            continue 
        
        if sim_t > 0 or comp_t > 0:
            labels.append(label)
            sim_times.append(sim_t)
            comp_times.append(comp_t)
    
    if not labels:
        print("Warning: No CPU time data found (need 'sim_time' and 'comp_time' in inr_errors.csv or pod_frobenius_errors.csv files).")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(labels))
    width = 0.5
    
    ax.bar(x, sim_times, width, label='Simulation Time', color='#1f77b4', edgecolor='black')
    ax.bar(x, comp_times, width, bottom=sim_times, label='Compression / Training Time', color='#ff7f0e', edgecolor='black')
    
    #add time labels on bars
    for i, (sim_t, comp_t) in enumerate(zip(sim_times, comp_times)):
        #write simulation time in the middle of the blue bar
        ax.text(i, sim_t / 2, f"{sim_t:.1f}s", ha='center', va='center', color='white', fontweight='bold', fontsize=11)
        # write compression time
        total_height = sim_t + comp_t
        if comp_t < 1.0:
            #if the compression time is small, we write it above the orange bar to avoid overlap
            ax.text(i, total_height + max(comp_times)*0.02, f"{comp_t:.2f}s", ha='center', va='bottom', color='black', fontweight='bold', fontsize=11)
        else:
            # if the time is large (like inr), we write it in the middle of the orange bar
            ax.text(i, sim_t + (comp_t / 2), f"{comp_t:.1f}s", ha='center', va='center', color='black', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Total CPU Time (s)', fontsize=14)
    ax.set_title('Computational Cost: Simulation vs Compression', fontsize=16, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12, frameon=True, edgecolor='black')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    #add some space on top of the highest bar for better visualization of labels
    ax.set_ylim(0, max(sim_times[i] + comp_times[i] for i in range(len(labels))) * 1.1)
    fig.tight_layout()
    save_figure(fig, figure_dir / "cpu_time_comparison", fmt)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate plots from diagnostics.csv")
    parser.add_argument("--params", type=str, nargs="+", required=True, help="Path(s) to YAML params file(s)")

    parser.add_argument("--epot", action="store_true", help="Plot Potential Energy")
    parser.add_argument("--ekin", action="store_true", help="Plot Kinetic Energy")
    parser.add_argument("--etot", action="store_true", help="Plot Total Energy")
    parser.add_argument("--mass", action="store_true", help="Plot Mass")
    parser.add_argument("--momentum", action="store_true", help="Plot Momentum")
    parser.add_argument("--l2norm", action="store_true", help="Plot L2 Norm")
    parser.add_argument("--frob", action="store_true", help="Plot Frobenius Error (POD + INR all architectures)")
    parser.add_argument("--frob-pod", action = "store_true", help="Plot only the Frobenius error of POD")
    parser.add_argument("--frob-inr", action = "store_true", help="Plot only the Frobenius error of INR (all architectures)")
    parser.add_argument("--inr-loss", action="store_true", help="Plot the INR final loss per segment and per architecture")
    parser.add_argument("--cpu-time", action="store_true", help="Plot Stacked CPU times")
    parser.add_argument("--all", action="store_true", help="Generate all plots (default if no specific flags)")
    parser.add_argument("--format", choices=["png", "tex", "both"], default="both", help="Output format (default: both)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")

    args = parser.parse_args()

    if not any((args.epot, args.ekin, args.etot, args.mass, args.momentum,
                args.l2norm, args.frob, args.frob_pod, args.frob_inr, args.inr_loss, args.cpu_time)):
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
        first_path = Path(args.params[0])
        case_folder = next((p for p in first_path.parts if p.startswith("case_")), "case_unknown")
        all_paths_str = " ".join(args.params)
        if "INR" in all_paths_str and "POD" not in all_paths_str:
            sub_folder = "baseline_vs_INR"
        elif "POD" in all_paths_str and "INR" not in all_paths_str:
            sub_folder = "baseline_vs_POD"
        else:
            sub_folder = "mixed_comparisons"
        
        figure_dir = Path("results") / case_folder / "comparisons" / sub_folder
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

    #physical curves
    for quantity in quantities:
        csv_key = name_mapping[quantity]
        fig, ax = plt.subplots(figsize=(17, 10))

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
        ax.set_ylabel(quantity.upper(), fontsize=14, labelpad=10)
        ax.set_title(f"{quantity.upper()} over Time", fontsize=16, pad=15)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=True, edgecolor='black')
        ax.grid(True, which='major', linestyle='-', alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)
        fig.tight_layout()

        if len(cases_data) > 1:
            filename = f"{quantity}_comparison"
        else:
            filename = f"{quantity}_{cases_data[0]['name']}"

        output_path = figure_dir / filename
        save_figure(fig, output_path, args.format)
        plt.close(fig)
        
    
    #Frobenius error plot
    #pod + inr on the same plot
    if args.all or args.frob:
        plot_frob_errors(cases_data, figure_dir, args.format,
                         show_inr=True, show_pod=True)
 
    # pod only
    if args.frob_pod:
        plot_frob_errors(cases_data, figure_dir, args.format,
                         show_inr=False, show_pod=True)
 
    # inr only, all architectures
    if args.frob_inr:
        plot_frob_errors(cases_data, figure_dir, args.format,
                         show_inr=True, show_pod=False)
 
    # final loss per segment
    if args.all or args.inr_loss:
        plot_inr_loss_curves(cases_data, figure_dir, args.format)
    
    #cpu time
    if args.all or args.cpu_time:
        plot_cpu_time(cases_data, figure_dir, args.format)
 
    print(f"\nPlots saved to {figure_dir}.")


if __name__ == "__main__":
    main()
