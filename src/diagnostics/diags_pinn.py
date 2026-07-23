import argparse
import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def save_figure(fig: plt.Figure, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved PINN diagnostic plot to '{png_path}'")
    
def load_pinn_metrics(csv_path: Path) -> dict:
    times, loss_adam, loss_lbfgs, frob_errors, train_times = [], [], [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            loss_adam.append(float(row["loss_adam"]))
            loss_lbfgs.append(float(row["loss_lbfgs"]))
            frob_errors.append(float(row["frobenius_error"]))
            train_times.append(float(row["train_time"]))
    
    return {
        "times": times,
        "loss_adam": loss_adam,
        "loss_lbfgs": loss_lbfgs,
        "frob_errors": frob_errors,
        "train_times": train_times
    }
    
def plot_reconstruction_error(metrics: dict, case_name: str, arch: str, ratio: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(
        metrics["times"], metrics["frob_errors"], 
        color="#800080", linestyle="-", linewidth=2.5, marker="^", markersize=8, 
        label=f"PINN Reconstruction ({ratio} data)"
    )
    
    ax.set_xlabel("Time $t$", fontsize=20, labelpad=10)
    ax.set_ylabel("Relative Frobenius Error", fontsize=20, labelpad=12)
    ax.set_title(f"PINN Data Assimilation Error - {case_name.upper()}\nArchitecture: {arch}", fontsize=18, pad=15)
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(loc="best", fontsize=16, frameon=True, edgecolor="black")
    ax.grid(True, which="major", linestyle="-", alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)
    fig.tight_layout()
    
    save_figure(fig, out_dir / "pinn_reconstruction_error")
    plt.close(fig)
    
def plot_optimization_losses(metrics: dict, case_name: str, arch: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.semilogy(metrics["times"], metrics["loss_adam"], color="#1f77b4", linestyle="--", linewidth=2.5, marker="o", label="Phase 1: ADAM Loss (Data Only)")
    ax.semilogy(metrics["times"], metrics["loss_lbfgs"], color="d62728", linestyle="-", linewidth=2.5, marker="s", label="Phase 2: L-BFGS Loss (Data + Physics)")
    
    ax.set_xlabel("Time Segment", fontsize=18, labelpad=10)
    ax.set_ylabel("Total Loss", fontsize=18, labelpad=10)
    ax.set_title(f"PINN Convergence per Segment - {case_name.upper()}", fontsize=20, pad=15)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc="best", fontsize=14, frameon=True, edgecolor="black")
    ax.grid(True, which="major", linestyle="-", alpha=0.4)
    fig.tight_layout()
    
    save_figure(fig, out_dir / "pinn_optimization_losses")
    plt.close(fig)
    
def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic plots for PINN Data Assimilation")
    parser.add_argument("--metrics", type=str, required=True, help="Path to pinn_metrics.csv")
    args = parser.parse_args()
    
    csv_path = Path(args.metrics)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return 
    
    #extraction of context
    data_dir = csv_path.parent
    pinn_folder = data_dir.parent.name 
    case_name = data_dir.parent.parent.parent.name.replace("case_", "")
    
    arch = pinn_folder.split("_ratio")[0] if "_ratio" in pinn_folder else pinn_folder
    ratio = pinn_folder.split("_ratio")[1] if "_ratio" in pinn_folder else "Unknown"
    ratio_str = f"{float(ratio)*100:.1f}%" if ratio != "Unknown" else ratio 
    
    plot_dir = data_dir.parent / "plots"
    metrics = load_pinn_metrics(csv_path)
    
    print(f"\n--- Generating PINN Diagnostics for {case_name} ---")
    plot_reconstruction_error(metrics, case_name, arch, ratio_str, plot_dir)
    plot_optimization_losses(metrics, case_name, arch, plot_dir)
    
    total_time = sum(metrics["train_times"])
    print(f"\nTotal PINN Training Time: {total_time:.2f} seconds")
    print(f"Diagnostics saved to {plot_dir}")
    
if __name__ == "__main__":
    main()