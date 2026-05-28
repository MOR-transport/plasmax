import argparse
from pathlib import Path
from .config import load_config, Paths
from .sim import run_time_loop
import jax.numpy as jnp
import matplotlib.pyplot as plt


def compress_pod(f: jnp.ndarray, rank: int, current_time: float, plot_dir: Path, data_dir: Path) -> jnp.ndarray:
    """Apply SVD, plot the spectrum, calculate the error and reconstruct the distribution f with rank r"""
    # SVD calculation
    U, s, VT = jnp.linalg.svd(f, full_matrices=False)

    # Computation of the error in Frobenius norm
    total_energy = jnp.sum(s**2)
    truncated_energy = jnp.sum(s[rank:]**2)
    error_frob = float(jnp.sqrt(truncated_energy / total_energy))
    # Error saving
    error_file = data_dir / "pod_frobenius_errors.csv"
    if not error_file.exists():
        with open(error_file, "w") as f_err:
            f_err.write("time,frobenius_error\n")

    with open(error_file, "a") as f_err:
        f_err.write(f"{current_time:.4f},{error_frob:.6e}\n")

    # Plot of the normalized singular spectrum
    s_norm = s / s[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(s_norm, marker='o', linestyle='', color='#1f77b4', markersize=5, alpha=0.8, label=r"Normalized $\sigma_i$")
    ax.axvline(x=rank, color='#d62728', linestyle='--', linewidth=2, label=f'Truncation $r={rank}$')
    ax.set_xlabel(r'Singular Value Index $i$', fontsize=14, labelpad=10)
    ax.set_ylabel(r'$\sigma_i / \sigma_1$', fontsize=14, labelpad=10)
    ax.set_title(f'Normalized SVD Spectrum at $t={current_time:.2f}$', fontsize=16, pad=15)
    ax.grid(True, which='major', linestyle='-', alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax.legend(loc='upper right', fontsize=12, frameon=True, edgecolor='black', fancybox=False, facecolor='white', framealpha=1.0)
    fig.tight_layout()
    fig.savefig(plot_dir / f"svd_spectrum_t{current_time:05.2f}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Truncation
    U_r = U[:, :rank]
    s_r = s[:rank]
    VT_r = VT[:rank, :]

    # Reconstruction
    f_comp = (U_r * s_r) @ VT_r

    print(f"-> Frobeniurs Error: {error_frob:.2e}")

    return f_comp


def main():
    parser = argparse.ArgumentParser(description="Run PlasmaX simulation in segments.")
    parser.add_argument("--params", type=str, required=True, help="Base yaml config file")
    parser.add_argument("--tend", type=float, default=None, help="Total simulation time")
    parser.add_argument("--rank", type=int, default=32, help="Rank for POD compression")
    args = parser.parse_args()

    # Base configuration
    cfg = load_config(args.params)

    final_time = args.tend if args.tend is not None else cfg.time.tend

    dt_seg = cfg.io.dt_save if cfg.io.dt_save is not None else 5.0

    original_case = cfg.inicond.case

    # Creating specific folders to be able to compare the results
    if cfg.io.compression_method == "POD":
        segmented_case = f"{original_case}_segmented_POD_r{args.rank}"
    else:
        segmented_case = f"{original_case}_segmented"

    cfg.inicond.case = segmented_case

    new_save_dir = cfg.paths.save_dir.parent / segmented_case
    cfg.paths = Paths.from_case(segmented_case, new_save_dir)
    cfg.paths.data_dir.mkdir(parents=True, exist_ok=True)

    # Clean up the old csv file if we restart the experiment from scratch
    diag_file = cfg.paths.data_dir / "diagnostics.csv"
    if diag_file.exists():
        diag_file.unlink()

    current_time = 0.0
    cfg.io.restart.enabled = False
    cfg.io.restart.file = None

    while current_time < final_time - 1e-9:
        next_time = min(current_time + dt_seg, final_time)

        print("\n" + "="*60)
        print(f"-> Running segment: t= {current_time:.2f} to t={next_time:.2f}")
        print("="*60 + "\n")

        # Update of end time for this segment
        cfg.time.tend = next_time

        run_time_loop(cfg)

        current_time = next_time

        # Interception and compression
        file_path = cfg.paths.data_dir / "f_final.npz"

        if cfg.io.compression_method == "POD":
            print(f"\n[POD] Applying compression (rank={args.rank}) to saved state...")
            data = jnp.load(file_path)
            f_full = data['f']
            t_saved = data['t']
            it_saved = data['it']

            plot_dir = cfg.paths.save_dir.parent / segmented_case / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)

            f_comp = compress_pod(f_full, args.rank, current_time, plot_dir, cfg.paths.data_dir)

            # Overwrite the saved state with the compressed version
            jnp.savez(file_path, f=f_comp, t=t_saved, it=it_saved)
            print("[POD] Done\n")

        cfg.io.restart.enabled = True
        cfg.io.restart.file = str(file_path)

    print(f"\n Segmented simulation finished successfully. Results saved in '{new_save_dir}' ")


if __name__ == "__main__":
    main()
