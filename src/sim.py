import math
import argparse
import matplotlib.pyplot as plt
import time as time_module
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp

from .inicond import get_inicond
from .predcorr import predictor_corrector_step
from .physics import compute_density, vpoisson
from .source import maxwell_distrib
from .plotting import plot_solution, plot_Efield, plot_profile, plot_inicond, plot_energy
from .diagnostics import measure
from .config import load_config

jax.config.update("jax_enable_x64", True)

# Default YAML next to project root `python/`, sibling of `src/` and `params/`.
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "params" / "landau_damping.yaml"


def step(f: jnp.ndarray, cfg, t: float, src=None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single time step"""
    method = cfg.method.lower()

    match method.split():
        case["predcorr"]:
            return predictor_corrector_step(f, cfg.grid, cfg, t, src)
        case["nufi"]:
            raise NotImplementedError(
                "NuFI time stepping is not implemented in Python; use method: "
                "predcorr in YAML."
            )
        case _:
            raise ValueError(f"Unknown cfg.method: {cfg.method!r} "
                             "(expected predcorr or nufi).")


def run_time_loop(cfg, src=None, inicond=None, format="png", nb_profile=0,
                  verbose=True) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Advance ``f`` until ``time >= cfg.time.tend`` or ``nt_max`` steps."""
    grid = cfg.grid

    # f and t initialization 
    if cfg.io.restart.enabled and cfg.io.restart.file:
        restart_path = Path(cfg.io.restart.file) 
        if not restart_path.exists():
            raise FileNotFoundError(f"Restart file '{restart_path}' not found.")
        
        # Loading .npz archive
        data = jnp.load(restart_path)
        f = data['f']
        t = float(data["t"])
        it_offset = int(data["it"])
        
        # Check the compatibility of the dimensions
        if f.shape != (grid.nv, grid.nx):
            raise ValueError(f"Restart file shape {f.shape} != grid shape ({grid.nv}, {grid.nx})")
        
        print(f"Restarting from {restart_path} at t = {t:.6g}")
    
    else:
        if inicond is None:
            f0 = get_inicond(cfg)
            f = f0(grid.X, grid.V)
        else:
            f = inicond
        t = 0.0
        it_offset = 0  # No offset if we start at 0
        if verbose:
            print(f"Starting from analytical initial condition")
        
    rho = compute_density(f, grid.dv)
    Efield = vpoisson(rho, grid, cfg.physics.charge)
    
    if cfg.time.plot_freq > 0:
        folder = Path("plots/simulation/sim_default")
        folder_f = folder / "iterations/solution"
        folder_E = folder / "iterations/Efield"

        if folder.exists() and folder.is_dir():
            shutil.rmtree(folder)

        folder_f.mkdir(parents=True)
        folder_E.mkdir(parents=True)

        plot_inicond(cfg, f, folder_f / f"solution_{0:04d}.{format}")
        plot_Efield(cfg, Efield, t, folder_E / f"Efield_{0:04d}.{format}")

    # Number of iterations from the initial time
    remaining = max(0.0, cfg.time.tend - t)
    nt_cap = min(cfg.time.nt_max, int(math.ceil(remaining / cfg.time.dt)))
    if nt_cap <=0: 
        print("Nothing to simulate (tend already reached).")
        return jnp.empty((0, grid.nv, grid.nx)), jnp.empty((0, grid.nx))        
    
    tcpu = []
    
    if nb_profile > 0:
        fig, axs = plt.subplots(2, 1, figsize=(16, 10))

    cfg.paths.plot_dir.mkdir(parents=True, exist_ok=True)

    f_hist = jnp.empty((nt_cap+1, grid.nv, grid.nx), dtype=jnp.float64)
    f_hist = f_hist.at[0, :, :].set(f)

    Efield_hist = jnp.empty((nt_cap+1, grid.nx), dtype=jnp.float64)
    Efield_hist = Efield_hist.at[0, :].set(Efield)
    
    global_it = it_offset 
    if global_it == 0:
        measure(cfg, f, Efield, global_it, t)
    for it in range(1, nt_cap + 1):

        t0 = time_module.perf_counter()
        global_it = it + it_offset
        f, Efield = step(f, cfg, t, src)
        t += cfg.time.dt
        tcpu.append(time_module.perf_counter() - t0)

        f_hist = f_hist.at[it, :, :].set(f)
        Efield_hist = Efield_hist.at[it, :].set(Efield)
        
        measure(cfg, f, Efield, global_it, t)

        print(f"iter: {it:3d}, time: {t:4.1f}, dt: {cfg.time.dt:.2f}, cpu_time: {tcpu[-1]:.2f} s", flush=True)
        if cfg.time.plot_freq > 0 and it % cfg.time.plot_freq == 0:
            plot_solution(cfg, f, t,
                          str(cfg.paths.plot_dir / f"solution_{it:04d}.{format}"))
            plot_Efield(cfg, Efield, t,
                        str(cfg.paths.plot_dir / f"Efield_{it:04d}.{format}"))
        if nb_profile > 0:
            if cfg.time.plot_freq > 0 and it % ((nt_cap-2) // nb_profile) == 0:
                plot_profile(cfg, f, t, axs)
        if t >= cfg.time.tend - 1e-15:
            break
    else:
        print("Warning: reached nt_max before Tend.", flush=True)

    total = sum(tcpu)
    if verbose:
        print(
            f"\n=== Simulation complete ===\n"
            f"Nb of iterations: {len(tcpu)}, final time: {t:.6g}, total CPU: {total:.3f} s\n"
            f"Average step time: {total / max(len(tcpu), 1):.4f} s",
            flush=True,
        )

    if nb_profile > 0:
        axs[0].legend()
        axs[1].legend()
        fig.savefig(cfg.paths.plot_dir / "profile.png")

    # Save final distribution function and time to an .npz archive
    save_path = cfg.paths.data_dir / "f_final.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    jnp.savez(save_path, f=f, t=t, it=global_it)
    print(f"Save state (f,t,it) to {save_path}")

    return f_hist, Efield_hist


def simulate(cfg=None) -> None:
    if cfg is None:
        parser = argparse.ArgumentParser(
            description="Vlasov–Poisson driver (predcorr / NuFI stub).")
        parser.add_argument(
            "--params",
            type=Path,
            nargs="?",
            default=_DEFAULT_CONFIG,
            help="Path to YAML (default: params/landau_damping.yaml next to src/)",
        )
        args = parser.parse_args()
        cfg = load_config(args.params)

    # Print device
    backend = jax.default_backend().lower()
    device = "GPU" if backend in ("gpu", "cuda") else "CPU"
    print(f"Device: {device}", flush=True)
    if cfg.physics.knudsen is not None:
        f_hist, Efield_hist = run_time_loop(cfg, src=maxwell_distrib, format="png")
    else:
        f_hist, Efield_hist = run_time_loop(cfg, format="png")
    plot_profile(cfg, f_hist, format="png", nb_profiles=3)
    plot_energy(cfg, Efield_hist)
