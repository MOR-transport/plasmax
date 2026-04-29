import math
import time as time_module
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp

from .inicond import get_inicond
from .periodic_grid import make_periodic_grid
from .predcorr import predictor_corrector_step
from .physics import compute_density, vpoisson
from .source import maxwell_distrib
from .plotting import plot_solution, plot_Efield, plot_profile, plot_inicond, make_anim_1d, make_anim_2d

jax.config.update("jax_enable_x64", True)


def step(f: jnp.ndarray, cfg, t: float, src=None) -> tuple[jnp.ndarray, jnp.ndarray]:
    method = cfg.method.lower()

    if method == "predcorr":
        return predictor_corrector_step(f, cfg.grid, cfg, t, src)
    if method == "nufi":
        raise NotImplementedError(
            "NuFI time stepping is not implemented in Python; use method: predcorr in YAML."
        )
    raise ValueError(f"Unknown cfg.method: {cfg.method!r} (expected predcorr or nufi).")


def run_time_loop(cfg, src=None, inicond=None, format="png") -> tuple[jnp.ndarray, jnp.ndarray, float]:
    grid = make_periodic_grid(cfg.grid)
    cfg.grid = grid

    if inicond is None:
        f0 = get_inicond(cfg)
        f = f0(grid.X, grid.V)
    else:
        f = inicond
    t = 0.0

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

    if cfg.time.plot_freq > 0:
        plot_inicond(cfg, f, folder_f / f"solution_{0:04d}.{format}")
        plot_Efield(cfg, Efield, t, folder_E / f"Efield_{0:04d}.{format}")

    nt_cap = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt))))
    tcpu = []        

    f_hist = jnp.empty((nt_cap+1, grid.nv, grid.nx), dtype=jnp.float64)
    f_hist = f_hist.at[0, :, :].set(f)

    Efield_hist = jnp.empty((nt_cap+1, grid.nx), dtype=jnp.float64)
    Efield_hist = Efield_hist.at[0, :].set(Efield)

    for it in range(1, nt_cap + 1):

        t0 = time_module.perf_counter()
    
        f, Efield = step(f, cfg, t, src)
        
        t += cfg.time.dt
        tcpu.append(time_module.perf_counter() - t0)

        f_hist = f_hist.at[it, :, :].set(f)
        Efield_hist = Efield_hist.at[it, :].set(Efield)

        print(f"iter: {it}, time: {t:.6g}, dt: {cfg.time.dt:.6g}, "f"cpu_time: {tcpu[-1]:.4f} s", flush=True)
        if cfg.time.plot_freq > 0 and it % cfg.time.plot_freq == 0:
            plot_solution(cfg, f, t, folder_f / f"solution_{it:04d}.{format}")
            plot_Efield(cfg, Efield, t, folder_E / f"Efield_{it:04d}.{format}")

    total = sum(tcpu)
    print(
        f"\n=== Simulation complete ===\n"
        f"iterations: {len(tcpu)}, final time: {t:.6g}, total CPU: {total:.3f} s\n"
        f"avg step: {total / max(len(tcpu), 1):.4f} s",
        flush=True,
    )

    return f_hist, Efield_hist


def simulate(cfg) -> None:
    # Print device
    backend = jax.default_backend().lower()
    device = "GPU" if backend in ("gpu", "cuda") else "CPU"
    print(f"Device: {device}", flush=True)

    f_hist, Efield_hist = run_time_loop(cfg, src=maxwell_distrib, format="tex")
    plot_profile(cfg, f_hist, format="tex", nb_profiles=3)

