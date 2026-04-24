import math
from pathlib import Path
import time as time_module

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .inicond import get_inicond
from .periodic_grid import make_periodic_grid
from .source import f_exp, filter_t, filter_xv
from .sim import run_time_loop
from .advect import advect_with_source_hist
from .inicond import plot_inicond

jax.config.update("jax_enable_x64", True)


def plot_optimisation(cfg, init_inicond, final_inicond, residuals, fname):
    cfg.grid = make_periodic_grid(cfg.grid)

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    pcm_init = axs[1].pcolormesh(cfg.grid.X, cfg.grid.V, init_inicond, shading="auto")
    pcm_final = axs[2].pcolormesh(cfg.grid.X, cfg.grid.V, final_inicond, shading="auto")

    fig.colorbar(pcm_init, ax=axs[1], label=r"$init(x,v)$")
    fig.colorbar(pcm_final, ax=axs[2], label=r"$final(x,v)$")

    axs[1].set_xlabel(r"$x$")
    axs[1].set_ylabel(r"$v$")
    axs[1].set_title("Intiale inicond")

    axs[2].set_xlabel(r"$x$")
    axs[2].set_ylabel(r"$v$")
    axs[2].set_title("Final inicond")

    axs[0].plot(jnp.arange(cfg.optim.Nopt), residuals, "-x")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Residual")
    axs[0].set_title("Evolution of residuals")

    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname)
        plt.close(fig)
    else:
        plt.show()


def run_time_loop_adjoint(cfg, Efield, src):
    grid = make_periodic_grid(cfg.grid)
    cfg.grid = grid

    f = jnp.zeros((grid.nv, grid.nx), dtype=jnp.float64)
    t = cfg.time.tend

    nt_cap = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt))))
    tcpu = []

    f_hist = jnp.empty((nt_cap, grid.nv, grid.nx), dtype=jnp.float64)
    f_hist = f_hist.at[-1, :, :].set(f)

    for it in range(1, nt_cap + 1):

        t0 = time_module.perf_counter()

        f = advect_with_source_hist(f, Efield[-it, :], grid, cfg.time.dt, cfg.interp.order, src, -it)
        
        t += cfg.time.dt
        tcpu.append(time_module.perf_counter() - t0)

        f_hist = f_hist.at[-it, :, :].set(f)
        print(f"iter: {it}, time: {t:.6g}, dt: {cfg.time.dt:.6g}, "f"cpu_time: {tcpu[-1]:.4f} s", flush=True)

    total = sum(tcpu)
    print(
        f"\n=== Simulation complete ===\n"
        f"iterations: {len(tcpu)}, final time: {t:.6g}, total CPU: {total:.3f} s\n"
        f"avg step: {total / max(len(tcpu), 1):.4f} s",
        flush=True,
    )

    return f_hist


def adjoint(cfg):
    cfg.grid = make_periodic_grid(cfg.grid)
    cfg.time.plot_freq = 0

    inicond_initiale = get_inicond(cfg)(cfg.grid.X, cfg.grid.V)  
    inicond = inicond_initiale.copy()
    residuals = []

    folder = Path(f"plots/optimization/step-{cfg.optim.lr}_Nopt-{cfg.optim.Nopt}/iterations")
    folder.mkdir(parents=True, exist_ok=True)

    for it in range(cfg.optim.Nopt):
        f_hist, Efield_hist = run_time_loop(cfg, inicond=inicond)
        src = (f_hist - f_exp(cfg)) * filter_xv(cfg)[None, :, :] * filter_t(cfg)[:, None, None]
        residuals.append(jnp.sqrt(jnp.mean(src ** 2)))
        cfg.time.dt = - cfg.time.dt
        adj_hist = run_time_loop_adjoint(cfg, Efield_hist, src)
        inicond += cfg.optim.lr * adj_hist[0, :, :]
        cfg.time.dt = - cfg.time.dt

        plot_inicond(cfg, inicond, folder / f"inicond_{it:04d}.png")
    
    plot_optimisation(cfg, inicond_initiale, inicond, residuals, f"plots/optimization/step-{cfg.optim.lr}_Nopt-{cfg.optim.Nopt}/result.png")


def optimize(cfg):
    # Print device
    backend = jax.default_backend().lower()
    device = "GPU" if backend in ("gpu", "cuda") else "CPU"
    print(f"Device: {device}", flush=True)

    adjoint(cfg)
