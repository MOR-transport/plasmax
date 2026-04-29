import math
from pathlib import Path
import shutil
import time as time_module

import jax
import jax.numpy as jnp

from .inicond import get_inicond
from .periodic_grid import make_periodic_grid
from .source import compute_src, functionnal
from .sim import run_time_loop
from .advect import advect_with_source_hist
from .plotting import make_anim_2d, plot_optimisation, plot_f_exp, plot_inicond

jax.config.update("jax_enable_x64", True)


def run_time_loop_adjoint(cfg, Efield, src):
    grid = make_periodic_grid(cfg.grid)
    cfg.grid = grid

    f = jnp.zeros((grid.nv, grid.nx), dtype=jnp.float64)
    t = cfg.time.tend
    cfg.time.dt = - cfg.time.dt

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

    cfg.time.dt = - cfg.time.dt
    total = sum(tcpu)
    print(
        f"\n=== Simulation complete ===\n"
        f"iterations: {len(tcpu)}, final time: {t:.6g}, total CPU: {total:.3f} s\n"
        f"avg step: {total / max(len(tcpu), 1):.4f} s",
        flush=True,
    )

    return f_hist


def line_search(cfg, inicond, f, adj, m=1e-4, theta=0.5):
    alpha = 100
    J = lambda f: functionnal(cfg, f)

    while True:
        inicond_tmp = inicond + alpha * adj

        f_new, Efield_new = run_time_loop(cfg, inicond=inicond_tmp)
        armijo_cond = J(f_new) <= J(f) - m*alpha*jnp.sum(adj**2)

        if armijo_cond:
            return inicond_tmp, f_new, Efield_new, alpha
        alpha *= theta

        if alpha <= 1e-15:
            raise("Line search err")


def adjoint(cfg, line_search_opt=True, tolerance=0.5):
    cfg.grid = make_periodic_grid(cfg.grid)
    cfg.time.plot_freq = 0

    inicond_initiale = get_inicond(cfg)(cfg.grid.X, cfg.grid.V)  
    inicond = inicond_initiale.copy()
    residuals = []
    alphas = []

    folder = Path(f"plots/optimization/default_optim/")
    folder_it = folder / "iterations"

    if folder.exists() and folder.is_dir():
            shutil.rmtree(folder)

    folder_it.mkdir(parents=True)

    if line_search_opt:
        f_hist, Efield_hist = run_time_loop(cfg, inicond=inicond)

    for it in range(cfg.optim.Nopt):
        
        if not line_search_opt:
            f_hist, Efield_hist = run_time_loop(cfg, inicond=inicond)

        residuals.append(functionnal(cfg, f_hist))
        
        src = compute_src(cfg, f_hist)
        adj_hist = run_time_loop_adjoint(cfg, Efield_hist, src)
        
        if jnp.sqrt(jnp.sum(adj_hist[0, :, :] ** 2)) <= tolerance:
            cfg.optim.Nopt = it + 1
            break

        if line_search_opt:
            inicond, f_hist, Efield_hist, alpha = line_search(cfg, inicond.copy(), f_hist, adj_hist[0, :, :])
        else:
            inicond += adj_hist[0, :, :]
            alpha = cfg.optim.lr
        
        alphas.append(alpha)
        plot_inicond(cfg, inicond, folder_it / f"inicond_{it:04d}.png")

    plot_optimisation(cfg, inicond_initiale, inicond, residuals, folder / "result.tex")
    plot_f_exp(cfg, folder / "f_exp.tex")
    make_anim_2d(cfg, f_hist, folder / "f_res.gif")

    print(alphas)


def optimize(cfg):
    # Print device
    backend = jax.default_backend().lower()
    device = "GPU" if backend in ("gpu", "cuda") else "CPU"
    print(f"Device: {device}", flush=True)

    adjoint(cfg, line_search_opt=True)
