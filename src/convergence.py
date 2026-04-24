from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .periodic_grid import make_periodic_grid
from .sim import run_time_loop
from .source import maxwell_distrib

jax.config.update("jax_enable_x64", True)


def plot_time_error(cfg, src=None):
    dt_backup = cfg.time.dt
    dt_ref = jnp.float32(jnp.log2(cfg.time.dt))
    if dt_ref - jnp.ceil(dt_ref) != 0:
        raise ValueError("Invalid time step: dt must be a negative power of 2")
    cfg.time.plot_freq = 0
    
    dt_ref = jnp.int32(dt_ref)
    print(f"{-dt_ref-1} computations needed for error in time")

    f_ref, _ = run_time_loop(cfg, src=src)
    f_ref = f_ref[-1, :, :]

    errors_dts = []
    dts = []

    powers_dts = jnp.arange(dt_ref + 1, -1)
    for power in powers_dts:
        dt = 2.0 ** power
        cfg.time.dt = dt

        f, _ = run_time_loop(cfg, src=src)
        f = f[-1, :, :]

        err = jnp.sqrt(jnp.sum((f - f_ref) ** 2))  # L2 error
        errors_dts.append(err)
        dts.append(dt)

    dts = jnp.array(dts)
    errors_dts = jnp.array(errors_dts)

    log_dt = jnp.log(dts)
    log_err_dt = jnp.log(errors_dts)
    p_dt, C_dt = jnp.polyfit(log_dt, log_err_dt, 1)

    print(f"Estimated convergence order in time: {p_dt:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(dts, errors_dts, "x-")
    ax.loglog(dts, jnp.exp(C_dt) * dts**p_dt, "--", label=fr"fit: $O(\Delta t^{{{p_dt:.2f}}})$")

    ax.set_xlabel("time step Δt")
    ax.set_ylabel("L2 error")
    ax.set_title(f"Errors in time for ({cfg.inicond.case}) (Kn = {cfg.physics.knudsen:.0e})")
    
    plt.minorticks_off()
    ax.set_xticks(dts)
    ax.set_xticklabels(dts)

    ax.legend()
    ax.grid(True, which="both")

    folder = Path(f"plots/errors")
    folder.mkdir(parents=True, exist_ok=True)

    fig.savefig(folder / f"time_error-Kn_{cfg.physics.knudsen:.0e}.png")
    plt.close(fig)

    cfg.time.dt = dt_backup


def plot_space_error(cfg, src=None):
    if cfg.grid.nx != cfg.grid.nv:
        raise ValueError("Invalid space step: dx and dv must be equal")
    
    nxv_backup = cfg.grid.nx
    nxv_ref = jnp.float32(jnp.log2(cfg.grid.nx))
    if nxv_ref - jnp.floor(nxv_ref) != 0:
        raise ValueError("Invalid space step: dx and dx must be a power of 2")
    cfg.time.plot_freq = 0

    nxv_ref = jnp.int32(nxv_ref)
    print(f"{nxv_ref - 3} computations needed for error in space")

    f_ref, _ = run_time_loop(cfg, src=src)
    f_ref = f_ref[-1, :, :]

    errors_nxvs = []
    nxvs = []

    powers_nxvs = jnp.arange(nxv_ref-1, 3, -1)
    for power in powers_nxvs:
        step = 2 ** power
        cfg.grid.nx = step
        cfg.grid.nv = step
        cfg.grid = make_periodic_grid(cfg.grid)

        f, _ = run_time_loop(cfg, src=src)
        f = f[-1, :, :]

        ratio = nxv_backup // step

        err = jnp.sqrt(jnp.mean((f - f_ref[::ratio, ::ratio]) ** 2))  
        errors_nxvs.append(err)
        nxvs.append(step)

    nxvs = jnp.array(nxvs)
    errors_nxvs = jnp.array(errors_nxvs)

    log_nxv = jnp.log(nxvs)
    log_err_nxv = jnp.log(errors_nxvs)
    p_nxv, C_nxv = jnp.polyfit(log_nxv, log_err_nxv, 1)

    order_dx = -p_nxv 
    print(f"Estimated convergence order in space: {order_dx:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(nxvs, errors_nxvs, "x-")
    ax.loglog(nxvs, jnp.exp(C_nxv) * nxvs**p_nxv, "--", label=fr"fit: $\mathcal{{O}}(N^{{{p_nxv:.2f}}})$ (Ordre $\sim {order_dx:.1f}$)")

    ax.set_xlabel("Number of points per axis")
    ax.set_ylabel("L2 error")
    ax.set_title(f"Errors in space for ({cfg.inicond.case}) (Kn = {cfg.physics.knudsen:.0e})")

    plt.minorticks_off()
    ax.set_xticks(nxvs)
    ax.set_xticklabels(nxvs)

    ax.legend()
    ax.grid(True, which="both")

    folder = Path(f"plots/errors")
    folder.mkdir(parents=True, exist_ok=True)

    fig.savefig(folder / f"space_error-Kn_{cfg.physics.knudsen:.0e}.png")
    plt.close(fig)

    cfg.grid.nx = nxv_backup
    cfg.grid.nv = nxv_backup
    cfg.grid = make_periodic_grid(cfg.grid)


def plot_errors(cfg):
    # Print device
    backend = jax.default_backend().lower()
    device = "GPU" if backend in ("gpu", "cuda") else "CPU"
    print(f"Device: {device}", flush=True)

    grid = make_periodic_grid(cfg.grid)
    cfg.grid = grid

    #plot_time_error(cfg, src=maxwell_distrib)
    plot_space_error(cfg, src=maxwell_distrib)
