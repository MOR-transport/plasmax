"""Predictor–corrector scheme."""

from __future__ import annotations

import jax.numpy as jnp

from .advect import advect, advect_with_source
from .config import Config, Grid
from .physics import compute_density, vpoisson
from .source import maxwell_distrib


def predictor_corrector_step(
    f: jnp.ndarray,
    grid: Grid,
    cfg: Config,
    time: float,
    src = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    del time  
    dt = cfg.time.dt
    ord_ = cfg.interp.order
    q_m = cfg.physics.charge / cfg.physics.mass

    if src is not None:
        advection = lambda f, efield, grid, dt, order: advect_with_source(f, efield, grid, dt, order, lambda p: src(cfg, p))
    else:
        advection = advect

    rho0 = compute_density(f, float(grid.dv))
    e0 = vpoisson(rho0, grid, cfg.physics.charge)
    f12 = advection(f, q_m * e0, grid, dt / 2.0, ord_)

    rho12 = compute_density(f12, float(grid.dv))
    e12 = vpoisson(rho12, grid, cfg.physics.charge)
    f_new = advection(f, q_m * e12, grid, dt, ord_)

    return f_new, e12
