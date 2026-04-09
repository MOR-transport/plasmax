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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    del time  
    dt = cfg.time.dt
    ord_ = cfg.interp.order
    q_m = cfg.physics.charge / cfg.physics.mass

    rho0 = compute_density(f, float(grid.dv))
    e0 = vpoisson(rho0, grid, cfg.physics.charge)
    f12 = advect_with_source(f, q_m * e0, grid, dt / 2.0, ord_, source=lambda fun: maxwell_distrib(fun,cfg))

    rho12 = compute_density(f12, float(grid.dv))
    e12 = vpoisson(rho12, grid, cfg.physics.charge)
    f_new = advect_with_source(f, q_m * e12, grid, dt, ord_, source=lambda fun: maxwell_distrib(fun,cfg))

    return f_new, e12
