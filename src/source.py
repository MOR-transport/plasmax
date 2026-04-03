import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from .config import Config

def maxwell_distrib(f: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    n = jnp.sum(f, axis=0) * cfg.grid.dv # jnp.full((256,), 1)
    u = (jnp.sum(f * cfg.grid.v, axis=0) * cfg.grid.dv) / n
    T = (jnp.sum(f * (cfg.grid.v - u[:, None])**2, axis=0) * cfg.grid.dv) / n
    n, u, T, v_row = n[:, None], u[:, None], T[:, None], cfg.grid.v[None, :]
    MF = (n / jnp.sqrt(2 * jnp.pi * T)) * jnp.exp(- ((v_row - u)**2) / (2 * T))
    return (MF - f) / cfg.physics.knudsen
