import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

from .config import Config
from .periodic_grid import make_periodic_grid


def plot_source(cfg: Config, source, f: jnp.ndarray, fname):
    cfg.grid = make_periodic_grid(cfg.grid)
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(cfg.grid.X, cfg.grid.V, source(f, cfg), shading="auto")
    fig.colorbar(pcm, ax=ax, label=r"$f(x,v)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$")
    ax.set_title(f"Source ({cfg.inicond.case})")
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname)
    else:
        plt.show()


def maxwell_distrib(f: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    n = jnp.sum(f, axis=0) * cfg.grid.dv # jnp.full((256,), 1)
    u = (jnp.sum(f * cfg.grid.v, axis=0) * cfg.grid.dv) / n
    T = (jnp.sum(f * (cfg.grid.v - u[:, None])**2, axis=0) * cfg.grid.dv) / n
    n, u, T, v_row = n[:, None], u[:, None], T[:, None], cfg.grid.v[None, :]
    MF = (n / jnp.sqrt(2 * jnp.pi * T)) * jnp.exp(- ((v_row - u)**2) / (2 * T))
    return (MF - f) / cfg.physics.knudsen
