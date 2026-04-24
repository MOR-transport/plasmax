import jax
import math

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

from .config import Config
from .periodic_grid import make_periodic_grid

import jax
jax.config.update("jax_enable_x64", True)


def plot_source(cfg: Config, source, f: jnp.ndarray, fname):
    cfg.grid = make_periodic_grid(cfg.grid)
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(cfg.grid.X, cfg.grid.V, source(cfg, f), shading="auto")
    fig.colorbar(pcm, ax=ax, label=r"$f(x,v)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$")
    ax.set_title(f"Source ({cfg.inicond.case})")
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname)
    else:
        plt.show()


def maxwell_distrib(cfg, f: jnp.ndarray) -> jnp.ndarray:
    v_col = cfg.grid.v[:, None]
    dv = cfg.grid.dv

    n = jnp.sum(f, axis=0) * dv
    u = (jnp.sum(f * v_col, axis=0) * dv) / n
    T = (jnp.sum(f * (v_col - u[None, :])**2, axis=0) * dv) / n

    n_row = n[None, :]
    u_row = u[None, :]
    T_row = T[None, :]

    MF = (n_row / jnp.sqrt(2 * jnp.pi * T_row)) * jnp.exp(- ((v_col - u_row)**2) / (2 * T_row))
    
    return (MF - f) / cfg.physics.knudsen


def filter_xv(cfg):    
    v_col = cfg.grid.v[:, None]
    res = (1 / (2*jnp.pi)) * jnp.exp((-v_col**2) / 20)
    
    return jnp.broadcast_to(res, (cfg.grid.nv, cfg.grid.nx))


def filter_t(cfg, t_star=0.6):
    Nt = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt)))) + 1
    t = jnp.linspace(0, cfg.time.tend, Nt)
    return 0.5 + 0.5 * jnp.tanh(t - t_star)


def f_exp(cfg, alpha=100, beta=0.1):
    Nt = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt)))) + 1   
    v_mat = cfg.grid.v[None, :, None]
    res = (alpha / (2*jnp.pi)) * jnp.exp((-v_mat**2) / (2*beta))    
    return jnp.broadcast_to(res, (Nt, cfg.grid.nv, cfg.grid.nx))
