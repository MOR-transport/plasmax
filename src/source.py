import math

import jax
import jax.numpy as jnp

from .config import Config

jax.config.update("jax_enable_x64", True)


def maxwell_distrib(cfg: Config, f: jnp.ndarray) -> jnp.ndarray:
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


def gaussian_xv(cfg, a=10):    
    v_col = cfg.grid.v[:, None]
    res = (1 / (2*jnp.pi)) * jnp.exp((-v_col**2) / 2*a)
    return jnp.broadcast_to(res, (cfg.grid.nv, cfg.grid.nx))[None, :, :]


def disk_xv(cfg, a=6, b=0, c=1):
    x_row = cfg.grid.x[None, :]
    v_col = cfg.grid.v[:, None]
    return jnp.float64((x_row-a)**2 + (v_col-b)**2 <= c**2)[None, :, :]


def tanh_t(cfg, t_star=0.6):
    Nt = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt)))) + 1
    t = jnp.linspace(0, cfg.time.tend, Nt)
    return 0.5 + 0.5 * jnp.tanh(t - t_star)[:, None, None]


def gate_t(cfg, a=0.2, b=0.5):
    Nt = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt)))) + 1
    t = jnp.linspace(0, cfg.time.tend, Nt)
    return jnp.float64((a <= t) & (t <= b))[:, None, None]


def ones(cfg):
    return jnp.full((cfg.grid.nv, cfg.grid.nx), 1)[None, :, :]


def compute_src(cfg, f_hist, fexp):
    sigxv = ones(cfg)
    sigt = ones(cfg)
    return (f_hist - fexp) * sigxv * sigt


def get_filters_exp(cfg):

    if cfg.optim.filter_xv == "gaussian_xv":
        sigxv = gaussian_xv(cfg)
        
    elif cfg.optim.filter_xv == "disk_xv":
        sigxv = disk_xv(cfg)
    
    elif cfg.optim.filter_xv == "ones":
        sigxv = ones(cfg)

    else:
        raise ValueError(f"Unknown optim.filter_xv.: {cfg.optim.filter_xv!r}")
    
    if cfg.optim.filter_t == "tanh_t":
        sigt = tanh_t(cfg)
        
    elif cfg.optim.filter_t == "gate_t":
        sigt = gate_t(cfg)

    elif cfg.optim.filter_t == "ones":
        sigt = ones(cfg)

    else:
        raise ValueError(f"Unknown optim.filter_xv.: {cfg.optim.filter_t!r}")
    
    return sigxv, sigt
