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


def gaussian_xv(cfg, a=5):    
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
    filter_xv_dic = {
        "gaussian_xv": gaussian_xv,
        "disk_xv": disk_xv,
        "ones": ones,
    }
    filter_t_dic = {
        "tanh_t": tanh_t,
        "gate_t": gate_t,
        "ones": ones,
    }

    try:
        func_xv = filter_xv_dic[cfg.optim.filter_xv]
        sigxv = func_xv(cfg)
    except KeyError:
        raise ValueError(f"Unknown optim.filter_xv: {cfg.optim.filter_xv!r}")

    try:
        func_t = filter_t_dic[cfg.optim.filter_t]
        sigt = func_t(cfg)
    except KeyError:
        raise ValueError(f"Unknown optim.filter_t: {cfg.optim.filter_t!r}")

    return sigxv, sigt
