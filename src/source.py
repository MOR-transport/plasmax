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


def gaussian_v(cfg, a=2):
    v_col = cfg.grid.v[:, None]
    res = jnp.exp((-v_col**2) / (2*a) )
    return jnp.broadcast_to(res, (cfg.grid.nv, cfg.grid.nx))[None, :, :]


def gaussian_x(cfg, a=500, b=30):
    x_row = cfg.grid.x[None, :]
    res = jnp.exp((-(x_row-b)**2) / (2*a))
    return jnp.broadcast_to(res, (cfg.grid.nv, cfg.grid.nx))[None, :, :]


def gate_x(cfg, a=15, b=31.41592653589793):
    x_row = cfg.grid.x[None, :]
    return jnp.float64((a <= x_row) & (x_row <= b))[None, :, :]


def disk_xv(cfg, a=6, b=0, c=1):
    x_row = cfg.grid.x[None, :]
    v_col = cfg.grid.v[:, None]
    return jnp.float64((x_row-a)**2 + (v_col-b)**2 <= c**2)[None, :, :]


def tanh_t(cfg, t_star=0.8):
    Nt = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt)))) + 1
    t = jnp.linspace(0, cfg.time.tend, Nt)
    return  0.5 + 0.5 * jnp.tanh(10 * (t - t_star))[:, None, None]


def gate_t(cfg, a=0.5, b=1):
    Nt = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt)))) + 1
    t = jnp.linspace(0, cfg.time.tend, Nt)
    return jnp.float64((a <= t) & (t <= b))[:, None, None]


def ones(cfg):
    return jnp.full((cfg.grid.nv, cfg.grid.nx), 1)[None, :, :]


def compute_src(cfg, f_hist, fexp):
    sigxv, sigt = get_filters_exp(cfg)
    return (f_hist - fexp) * sigxv * sigt


def density(cfg, f):
    res = jnp.sum(f, axis=2, keepdims=True) * cfg.grid.dx
    return jnp.nan_to_num(res, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

def n(cfg, f):
    res = jnp.sum(f, axis=1, keepdims=True) * cfg.grid.dv
    return jnp.nan_to_num(res, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

def macro_vel(cfg, f):
    res = (1/n(cfg, f)) * jnp.sum(cfg.grid.v * f, axis=1, keepdims=True) * cfg.grid.dv
    return jnp.nan_to_num(res, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

def temperature(cfg, f):
    res = (1/n(cfg, f)) * jnp.sum(((cfg.grid.v -macro_vel(cfg, f))**2 )* f, axis=1, keepdims=True) * cfg.grid.dv
    return jnp.nan_to_num(res, copy=True, nan=0.0, posinf=0.0, neginf=0.0)


def compute_src_control(cfg, f_hist, fexp):
    sigxv, sigt = get_filters_exp(cfg)

    rho_hist = density(cfg, f_hist)
    rho_exp  = density(cfg, fexp)
    T_hist   = temperature(cfg, f_hist)
    T_exp    = temperature(cfg, fexp)
    u_hist   = macro_vel(cfg, f_hist)
    
    v_3d = cfg.grid.v.reshape(1, -1, 1)

    dT_df = ((v_3d - u_hist)**2 - T_hist) / (rho_hist + 1e-12)
    src = (rho_hist - rho_exp) + (T_hist - T_exp) * dT_df
    
    src = jnp.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0)  
    return src * sigxv * sigt


def get_filters_exp(cfg):
    filter_xv_dic = {
        "gaussian_x": gaussian_x,
        "gaussian_v": gaussian_v,
        "disk_xv": disk_xv,
        "ones": ones,
        "gate_x": gate_x,
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
