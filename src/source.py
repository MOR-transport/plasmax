import math

import jax
import jax.numpy as jnp

from .config import Config

jax.config.update("jax_enable_x64", True)

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


def compute_src(cfg, f_hist):
    fexp = get_fexp(cfg)
    sigxv = get_sigxv(cfg)
    sigt = get_sigt(cfg)
    return (f_hist - fexp) * sigxv * sigt


def functionnal(cfg, f_hist):
    fexp = get_fexp(cfg)
    sigxv = get_sigxv(cfg)
    sigt = get_sigt(cfg)
    integrant = (1/2) * (f_hist - fexp)**2 * sigxv * sigt
    return jnp.sum(integrant) * cfg.time.dt * cfg.grid.dx * cfg.grid.dv


########################################################################################################
##################################       SPACE FILTERS       ###########################################
########################################################################################################


def gaussian_xv(cfg):    
    v_col = cfg.grid.v[:, None]
    res = (1 / (2*jnp.pi)) * jnp.exp((-v_col**2) / 20)
    return jnp.broadcast_to(res, (cfg.grid.nv, cfg.grid.nx))[None, :, :]


def disk(cfg, a=6, b=0, c=1):
    x_row = cfg.grid.x[None, :]
    v_col = cfg.grid.v[:, None]
    return jnp.float64((x_row-a)**2 + (v_col-b)**2 <= c**2)[None, :, :]


########################################################################################################
###################################       TIME FILTERS       ###########################################
########################################################################################################


def filter_t(cfg, t_star=0.6):
    Nt = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt)))) + 1
    t = jnp.linspace(0, cfg.time.tend, Nt)
    return 0.5 + 0.5 * jnp.tanh(t - t_star)[:, None, None]


def gate(cfg, a=0.2, b=0.5):
    Nt = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt)))) + 1
    t = jnp.linspace(0, cfg.time.tend, Nt)
    return jnp.float64((a <= t) & (t <= b))[:, None, None]


########################################################################################################
#####################################       FUNC EXP       #############################################
########################################################################################################


def gaussian_fexp(cfg, alpha=100, beta=0.1):
    Nt = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt)))) + 1   
    v_mat = cfg.grid.v[None, :, None]
    res = (alpha / (2*jnp.pi)) * jnp.exp((-v_mat**2) / (2*beta))    
    return jnp.broadcast_to(res, (Nt, cfg.grid.nv, cfg.grid.nx))


########################################################################################################
########################################################################################################
########################################################################################################


def get_fexp(cfg: Config):
    if cfg.optim.fexp == "gaussian":
        return gaussian_fexp(cfg)
    raise ValueError(f"Unknown optim.fexp: {cfg.optim.fexp!r}")


def get_sigxv(cfg: Config):
    if cfg.optim.sigxv == "gaussian":
        return gaussian_xv(cfg)
    elif cfg.optim.sigxv == "disk":
        return disk(cfg)
    raise ValueError(f"Unknown optim.sigxv: {cfg.optim.sigxv!r}")


def get_sigt(cfg: Config):
    if cfg.optim.sigt == "gate":
        return gate(cfg)
    raise ValueError(f"Unknown optim.sigt: {cfg.optim.sigt!r}")
