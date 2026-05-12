import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from .config import Config


def landau_damping(x, v, alpha: float, k: float) -> jnp.ndarray:
    return (1 + alpha * jnp.cos(x * k)) / jnp.sqrt(2 * jnp.pi) * jnp.exp(-(v**2) / 2)


def two_stream(x, v, k: float, eps: float, v0: float) -> jnp.ndarray:
    gauss_sum = jnp.exp(-((v - v0) ** 2) / 2) + jnp.exp(-((v + v0) ** 2) / 2)
    return (1 + eps * jnp.cos(k * x)) / (2 * jnp.sqrt(2 * jnp.pi)) * gauss_sum


def bump_on_tail(x, v, k: float, eps: float, vd: float, vt: float, nb: float):
    gauss_1 = jnp.exp(-(v ** 2) / 2) * ( (1-nb) / jnp.sqrt(2 * jnp.pi) )
    gauss_2 = jnp.exp(-((v - vd) ** 2) / (2*vt**2)) * ( nb / (jnp.sqrt(2 * jnp.pi)*vt) )
    return (1 + eps * jnp.cos(k * x)) * (gauss_1 + gauss_2)


def get_inicond(cfg: Config):
    ic = cfg.inicond

    if ic.case == "landau_damping":
        if ic.alpha is None or ic.k is None:
            raise ValueError("landau_damping requires inicond.alpha and inicond.k")
        alpha, k = ic.alpha, ic.k
        return lambda x, v: landau_damping(x, v, alpha, k)
        
    if ic.case == "two_stream":
        if ic.k is None or ic.eps is None or ic.v0 is None:
            raise ValueError("two_stream requires inicond.k, inicond.eps, and inicond.v0")
        k, eps, v0 = ic.k, ic.eps, ic.v0
        return lambda x, v: two_stream(x, v, k, eps, v0)
    
    if ic.case == "bump_on_tail":
        if ic.k is None or ic.eps is None or ic.vd is None or ic.vt is None or ic.nb is None:
            raise ValueError("bump_on_tail requires inicond.k, inicond.eps, inicond.vd, inicond.vt, and inicond.nb")
        k, eps, vd, vt, nb = ic.k, ic.eps, ic.vd, ic.vt, ic.nb
        return lambda x, v: bump_on_tail(x, v, k, eps, vd, vt, nb)

    raise ValueError(f"Unknown inicond.case: {ic.case!r}")


def get_inicond_exp(cfg):

    if cfg.optim.target == "landau_damping":
        if cfg.optim.alpha is None or cfg.optim.k is None:
            raise ValueError("landau_damping requires inicond.alpha and inicond.k")
        alpha, k = cfg.optim.alpha, cfg.optim.k
        return lambda x, v: landau_damping(x, v, alpha, k)
        
    if cfg.optim.target == "two_stream":
        if cfg.optim.k is None or cfg.optim.eps is None or cfg.optim.v0 is None:
            raise ValueError("two_stream requires inicond.k, inicond.eps, and inicond.v0")
        k, eps, v0 = cfg.optim.k, cfg.optim.eps, cfg.optim.v0
        return lambda x, v: two_stream(x, v, k, eps, v0)
    
    if cfg.optim.target == "bump_on_tail":
        if cfg.optim.k is None or cfg.optim.eps is None or cfg.optim.vd is None or cfg.optim.vt is None or cfg.optim.nb is None:
            raise ValueError("bump_on_tail requires inicond.k, inicond.eps, inicond.vd, inicond.vt, and inicond.nb")
        k, eps, vd, vt, nb = cfg.optim.k, cfg.optim.eps, cfg.optim.vd, cfg.optim.vt, cfg.optim.nb
        return lambda x, v: bump_on_tail(x, v, k, eps, vd, vt, nb)

    raise ValueError(f"Unknown inicond.case: {cfg.optim.case!r}")
