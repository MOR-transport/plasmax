import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

from .config import Config, load_config
from .periodic_grid import make_periodic_grid

def plot_inicond(cfg, inicond: jnp.ndarray, fname: str) -> None:
    cfg.grid = make_periodic_grid(cfg.grid)
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(cfg.grid.X, cfg.grid.V, inicond, shading="auto")
    fig.colorbar(pcm, ax=ax, label=r"$f(x,v)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$")
    ax.set_title(f"Initial condition ({cfg.inicond.case})")
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname)
        plt.close(fig)
    else:
        plt.show()


def landau_damping(x, v, alpha: float, k: float) -> jnp.ndarray:
    return (1 + alpha * jnp.cos(x * k)) / jnp.sqrt(2 * jnp.pi) * jnp.exp(-(v**2) / 2)

def two_stream(x, v, k: float, eps: float, v0: float) -> jnp.ndarray:
    gauss_sum = jnp.exp(-((v - v0) ** 2) / 2) + jnp.exp(-((v + v0) ** 2) / 2)
    return (1 + eps * jnp.cos(k * x)) / (2 * jnp.sqrt(2 * jnp.pi)) * gauss_sum


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

    raise ValueError(f"Unknown inicond.case: {ic.case!r}")
