import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from abc import ABC, abstractmethod

from .config import Config


class TestCase(ABC):
    @abstractmethod
    def get_initcond(self, x, v) -> jnp.ndarray:
        """Return the initial condition of the test case"""
        pass


class LandauDamping(TestCase):
    def __init__(self, alpha: float, k: float):
        if alpha is None or k is None:
            raise ValueError("landau_damping requires alpha and k")
        self.alpha = alpha
        self.k = k

    def get_initcond(self, x, v) -> jnp.ndarray:
        return (1 + self.alpha * jnp.cos(x * self.k)) / jnp.sqrt(2 * jnp.pi) * jnp.exp(-(v**2) / 2)


class TwoStream(TestCase):
    def __init__(self, k: float, eps: float, v0: float):
        if k is None or eps is None or v0 is None:
            raise ValueError("two_stream requires k, eps, and v0")
        self.k = k
        self.eps = eps
        self.v0 = v0

    def get_initcond(self, x, v) -> jnp.ndarray:
        gauss_sum = jnp.exp(-((v - self.v0) ** 2) / 2) + jnp.exp(-((v + self.v0) ** 2) / 2)
        return (1 + self.eps * jnp.cos(self.k * x)) / (2 * jnp.sqrt(2 * jnp.pi)) * gauss_sum


class BumpOnTail(TestCase):
    def __init__(self, k: float, eps: float, vd: float, vt: float, nb: float):
        if k is None or eps is None or vd is None or vt is None or nb is None:
            raise ValueError("bump_on_tail requires k, eps, vd, vt, and nb")
        self.k = k
        self.eps = eps
        self.vd = vd
        self.vt = vt
        self.nb = nb

    def get_initcond(self, x, v) -> jnp.ndarray:
        gauss_1 = jnp.exp(-(v ** 2) / 2) * ( (1 - self.nb) / jnp.sqrt(2 * jnp.pi) )
        gauss_2 = jnp.exp(-((v - self.vd) ** 2) / (2 * self.vt**2)) * ( self.nb / (jnp.sqrt(2 * jnp.pi) * self.vt) )
        return (1 + self.eps * jnp.cos(self.k * x)) * (gauss_1 + gauss_2)


def _create_experiment(ic) -> TestCase:
    expname = getattr(ic, "case", getattr(ic, "target", None))
    
    if expname == "landau_damping":
        return LandauDamping(ic.alpha, ic.k)
    elif expname == "two_stream":
        return TwoStream(ic.k, ic.eps, ic.v0)
    elif expname == "bump_on_tail":
        return BumpOnTail(ic.k, ic.eps, ic.vd, ic.vt, ic.nb)
    else:
        raise ValueError(f"Unknown case: {expname!r}")


def get_inicond(cfg: Config):
    case_name = cfg.inicond.case
    
    #if the name contains "_segmented"
    if case_name.endswith("_segmented"):
        exp_name_base = case_name.replace("_segmented", "")
        if exp_name_base == "landau_damping":
            experiment = LandauDamping(cfg.inicond.alpha, cfg.inicond.k)
        elif exp_name_base == "two_stream":
            experiment = TwoStream(cfg.inicond.k, cfg.inicond.eps, cfg.inicond.v0)
        elif exp_name_base == "bump_on_tail":
            experiment = BumpOnTail(cfg.inicond.k, cfg.inicond.eps, cfg.inicond.vd, cfg.inicond.vt, cfg.inicond.nb)
        else:
            raise ValueError(f"Unknown base case for segmented run: {exp_name_base!r}")
        return experiment.get_initcond
    
    #if it is the normal simulation case
    experiment = _create_experiment(cfg.inicond)
    return experiment.get_initcond


def get_inicond_exp(cfg: Config):
    experiment = _create_experiment(cfg.optim)
    return experiment.get_initcond
