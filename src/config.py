from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Grid:
    """Spatial / velocity resolution and domain (arrays filled by ``make_periodic_grid``)."""

    nx: int
    nv: int
    lx: float
    lv: float
    dx: float | None = None
    dv: float | None = None
    x: Any = None
    v: Any = None
    X: Any = None
    V: Any = None
    kx: Any = None
    kx2_inverse: Any = None


@dataclass
class IniCond:
    case: str
    k: float | None = None
    alpha: float | None = None
    eps: float | None = None
    v0: float | None = None
    #: For ``landau_damping``: ``"f0"`` or ``"f1"``; ``None`` matches MATLAB ``params.fini`` (``f1``).
    landau_profile: str | None = None


@dataclass
class Time:
    nt_max: int
    dt: float
    tend: float
    plot_freq: int = 100


@dataclass
class Physics:
    """Single-species Vlasov–Poisson constants"""
    charge: float = -1.0
    mass: float = 1.0
    knudsen: float = 1


@dataclass
class Interp:
    """Advection interpolation settings (e.g. Lagrange ``order``)."""

    order: int = 3


@dataclass
class Optim:
    case: str
    k: float | None = None
    alpha: float | None = None
    eps: float | None = None
    v0: float | None = None
    filter_xv: str = "ones"
    filter_t: str = "ones"
    lr: float = 10
    Nopt: int = 10


@dataclass
class Config:
    inicond: IniCond
    grid: Grid
    time: Time
    optim: Optim
    method: str = "predcorr"
    physics: Physics = field(default_factory=Physics)
    interp: Interp = field(default_factory=Interp)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    return Config(
        inicond=IniCond(**data["inicond"]),
        grid=Grid(**data["grid"]),
        time=Time(**data["time"]),
        method=str(data.get("method", "predcorr")),
        physics=Physics(**(data.get("physics") or {})),
        interp=Interp(**(data.get("interp") or data.get("opt_interp") or {})),
        optim=Optim(**data["optim"])
    )
