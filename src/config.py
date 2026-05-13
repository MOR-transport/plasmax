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
class Restart:
    """Configuration for restarting from a saved state"""
    enabled: bool = False 
    file: str | None = None


@dataclass(frozen=True)
class Paths:
    """
    Sets up the paths <save_dir> as root:
    - plot_dir: <save_dir>/plots
    - data_dir: <save_dir>/data
    """

    save_dir: Path
    plot_dir: Path
    data_dir: Path

    @classmethod
    def from_case(cls, case: str, save_dir: str | Path | None = None) -> Paths:
        save = Path(save_dir) if save_dir is not None else Path("results") / case
        return cls(save_dir=save, plot_dir=save / "plots", data_dir=save / "data")


@dataclass
class Config:
    inicond: IniCond
    grid: Grid
    time: Time
    paths: Paths
    method: str = "predcorr"
    physics: Physics = field(default_factory=Physics)
    interp: Interp = field(default_factory=Interp)
    restart: Restart = field(default_factory=Restart)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    
    # manage the restart section (may be absent)
    restart_data = data.get("restart", {})
    restart = Restart(**restart_data) if restart_data else Restart()

    inicond = IniCond(**data["inicond"])
    save_dir = (data.get("io") or {}).get("save_dir")
    paths = Paths.from_case(inicond.case, save_dir)

    return Config(
        inicond=inicond,
        grid=Grid(**data["grid"]),
        time=Time(**data["time"]),
        paths=paths,
        method=str(data.get("method", "predcorr")),
        physics=Physics(**(data.get("physics") or {})),
        interp=Interp(**(data.get("interp") or data.get("opt_interp") or {})),
        restart=restart,
    )
