from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import jax.numpy as jnp
import yaml


@dataclass
class Grid:
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

    def __post_init__(self) -> None:
        x = jnp.linspace(0, self.lx, self.nx, endpoint=False)
        v = jnp.linspace(-self.lv, self.lv, self.nv, endpoint=False)
        self.dx = float(x[1] - x[0])
        self.dv = float(v[1] - v[0])
        self.x = x
        self.v = v
        self.X, self.V = jnp.meshgrid(self.x, self.v, indexing="xy")
        self.kx = 2 * jnp.pi * jnp.fft.fftfreq(self.nx, d=self.dx)
        self.kx2 = self.kx * self.kx
        self.kx2_inverse = self.kx2.at[0].set(1.0)
        self.kx2_inverse = 1 / self.kx2_inverse


@dataclass
class IniCond:
    case: str
    k: float | None = None
    alpha: float | None = None
    eps: float | None = None
    v0: float | None = None
    vd: float | None = None
    vt: float | None = None
    nb: float | None = None
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
    knudsen: float = None
    source = None


@dataclass
class Interp:
    """Advection interpolation settings (e.g. Lagrange ``order``)."""
    order: int = 3


@dataclass
class Restart:
    """Configuration for restarting from a saved state"""
    enabled: bool = False
    file: str | None = None


@dataclass
class IoConfig:
    save_dir: str | None = None
    compression_method: str | None = None
    dt_save: float | None = None
    restart: Restart = field(default_factory=Restart)


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
    def from_case(cls, testcase: str, save_dir: str | Path | None = None) -> Paths:
        save = Path(save_dir) if save_dir is not None else Path("results") / testcase
        return cls(save_dir=save, plot_dir=save / "plots", data_dir=save / "data")


@dataclass
class Optim:
    target: str
    k: float | None = None
    alpha: float | None = None
    eps: float | None = None
    v0: float | None = None
    vd: float | None = None
    vt: float | None = None
    nb: float | None = None
    filter_xv: str = "ones"
    filter_t: str = "ones"
    lr: float = 10
    Nopt: int = 10


@dataclass
class Config:
    inicond: IniCond
    grid: Grid
    time: Time
    paths: Paths
    io: IoConfig
    optim: Optim
    method: str = "predcorr"
    physics: Physics = field(default_factory=Physics)
    interp: Interp = field(default_factory=Interp)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    io_data = data.get("io", {})
    restart_data = io_data.get("restart", {})
    restart = Restart(**restart_data) if restart_data else Restart()

    io_cfg = IoConfig(
        save_dir=io_data.get("save_dir"),
        compression_method=io_data.get("compression_method"),
        dt_save=io_data.get("dt_save"),
        restart=restart
    )

    inicond_data = data.get("inicond")
    if inicond_data:
        inicond = IniCond(**inicond_data)
    else:
        params_path = Path(path)
        inferred_case = params_path.stem
        inicond = IniCond(case=inferred_case)

    paths = Paths.from_case(inicond.case, io_cfg.save_dir)

    optim_data = data.get("optim", {})
    optim = Optim(**optim_data) if optim_data else Optim(target=inicond.case)
    cfg = Config(inicond=inicond, grid=Grid(**data["grid"]),
                 time=Time(**data["time"]),
                 paths=paths, io=io_cfg, optim=optim,
                 method=str(data.get("method", "predcorr")),
                 physics=Physics(**(data.get("physics") or {})),
                 interp=Interp(**(data.get("interp") or data.get("opt_interp") or {})))
    # test if inicondition fits periodically in domain
    if cfg.inicond.k is not None:
        assert jnp.abs(cfg.grid.lx - 2*jnp.pi/cfg.inicond.k) < 1e-12, f"grid.lx = {cfg.grid.lx} != 2*pi/k = {2*jnp.pi/cfg.inicond.k}"
    return cfg
