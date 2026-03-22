import argparse
import math
import time as time_module
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .config import load_config
from .inicond import get_inicond
from .periodic_grid import make_periodic_grid
from .predcorr import predictor_corrector_step
from .physics import compute_density, vpoisson

# Default YAML next to project root `python/`, sibling of `src/` and `params/`.
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "params" / "landau_damping.yaml"


def step(f: jnp.ndarray, cfg, t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single time step"""
    method = cfg.method.lower()

    if method == "predcorr":
        return predictor_corrector_step(f, cfg.grid, cfg, t)
    if method == "nufi":
        raise NotImplementedError(
            "NuFI time stepping is not implemented in Python; use method: predcorr in YAML."
        )
    raise ValueError(f"Unknown cfg.method: {cfg.method!r} (expected predcorr or nufi).")


def run_time_loop(cfg) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Advance ``f`` until ``time >= cfg.time.tend`` or ``nt_max`` steps."""
    grid = make_periodic_grid(cfg.grid)
    cfg.grid = grid
    f0 = get_inicond(cfg)
    f = f0(grid.X, grid.V)
    t = 0.0
    rho = compute_density(f, grid.dv)
    Efield = vpoisson(rho, grid, cfg.physics.charge)

    nt_cap = min(cfg.time.nt_max, int(math.ceil(cfg.time.tend / cfg.time.dt)) + 2)
    tcpu = []

    for it in range(1, nt_cap + 1):

        t0 = time_module.perf_counter()
        f, Efield = step(f, cfg, t)
        t += cfg.time.dt
        tcpu.append(time_module.perf_counter() - t0)

        print(
            f"iter: {it}, time: {t:.6g}, dt: {cfg.time.dt:.6g}, "
            f"cpu_time: {tcpu[-1]:.4f} s",
            flush=True,
        )
        if cfg.time.plot_freq > 0 and it % cfg.time.plot_freq == 0:
            plot_solution(cfg, f, f"plots/solution_{it:04d}.png")
        if t >= cfg.time.tend - 1e-15:
            break
    else:
        print("Warning: reached nt_max before Tend.", flush=True)

    total = sum(tcpu)
    print(
        f"\n=== Simulation complete ===\n"
        f"iterations: {len(tcpu)}, final time: {t:.6g}, total CPU: {total:.3f} s\n"
        f"avg step: {total / max(len(tcpu), 1):.4f} s",
        flush=True,
    )
    return f, Efield, t


def plot_solution(cfg, f: jnp.ndarray,fname: str) -> None:
    cfg.grid = make_periodic_grid(cfg.grid)
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(cfg.grid.X, cfg.grid.V, f, shading="auto")
    fig.colorbar(pcm, ax=ax, label=r"$f(x,v)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$")
    ax.set_title(f"Solution ({cfg.inicond.case})")
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname)
    else:
        plt.show()


def simulate() -> None:
    parser = argparse.ArgumentParser(description="Vlasov–Poisson driver (predcorr / NuFI stub).")
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=_DEFAULT_CONFIG,
        help="Path to YAML (default: params/landau_damping.yaml next to src/)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Print device
    backend = jax.default_backend().lower()
    device = "GPU" if backend in ("gpu", "cuda") else "CPU"
    print(f"Device: {device}", flush=True)

    run_time_loop(cfg)


if __name__ == "__main__":
    simulate()
