import math
import time as time_module
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp

from .inicond import get_inicond
from .periodic_grid import make_periodic_grid
from .predcorr import predictor_corrector_step
from .physics import compute_density, vpoisson
from .source import maxwell_distrib
from .plotting import plot_solution, plot_Efield, plot_profile, plot_inicond, plot_energy
from .diagnostics import measure

jax.config.update("jax_enable_x64", True)

jax.config.update("jax_enable_x64", True)

# Default YAML next to project root `python/`, sibling of `src/` and `params/`.
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "params" / "landau_damping.yaml"


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
        plt.close(fig)
    else:
        plt.show()

def plot_Efield(cfg, Efield: jnp.ndarray,fname: str) -> None:
    cfg.grid = make_periodic_grid(cfg.grid)
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(cfg.grid.x, Efield)
    ax.set_xlabel(r"$x$")
    ax.set_title(f"Electric field ({cfg.inicond.case})")
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname)
        plt.close(fig)
    else:
        plt.show()

def plot_profile(cfg, f: jnp.ndarray, t:float, axs: jax.typing.ArrayLike):
    cfg.grid = make_periodic_grid(cfg.grid)
    axs[0].plot(cfg.grid.v, f[:, cfg.grid.nx//2], label=f"t = {t:.2f}")
    axs[0].set_xlabel(r"$v$")
    axs[0].set_ylabel(r"$f(x=" + str(cfg.grid.dx * cfg.grid.nx//2) + ", v, t= .)$")
    axs[0].set_title(f"Profile in v ({cfg.inicond.case})")
    axs[1].plot(cfg.grid.x, f[cfg.grid.nv//2, :], label=f"t = {t:.2f}")
    axs[1].set_xlabel(r"$x$")
    axs[1].set_ylabel(r"$f(x, v=" + str(cfg.grid.dv * cfg.grid.nv//2) + ", t= .)$")
    axs[1].set_title(f"Profile in x ({cfg.inicond.case})")


def plot_time_error(cfg):
    dt_backup = cfg.time.dt
    dt_ref = jnp.float32(jnp.log2(cfg.time.dt))
    if dt_ref - jnp.ceil(dt_ref) != 0:
        raise ValueError("Invalid time step: dt must be a negative power of 2")
    cfg.time.plot_freq = 0
    
    dt_ref = jnp.int32(dt_ref)
    print(f"{-dt_ref} computations needed for error in time")

    f_ref, _ = run_time_loop(cfg)
    f_ref = f_ref[-1, :, :]
    print(f_ref)

    errors_dts = []
    dts = []

    powers_dts = jnp.arange(dt_ref + 1, 1)
    for power in powers_dts:
        dt = 2.0 ** power
        cfg.time.dt = dt

        f, _ = run_time_loop(cfg)
        f = f[-1, :, :]

        err = jnp.sqrt(jnp.sum((f - f_ref) ** 2))  # L2 error
        errors_dts.append(err)
        dts.append(dt)

    dts = jnp.array(dts)
    errors_dts = jnp.array(errors_dts)

    log_dt = jnp.log(dts)
    log_err_dt = jnp.log(errors_dts)
    p_dt, C_dt = jnp.polyfit(log_dt, log_err_dt, 1)

    print(f"Estimated convergence order in time: {p_dt:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(dts, errors_dts, "x-")
    ax.loglog(dts, jnp.exp(C_dt) * dts**p_dt, "--", label=fr"fit: $O(\Delta t^{{{p_dt:.2f}}})$")

    ax.set_xlabel("time step Δt")
    ax.set_ylabel("L2 error")
    ax.set_title(f"Errors in time for ({cfg.inicond.case}) (Kn = {cfg.physics.knudsen})")

    ax.legend()
    ax.grid(True, which="both")
    cfg.paths.plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.paths.plot_dir / "time_error.png")

    cfg.time.dt = dt_backup


def plot_space_error(cfg):
    if cfg.grid.nx != cfg.grid.nv:
        raise ValueError("Invalid space step: dx and dv must be equal")
    
    nxv_backup = cfg.grid.nx
    nxv_ref = jnp.float32(jnp.log2(cfg.grid.nx))
    if nxv_ref - jnp.floor(nxv_ref) != 0:
        raise ValueError("Invalid space step: dx and dx must be a power of 2")
    cfg.time.plot_freq = 0

    nxv_ref = jnp.int32(nxv_ref)
    print(f"{nxv_ref - 5} computations needed for error in space")

    # 1. Calcul de la solution de référence (Grille fine)
    f_ref, _ = run_time_loop(cfg)
    f_ref = f_ref[-1, :, :]

    errors_nxvs = []
    nxvs = []

    powers_nxvs = jnp.arange(nxv_ref-1, 5, -1)
    for power in powers_nxvs:
        step = 2 ** power
        cfg.grid.nx = step
        cfg.grid.nv = step
        cfg.grid = make_periodic_grid(cfg.grid)

        f, _ = run_time_loop(cfg)
        f = f[-1, :, :]

        ratio = nxv_backup // step

        err = jnp.sqrt(jnp.mean((f - f_ref[::ratio, ::ratio]) ** 2))  
        errors_nxvs.append(err)
        nxvs.append(step)

    nxvs = jnp.array(nxvs)
    errors_nxvs = jnp.array(errors_nxvs)

    log_nxv = jnp.log(nxvs)
    log_err_nxv = jnp.log(errors_nxvs)
    p_nxv, C_nxv = jnp.polyfit(log_nxv, log_err_nxv, 1)

    order_dx = -p_nxv 
    print(f"Estimated convergence order in space: {order_dx:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(nxvs, errors_nxvs, "x-")
    ax.loglog(nxvs, jnp.exp(C_nxv) * nxvs**p_nxv, "--", label=fr"fit: $\mathcal{{O}}(N^{{{p_nxv:.2f}}})$ (Ordre $\sim {order_dx:.1f}$)")

    ax.set_xlabel("number of cells (N)")
    ax.set_ylabel("L2 error")
    ax.set_title(f"Errors in space for ({cfg.inicond.case})")

    ax.legend()
    ax.grid(True, which="both")
    cfg.paths.plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.paths.plot_dir / "space_error.png")

    cfg.grid.nx = nxv_backup
    cfg.grid.nv = nxv_backup
    cfg.grid = make_periodic_grid(cfg.grid)


def step(f: jnp.ndarray, cfg, t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single time step"""
    method = cfg.method.lower()

    if method == "predcorr":
        return predictor_corrector_step(f, cfg.grid, cfg, t, src)
    if method == "nufi":
        raise NotImplementedError(
            "NuFI time stepping is not implemented in Python; use method: predcorr in YAML."
        )
    raise ValueError(f"Unknown cfg.method: {cfg.method!r} (expected predcorr or nufi).")


def run_time_loop(cfg, src=None, inicond=None, format="png", nb_profile=0) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Advance ``f`` until ``time >= cfg.time.tend`` or ``nt_max`` steps."""
    grid = make_periodic_grid(cfg.grid)
    cfg.grid = grid
    
    # f and t initialization 
    if cfg.restart.enabled and cfg.restart.file:
        restart_path = Path(cfg.restart.file) 
        if not restart_path.exists():
            raise FileNotFoundError(f"Restart file '{restart_path}' not found.")
        
        #loading .npz archive
        data = jnp.load(restart_path)
        f = data['f']
        t = float(data["t"])
        it_offset = int(data["it"])
        
        #check the compatibility of the dimensions
        if f.shape != (grid.nv, grid.nx):
            raise ValueError(f"Restart file shape {f.shape} != grid shape ({grid.nv}, {grid.nx})")
        
        print(f"Restarting from {restart_path} at t = {t:.6g}")
    
    else:
        f0 = get_inicond(cfg)
        f = f0(grid.X, grid.V)
        t = 0.0
        it_offset = 0 #no offset if we start at 0
        print(f"Starting from analytical initial condition")
        
    rho = compute_density(f, grid.dv)
    Efield = vpoisson(rho, grid, cfg.physics.charge)
    
    if cfg.time.plot_freq > 0:
        folder = Path("plots/simulation/sim_default")
        folder_f = folder / "iterations/solution"
        folder_E = folder / "iterations/Efield"

        if folder.exists() and folder.is_dir():
            shutil.rmtree(folder)

        folder_f.mkdir(parents=True)
        folder_E.mkdir(parents=True)

        plot_inicond(cfg, f, folder_f / f"solution_{0:04d}.{format}")
        plot_Efield(cfg, Efield, t, folder_E / f"Efield_{0:04d}.{format}")

    #number of iterations from the initial time
    remaining = max(0.0, cfg.time.tend - t)
    nt_cap = min(cfg.time.nt_max, int(math.ceil(remaining / cfg.time.dt)))
    if nt_cap <=0: 
        print("Nothing to simulate (tend already reached).")
        return jnp.empty((0, grid.nv, grid.nx)), jnp.empty((0, grid.nx))        
    
    tcpu = []
    
    if nb_profile > 0:
        fig, axs = plt.subplots(2, 1, figsize=(16, 10))

    cfg.paths.plot_dir.mkdir(parents=True, exist_ok=True)

    f_hist = jnp.empty((nt_cap+1, grid.nv, grid.nx), dtype=jnp.float64)
    f_hist = f_hist.at[0, :, :].set(f)

    Efield_hist = jnp.empty((nt_cap+1, grid.nx), dtype=jnp.float64)
    Efield_hist = Efield_hist.at[0, :].set(Efield)
    
    global_it = it_offset 

    for it in range(1, nt_cap + 1):

        t0 = time_module.perf_counter()
        global_it = it + it_offset
        f, Efield = step(f, cfg, t, src)
        t += cfg.time.dt
        tcpu.append(time_module.perf_counter() - t0)

        f_hist = f_hist.at[it, :, :].set(f)
        Efield_hist = Efield_hist.at[it, :].set(Efield)
        
        measure(cfg, f, Efield, global_it, t)

        print(f"iter: {it}, time: {t:.6g}, dt: {cfg.time.dt:.6g}, "f"cpu_time: {tcpu[-1]:.4f} s", flush=True)
        if cfg.time.plot_freq > 0 and it % cfg.time.plot_freq == 0:
            plot_solution(cfg, f, str(cfg.paths.plot_dir / f"solution_{it:04d}.{format}"))
            plot_Efield(cfg, Efield, str(cfg.paths.plot_dir / f"Efield_{it:04d}.{format}"))
        if nb_profile > 0:
            if cfg.time.plot_freq > 0 and it % ((nt_cap-2) // nb_profile) == 0:
                plot_profile(cfg, f, t, axs)
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

    if nb_profile > 0:
        axs[0].legend()
        axs[1].legend()
        fig.savefig(cfg.paths.plot_dir / "profile.png")

    # save final distribution function and time to an .npz archive
    save_path = cfg.paths.data_dir / "f_final.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    jnp.savez(save_path, f=f, t=t, it=global_it)
    print(f"Save state (f,t,it) to {save_path}")

    return f_hist, Efield_hist


def simulate(cfg) -> None:
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

    f_hist, Efield_hist = run_time_loop(cfg, src=maxwell_distrib, format="png")
    plot_profile(cfg, f_hist, format="png", nb_profiles=3)
    plot_energy(cfg, Efield_hist)