import math
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tikzplotlib import save

from .periodic_grid import make_periodic_grid
from .config import Config
from .source import get_fexp, get_sigxv, get_sigt


def plot_solution(cfg, f: jnp.ndarray, t: float, fname: str) -> None:
    cfg.grid = make_periodic_grid(cfg.grid)
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(cfg.grid.X, cfg.grid.V, f, shading="auto")
    fig.colorbar(pcm, ax=ax, label=r"$f(x,v)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$")
    ax.set_title(f"Solution at t = {t:.2f}  ({cfg.inicond.case}) (Kn = {cfg.physics.knudsen:.0e})")
    fig.tight_layout()
    if fname is not None:
        if str(fname)[-3:] == "png":
            fig.savefig(fname)
        elif str(fname)[-3:] == "tex":
            save(fname, encoding="utf-8")
        plt.close(fig)
    else:
        plt.show()


def plot_Efield(cfg, Efield: jnp.ndarray, t, fname: str) -> None:
    cfg.grid = make_periodic_grid(cfg.grid)
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(cfg.grid.x, Efield)
    ax.set_xlabel(r"$x$")
    ax.set_title(f"Electric field at t = {t:.2f} ({cfg.inicond.case}) (Kn = {cfg.physics.knudsen:.0e})")
    fig.tight_layout()
    if fname is not None:
        if str(fname)[-3:] == "png":
            fig.savefig(fname)
        elif str(fname)[-3:] == "tex":
            save(fname, encoding="utf-8")
        plt.close(fig)
    else:
        plt.show()


def plot_profile(cfg, f_hist: jnp.ndarray, format="png", nb_profiles=3) -> None:
    cfg.grid = make_periodic_grid(cfg.grid)
    
    fig, axs = plt.subplots(1, 2, figsize=(25, 10))
    
    axs[0].set_xlabel(r"$v$")
    axs[0].set_ylabel(r"$f(x=" + str(cfg.grid.dx * cfg.grid.nx//2) + ", v, t= .)$")
    axs[0].set_title(f"Profile in v ({cfg.inicond.case}) (Kn = {cfg.physics.knudsen:.0e})")
    
    axs[1].set_xlabel(r"$x$")
    axs[1].set_ylabel(r"$f(x, v=" + str(cfg.grid.dv * cfg.grid.nv//2) + ", t= .)$")
    axs[1].set_title(f"Profile in x ({cfg.inicond.case}) (Kn = {cfg.physics.knudsen:.0e})")

    nt = f_hist.shape[0]
    for it in range(1, nb_profiles+1):
        t_index = it * nt // (nb_profiles+1)
        t = t_index * cfg.time.dt
        axs[0].plot(cfg.grid.v, f_hist[t_index, :, cfg.grid.nx//2], label=f"t = {t:.2f}")
        axs[1].plot(cfg.grid.x, f_hist[t_index, cfg.grid.nv//2, :], label=f"t = {t:.2f}")

    axs[0].legend()
    axs[1].legend()

    folder = Path(f"plots/simulation/sim_default")
    if not folder.exists():
        raise("Error : simulation folder doesn't exists")
    if format == "png":
        fig.savefig(folder / "profile.png")
    elif format == "tex":
        save(folder / "profile.tex", encoding="utf-8")
    plt.close(fig)


def make_anim_1d(cfg, hist, fname):
    fig, ax = plt.subplots()
    frames = []
    
    ax.set_xlim(0, cfg.grid.lx)
    ax.set_xlabel('x')
    ax.set_ylabel('Valeur')

    for frame_data in hist:
        line, = ax.plot(cfg.grid.x, frame_data, color='blue', animated=True)
        frames.append([line])

    anim = animation.ArtistAnimation(fig, frames, interval=50)

    folder = Path(f"plots/simulation/inicond-{cfg.inicond.case}_dt-{cfg.time.dt}_nx-{cfg.grid.nx}_nv-{cfg.grid.nv}")
    folder.mkdir(parents=True, exist_ok=True)

    anim.save(folder / fname, writer="pillow")
    plt.close(fig)


def make_anim_2d(cfg, hist, fname):
    val_min = jnp.min(hist)
    val_max = jnp.max(hist)

    fig, ax = plt.subplots()
    frames = []

    limits = [0, cfg.grid.lx, -cfg.grid.lv, cfg.grid.lv]

    for frame_data in hist:
        im = ax.imshow(frame_data, extent=limits, origin='lower', cmap='viridis', vmin=val_min, vmax=val_max, animated=True)
        frames.append([im])

    fig.colorbar(im, ax=ax, label='Valeur')

    anim = animation.ArtistAnimation(fig, frames, interval=50)

    anim.save(fname, writer="pillow")
    plt.close(fig)


def plot_source(cfg: Config, source, f: jnp.ndarray, fname):
    cfg.grid = make_periodic_grid(cfg.grid)
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(cfg.grid.X, cfg.grid.V, source(cfg, f), shading="auto")
    fig.colorbar(pcm, ax=ax, label=r"$f(x,v)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$")
    ax.set_title(f"Source ({cfg.inicond.case})")
    fig.tight_layout()
    if fname is not None:
        if str(fname)[-3:] == "png":
            fig.savefig(fname)
        elif str(fname)[-3:] == "tex":
            save(fname, encoding="utf-8")
    else:
        plt.show()


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
        if str(fname)[-3:] == "png":
            fig.savefig(fname)
        elif str(fname)[-3:] == "tex":
            save(fname, encoding="utf-8")
    else:
        plt.show()


def plot_f_exp(cfg, fname):
    cfg.grid = make_periodic_grid(cfg.grid)

    fexp = get_fexp(cfg)
    sigxv = get_sigxv(cfg)
    sigt = get_sigt(cfg)

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    pcm_fe = axs[1].pcolormesh(cfg.grid.X, cfg.grid.V, fexp[0, :, :], shading="auto")
    pcm_sigxv = axs[2].pcolormesh(cfg.grid.X, cfg.grid.V, sigxv.squeeze(), shading="auto")

    fig.colorbar(pcm_fe, ax=axs[1], label=r"$f^{exp}$")
    fig.colorbar(pcm_sigxv, ax=axs[2], label=r"$\sigma_{xv}$")

    axs[1].set_xlabel(r"$x$")
    axs[1].set_ylabel(r"$v$")
    axs[1].set_title(r"$f^{exp}$")

    axs[2].set_xlabel(r"$x$")
    axs[2].set_ylabel(r"$v$")
    axs[2].set_title(r"$\sigma_{xv}$")
    
    Nt = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt)))) + 1
    axs[0].plot(jnp.linspace(0, cfg.time.tend, Nt), sigt.squeeze())
    axs[0].set_xlabel("$t$")
    axs[0].set_title(r"$\sigma_t$")

    fig.tight_layout()
    if fname is not None:
        if str(fname)[-3:] == "png":
            fig.savefig(fname)
        elif str(fname)[-3:] == "tex":
            save(fname, encoding="utf-8")
    else:
        plt.show()


def plot_optimisation(cfg, init_inicond, final_inicond, residuals, fname):
    cfg.grid = make_periodic_grid(cfg.grid)

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    pcm_init = axs[1].pcolormesh(cfg.grid.X, cfg.grid.V, init_inicond, shading="auto")
    pcm_final = axs[2].pcolormesh(cfg.grid.X, cfg.grid.V, final_inicond, shading="auto")

    fig.colorbar(pcm_init, ax=axs[1], label=r"$init(x,v)$")
    fig.colorbar(pcm_final, ax=axs[2], label=r"$final(x,v)$")

    axs[1].set_xlabel(r"$x$")
    axs[1].set_ylabel(r"$v$")
    axs[1].set_title("Intiale inicond")

    axs[2].set_xlabel(r"$x$")
    axs[2].set_ylabel(r"$v$")
    axs[2].set_title("Final inicond")

    axs[0].plot(jnp.arange(cfg.optim.Nopt), residuals, "-x")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Residual")
    axs[0].set_title("Evolution of residuals")
    axs[0].set_xticks(jnp.arange(cfg.optim.Nopt))
    axs[0].set_xticklabels(jnp.arange(cfg.optim.Nopt))

    fig.tight_layout()
    if fname is not None:
        if str(fname)[-3:] == "png":
            fig.savefig(fname)
        elif str(fname)[-3:] == "tex":
            save(fname, encoding="utf-8")
    else:
        plt.show()
