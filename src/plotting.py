import math
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from tikzplotlib import save

from .periodic_grid import make_periodic_grid
from .config import Config


def plot_solution(cfg, f: jnp.ndarray, t: float, fname: str) -> None:
    cfg.grid = make_periodic_grid(cfg.grid)
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(cfg.grid.X, cfg.grid.V, f, shading="auto", cmap='turbo')
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
    ax.plot(cfg.grid.x, Efield)
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


def plot_energy(cfg, Efield_hist, format="png"):
    cfg.grid = make_periodic_grid(cfg.grid)

    energy = (1/2) * jnp.sum(Efield_hist ** 2, axis=1) * cfg.grid.dx

    Nt = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt)))) + 1
    t = jnp.linspace(0, cfg.time.tend, Nt)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(t, energy)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\log E_{pe}$")
    ax.set_title(f"Potential electrostatic energy (Kn = {cfg.physics.knudsen:.0e})")

    folder = Path(f"plots/simulation/sim_default")
    if format == "png":
        fig.savefig(folder / "energy.png")
    elif format == "tex":
        save(folder / "energy.tex", encoding="utf-8")
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
        im = ax.imshow(frame_data, extent=limits, origin='lower', cmap='turbo', vmin=val_min, vmax=val_max, animated=True)
        frames.append([im])

    fig.colorbar(im, ax=ax, label='Valeur')

    anim = animation.ArtistAnimation(fig, frames, interval=50)

    anim.save(fname, writer="pillow")
    plt.close(fig)


def plot_source(cfg: Config, source, f: jnp.ndarray, fname):
    cfg.grid = make_periodic_grid(cfg.grid)
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(cfg.grid.X, cfg.grid.V, source(cfg, f), shading="auto", cmap='turbo')
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
    pcm = ax.pcolormesh(cfg.grid.X, cfg.grid.V, inicond, shading="auto", cmap='turbo')
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
    plt.close(fig)


def plot_optimisation(cfg, residual, grad, alphas, inicond, f, f_exp, fname):
    cfg.grid = make_periodic_grid(cfg.grid)

    fig, axs = plt.subplots(2, 3, figsize=(35, 18))

    iterations = jnp.arange(1, len(residual)+1)
    axs[0][0].semilogy(iterations, residual, "x-")
    axs[0][0].set_xlabel("Iteration")
    axs[0][0].set_ylabel("Residual")
    axs[0][0].set_title("Evolution of residuals")
    axs[0][0].set_xticks(iterations)

    axs[0][1].plot(iterations, grad, "x-")
    axs[0][1].set_xlabel("Iteration")
    axs[0][1].set_ylabel(r"$\left\| \nabla J(f) \right\|$")
    axs[0][1].set_title(r"Evolution of $\left\| \nabla J(f) \right\|$")
    axs[0][1].set_xticks(iterations)

    axs[0][2].plot(iterations, alphas, "x-")
    axs[0][2].set_xlabel("Iteration")
    axs[0][2].set_ylabel(r"$\alpha$")
    axs[0][2].set_title(r"Evolution of $\alpha$")
    axs[0][2].set_xticks(iterations)

    vmin = jnp.min(f_exp[-1, :, :])
    vmax = jnp.max(f_exp[-1, :, :])

    pcm_inicond = axs[1][0].pcolormesh(cfg.grid.X, cfg.grid.V, inicond, shading="auto", cmap='turbo', vmin=vmin, vmax=vmax) # norm=mcolors.PowerNorm(gamma=0.3, vmin=vmin, vmax=vmax))
    pcm_f = axs[1][1].pcolormesh(cfg.grid.X, cfg.grid.V, f[-1, :, :], shading="auto", cmap='turbo', vmin=vmin, vmax=vmax) # norm=mcolors.PowerNorm(gamma=0.3, vmin=vmin, vmax=vmax))
    pcm_fexp = axs[1][2].pcolormesh(cfg.grid.X, cfg.grid.V, f_exp[-1, :, :], shading="auto", cmap='turbo', vmin=vmin, vmax=vmax) # norm=mcolors.PowerNorm(gamma=0.3, vmin=vmin, vmax=vmax))

    fig.colorbar(pcm_inicond, ax=axs[1][0])
    fig.colorbar(pcm_f, ax=axs[1][1])
    fig.colorbar(pcm_fexp, ax=axs[1][2])

    axs[1][0].set_xlabel(r"$x$")
    axs[1][0].set_ylabel(r"$v$")
    axs[1][0].set_title("Intiale condition")

    axs[1][1].set_xlabel(r"$x$")
    axs[1][1].set_ylabel(r"$v$")
    axs[1][1].set_title(r"Function $f$")

    axs[1][2].set_xlabel(r"$x$")
    axs[1][2].set_ylabel(r"$v$")
    axs[1][2].set_title(r"Function $f^{exp}$")

    fig.tight_layout()

    if fname is not None:
        if str(fname)[-3:] == "png":
            fig.savefig(fname)
        elif str(fname)[-3:] == "tex":
            save(fname, encoding="utf-8")
        plt.close(fig)
    else:
        plt.show()
