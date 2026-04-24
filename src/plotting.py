from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .periodic_grid import make_periodic_grid


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
        fig.savefig(fname)
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
        fig.savefig(fname)
        plt.close(fig)
    else:
        plt.show()


def plot_profile(cfg, f_hist: jnp.ndarray, nb_profiles=3) -> None:
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
    fig.savefig(folder / f"profile_Kn_{cfg.physics.knudsen:.0e}.png")
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

    for frame_data in hist:
        im = ax.imshow(frame_data, cmap='viridis', vmin=val_min, vmax=val_max, animated=True)
        frames.append([im])

    fig.colorbar(im, ax=ax, label='Valeur')

    anim = animation.ArtistAnimation(fig, frames, interval=50)

    folder = Path(f"plots/simulation/inicond-{cfg.inicond.case}_dt-{cfg.time.dt}_nx-{cfg.grid.nx}_nv-{cfg.grid.nv}")
    folder.mkdir(parents=True, exist_ok=True)

    anim.save(folder / fname, writer="pillow")
    plt.close(fig)
