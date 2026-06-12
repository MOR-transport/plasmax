import math
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Compatibility patch for tikzplotlib
import matplotlib.backends.backend_pgf as backend_pgf
if not hasattr(backend_pgf, "common_texification") and hasattr(backend_pgf, "_tex_escape"):
    backend_pgf.common_texification = backend_pgf._tex_escape
from tikzplotlib import save

from .config import Config


import scienceplots
plt.style.use(['science'])

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 22,
})


def plot_solution(cfg, f: jnp.ndarray, t: float, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(cfg.grid.X, cfg.grid.V, f, shading="auto", cmap='turbo')
    fig.colorbar(pcm, ax=ax, label=r"$f(x,v)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$")
    ax.set_title(f"Solution at t = {t:.2f}")
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
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cfg.grid.x, Efield)
    ax.set_xlabel(r"$x$")
    ax.set_title(f"Electric field at t = {t:.2f}")
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
    fig, axs = plt.subplots(1, 2, figsize=(25, 10))

    axs[0].set_xlabel(r"$v$")
    axs[0].set_ylabel(r"$f(x=" + str(cfg.grid.dx * cfg.grid.nx//2) + ", v, t= .)$")
    axs[0].set_title("Profile in v")

    axs[1].set_xlabel(r"$x$")
    axs[1].set_ylabel(r"$f(x, v=" + str(cfg.grid.dv * cfg.grid.nv//2) + ", t= .)$")
    axs[1].set_title("Profile in x")

    nt = f_hist.shape[0]
    for it in range(1, nb_profiles+1):
        t_index = it * nt // (nb_profiles+1)
        t = t_index * cfg.time.dt
        axs[0].plot(cfg.grid.v, f_hist[t_index, :, cfg.grid.nx//2], label=f"t = {t:.2f}")
        axs[1].plot(cfg.grid.x, f_hist[t_index, cfg.grid.nv//2, :], label=f"t = {t:.2f}")

    axs[0].legend()
    axs[1].legend()

    if format == "png":
        fig.savefig(str(cfg.paths.plot_dir / f"profile.{format}"))
    elif format == "tex":
        save(str(cfg.paths.plot_dir / f"profile.{format}"), encoding="utf-8")
    plt.close(fig)


def plot_energy(cfg, Efield_hist, format="png"):
    energy = (1/2) * jnp.sum(Efield_hist ** 2, axis=1) * cfg.grid.dx

    Nt = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt)))) + 1
    t = jnp.linspace(0, cfg.time.tend, Nt)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(t, energy)

    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"$E_\text{pot}$")
    ax.set_title("Potential energy")

    if format == "png":
        fig.savefig(str(cfg.paths.plot_dir / f"energy.{format}"))
    elif format == "tex":
        save(str(cfg.paths.plot_dir / f"energy.{format}"), encoding="utf-8")
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
        im = ax.imshow(frame_data, extent=limits, origin='lower', cmap='turbo',
                       vmin=val_min, vmax=val_max, animated=True)
        frames.append([im])

    fig.colorbar(im, ax=ax, label='Valeur')

    anim = animation.ArtistAnimation(fig, frames, interval=50)

    anim.save(fname, writer="pillow")
    plt.close(fig)


def plot_source(cfg: Config, source, f: jnp.ndarray, fname):
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
    fig, axs = plt.subplots(2, 3, figsize=(35, 18))

    iterations = jnp.arange(1, len(residual)+1)
    axs[0][0].semilogy(iterations, residual, "x-")
    axs[0][0].set_xlabel("Iteration")
    axs[0][0].set_ylabel(r"$\log J(f) $")
    axs[0][0].set_title(r"Evolution of the functional $ J(f) $")
    axs[0][0].set_xticks(iterations)
    axs[0][0].set_yticks(residual)
    axs[0][0].set_yticklabels(residual)

    axs[0][1].semilogy(iterations, grad, "x-")
    axs[0][1].set_xlabel("Iteration")
    axs[0][1].set_ylabel(r"$\log \left\| \nabla J(f) \right\|$")
    axs[0][1].set_title(r"Evolution of the gradient $\left\| \nabla J(f) \right\|$")
    axs[0][1].set_xticks(iterations)
    axs[0][1].set_yticks(grad)
    axs[0][1].set_yticklabels(grad)

    if jnp.isscalar(alphas):
        alphas = jnp.full(iterations.shape, alphas)

    axs[0][2].plot(iterations, alphas, "x-")
    axs[0][2].set_xlabel("Iteration")
    axs[0][2].set_ylabel(r"$\alpha$")
    axs[0][2].set_title(r"Evolution of the optimization step $\alpha$")
    axs[0][2].set_xticks(iterations)
    axs[0][2].set_yticks(alphas)
    axs[0][2].set_yticklabels(alphas)

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
    axs[1][0].set_title("Initial condition")

    axs[1][1].set_xlabel(r"$x$")
    axs[1][1].set_ylabel(r"$v$")
    axs[1][1].set_title(r"Function $f$ at final time")

    axs[1][2].set_xlabel(r"$x$")
    axs[1][2].set_ylabel(r"$v$")
    axs[1][2].set_title(r"Function $f^{exp}$ at final time")

    fig.tight_layout()

    if fname is not None:
        if str(fname)[-3:] == "png":
            fig.savefig(fname)
        elif str(fname)[-3:] == "tex":
            save(fname, encoding="utf-8")
        plt.close(fig)
    else:
        plt.show()


def plot_grad_info(cfg, inicond, adj_grad, fname):
    fig, axs = plt.subplots(1, 3, figsize=(40, 15))

    axs[0].plot(cfg.grid.x, inicond[cfg.grid.nv // 2, :], label="Initial condition")
    axs[0].plot(cfg.grid.x, adj_grad[cfg.grid.nv // 2, :], label="Gradient")
    axs[0].legend()
    axs[0].set_title(f"Comparison of initial condition and gradient at v = {-cfg.grid.lv + cfg.grid.dv * (cfg.grid.nv // 2):.2f}")
    axs[0].set_xlabel(r"$x$")

    axs[1].plot(cfg.grid.v, inicond[:, cfg.grid.nx // 2], label="Initial condition")
    axs[1].plot(cfg.grid.v, adj_grad[:, cfg.grid.nx // 2], label="Gradient")
    axs[1].legend()
    axs[1].set_title(f"Comparison of initial condition and gradient at x = {cfg.grid.dx * (cfg.grid.nx // 2):.2f}")
    axs[1].set_xlabel(r"$v$")

    pcm_grad = axs[2].pcolormesh(cfg.grid.X, cfg.grid.V, adj_grad, shading="auto", cmap='turbo')
    fig.colorbar(pcm_grad, ax=axs[2])
    axs[2].set_xlabel(r"$x$")
    axs[2].set_ylabel(r"$v$")

    fig.tight_layout()

    if fname is not None:
        if str(fname)[-3:] == "png":
            fig.savefig(fname)
        elif str(fname)[-3:] == "tex":
            save(fname, encoding="utf-8")
        plt.close(fig)
    else:
        plt.show()


def compare_auto_grad(cfg, adj_grad, auto_grad, fname):
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    
    pcm_adj_grad = axs[0][0].pcolormesh(cfg.grid.X, cfg.grid.V, adj_grad, shading="auto", cmap='turbo')
    pcm_auto_grad = axs[0][1].pcolormesh(cfg.grid.X, cfg.grid.V, auto_grad, shading="auto", cmap='turbo')
    pcm_diff_grad = axs[1][1].pcolormesh(cfg.grid.X, cfg.grid.V, jnp.abs(adj_grad-auto_grad), shading="auto", cmap='turbo')

    fig.colorbar(pcm_adj_grad, ax=axs[0][0])
    fig.colorbar(pcm_auto_grad, ax=axs[0][1])
    fig.colorbar(pcm_diff_grad, ax=axs[1][1])

    axs[0][0].set_title("Adjoint method differentiation")
    axs[0][1].set_title("Automatic differentiation")
    axs[1][1].set_title("Difference between")

    axs[1][0].plot(cfg.grid.x, adj_grad[:, cfg.grid.nv //2], label="Adjoint method")
    axs[1][0].plot(cfg.grid.x, auto_grad[:, cfg.grid.nv //2], label="Auto differentiation")
    
    axs[1][0].set_title(f"Comparison at v index {cfg.grid.nv // 2}")
    axs[1][0].legend()

    fig.tight_layout()
    if fname is not None:
        if str(fname)[-3:] == "png":
            fig.savefig(fname)
        elif str(fname)[-3:] == "tex":
            save(fname, encoding="utf-8")
        plt.close(fig)
    else:
        plt.show()


def plot_opt_source(cfg, src, fname):
    fig, axs = plt.subplots(1, 3, figsize=(40, 15))

    pcm_src = axs[0].pcolormesh(cfg.grid.X, cfg.grid.V, src[0], shading="auto", cmap='turbo')
    fig.colorbar(pcm_src, ax=axs[0])
    axs[0].set_title(f"Optimization source at time t = {0:.2f}")

    Nt = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt)))) + 1
    pcm_src = axs[1].pcolormesh(cfg.grid.X, cfg.grid.V, src[Nt // 2], shading="auto", cmap='turbo')
    fig.colorbar(pcm_src, ax=axs[1])
    axs[1].set_title(f"Optimization source at time t = {(Nt // 2)*cfg.time.dt:.2f}")

    pcm_src = axs[2].pcolormesh(cfg.grid.X, cfg.grid.V, src[-1], shading="auto", cmap='turbo')
    fig.colorbar(pcm_src, ax=axs[2])
    axs[2].set_title(f"Optimization source at time t = {cfg.time.tend:.2f}")

    fig.tight_layout()
    if fname is not None:
        if str(fname)[-3:] == "png":
            fig.savefig(fname)
        elif str(fname)[-3:] == "tex":
            save(fname, encoding="utf-8")
        plt.close(fig)
    else:
        plt.show()