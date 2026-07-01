import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
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

    folder = Path("plots/simulation/sim_default")
    if not folder.exists():
        raise FileNotFoundError("Error : simulation folder doesn't exists")
    if format == "png":
        fig.savefig(folder / "profile.png")
    elif format == "tex":
        save(folder / "profile.tex", encoding="utf-8")
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

    folder = Path("plots/simulation/sim_default")
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

    axs[0][1].semilogy(iterations, grad, "x-")
    axs[0][1].set_xlabel("Iteration")
    axs[0][1].set_ylabel(r"$\log \left\| \nabla J(f) \right\|$")
    axs[0][1].set_title(r"Evolution of the gradient $\left\| \nabla J(f) \right\|$")
    axs[0][1].set_xticks(iterations)

    if jnp.isscalar(alphas):
        alphas = jnp.full(iterations.shape, alphas)

    axs[0][2].plot(iterations, alphas, "x-")
    axs[0][2].set_xlabel("Iteration")
    axs[0][2].set_ylabel(r"$\alpha$")
    axs[0][2].set_title(r"Evolution of the optimization step $\alpha$")
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
    axs[1][0].set_title("Initial condition")

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

def plot_inr_benchmark_comparison(f_sim, f_net, grid_X, grid_V, t, arch_name, save_dir):
    """ 
    Plots f_sim, f_net and the difference side by side
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    diff = f_sim - f_net 
    vmax = max(np.max(f_sim), np.max(f_net))
    vmin = min(np.min(f_sim), np.min(f_net))
    
    abs_max_diff = np.max(np.abs(diff))
    
    #exact matrix (simulation)
    im0 = axs[0].pcolormesh(grid_X, grid_V, f_sim, cmap='turbo', shading="auto", vmin=vmin, vmax=vmax)
    axs[0].set_title(f'Exact Simulation (t={t:.1f})')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('v')
    fig.colorbar(im0, ax=axs[0])
    
    #Network prediction
    im1 = axs[1].pcolormesh(grid_X, grid_V, f_net, cmap='turbo', shading='auto', vmin=vmin, vmax=vmax)
    axs[1].set_title(f'INR Network ({arch_name})')
    axs[1].set_xlabel('x')
    fig.colorbar(im1, ax=axs[1])
    
    #Difference
    im2 = axs[2].pcolormesh(grid_X, grid_V, diff, cmap='coolwarm', shading='auto', vmin=-abs_max_diff, vmax=abs_max_diff)
    axs[2].set_title('Error (f_sim - f_network)')
    axs[2].set_xlabel('x')
    fig.colorbar(im2, ax=axs[2])
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / f'comparison_t{t:.1f}_{arch_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_loss_history(loss_history, t, arch_name, save_dir):
    """Plots the evolution of MSE (in logarithmic scale)"""
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label=arch_name, color='blue', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.title(f'Convergence of {arch_name} network at t={t:.1f}')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    save_path = Path(save_dir) / f'loss_t{t:.1f}_{arch_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_pod_benchmark_comparison(f_sim, f_pod, grid_X, grid_V, t, rank, save_dir):
    """
    Plots a 3-panel comparison (Exact, POD Reconstructed, Signed Error)
    at a specific checkpoint time t, PERFECTLY matching the INR layout and colors.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    # conversion in numpy if they are jax.array
    f_sim_np = np.array(f_sim)
    f_pod_np = np.array(f_pod)
    
    #calculate the error 
    diff = f_sim_np - f_pod_np
    
    vmax = max(np.max(f_sim_np), np.max(f_pod_np))
    vmin = min(np.min(f_sim_np), np.min(f_pod_np))
    
    abs_max_diff = np.max(np.abs(diff))
    if abs_max_diff == 0:  # avoid division by zero in case of perfect match
        abs_max_diff = 1e-5
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Exact Simulation
    im0 = axes[0].pcolormesh(grid_X, grid_V, f_sim_np, cmap='turbo', shading="auto", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Exact Simulation ($t={t:.1f}$)", fontsize=12)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("v")
    fig.colorbar(im0, ax=axes[0])
    
    # POD reconstruction
    im1 = axes[1].pcolormesh(grid_X, grid_V, f_pod_np, cmap='turbo', shading='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"POD Reconstruction (r={rank})", fontsize=12)
    axes[1].set_xlabel("x")
    fig.colorbar(im1, ax=axes[1])
    
    # Error 
    im2 = axes[2].pcolormesh(grid_X, grid_V, diff, cmap='coolwarm', shading='auto', vmin=-abs_max_diff, vmax=abs_max_diff)
    axes[2].set_title("Error (f_sim - f_pod)", fontsize=12)
    axes[2].set_xlabel("x")
    fig.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / f"comparison_t{t:.1f}_r{rank}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   [Benchmark] Saved POD plot to {save_path.name}")    