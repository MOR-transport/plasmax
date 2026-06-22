"""Minimal autodiff + L-BFGS-B demo for talks."""

import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from src.adjoint_method import functionnal
from src.config import load_config
from src.inicond import get_inicond_exp
from src.sim import run_time_loop

jax.config.update("jax_enable_x64", True)

folder = Path("optimization/talk_optim")
if folder.exists():
    shutil.rmtree(folder)
folder.mkdir(parents=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_progress(cfg, losses, inicond, f_hist, f_exp, fname):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    ax = axs[0, 0]
    ax.semilogy(range(1, len(losses) + 1), losses, "x-")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\mathcal{L}[f]$")
    ax.set_title(r"Loss $\mathcal{L}[f]$")

    vmin = float(jnp.min(f_exp[0]))
    vmax = float(jnp.max(f_exp[0]))
    ax = axs[0, 1]
    pcm = ax.pcolormesh(cfg.grid.X, cfg.grid.V, inicond, shading="auto",
                        cmap="turbo", vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$")
    ax.set_title(r"Current $f_0$")

    vmin = float(jnp.min(f_exp[-1]))
    vmax = float(jnp.max(f_exp[-1]))

    ax = axs[1, 0]
    pcm = ax.pcolormesh(cfg.grid.X, cfg.grid.V, f_hist[-1], shading="auto",
                        cmap="turbo", vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$")
    ax.set_title(r"$f(t_\mathrm{end})$")

    ax = axs[1, 1]
    pcm = ax.pcolormesh(cfg.grid.X, cfg.grid.V, f_exp[-1], shading="auto",
                        cmap="turbo", vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$")
    ax.set_title(r"$f^\mathrm{exp}(t_\mathrm{end})$")

    fig.tight_layout()
    fig.savefig(fname, dpi=100)
    plt.close(fig)


def plot_all(data_folder=folder):
    cfg = load_config("params/landau_damping.yaml")
    cfg.time.plot_freq = 0

    f_exp = generate_target(cfg)

    files = sorted(data_folder.glob("data_*.npz"))
    print(f"Plotting {len(files)} snapshots …", flush=True)
    for path in files:
        data = np.load(path)
        inicond = jnp.array(data["inicond"], dtype=jnp.float64)
        losses = list(data["losses"])
        f_hist, _ = run_time_loop(cfg, inicond=inicond, verbose=False)
        it = int(path.stem.split("_")[1])
        plot_progress(cfg, losses, inicond, f_hist, f_exp,
                      data_folder / f"plot_{it:04d}.png")
        print(f"  saved plot_{it:04d}.png", flush=True)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def generate_target(cfg):
    """Forward run with the experimental initial condition."""
    inicond_exp = get_inicond_exp(cfg)(cfg.grid.X, cfg.grid.V)
    print("Running target simulation …", flush=True)
    f_exp, _ = run_time_loop(cfg, inicond=inicond_exp, verbose=False)
    return f_exp


def make_loss(cfg, f_exp):
    """J(f0): run the simulation and compare to the target trajectory."""

    def loss(inicond):
        f_hist, _ = run_time_loop(cfg, inicond=inicond, auto_grad=True, verbose=False)
        return functionnal(cfg, f_hist, f_exp)

    return loss


def make_scipy_objective(loss_and_grad, shape, cache):
    """SciPy expects a 1D vector; JAX works on the 2D grid (nv, nx).
    Stores the last (value, grad) in cache so the callback reads it for free."""

    def objective(x_flat):
        inicond = jnp.array(x_flat, dtype=jnp.float64).reshape(shape)
        value, grad = loss_and_grad(inicond)
        cache["value"] = float(value)
        cache["grad"] = grad
        cache["inicond"] = inicond
        cache["losses"].append(float(value))
        return float(value), np.asarray(grad, dtype=np.float64).ravel()

    return objective


def make_callback(cache):
    state = {"it": 0}

    def on_step(_x_flat):
        it = state["it"]
        print(f"  iter {it:3d} | J = {cache['value']:.6e}")
        np.savez(
            folder / f"data_{it:04d}.npz",
            losses=np.asarray(cache["losses"]),
            inicond=np.asarray(cache["inicond"]),
        )
        state["it"] += 1

    return on_step


def main():
    cfg = load_config("params/landau_damping.yaml")
    cfg.time.plot_freq = 0

    f_exp = generate_target(cfg)
    inicond0 = jnp.exp(-cfg.grid.V**2 / 2) / jnp.sqrt(2 * jnp.pi)
    shape = inicond0.shape

    loss_fn = make_loss(cfg, f_exp)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    cache = {"losses": []}
    objective = make_scipy_objective(loss_and_grad, shape, cache)
    x0 = np.asarray(inicond0, dtype=np.float64).ravel()

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        callback=make_callback(cache),
        options={"maxiter": 500},
    )

    inicond_opt = jnp.array(result.x, dtype=jnp.float64).reshape(shape)
    print(f"Done: {result.message}")
    print(f"Final J = {result.fun:.6e}")
    return inicond_opt


if __name__ == "__main__":
    main()
    plot_all()
