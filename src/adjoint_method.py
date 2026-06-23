import math
import argparse
from pathlib import Path
import shutil
import time as time_module

import jax
import jax.numpy as jnp

from .config import load_config
from .inicond import get_inicond, get_inicond_exp
from .sim import run_time_loop
from .advect import advect_with_source_hist
from .plotting import make_anim_2d, plot_optimisation, plot_grad_info, compare_auto_grad, plot_opt_source
from .source import get_filters_exp, compute_src 

jax.config.update("jax_enable_x64", True)


def functionnal(cfg, f_hist, fexp):
    sigxv, sigt = get_filters_exp(cfg)
    integrant = (1/2) * (f_hist - fexp)**2 * ( sigxv * sigt )
    return jnp.sum(integrant) * cfg.time.dt * cfg.grid.dx * cfg.grid.dv


def run_time_loop_adjoint(cfg, Efield, src, verbose=False):
    grid = cfg.grid

    f = jnp.zeros((grid.nv, grid.nx), dtype=jnp.float64)
    t = cfg.time.tend
    cfg.time.dt = - cfg.time.dt

    nt_cap = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt))))
    tcpu = []

    f_hist = jnp.empty((nt_cap, grid.nv, grid.nx), dtype=jnp.float64)
    f_hist = f_hist.at[-1, :, :].set(f)

    for it in range(1, nt_cap + 1):

        t0 = time_module.perf_counter()

        f = advect_with_source_hist(f, Efield[-it, :], grid, cfg.time.dt, cfg.interp.order, src, -it)

        t += cfg.time.dt
        tcpu.append(time_module.perf_counter() - t0)

        f_hist = f_hist.at[-it, :, :].set(f)
        if verbose:
            print(f"iter: {it:3d}, time: {t:4.1f}, dt: {cfg.time.dt:.6g}, "
                  f"cpu_time: {tcpu[-1]:.4f} s", flush=True)

    cfg.time.dt = - cfg.time.dt
    total = sum(tcpu)
    if verbose:
        print(
            f"\n=== Simulation complete ===\n"
            f"iterations: {len(tcpu)}, final time: {t:.6g}, total CPU: {total:.3f} s\n"
            f"avg step: {total / max(len(tcpu), 1):.4f} s",
            flush=True,
        )

    return f_hist


def line_search_step(cfg, alpha, inicond, grad):

    inicond_new = inicond - alpha * grad
    f_new, Efield_new = run_time_loop(cfg, inicond=inicond_new, disabled_save=True, verbose=False)

    return inicond_new, f_new, Efield_new, alpha


def line_search(cfg, inicond, f, fexp, grad, alpha_init, m=1e-4, theta=0.5):
    alpha = alpha_init

    if alpha <= 1e-8:
        return line_search_step(cfg, alpha, inicond, grad)

    J = lambda f: functionnal(cfg, f, fexp)

    while True:
        print(f"\nTRY ALPHA = {alpha}")

        inicond_tmp, f_new, Efield_new, alpha = line_search_step(cfg, alpha,
                                                                 inicond, grad)

        armijo_cond = J(f_new) <= J(f) - m*alpha*jnp.sum(grad**2)
        if armijo_cond:
            return inicond_tmp, f_new, Efield_new, alpha

        alpha *= theta
        if alpha <= 1e-5:
            return line_search_step(cfg, alpha, inicond, grad)


def adjoint(cfg, line_search_opt=True, tolerance=1E-4, format="png"):
    cfg.time.plot_freq = 0

    inicond_exp = get_inicond_exp(cfg)(cfg.grid.X, cfg.grid.V)
    f_exp, _ = run_time_loop(cfg, inicond=inicond_exp, disabled_save=True, verbose=False)

    inicond_initiale = jnp.zeros((cfg.grid.nv, cfg.grid.nx))
    inicond = inicond_initiale.copy()
    stepsize = cfg.optim.lr

    residuals = []
    norm_grads = []
    gradients = []
    alphas = []
    iniconds = [inicond.copy()]
    f_hists = []
    opt_srcs = []

    print("\n# ========== Start running the simulation framework ========== %")

    f_hist, Efield_hist = run_time_loop(cfg, inicond=inicond, disabled_save=True, verbose=False)
    f_hists.append(f_hist)

    for it in range(1, cfg.optim.Nopt+1):
        print(f"##### Iteration {it:3d} #####")
        
        residuals.append(functionnal(cfg, f_hist, f_exp))
        
        # Halting condition : progress in the objective function
        if it > 1 and residuals[-2]-residuals[-1] < tolerance*residuals[-2]:
            cfg.optim.Nopt = it
            print("Insufficient progression")
            break

        src = compute_src(cfg, f_hist, f_exp)
        opt_srcs.append(src)
        adj_hist = run_time_loop_adjoint(cfg, Efield_hist, src, verbose=False)
        grad = - adj_hist[0, :, :]

        gradients.append(grad.copy())
        norm_grads.append(jnp.sqrt(jnp.sum(grad ** 2) / (cfg.grid.lx * cfg.grid.lv)))
        
        # Halting condition : norm of the gradient
        # if norm_grads[-1] <= tolerance:
        #     cfg.optim.Nopt = it + 1
        #    break

        if line_search_opt:
            inicond, f_hist, Efield_hist, alpha = line_search(cfg, inicond.copy(),
                                                              f_hist, f_exp,
                                                              grad,
                                                              stepsize)
            stepsize = alpha

            alphas.append(alpha)
            iniconds.append(inicond.copy())
            f_hists.append(f_hist)
        else:
            inicond -= cfg.optim.lr * grad
            f_hist, Efield_hist = run_time_loop(cfg, inicond=inicond, disabled_save=True, verbose=False)

            alphas.append(stepsize)
            iniconds.append(inicond.copy())
            f_hists.append(f_hist)

        print("\n")

    print("# ============= Simulation framework terminates ============= %")
    return residuals, norm_grads, gradients, alphas, iniconds, f_hists, f_exp, opt_srcs


def optimize(cfg):
    # Print device
    backend = jax.default_backend().lower()
    device = "GPU" if backend in ("gpu", "cuda") else "CPU"
    print(f"Device: {device}", flush=True)

    residuals, norm_grads, gradients, alphas, iniconds, f_hists, f_exp, opt_srcs = adjoint(cfg, line_search_opt=True)

    save_path = cfg.paths.data_dir / "adj_diff_optim.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    jnp.savez(save_path, 
              residuals=residuals, 
              norm_grads=norm_grads, 
              gradients=gradients, 
              alphas=alphas, 
              iniconds=iniconds, 
              f_hists=f_hists,
              opt_srcs=opt_srcs,
              f_exp=f_exp)
    print(f"Save optimization datas to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Vlasov–Poisson driver (predcorr / NuFI stub).")
    parser.add_argument("--params", type=str, required=True, help="Base yaml config file")
    args = parser.parse_args()

    cfg = load_config(args.params)
    optimize(cfg)
