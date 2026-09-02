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
from .advect import advect_with_source_hist, _adv_x, _adv_v, advect
from .advect_adj import advect_adj, _adv_x_adj, _adv_v_adj, compute_mu
from .plotting import make_anim_2d, plot_optimisation, plot_grad_info, compare_auto_grad, plot_opt_source
from .source import get_filters_exp, compute_src, density, temperature, compute_src_control
from .physics import compute_density_adj, vpoisson_adj, compute_density, vpoisson

jax.config.update("jax_enable_x64", True)

def functional(cfg, f_hist, fexp):
    sigxv, sigt = get_filters_exp(cfg)
    integrant = (1/2) * (f_hist - fexp)**2 * ( sigxv * sigt )
    return jnp.sum(integrant) * cfg.time.dt * cfg.grid.dx * cfg.grid.dv

def functional_control(cfg, f_hist, fexp):
    sigxv, sigt = get_filters_exp(cfg)
    integrant = (1/2) * ( (density(cfg, f_hist) - density(cfg, fexp))**2 +  (temperature(cfg, f_hist) - temperature(cfg, fexp))**2) * ( sigxv * sigt )
    return jnp.sum(integrant) * cfg.time.dt * cfg.grid.dx * cfg.grid.dv

def run_time_loop_adjoint_dto(cfg, f_hist, Efield_hist, src):
    grid = cfg.grid
    nt_cap = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt)))) + 1
    dt = cfg.time.dt 
    ord_ = cfg.interp.order
    q_m = cfg.physics.charge / cfg.physics.mass

    lambda_hist = jnp.zeros((nt_cap, grid.nv, grid.nx), dtype=jnp.float64)
    lambda_hist = lambda_hist.at[-1, :, :].set(- src[-1] * cfg.time.dt)

    for n in range(nt_cap - 2, -1, -1):
        f_primal = f_hist[n, :, :]
        f_star = _adv_x(f_primal, grid, dt / 2.0)
        lambda_next = lambda_hist[n + 1, :, :]

        rho0 = compute_density(f_primal, float(grid.dv))
        e0 = vpoisson(rho0, grid, cfg.physics.charge)
        
        # ==========================================
        # 1. CORRECTOR
        # ==========================================
        lam_f_corr, lam_e12_scaled = advect_adj(lambda_next, f_star, q_m * Efield_hist[n], grid, dt, ord_)
        lam_e12 = lam_e12_scaled * q_m
        # ==========================================

        # ==========================================
        # 2. PREDICTOR
        # ==========================================
        lam_rho12 = vpoisson_adj(lam_e12, grid, cfg.physics.charge)
        lam_f12 = compute_density_adj(lam_rho12, grid)
        
        lam_f_pred, lam_e0_scaled = advect_adj(lam_f12, f_star, q_m * e0, grid, dt / 2.0, ord_)
        lam_e0 = lam_e0_scaled * q_m
        # ==========================================

        lam_rho0 = vpoisson_adj(lam_e0, grid, cfg.physics.charge)
        lam_f_rho0 = compute_density_adj(lam_rho0, grid)

        lambda_f = lam_f_corr + lam_f_pred + lam_f_rho0 - src[n] * cfg.time.dt
        lambda_hist = lambda_hist.at[n, :, :].set(lambda_f)

    return lambda_hist


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
    inicond_new = jnp.maximum(inicond_new, 0.0)
    f_new, Efield_new = run_time_loop(cfg, inicond=inicond_new, disabled_save=True, verbose=False)

    return inicond_new, f_new, Efield_new, alpha


def line_search(cfg, inicond, functional_to_minimize, f, fexp, grad, alpha_init, m=1e-4, theta=0.5):
    alpha = alpha_init

    if alpha <= 1e-8:
        return line_search_step(cfg, alpha, inicond, grad)

    J = lambda f: functional_to_minimize(cfg, f, fexp)

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


def adjoint(cfg, compute_grad, functional_to_minimize, line_search_opt=True, tolerance=1E-4, format="png"):
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
    lbdas = []

    def func_to_minimize(inicond):
        f, _ = run_time_loop(cfg, inicond=inicond, disabled_save=True, verbose=False)
        return functional_to_minimize(cfg, f, f_exp)

    print("\n# ========== Start running the simulation framework ========== %")

    f_hist, Efield_hist = run_time_loop(cfg, inicond=inicond, disabled_save=True, verbose=False)
    f_hists.append(f_hist)

    for it in range(1, cfg.optim.Nopt+1):
        print(f"##### Iteration {it:3d} #####")
        
        residuals.append(functional_to_minimize(cfg, f_hist, f_exp))
        print("FUNCTIONAL : ", residuals[-1])
        
        # Halting condition : progress in the objective function
        if it > 1 and residuals[-2]-residuals[-1] < tolerance*residuals[-2]:
            cfg.optim.Nopt = it
            print("Insufficient progression")
            break

        if functional_to_minimize is functional:
            src = compute_src(cfg, f_hist, f_exp)
        elif functional_to_minimize is functional_control:
            src = compute_src_control(cfg, f_hist, f_exp)
        else:
            print("Error : functional to minimize not recognised")
            return

        opt_srcs.append(src)

        if compute_grad is run_time_loop_adjoint:
            c_grad = lambda cfg, f_hist, Efield, src: compute_grad(cfg, Efield, src)
        if compute_grad is run_time_loop_adjoint_dto:
                c_grad = lambda cfg, f_hist, Efield, src: compute_grad(cfg, f_hist, Efield, src)

        adj_hist = c_grad(cfg, f_hist, Efield_hist, src)
        lbdas.append(adj_hist)
        grad = - adj_hist[0, :, :]

        # auto_grad = jax.grad(func_to_minimize)(inicond) / ( cfg.grid.dx * cfg.grid.dv )
        # print("Comparison with auto differentiation : ", jnp.max(jnp.abs(auto_grad - grad)))

        gradients.append(grad.copy())
        norm_grads.append(jnp.sqrt(jnp.sum(grad ** 2) / (cfg.grid.lx * cfg.grid.lv)))
        
        # Halting condition : norm of the gradient
        # if norm_grads[-1] <= tolerance:
        #     cfg.optim.Nopt = it + 1
        #    break

        if line_search_opt:
            inicond, f_hist, Efield_hist, alpha = line_search(cfg, inicond.copy(), functional_to_minimize,
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
    return residuals, norm_grads, gradients, alphas, iniconds, f_hists, f_exp, opt_srcs, lbdas


def optimize(cfg):
    # Print device
    backend = jax.default_backend().lower()
    device = "GPU" if backend in ("gpu", "cuda") else "CPU"
    print(f"Device: {device}", flush=True)

    compute_grad = run_time_loop_adjoint
    functional_to_minimize = functional_control
    residuals, norm_grads, gradients, alphas, iniconds, f_hists, f_exp, opt_srcs, lbdas = adjoint(cfg, compute_grad=compute_grad, functional_to_minimize=functional_to_minimize, line_search_opt=True)

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
              f_exp=f_exp,
              lbdas=lbdas)
    print(f"Save optimization datas to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Vlasov–Poisson driver (predcorr / NuFI stub).")
    parser.add_argument("--params", type=str, required=True, help="Base yaml config file")
    args = parser.parse_args()

    cfg = load_config(args.params)
    optimize(cfg)
