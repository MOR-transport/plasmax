import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from .config import load_config
from .inicond import get_inicond_exp
from .sim import run_time_loop
from .source import get_filters_exp 
from .adjoint_method import line_search

jax.config.update("jax_enable_x64", True)


def functionnal(cfg, f_hist, fexp):
    sigxv, sigt = get_filters_exp(cfg)
    integrant = (1/2) * (f_hist - fexp)**2 * ( sigxv * sigt )
    return jnp.sum(integrant) * cfg.time.dt * cfg.grid.dx * cfg.grid.dv



def optimize_loop(cfg, line_search_opt=True, tolerance=1E-4):
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

    def func_to_minimize(inicond):
        f, _ = run_time_loop(cfg, inicond=inicond, disabled_save=True, verbose=False)
        return functionnal(cfg, f, f_exp)

    print("\n# ========== Start running the simulation framework ========== %")

    f_hist, _ = run_time_loop(cfg, inicond=inicond, disabled_save=True, verbose=False)
    f_hists.append(f_hist)

    for it in range(1, cfg.optim.Nopt+1):
        print(f"##### Iteration {it:3d} #####")
        
        residuals.append(functionnal(cfg, f_hist, f_exp))
        
        # Halting condition : progress in the objective function
        if it > 1 and residuals[-2]-residuals[-1] < tolerance*residuals[-2]:
            cfg.optim.Nopt = it
            print("Insufficient progression")
            break

        auto_grad = jax.grad(func_to_minimize)(inicond) / ( cfg.grid.dx * cfg.grid.dv )

        norm_grads.append(jnp.sqrt(jnp.sum(auto_grad ** 2) / (cfg.grid.lx * cfg.grid.lv)))
        gradients.append(auto_grad)

        # Halting condition : norm of the gradient
        # if norm_grads[-1] <= tolerance:
        #     cfg.optim.Nopt = it + 1
        #    break

        if line_search_opt:
            inicond, f_hist, _, alpha = line_search(cfg, inicond.copy(),
                                                              f_hist, f_exp,
                                                              auto_grad,
                                                              stepsize)
            stepsize = alpha
            
            alphas.append(alpha)
            iniconds.append(inicond.copy())
            f_hists.append(f_hist)
        else:
            inicond += stepsize * auto_grad
            f_hist, _ = run_time_loop(cfg, inicond=inicond, disabled_save=True, verbose=False)

            alphas.append(stepsize)
            iniconds.append(inicond.copy())
            f_hists.append(f_hist)

        print("\n")

    print("# ============= Simulation framework terminates ============= %")
    return residuals, norm_grads, gradients, alphas, iniconds, f_hists, f_exp


def optimize_autodiff(cfg):
    # Print device
    backend = jax.default_backend().lower()
    device = "GPU" if backend in ("gpu", "cuda") else "CPU"
    print(f"Device: {device}", flush=True)

    residuals, norm_grads, gradients, alphas, iniconds, f_hists, f_exp = optimize_loop(cfg, line_search_opt=True)

    save_path = cfg.paths.data_dir / "auto_diff_optim.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    jnp.savez(save_path, 
              residuals=residuals, 
              norm_grads=norm_grads, 
              gradients=gradients, 
              alphas=alphas, 
              iniconds=iniconds, 
              f_hists=f_hists,
              f_exp=f_exp)
    print(f"Save optimization datas to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Vlasov–Poisson driver (predcorr / NuFI stub).")
    parser.add_argument("--params", type=str, required=True, help="Base yaml config file")
    args = parser.parse_args()

    cfg = load_config(args.params)
    optimize_autodiff(cfg)
