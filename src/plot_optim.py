import math
import argparse

import jax.numpy as jnp

from .config import load_config
from .plotting import plot_optimisation, plot_grad_info, plot_opt_source, plot_norm_lambda

def plot_datas(cfg, datas, suffix, format):
    Nt = min(cfg.time.nt_max, int(math.ceil(abs(cfg.time.tend / cfg.time.dt)))) + 1

    residuals = list(datas["residuals"])
    norm_grads = list(datas["norm_grads"])
    gradients = list(datas["gradients"])
    alphas = list(datas["alphas"])
    iniconds = list(datas["iniconds"])
    f_hists = list(datas["f_hists"])
    f_exp = datas["f_exp"]
    lbdas = datas["lbdas"]

    if suffix == "adj":
        opt_srcs = list(datas["opt_srcs"])

    print(f"### {len(residuals)} optimization step recovered ###")

    plot_optimisation(cfg, residuals[:0], [], [], iniconds[0], f_hists[0], f_exp, cfg.paths.plot_dir / f"opt_{0:04d}_{suffix}.{format}")
    for it in range(1, len(iniconds)):
        plot_optimisation(cfg, residuals[:it], norm_grads[:it], alphas[:it], iniconds[it], f_hists[it], f_exp, cfg.paths.plot_dir / f"opt_{it:04d}_{suffix}.{format}")
        plot_grad_info(cfg, iniconds[it], gradients[it-1], cfg.paths.plot_dir / f"grad_{it:04d}_{suffix}.{format}")
        #plot_norm_lambda(cfg, lbdas[it-1], cfg.paths.plot_dir / f"norm_lambda_{it:04d}_{suffix}.{format}")
        if suffix == "adj":
            plot_opt_source(cfg, opt_srcs[it-1], cfg.paths.plot_dir / f"opt_src_{it:04d}_{suffix}.{format}")


def get_datas(cfg):
    adj_save_path = cfg.paths.data_dir / "adj_diff_optim.npz"
    auto_save_path = cfg.paths.data_dir / "auto_diff_optim.npz"

    try: 
        adj_datas = jnp.load(adj_save_path)
        plot_datas(cfg, adj_datas, "adj", "png")
    except FileNotFoundError:
        print("No adjoint method datas found")

    try: 
        auto_datas = jnp.load(auto_save_path)
        plot_datas(cfg, auto_datas, "auto", "png")
    except FileNotFoundError:
        print("No autodiff method datas found")


def main():
    parser = argparse.ArgumentParser(description="Vlasov–Poisson driver (predcorr / NuFI stub).")
    parser.add_argument("--params", type=str, required=True, help="Base yaml config file")
    args = parser.parse_args()

    cfg = load_config(args.params)
    get_datas(cfg)
