import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from scimba_jax.domains.meshless_domains.domains_1d import Segment1D
from scimba_jax.domains.meshless_domains.domains_nd import CartesianProduct 
from scimba_jax.nonlinear_approximation.approximation_spaces.approximation_spaces import ApproximationSpace
from scimba_jax.nonlinear_approximation.numerical_solvers.projectors import Projector
from scimba_jax.physical_models.data_residuals import CollocDataResidual

from ..config import load_config
from .pinn_models import get_pinn_network
from .pinn_physics import VlasovPoissonModel
from .data_assimilation import (
    build_E_interpolator,
    create_collocation_sampler,
    create_data_sampler
)

jax.config.update("jax_enable_x64", True)

def main():
    parser = argparse.ArgumentParser(description="PINN Data Assimilation runner for PlasmaX.")
    parser.add_argument("--params", type=str, required=True, help="Base yaml config file")
    parser.add_argument("--tend", type=float, default=None, help="Total simulation time")
    parser.add_argument("--ratio", type=float, default=0.05, help="Ratio of experimental data kept (e.g. 0.05)")
    parser.add_argument("--arch", type=str, default="periodic_siren_deep_128", help="Architecture name")
    parser.add_argument("--lambda-data", type=float, default=100.0, help="Weight for the data assimilation loss")
    args = parser.parse_args()
    
    # 1. Base configuration and path management
    cfg = load_config(args.params)
    final_time = args.tend if args.tend is not None else cfg.time.tend
    dt_seg = cfg.io.dt_save if cfg.io.dt_save is not None else 5.0
    
    original_case = cfg.inicond.case
    case_root = Path("results") / f"case_{original_case}"
    gt_data_dir = case_root / "segmented" / "NO_COMPRESSION" / "data"
    
    folder_name = f"{args.arch}_ratio{args.ratio}"
    pinn_save_dir = case_root / "pinn" / folder_name
    pinn_data_dir = pinn_save_dir / "data"
    pinn_plot_dir = pinn_save_dir / "plots"
    
    pinn_data_dir.mkdir(parents=True, exist_ok=True)
    pinn_plot_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = pinn_data_dir / "pinn_metrics.csv"
    if metrics_file.exists():
        metrics_file.unlink()
    with open(metrics_file, "w") as f:
        f.write("time,loss_adam,loss_lbfgs,frobenius_error,train_time\n")
    
    print("\n" + "="*70)
    print(f"STARTING PINN DATA ASSIMILATION")
    print(f"Target: {original_case} | Ratio: {args.ratio*100}% | Arch: {args.arch}")
    print("="*70 + "\n")
    
    # 2. Network and space initialization (Cold Start at t=0)
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    
    # Utilisation de la Factory propre
    network = get_pinn_network(args.arch, cfg.grid.lx, subkey)
    
    # Correction : dims={} pour que scimba_jax les déduise de "t_x_v"
    space = ApproximationSpace(
        dims={"t": 1, "x": 1, "v": 1},
        list_models=[(network, "scalar", None)],
        model_type="t_x_v",
    )
    
    current_time = 0.0
    
    # 3. Temporal loop over segments
    while current_time < final_time - 1e-9:
        next_time = min(current_time + dt_seg, final_time)
        history_file = gt_data_dir / f"history_t{next_time:05.2f}.npz"
        
        if not history_file.exists():
            raise FileNotFoundError(f"Ground Truth missing: {history_file}")
        
        print(f"\n--- Assimilation segment: t={current_time:.2f} to t={next_time:.2f} ---")
        
        data = jnp.load(history_file)
        f_hist, Efield_hist, t_grid = data["f_hist"], data["Efield_hist"], data["t_grid"]
        t_min, t_max = float(t_grid[0]), float(t_grid[-1])
        x_min, x_max = 0.0, cfg.grid.lx
        v_min, v_max = -cfg.grid.lv, cfg.grid.lv
        
        t0_train = time.perf_counter()
        
        #domain definitions for the Vlasov-Poisson model
        x_domain = Segment1D((x_min, x_max))
        v_domain = Segment1D((v_min, v_max))
        main_domain = CartesianProduct([x_domain, v_domain], is_main_domain=True)
        
        E_interpolator = build_E_interpolator(t_grid, cfg.grid.x, Efield_hist)
        pde_model = VlasovPoissonModel(main_domain=main_domain, time_domain=(t_min, t_max), E_interpolator=E_interpolator)
        pde_model.data_residuals = {"data_assim": CollocDataResidual(size=1)}
        
        colloc_sampler = create_collocation_sampler(t_min, t_max, x_min, x_max, v_min, v_max)
        data_sampler = create_data_sampler(cfg, t_grid, f_hist, data_ratio=args.ratio)
        
        class PINNSampler:
            def build_sample_func(self, n_colloc, n_bc, n_ic, n_dl):
                colloc_func = colloc_sampler.build_sample_func(n_colloc, n_bc=0, n_ic=0, n_dl=0)
                data_func = data_sampler.build_sample_func(n_dl)
                
                def sample_func(k):
                    k, k1, k2 = jax.random.split(k, 3)
                    _, samples = colloc_func(k1)
                    interior_pts = samples["interior"]
                    
                    _, data_pts = data_func(k2)
                    inputs, outputs = data_pts
                    
                    return k, {
                        "interior": interior_pts,
                        "data_assim": (inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], outputs) 
                    }
                return sample_func
        
        sampler = PINNSampler()
        loss_weights = {"interior": [1.0], "data_assim": [args.lambda_data]}
        
        # Phase 1: ADAM
        print("Phase 1: ADAM optimization...")
        proj_adam = Projector(model=pde_model, space=space, sampler=sampler, optimizer="Adam", learning_rate=1e-3, weights=loss_weights)
        key, proj_adam = proj_adam.project(key=key, space=space, n_epochs=2000, n_colloc=10000, n_dl_colloc=2000, verbose=False)
        loss_adam = float(proj_adam.best_loss["total"])
        
        # Phase 2: L-BFGS
        print("Phase 2: L-BFGS optimization...")
        space_after_adam = proj_adam.space
        proj_lbfgs = Projector(model=pde_model, space=space_after_adam, sampler=sampler, optimizer="L-BFGS", weights=loss_weights)
        key, proj_lbfgs = proj_lbfgs.project(key=key, space=space_after_adam, n_epochs=50, n_colloc=15000, n_dl_colloc=5000, verbose=False)
        loss_lbfgs = float(proj_lbfgs.best_loss["total"])
        
        train_time = time.perf_counter() - t0_train
        
        # warm-starting (in-memory)
        space = proj_lbfgs.space
        
        # evaluation
        T_grid, V_grid, X_grid = jnp.meshgrid(t_grid, cfg.grid.v, cfg.grid.x, indexing='ij')
        inputs_flat = jnp.concatenate([T_grid.flatten()[:, None], X_grid.flatten()[:, None], V_grid.flatten()[:, None]], axis=1)
        f_pred_flat = jax.vmap(space.models[0])(inputs_flat)
        f_pred = f_pred_flat.reshape(f_hist.shape)
        
        frob_error = float(jnp.linalg.norm(f_pred - f_hist) / jnp.linalg.norm(f_hist))
        print(f"  -> Loss ADAM: {loss_adam:.2e} | Loss L-BFGS: {loss_lbfgs:.2e} | Frob Error: {frob_error:.2e} | Time: {train_time:.1f}s")
        
        # Save
        weights_file = pinn_data_dir / f"weights_t{next_time:05.2f}.pkl"
        with open(weights_file, "wb") as f:
            pickle.dump(space.models[0], f)
            
        with open(metrics_file, "a") as f:
            f.write(f"{next_time:.4f},{loss_adam:.6e},{loss_lbfgs:.6e},{frob_error:.6e},{train_time:.6e}\n")
            
        current_time = next_time 
    
    print(f"\nPINN Simulation finished successfully. Results saved in '{pinn_save_dir}'")

if __name__ == "__main__":
    main()