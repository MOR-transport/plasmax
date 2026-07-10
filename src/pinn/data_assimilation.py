#phase 2 and 3

import jax
import jax.numpy as jnp

from scimba_jax.domains.meshless_domains.domains_1d import Segment1D
from scimba_jax.nonlinear_approximation.integration.monte_carlo import DomainSampler, TensorizedSampler
from scimba_jax.nonlinear_approximation.integration.monte_carlo_time import UniformTimeSampler
from scimba_jax.nonlinear_approximation.integration.monte_carlo_parameters import UniformParametricSampler
from scimba_jax.nonlinear_approximation.integration.data_sampler import DataSampler
from scimba_jax.physical_models.data_residuals import CollocDataResidual

from src.config import Config

def build_E_interpolator(t_grid: jnp.ndarray, x_grid: jnp.ndarray, E_hist: jnp.ndarray):
    """Function to interpolate E(t,x) in all continuous point """
    t_min, t_max = t_grid[0], t_grid[-1]
    x_min, x_max = x_grid[0], x_grid[-1]
    Nt, Nx = E_hist.shape
    
    def E_interp(x_eval, t_eval):
        #conversion of physical coordinates into fractional grid indices
        t_idx = (t_eval - t_min) / (t_max - t_min) * (Nt - 1)
        x_idx = (x_eval - x_min) / (x_max - x_min) * (Nx - 1)
        # coordinate overlay for map_coordinates (shape: [2, N_points])
        coords = jnp.stack([t_idx.flatten(), x_idx.flatten()], axis=0) 
        #bilinear interpolation
        E_vals = jax.scipy.ndimage.map_coordinates(E_hist, coords, order=1, mode='nearest') 
        #restore the result to the original shape of the evaluation points.
        return E_vals.reshape(t_eval.shape)
    
    return jax.jit(E_interp)

def create_collocation_sampler(t_min: float, t_max: float, x_min: float, x_max: float, v_min: float, v_max: float):
    """Create the Sampler for the continuous collocation points (L_physics)"""
    sampler_t = UniformTimeSampler((t_min, t_max))
    domain_x = Segment1D((x_min, x_max), is_main_domain=True)
    sampler_x = DomainSampler(domain_x)
    sampler_v = UniformParametricSampler([(v_min, v_max)])
    
    sampler_physics = TensorizedSampler(
        list_sampler=[sampler_t, sampler_x, sampler_v],
        model_type="t_x_v",
        bc=False,
        ic=False   
    )
    
    return sampler_physics 

def create_data_sampler(cfg: Config, t_grid: jnp.ndarray, f_hist: jnp.ndarray, data_ratio: float = 0.05, key_seed: int = 42):
    """
    Prepares experimental data for the data assimilation (L_data)
    Takes the full history (N_t, N_v, N_x), generates the points (t, x, v),
    applies a mask (retaining only data_ratio of the points), and returns a DataSampler.
    """
    Nt, Nv, Nx = f_hist.shape 
    
    #create the complete grid of points (t, x, v)
    T_grid, V_grid, X_grid = jnp.meshgrid(t_grid, cfg.grid.v, cfg.grid.x, indexing='ij') 
    
    #flattening
    T_flat = T_grid.flatten()[:, None] # shape (N, 1)
    V_flat = V_grid.flatten()[:, None] 
    X_flat = X_grid.flatten()[:, None]
    f_flat = f_hist.flatten()[:, None]
    
    inputs_flat = jnp.concatenate([T_flat, X_flat, V_flat], axis=1) # shape (N, 3)
    
    #application of the assimilation mask
    N_total = inputs_flat.shape[0]
    N_keep = int(N_total * data_ratio)
    
    key = jax.random.PRNGKey(key_seed)
    indices_to_keep = jax.random.choice(key, N_total, shape=(N_keep,), replace=False)
    
    inputs_masked = inputs_flat[indices_to_keep]
    outputs_masked = f_flat[indices_to_keep]
    
    print(f"Data assimilation: Keeping {N_keep} points out of {N_total} ({data_ratio*100:.1f}%)")
    
    inputs_masked_jax = jnp.array(inputs_masked)
    outputs_masked_jax = jnp.array(outputs_masked)
    
    data_sampler = DataSampler(data=(inputs_masked_jax, outputs_masked_jax))
    
    return data_sampler