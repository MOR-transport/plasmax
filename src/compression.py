import jax
import jax.numpy as jnp 
import flax.linen as nn 
import optax 
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Tuple
from functools import partial 

# POD
def compress_pod(f: jnp.ndarray, rank: int, current_time: float, plot_dir: Path, data_dir: Path) -> jnp.ndarray:
    """Apply SVD, plot the spectrum, calculate the error and reconstruct the distribution f with rank r"""
    #SVD calculation
    U, s, VT = jnp.linalg.svd(f, full_matrices=False)
    
    #calculation of the error in Frobenius norm
    total_energy = jnp.sum(s**2)
    truncated_energy = jnp.sum(s[rank:]**2)
    error_frob = float(jnp.sqrt(truncated_energy / total_energy))
    #error saving
    error_file = data_dir / "pod_frobenius_errors.csv"
    if not error_file.exists():
        with open(error_file, "w") as f_err:
            f_err.write("time,frobenius_error\n")
    
    with open(error_file, "a") as f_err:
        f_err.write(f"{current_time:.4f},{error_frob:.6e}\n")
    
    #plot of the normalized singular spectrum
    s_norm = s / s[0]
    
    fig, ax = plt.subplots(figsize=(8, 6))    
    ax.semilogy(s_norm, marker='o', linestyle='', color='#1f77b4', markersize=5, alpha=0.8, label=r"Normalized $\sigma_i$")    
    ax.axvline(x=rank, color='#d62728', linestyle='--', linewidth=2, label=f'Truncation $r={rank}$')    
    ax.set_xlabel(r'Singular Value Index $i$', fontsize=14, labelpad=10)
    ax.set_ylabel(r'$\sigma_i / \sigma_1$', fontsize=14, labelpad=10)
    ax.set_title(f'Normalized SVD Spectrum at $t={current_time:.2f}$', fontsize=16, pad=15)    
    ax.grid(True, which='major', linestyle='-', alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax.legend(loc='upper right', fontsize=12, frameon=True, edgecolor='black', fancybox=False, facecolor='white', framealpha=1.0)
    fig.tight_layout()
    fig.savefig(plot_dir / f"svd_spectrum_t{current_time:05.2f}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    #Truncation
    U_r = U[:, :rank]
    s_r = s[:rank]
    VT_r = VT[:rank, :]
    
    #reconstruction
    f_comp = (U_r * s_r) @ VT_r
    
    print(f"-> Frobeniurs Error: {error_frob:.2e}")
    
    return f_comp   

#INR
class INR(nn.Module):
    @nn.compact 
    def __call__(self, x):
        #x is of dimension (N, 2) where N is the number of grid points and 2 corresponds to (x,v
        x = nn.Dense(16)(x)
        x = nn.tanh(x)
        x = nn.Dense(16)(x)
        x = nn.tanh(x)
        x = nn.Dense(16)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x
    
def mse_loss(params, model, inputs, targets):
    predictions = model.apply(params, inputs)
    
    return jnp.mean((predictions - targets) ** 2)

@partial(jax.jit, static_argnames=["model", "optimizer"])
def train_step(params, opt_state, model, optimizer, inputs, targets):
    loss, grads = jax.value_and_grad(mse_loss)(params, model, inputs, targets)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss 

def compress_inr(f_full: jnp.ndarray, grid_X: jnp.ndarray, grid_V: jnp.ndarray, params_init: Any = None, lr: float = 1e-3, max_iters: int = 2000, batch_size: int =2000, treshold: float = 1e-8):
    """
    Fit an INR network to approximate f_full.
    If params_init is provided, a Warm Start is performed.
    """
    
    # Flatten the grid and function for training
    # X_flat, V_flat, and f_flat will have the shape (N_x * N_v,)
    inputs = jnp.stack([grid_X.flatten(), grid_V.flatten()], axis=-1)
    targets = f_full.flatten()[:, None] # shape (N,1)
    
    #Initialization of the model and optimizer 
    total_points = inputs.shape[0]
    model = INR()
    optimizer = optax.adam(learning_rate=lr)
    
    key = jax.random.PRNGKey(42)
    if params_init is None:
        key, subkey = jax.random.split(key)
        params = model.init(subkey, inputs[:1, :])
    else:
        params = params_init
        
    opt_state = optimizer.init(params)
    
    #training loop 
    for i in range(max_iters):
        #generate a new key for the random draw in this iteration
        key, subkey = jax.random.split(key)
        
        #select batch_size indices at random
        batch_indices = jax.random.choice(subkey, total_points, shape=(batch_size,), replace= False)
        
        batch_inputs = inputs[batch_indices]
        batch_targets = targets[batch_indices]
        
        #training on the mini-batch
        params, opt_state, loss = train_step(params, opt_state, model, optimizer, batch_inputs, batch_targets)
        
        #check of treshold every 100 iterations
        if i % 100 == 0:
            if loss < treshold:
                print(f"[INR] Anticipated convergence at iteration {i} (Loss: {loss:.2e})")
                break
            
    print(f"[INR] Final Loss: {loss:.2e}")
    
    #Reconstruction of f from the network for the error
    f_comp_flat = model.apply(params, inputs)
    f_comp = f_comp_flat.reshape(f_full.shape)
    
    return f_comp, params
    