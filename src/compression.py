import jax
import jax.numpy as jnp 
import flax.linen as nn 
import optax 
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any
from functools import partial 
jax.config.update("jax_enable_x64", True)

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

#INR Architecture Registry
def make_mlp(hidden_size: int, n_layers: int = 3):
    class MLP(nn.Module):
        @nn.compact 
        def __call__(self, x):
            for _ in range(n_layers):
                x = nn.Dense(hidden_size)(x)
                x = nn.tanh(x)
            return nn.Dense(1)(x)
    return MLP

def make_siren(features: tuple, omega_0: float):
    class SIREN(nn.Module):
        @nn.compact 
        def __call__(self, x):
            x = nn.Dense(features[0])(x)
            x = jnp.sin(omega_0 * x)
            for feat in features[1:]:
                x = nn.Dense(feat)(x)
                x = jnp.sin(x)
            x = nn.Dense(1)(x)
            return x
    return SIREN

def make_fourier_mlp(n_freqs: int, hidden: int, n_layers: int, sigma: float):
    class FourierMLP(nn.Module):
        """ 
        Random Fourier Features + MLP tanh
        """
        @nn.compact 
        def __call__(self, x):
            #fixed fourier features 
            B = self.param('B', lambda rng, shape: jax.random.normal(rng, shape) * sigma,  (2, n_freqs))
            proj = x @ B # (N, n_freqs)
            x = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1) # (N, 2*n_freqs)
            #classic MLP
            for _ in range(n_layers):
                x = nn.Dense(hidden)(x)
                x = nn.tanh(x)
            return nn.Dense(1)(x)
    return FourierMLP   

def make_periodic_mlp(hidden_size: int, n_layers: int):
    class PeriodicMLP(nn.Module):
        @nn.compact 
        def __call__(self, x_inputs):
            #x_inputs is of shape (N, 2) with (x, v) 
            x_coord = x_inputs[:, 0:1] # (N, 1) : only x 
            v_coord = x_inputs[:, 1:2] # (N, 1) : only v
            
            #periodic embedding of x
            kx = 2.0 * jnp.pi #no division by lx because x is already divided by lx for normalization
            x_embedded = jnp.concatenate([
                jnp.cos(kx * x_coord),
                jnp.sin(kx * x_coord)
            ], axis=-1)
            
            #concatenate with v
            h = jnp.concatenate([x_embedded, v_coord], axis=-1)
            
            #classic MLP
            for _ in range(n_layers):
                h = nn.Dense(hidden_size)(h)
                h = nn.tanh(h)
            return nn.Dense(1)(h)
    return PeriodicMLP

def make_periodic_siren(features: tuple, omega_0: float):
    class PeriodicSIREN(nn.Module):
        @nn.compact 
        def __call__(self, x_inputs):
            x_coord = x_inputs[:, 0:1]
            v_coord = x_inputs[:, 1:2]
            
            #periodic embedding of x
            kx = 2.0 * jnp.pi
            x_embedded = jnp.concatenate([
                jnp.cos(kx * x_coord),
                jnp.sin(kx * x_coord)
            ], axis=-1)
            
            h = jnp.concatenate([x_embedded, v_coord], axis=-1)
            
            #classic SIREN
            h = nn.Dense(features[0])(h)
            h = jnp.sin(omega_0 * h)
            for feat in features[1:]:
                h = nn.Dense(feat)(h)
                h = jnp.sin(h)
            return nn.Dense(1)(h)
    return PeriodicSIREN

def make_periodic_fourier_mlp(n_freqs: int, hidden: int, n_layers: int, sigma: float):
    class PeriodicFourierMLP(nn.Module):
        @nn.compact 
        def __call__(self, x_inputs):
            x_coord = x_inputs[:, 0:1]
            v_coord = x_inputs[:, 1:2]
            
            kx = 2.0 * jnp.pi
            x_embedded = jnp.concatenate([
                jnp.cos(kx * x_coord),
                jnp.sin(kx * x_coord)
            ], axis=-1)
            
            h = jnp.concatenate([x_embedded, v_coord], axis=-1)
            
            #the matrix B must now project 3 features (cos_x, sin_x, v) to n_freqs instead of 2
            B = self.param('B', lambda rng, shape: jax.random.normal(rng, shape) * sigma, (3, n_freqs))
            proj = h @ B # (N, n_freqs)
            
            h = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1) # (N, 2*n_freqs)
            
            for _ in range(n_layers):
                h = nn.Dense(hidden)(h)
                h = nn.tanh(h)
            return nn.Dense(1)(h)
    return PeriodicFourierMLP
            
            
def get_inr_registry():
    return {
        #mlp
        "mlp_16": make_mlp(16, 3),
        "mlp_64": make_mlp(64, 3),
        "mlp_128": make_mlp(128, 3),
        "deep_128": make_mlp(128, 5),
        #siren
        "siren": make_siren((64, 64, 64), 30.0),
        "siren_128": make_siren((128, 128, 128), 30.0),
        "siren_deep_128": make_siren((128, 128, 128, 128, 128), 30.0),
        #fourier mlp
        "fourier_mlp": make_fourier_mlp(16, 64, 3, 10.0),
        "fourier_mlp_128": make_fourier_mlp(16, 128, 3, 10.0),
        "fourier_mlp_deep_128": make_fourier_mlp(16, 128, 5, 10.0),
        #periodic versions
        #periodic mlp
        "periodic_mlp_16": make_periodic_mlp(16, 3),
        "periodic_mlp_64": make_periodic_mlp(64, 3),
        #periodic siren
        "periodic_siren": make_periodic_siren((64, 64, 64), 30.0),
        "periodic_siren_128": make_periodic_siren((128, 128, 128), 30.0),
        "periodic_siren_deep_128": make_periodic_siren((128, 128, 128, 128, 128), 30.0),
        #periodic fourier mlp
        "periodic_fourier_mlp": make_periodic_fourier_mlp(16, 64, 3, 10.0),
        "periodic_fourier_mlp_128": make_periodic_fourier_mlp(16, 128, 3, 10.0),
        "periodic_fourier_mlp_deep_128": make_periodic_fourier_mlp(16, 128, 5, 10.0)
    }

#list for argparse choices
AVAILABLE_INR_ARCHS = list(get_inr_registry().keys()) 

def get_inr_model(arch:str) -> nn.Module:
    registry = get_inr_registry()
    if arch not in registry: 
        raise ValueError(
            f"Unknown INR Architecture : '{arch}'.\n"
            f"Available choices : {list(registry.keys())}"
        )
    return registry[arch]()

def mse_loss(params, model, inputs, targets):
    predictions = model.apply(params, inputs)
    
    return jnp.mean((predictions - targets) ** 2)

@partial(jax.jit, static_argnames=["model", "optimizer"])
def train_step(params, opt_state, model, optimizer, inputs, targets):
    loss, grads = jax.value_and_grad(mse_loss)(params, model, inputs, targets)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss

def log_inr_error(data_dir: Path, arch: str, current_time: float, final_loss: float, frob_error: float):
    """Log loss + Frobenius error in inr_errors.csv """
    error_file = data_dir / "inr_errors.csv"
    if not error_file.exists():
        with open(error_file, "w") as f:
            f.write("time,arch,final_loss,frobenius_error\n")
    with open(error_file, "a") as f:
        f.write(f"{current_time:.4f},{arch},{final_loss:.6e},{frob_error:.6e}\n")
    

def compress_inr(
    f_full: jnp.ndarray, 
    grid_X: jnp.ndarray, 
    grid_V: jnp.ndarray, 
    lx: float,
    lv: float,
    current_time: float,
    data_dir: Path,
    arch: str = "periodic_mlp_64",
    params_init: Any = None, 
    lr: float = 1e-3, 
    max_iters: int = 2000, 
    batch_size: int =2000, 
    threshold: float = 1e-8):
    """
    Fit an INR network to approximate f_full.
    """
    #kx = 2.0 * jnp.pi / lx #dynamic calculation of kx for periodic embedding
    x_raw = grid_X.flatten() / lx #normalization of x to [0, 1]
    v_norm = grid_V.flatten() / lv #normalization of v to [-1, 1]
    
    inputs = jnp.stack([x_raw, v_norm], axis=-1) # (N, 2)
    targets = f_full.flatten()[:, None]
    
    total_points = inputs.shape[0]
    
    model = get_inr_model(arch)
    optimizer = optax.adam(learning_rate=lr)
    
    key = jax.random.PRNGKey(42)
    if params_init is None:
        key, subkey = jax.random.split(key)
        params = model.init(subkey, inputs[:1, :])
    else: 
        params = params_init
    
    opt_state = optimizer.init(params)
    loss = jnp.inf 
    
    loss_history = []
    
    #training loop 
    for i in range(max_iters):
        #generate a new key for the random draw in this iteration
        key, subkey = jax.random.split(key)
        #select batch_size indices at random
        batch_idx = jax.random.choice(subkey, total_points, shape=(batch_size,), replace= False)
        
        params, opt_state, loss = train_step(
            params, opt_state, model, optimizer, inputs[batch_idx], targets[batch_idx]
        )
        loss_history.append(float(loss))
        #check of treshold every 100 iterations
        if i % 100 == 0:
            print(f"  [INR/{arch}] iter {i:4d} — loss: {loss:.2e}")
            if loss < threshold:
                print(f"[INR] Anticipated convergence at iteration {i} (Loss: {loss:.2e})")
                break
    
    #Reconstruction of f from the network for the error
    f_comp_flat = model.apply(params, inputs)
    f_comp = f_comp_flat.reshape(f_full.shape)
    
    #Frobenius error
    frob_error = float(jnp.linalg.norm(f_comp - f_full) / jnp.linalg.norm(f_full))
    print(f"[INR/{arch}] Loss finale: {float(loss):.2e} | Erreur Frobenius: {frob_error:.2e}")
    
    log_inr_error(data_dir, arch, current_time, float(loss), frob_error)
    
    return f_comp, params, jnp.array(loss_history)