import jax
import jax.numpy as jnp 
import flax.linen as nn 
import optax 
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any
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

#INR Architecture Registry
class MLP_16(nn.Module):
    @nn.compact 
    def __call__(self, x):
        for _ in range(3):
            x = nn.Dense(16)(x)
            x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x

class MLP_64(nn.Module):
    @nn.compact 
    def __call__(self, x):
        for _ in range(3):
            x = nn.Dense(64)(x)
            x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x
    
class MLP_128(nn.Module):
    @nn.compact 
    def __call__(self, x):
        for _ in range(3):
            x = nn.Dense(128)(x)
            x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x

class Deep_128(nn.Module):
    @nn.compact 
    def __call__(self, x):
        for _ in range(5):
            x = nn.Dense(128)(x)
            x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x

class SIREN(nn.Module):
    """
    Sinusoidal Representation Network
    Activation sin(omega*x), special init for the first layer
    """
    features: tuple = (64, 64, 64)
    omega_0: float = 30.0
    
    @nn.compact 
    def __call__(self, x):
        #première couche : scale différent
        x = nn.Dense(self.features[0])(x)
        x = jnp.sin(self.omega_0 * x)
        #couches cachées
        for feat in self.features[1:]:
            x = nn.Dense(feat)(x)
            x = jnp.sin(x)
        x = nn.Dense(1)(x)
        return x

class SIREN_128(nn.Module):
    features: tuple = (128, 128, 128)
    omega_0: float = 30.0
    
    @nn.compact 
    def __call__(self, x):
        
        x = nn.Dense(self.features[0])(x)
        x = jnp.sin(self.omega_0 * x)
        for feat in self.features[1:]:
            x = nn.Dense(feat)(x)
            x =jnp.sin(x)
        x = nn.Dense(1)(x)
        return x 
    
class FourierMLP(nn.Module):
    """ 
    Random FOurier Features + MLP tanh
    """
    n_freqs: int = 16 #number of random frequencies
    hidden: int = 64
    n_layers: int = 3
    sigma: float = 10.0 #scale of the random frequencies
    
    @nn.compact 
    def __call__(self, x):
        #fixed fourier features 
        B = self.param('B', lambda rng, shape: jax.random.normal(rng, shape) * self.sigma,  (2, self.n_freqs))
        proj = x @ B # (N, n_freqs)
        x = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1) # (N, 2*n_freqs)
        #classic MLP
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden)(x)
            x = nn.tanh(x)
        return nn.Dense(1)(x)
    
INR_REGISTRY = {
    "mlp_16": MLP_16,
    "mlp_64": MLP_64,
    "mlp_128": MLP_128,
    "deep_128": Deep_128,
    "siren": SIREN,
    "siren_128": SIREN_128,
    "fourier_mlp": FourierMLP,
}   

def get_inr_model(arch: str) -> nn.Module:
    if arch not in INR_REGISTRY:
        raise ValueError(
            f"Architecture INR inconnue : '{arch}'."
            f"Choix disponibles : {list(INR_REGISTRY.keys())}"
        )
    return INR_REGISTRY[arch]()
    
        
    
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
    current_time: float,
    data_dir: Path,
    arch: str = "mlp_16",
    params_init: Any = None, 
    lr: float = 1e-3, 
    max_iters: int = 2000, 
    batch_size: int =2000, 
    threshold: float = 1e-8):
    """
    Fit an INR network to approximate f_full.
    """
    #Normalizing entries in [-1, 1]
    x_norm = grid_X / grid_X.max()
    v_norm = grid_V / jnp.abs(grid_V).max() 
    inputs = jnp.stack([x_norm.flatten(), v_norm.flatten()], axis=-1) 
    targets = f_full.flatten()[:, None]
    
    total_points = inputs.shape[0]
    model = get_inr_model(arch)
    optimizer = optax.adam(learning_rate=lr)
    
    key = jax.random.PRNGKey(42)
    if params_init is None:
        key, subkey = jax.random.split(key)
        params = model.init(subkey, inputs[:1, :])
    else: params = params_init
    
    opt_state = optimizer.init(params)
    loss = jnp.inf 
    
    #training loop 
    for i in range(max_iters):
        #generate a new key for the random draw in this iteration
        key, subkey = jax.random.split(key)
        #select batch_size indices at random
        batch_idx = jax.random.choice(subkey, total_points, shape=(batch_size,), replace= False)
        
        params, opt_state, loss = train_step(
            params, opt_state, model, optimizer, inputs[batch_idx], targets[batch_idx]
        )
                
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
    
    return f_comp, params
    