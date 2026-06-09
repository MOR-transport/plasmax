import jax
import jax.numpy as jnp 
#import flax.linen as nn 
#import optax 
import matplotlib.pyplot as plt
import equinox as eqx
import time
from pathlib import Path
from typing import Any
#from functools import partial 
from scimba_jax.nonlinear_approximation.networks.mlp import MLP
from scimba_jax.nonlinear_approximation.optimizers.optimizers import ScimbaAdam, ScimbaLBfgs
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

# POD
def compress_pod(f: jnp.ndarray, rank: int, current_time: float, plot_dir: Path, data_dir: Path, sim_time: float = 0.0) -> jnp.ndarray:
    """Apply SVD, plot the spectrum, calculate the error and reconstruct the distribution f with rank r"""
    t0 = time.perf_counter()
    #SVD calculation
    U, s, VT = jnp.linalg.svd(f, full_matrices=False)
    
    #calculation of the error in Frobenius norm
    total_energy = jnp.sum(s**2)
    truncated_energy = jnp.sum(s[rank:]**2)
    error_frob = float(jnp.sqrt(truncated_energy / total_energy))
    
    #Truncation
    U_r = U[:, :rank]
    s_r = s[:rank]
    VT_r = VT[:rank, :]
    #reconstruction
    f_comp = (U_r * s_r) @ VT_r
    
    t1 = time.perf_counter()
    comp_time = t1 - t0
    print(f"-> Frobenius Error: {error_frob:.2e}s | Sim Time: {sim_time:.2f} | Comp Time: {comp_time:.2f}s")

    #error saving
    error_file = data_dir / "pod_frobenius_errors.csv"
    if not error_file.exists():
        with open(error_file, "w") as f_err:
            f_err.write("time,frobenius_error,sim_time,comp_time\n")
    
    with open(error_file, "a") as f_err:
        f_err.write(f"{current_time:.4f},{error_frob:.6e},{sim_time:.6e},{comp_time:.6e}\n")
    
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
    
    return f_comp   

#INR Architecture Registry   

class PeriodicMLPScimbaINR(eqx.Module):
    """ 
    Wrapper to force periodic boundary conditions on the x axis
    """
    network: MLP
    
    def __init__(self, hidden_sizes: list[int], activation: str, key: jax.Array):
        """ 
        The raw input is (x, v)
        After embedding, we have (cos(kx*x), sin(kx*x), v)
        The input size of the MLP must therefore be 3
        """
        self.network = MLP(
            in_size=3,
            out_size=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
            key=key
        )    
        
    def __call__(self, x_inputs: jnp.ndarray) -> jnp.ndarray:
        x_coord = x_inputs[0:1]
        v_coord = x_inputs[1:2]
        
        kx = 2.0 * jnp.pi
        x_embedded = jnp.concatenate([
            jnp.cos(kx * x_coord),
            jnp.sin(kx * x_coord)
        ], axis=-1)
        h = jnp.concatenate([x_embedded, v_coord], axis=-1)
        return self.network(h)


class SIRENScimbaINR(eqx.Module):
    """Wrapper for a SIREN architecture"""
    layers: tuple
    omega_0: float

    def __init__(self, in_size: int, out_size: int, hidden_sizes: list[int], omega_0: float, key: jax.Array):
        self.omega_0 = omega_0
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        sizes = [in_size] + hidden_sizes + [out_size]
        
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i]))
        self.layers = tuple(layers)

    def __call__(self, x_input: jnp.ndarray) -> jnp.ndarray:
        x = jnp.sin(self.omega_0 * self.layers[0](x_input))
        
        for layer in self.layers[1:-1]:
            x = jnp.sin(layer(x))
            
        return self.layers[-1](x)

    def ndof(self) -> int:
        flat_params, _ = jax.tree_util.tree_flatten(self)
        return sum(p.size for p in flat_params if isinstance(p, jnp.ndarray))


class PeriodicSIRENScimbaINR(eqx.Module):
    """Wrapper pour forcer la périodicité spatiale sur un SIREN."""
    network: SIRENScimbaINR

    def __init__(self, hidden_sizes: list[int], omega_0: float, key: jax.Array):
        self.network = SIRENScimbaINR(
            in_size=3,
            out_size=1,
            hidden_sizes=hidden_sizes,
            omega_0=omega_0,
            key=key
        )

    def __call__(self, x_input: jnp.ndarray) -> jnp.ndarray:
        x_coord = x_input[0:1]
        v_coord = x_input[1:2]
        
        kx = 2.0 * jnp.pi
        x_embedded = jnp.concatenate([
            jnp.cos(kx * x_coord),
            jnp.sin(kx * x_coord)
        ], axis=-1)
        
        h = jnp.concatenate([x_embedded, v_coord], axis=-1)
        return self.network(h)

class FourierScimbaINR(eqx.Module):
    """
    Wrapper for Fourier features
    """
    network: MLP
    B: jnp.ndarray 
    
    def __init__(self, in_features: int, n_freqs: int, hidden_sizes: list[int], sigma: float, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        self.B = jax.random.normal(k1, (in_features, n_freqs)) * sigma
        
        self.network = MLP(
            in_size=2*n_freqs,
            out_size=1,
            hidden_sizes=hidden_sizes,
            activation="tanh",
            key=k2
        )
    
    def __call__(self, x_input: jnp.ndarray) -> jnp.ndarray:
        proj = x_input @ self.B
        h = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
        return self.network(h)
    
class PeriodicFourierScimbaINR(eqx.Module):
    """
    Wrapper for Fourier features + periodic embedding
    """
    network: MLP
    B: jnp.ndarray
    
    def __init__(self, n_freqs: int, hidden_sizes: list[int], sigma: float, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        #The matrix projects a vector of size 3 (cos x, sin x, v)
        self.B = jax.random.normal(k1, (3, n_freqs)) * sigma
        
        self.network = MLP(
            in_size=2*n_freqs,
            out_size=1,
            hidden_sizes=hidden_sizes,
            activation="tanh",
            key=k2
        )
    
    def __call__(self, x_inputs: jnp.ndarray) -> jnp.ndarray:
        x_coord = x_inputs[0:1]
        v_coord = x_inputs[1:2]
        
        kx = 2.0 * jnp.pi
        x_embedded = jnp.concatenate([
            jnp.cos(kx * x_coord),
            jnp.sin(kx * x_coord)
        ], axis=-1)
        
        h = jnp.concatenate([x_embedded, v_coord], axis=-1)
        proj = h @ self.B
        h_fourier = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
        
        return self.network(h_fourier)

AVAILABLE_INR_ARCHS = [
    "mlp_16", "mlp_64", "mlp_128", "deep_128",
    "siren", "siren_128", "siren_deep_128",
    "fourier_mlp", "fourier_mlp_128", "fourier_mlp_deep_128",
    "periodic_mlp_16", "periodic_mlp_64",
    "periodic_siren", "periodic_siren_128", "periodic_siren_deep_128",
    "periodic_fourier_mlp", "periodic_fourier_mlp_128", "periodic_fourier_mlp_deep_128"
] 

def get_inr_model(arch: str, key: jax.Array) -> eqx.Module:
    if arch == "mlp_16": return MLP(2, 1, [16]*3, "tanh", key)
    elif arch == "mlp_64": return MLP(2, 1, [64]*3, "tanh", key)
    elif arch == "mlp_128": return MLP(2, 1, [128]*3, "tanh", key)
    elif arch == "deep_128": return MLP(2, 1, [128]*5, "tanh", key)

    elif arch == "siren": return SIRENScimbaINR(2, 1, [64]*3, 30.0, key)
    elif arch == "siren_128": return SIRENScimbaINR(2, 1, [128]*3, 30.0, key)
    elif arch == "siren_deep_128": return SIRENScimbaINR(2, 1, [128]*5, 30.0, key)
    
    elif arch == "fourier_mlp": return FourierScimbaINR(2, 16, [64]*3, 10.0, key)
    elif arch == "fourier_mlp_128": return FourierScimbaINR(2, 16, [128]*3, 10.0, key)
    elif arch == "fourier_mlp_deep_128": return FourierScimbaINR(2, 16, [128]*5, 10.0, key)
    
    elif arch == "periodic_mlp_16": return PeriodicMLPScimbaINR([16]*3, "tanh", key)
    elif arch == "periodic_mlp_64": return PeriodicMLPScimbaINR([64]*3, "tanh", key)
    
    elif arch == "periodic_siren": return PeriodicSIRENScimbaINR([64]*3, 30.0, key)
    elif arch == "periodic_siren_128": return PeriodicSIRENScimbaINR([128]*3, 30.0, key)
    elif arch == "periodic_siren_deep_128": return PeriodicSIRENScimbaINR([128]*5, 30.0, key)
    
    elif arch == "periodic_fourier_mlp": return PeriodicFourierScimbaINR(16, [64]*3, 10.0, key)
    elif arch == "periodic_fourier_mlp_128": return PeriodicFourierScimbaINR(16, [128]*3, 10.0, key)
    elif arch == "periodic_fourier_mlp_deep_128": return PeriodicFourierScimbaINR(16, [128]*5, 10.0, key)
    
    else:
        raise ValueError(f"Unknown Architecture : {arch}")    


@jax.jit 
def losses_function(model: eqx.Module, batch: tuple) -> dict:
    """Calculate the MSE and returns the dictionary expected by scimba """
    inputs, targets = batch
    predictions = jax.vmap(model)(inputs)
    mse = jnp.mean((predictions - targets) ** 2)
    return {"total": mse}

@jax.jit
def grad_loss_function(model: eqx.Module, batch: tuple) -> jnp.ndarray:
    """Calculate the gradient with respect to the model and flattens it into a 1D vectot"""
    def loss_fn(m):
        return losses_function(m, batch)["total"]

    grads = eqx.filter_grad(loss_fn)(model) #eqx.filter_grad is the recommended way to derive an equinox model
    # flattening gradients so that ScimbaLBfgs can use them
    flat_grads, _ = ravel_pytree(grads)
    return flat_grads 

def log_inr_error(data_dir: Path, arch: str, current_time: float, final_loss: float, frob_error: float, sim_time: float, comp_time: float):
    """Log loss + Frobenius error + CPU times in inr_errors.csv """
    error_file = data_dir / "inr_errors.csv"
    if not error_file.exists():
        with open(error_file, "w") as f:
            f.write("time,arch,final_loss,frobenius_error,sim_time,comp_time\n")
    with open(error_file, "a") as f:
        f.write(f"{current_time:.4f},{arch},{final_loss:.6e},{frob_error:.6e},{sim_time:.6e},{comp_time:.6e}\n")

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
    batch_size: int = 2000,
    threshold: float = 1e-8,
    lbfgs_iters: int = 50,
    sim_time: float = 0.0
):
    """Fit an INR network to approximate f_full using ADAM then L-BFGS"""
    t0 = time.perf_counter()
    
    x_raw = grid_X.flatten() / lx
    v_norm = grid_V.flatten() / lv
    
    inputs = jnp.stack([x_raw, v_norm], axis=-1)
    targets = f_full.flatten()[:, None]
    total_points = inputs.shape[0]
    
    key = jax.random.PRNGKey(42)
    
    if params_init is None:
        key, subkey = jax.random.split(key)
        model = get_inr_model(arch, subkey)
    else:
        model = params_init
        
    loss_history = []
    
    #ADAM optimization (mini-batchs)
    print(f"\n[INR/{arch}] --- Beginning Phase 1: ADAM ({max_iters} iters) ---")
    adam_opt = ScimbaAdam(model, losses_function, grad_loss_function, learning_rate=lr)
    
    for i in range(max_iters):
        key, subkey = jax.random.split(key)
        batch_idx = jax.random.choice(subkey, total_points, shape=(batch_size,), replace=False)
        batch = (inputs[batch_idx], targets[batch_idx])
        
        loss_dict, model, adam_opt = adam_opt.update(model, batch)
        loss_val = float(loss_dict["total"])
        loss_history.append(loss_val)
        
        if i % 100 == 0:
            print(f" [ADAM] iter {i:4d} - loss: {loss_val:.2e}")
            if loss_val < threshold:
                print(f" [ADAM] Anticipated convergence at iteration {i}")
                break
    
    #L-BFGS optimization (full batch)
    print(f"[INR/{arch}] --- Beginning Phase 2: L-BFGS ---")
    full_batch = (inputs, targets)
    lbfgs_opt = ScimbaLBfgs(model, losses_function, grad_loss_function)
    
    for i in range(lbfgs_iters):
        loss_dict, model, lbfgs_opt = lbfgs_opt.update(model, full_batch)
        loss_val = float(loss_dict["total"])
        loss_history.append(loss_val)
        print(f" [L-BFGS] iter {i:4d} - loss: {loss_val:.2e}")
        
        if loss_val < threshold:
            print(f" [L-BFGS] Perfect convergence achieved")
            break
    
    #evaluation and backup 
    f_comp_flat = jax.vmap(model)(inputs)
    f_comp = f_comp_flat.reshape(f_full.shape)
    
    frob_error = float(jnp.linalg.norm(f_comp - f_full) / jnp.linalg.norm(f_full))
    
    t1 = time.perf_counter()
    comp_time = t1 - t0
    
    print(f"[INR/{arch}] Final Loss: {loss_val:.2e} | Frobenius Error: {frob_error:.2e}\n")
    print(f"[INR/{arch}] Sim Time: {sim_time:.2f}s | Comp Time: {comp_time:.2f}s\n")
    
    log_inr_error(data_dir, arch, current_time, loss_val, frob_error, sim_time, comp_time)
    
    return f_comp, model, jnp.array(loss_history)