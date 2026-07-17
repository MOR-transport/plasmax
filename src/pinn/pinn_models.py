import jax
import jax.numpy as jnp
import equinox as eqx
from scimba_jax.nonlinear_approximation.networks.mlp import MLP

def apply_spatial_periodic_embedding(txv_inputs: jnp.ndarray, lx: float) -> jnp.ndarray:
    """
    Apply periodic embedding to the spatial coordinate x in the input.
    Input : [t, x, v]
    Output : [t, cos(kx), sin(kx), v]
    """
    t_coord = txv_inputs[..., 0:1]
    x_coord = txv_inputs[..., 1:2]
    v_coord = txv_inputs[..., 2:3]

    kx = 2.0 * jnp.pi / lx
    x_embedded = jnp.concatenate([
        jnp.cos(kx * x_coord),
        jnp.sin(kx * x_coord)
    ], axis=-1)
    
    return jnp.concatenate([t_coord, x_embedded, v_coord], axis=-1)

class PeriodicMLPScimbaINR(eqx.Module):
    """Spatio-temporal MLP with hard periodic boundaries on x."""
    network: MLP
    lx: float
    
    def __init__(self, hidden_sizes: list[int], activation: str,lx: float, key: jax.Array):
        self.lx = lx
        self.network = MLP(
            in_size=4, #[t, cos(kx), sin(kx), v]
            out_size=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
            key=key
        )

    def __call__(self, txv_inputs: jnp.ndarray) -> jnp.ndarray:
        h = apply_spatial_periodic_embedding(txv_inputs, lx=self.lx)
        return self.network(h)
    
class SiRENScimbaINR(eqx.Module):
    """Spatio-temporal SIREN architecture"""
    layers: tuple
    omega_0: float = eqx.field(static=True)
    
    def __init__(self, in_size: int, out_size: int, hidden_sizes: list[int], omega_0: float, key: jax.Array):
        self.omega_0 = omega_0
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        sizes = [in_size] + hidden_sizes + [out_size]
        
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i]))
        self.layers = tuple(layers)
        
    def __call__(self, txv_inputs: jnp.ndarray) -> jnp.ndarray:
        x = jnp.sin(self.omega_0 * self.layers[0](txv_inputs)) #multiply by omega_0 for first layer
        for layer in self.layers[1:-1]: #all layers except the last one and the first one
            x = jnp.sin(layer(x))
        return self.layers[-1](x) #return the last layer of size out_size without activation
    
    def ndof(self) -> int:
        flat_params, _ = jax.tree_util.tree_flatten(self)
        return sum(p.size for p in flat_params if isinstance(p, jnp.ndarray))
    
class PeriodicSIRENScimbaINR(eqx.Module):
    """Spatio-temporel SIREN with hard periodic boundaries on x"""
    network: SiRENScimbaINR
    lx: float
    def __init__(self, hidden_sizes: list[int], omega_0: float,lx: float, key: jax.Array):
        self.lx = lx
        self.network = SiRENScimbaINR(
            in_size=4,
            out_size=1,
            hidden_sizes=hidden_sizes,
            omega_0=omega_0,
            key=key
        )
        
    def __call__(self, txv_inputs: jnp.ndarray) -> jnp.ndarray:
        h = apply_spatial_periodic_embedding(txv_inputs, lx=self.lx)
        return self.network(h)
    
    def ndof(self) -> int:
        return self.network.ndof()
    
class FourierMLPScimbaINR(eqx.Module):
    """Spatio-temporal Fourier features"""
    network: MLP
    B: jnp.ndarray
    
    def __init__(self, in_features: int, n_freqs: int, hidden_sizes: list[int], sigma: float, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        self.B = jax.random.normal(k1, (in_features, n_freqs)) * sigma
        
        self.network = MLP(
            in_size=2 * n_freqs,# 2 * n_freqs because we have both sin and cos features
            out_size=1,
            hidden_sizes=hidden_sizes,
            activation='tanh',
            key=k2
        )
        
    def __call__(self, txv_inputs: jnp.ndarray) -> jnp.ndarray:
        proj = txv_inputs @ jax.lax.stop_gradient(self.B)
        h = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
        return self.network(h)
    
class PeriodicFourierMLPScimbaINR(eqx.Module):
    """Spatio-temporal Fourier features + periodic embedding on x"""
    network: MLP
    B: jnp.ndarray
    lx: float
    
    def __init__(self, n_freqs: int, hidden_sizes: list[int], sigma: float,lx: float, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        self.B = jax.random.normal(k1, (4, n_freqs)) * sigma #4 input features embedding
        self.lx = lx
        
        self.network = MLP(
            in_size=2 * n_freqs,
            out_size=1,
            hidden_sizes=hidden_sizes,
            activation='tanh',
            key=k2
        )
        
    def __call__(self, txv_inputs: jnp.ndarray) -> jnp.ndarray:
        h = apply_spatial_periodic_embedding(txv_inputs, lx=self.lx)
        proj = h @ jax.lax.stop_gradient(self.B)
        h_fourier = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
        return self.network(h_fourier)
    

def get_pinn_network(arch: str, lx: float, key: jax.Array) -> eqx.Module:
        """Factory function to create a PINN network based on the specified architecture"""
        if arch == "periodic_siren_deep_128":
            return PeriodicSIRENScimbaINR(hidden_sizes=[128]*5, omega_0=30.0,lx=lx, key=key)
        #elif arch == "periodic_fourier_mlp_deep_128":
        else:
            raise ValueError(f"Unknown architecture: {arch}")