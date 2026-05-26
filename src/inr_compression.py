import jax
import jax.numpy as jnp 
import flax.linen as nn 
import optax 
from typing import Any 
from functools import partial 

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
    