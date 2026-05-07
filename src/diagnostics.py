import os 
import csv 
import jax.numpy as jnp 
from .config import Config

def measure(cfg: Config, f: jnp.ndarray, Efield: jnp.ndarray, it: int):
    """Calculates and saves the physical quantities of the simulation at iteration 'it'  """
    save_dir = f"results/{cfg.inicond.case}"  
    os.makedirs(save_dir, exist_ok=True) 
    
    diag_file_path = f"{save_dir}/diagnostics.csv" 
    
    t = it * cfg.time.dt 
    dx = cfg.grid.dx 
    dv = cfg.grid.dv
    dxdv = dx * dv
    
    epot = 0.5 * jnp.sum(Efield**2) * dx
    ekin = 0.5 * jnp.sum(f * (cfg.grid.V**2)) * dxdv
    etot = ekin + epot
    
    l2norm = jnp.sqrt(jnp.sum(f**2) * dxdv)
    mass = jnp.sum(f) * dxdv
    momentum = jnp.sum(f * cfg.grid.V) * dxdv
    
    data = {
        "iter": it,
        "time": float(t),
        "ekin": float(ekin),
        "epot": float(epot),
        "etot": float(etot),
        "l2norm": float(l2norm),
        "mass": float(mass),
        "momentum": float(momentum)
    }
    
    file_exists = os.path.isfile(diag_file_path)
    with open(diag_file_path, mode='a', newline='') as file: 
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
    