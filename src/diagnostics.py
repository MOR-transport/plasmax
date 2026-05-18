import csv
import jax.numpy as jnp 
from .config import Config
from pathlib import Path

def measure(cfg: Config, f: jnp.ndarray, Efield: jnp.ndarray, it: int, t_actual: float):
    """Calculates and saves the physical quantities of the simulation"""
    data_dir = cfg.paths.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    diag_file_path = data_dir / "diagnostics.csv" 
    
    t = t_actual
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
    
    file_exists = diag_file_path.is_file()
    with open(diag_file_path, mode="a", newline="") as file: 
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
        

def save_distribution(f: jnp.ndarray, filepath: Path) -> None:
    """Save the distribution function f to a .npy file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    jnp.save(filepath, f)
    print(f"Saved distribution function to {filepath}")
    