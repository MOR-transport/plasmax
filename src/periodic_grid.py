from .config import Grid
import jax.numpy as jnp

import jax
jax.config.update("jax_enable_x64", True)

def make_periodic_grid(grid: Grid) -> Grid:
    nx, nv = grid.nx, grid.nv
    x = jnp.linspace(0, grid.lx, nx, endpoint=False)
    v = jnp.linspace(-grid.lv, grid.lv, nv, endpoint=False)
    dx = x[1] - x[0]
    dv = v[1] - v[0]
    grid.x = x
    grid.v = v
    grid.X, grid.V = jnp.meshgrid(x, v, indexing="xy")
    grid.dx = dx
    grid.dv = dv

    grid.kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    grid.kx2 = grid.kx * grid.kx
    grid.kx2_inverse = grid.kx2.at[0].set(1.0)
    grid.kx2_inverse = 1 / grid.kx2_inverse
    return grid