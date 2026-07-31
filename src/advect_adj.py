


"""Strang-split advection."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Grid

jax.config.update("jax_enable_x64", True)


def _cubic_periodic_weights(xi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Uniform cubic Lagrange weights for nodes at -1, 0, 1, 2 in index space; xi in [0,1)."""
    w0 = -(xi * (xi - 1) * (xi - 2)) / 6.0
    w1 = ((xi + 1) * (xi - 1) * (xi - 2)) / 2.0
    w2 = -((xi + 1) * xi * (xi - 2)) / 2.0
    w3 = ((xi + 1) * xi * (xi - 1)) / 6.0
    return w0, w1, w2, w3


def _cubic_periodic_weights_deriv(xi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Analytical derivatives of the cubic Lagrange weights with respect to xi."""
    dw0 = -(3.0 * xi**2 - 6.0 * xi + 2.0) / 6.0
    dw1 = (3.0 * xi**2 - 4.0 * xi - 1.0) / 2.0
    dw2 = -(3.0 * xi**2 - 2.0 * xi - 2.0) / 2.0
    dw3 = (3.0 * xi**2 - 1.0) / 6.0
    return dw0, dw1, dw2, dw3


def _cubic_periodic_scatter(lambda_in: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    """
    Adjoint of 1D sampling: scatter/deposit a periodic 1D adjoint field 
    back to the grid using fractional grid indices ``s``.
    """
    n = lambda_in.shape[0]
    s = jnp.mod(s, n)
    k = jnp.floor(s).astype(jnp.int32)
    xi = s - k
    
    idx0 = (k - 1) % n
    idx1 = k % n
    idx2 = (k + 1) % n
    idx3 = (k + 2) % n
    
    w0, w1, w2, w3 = _cubic_periodic_weights(xi)
    
    lambda_out = jnp.zeros_like(lambda_in)

    lambda_out = lambda_out.at[idx0].add(w0 * lambda_in)
    lambda_out = lambda_out.at[idx1].add(w1 * lambda_in)
    lambda_out = lambda_out.at[idx2].add(w2 * lambda_in)
    lambda_out = lambda_out.at[idx3].add(w3 * lambda_in)
    
    return lambda_out


def _cubic_dirichlet_scatter(lambda_in: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    n = lambda_in.shape[0]
    k = jnp.floor(s).astype(jnp.int32)
    xi = s - k
    w0, w1, w2, w3 = _cubic_periodic_weights(xi)
    
    lambda_out = jnp.zeros_like(lambda_in)
    
    indices = [k-1, k, k+1, k+2]
    weights = [w0, w1, w2, w3]

    for idx, w in zip(indices, weights):
        mask = (idx >= 0) & (idx < n)
        lambda_out = lambda_out.at[jnp.where(mask, idx, 0)].add(jnp.where(mask, w * lambda_in, 0.0))
        
    return lambda_out

def _adv_x_adj(lambda_f: jnp.ndarray, grid: Grid, dt: float) -> jnp.ndarray:
    """Adjoint advection along x. ``lambda_f`` shape (Nv, Nx)."""
    x_new = grid.X - grid.V * dt
    s = jnp.mod(x_new, grid.lx) / float(grid.dx)

    return jax.vmap(_cubic_periodic_scatter)(lambda_f, s)


def _adv_v_adj(lambda_f: jnp.ndarray, grid: Grid, efield: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Adjoint advection along v. ``lambda_f`` shape (Nv, Nx)."""
    lv = float(grid.lv)
    dv = float(grid.dv)
    
    v_new = grid.V + efield[jnp.newaxis, :] * dt
    s = (v_new + lv) / dv

    return jax.vmap(_cubic_dirichlet_scatter, in_axes=1, out_axes=1)(lambda_f, s)


def compute_mu(lambda_in: jnp.ndarray, f_star: jnp.ndarray, grid: Grid, efield: jnp.ndarray, dt: float) -> jnp.ndarray:
    lv = float(grid.lv)
    dv = float(grid.dv)
    
    # PLUS DE MODULO : On utilise la même coordonnée que _adv_v_adj
    v_new = grid.V + efield[jnp.newaxis, :] * dt
    s = (v_new + lv) / dv

    def process_col(l_col, f_col, s_col):
        n = f_col.shape[0]
        
        # k et xi sans jnp.mod(..., n) !
        k = jnp.floor(s_col).astype(jnp.int32)
        xi = s_col - k
        
        # Indices purs (peuvent sortir des bornes)
        idx0 = k - 1
        idx1 = k
        idx2 = k + 1
        idx3 = k + 2
        
        dw0, dw1, dw2, dw3 = _cubic_periodic_weights_deriv(xi)
        
        # Masquage Dirichlet STRICTEMENT identique à _cubic_dirichlet_scatter
        m0 = (idx0 >= 0) & (idx0 < n)
        m1 = (idx1 >= 0) & (idx1 < n)
        m2 = (idx2 >= 0) & (idx2 < n)
        m3 = (idx3 >= 0) & (idx3 < n)
        
        # On lit f_col seulement si l'indice est valide, sinon 0.0
        f0 = jnp.where(m0, f_col[jnp.where(m0, idx0, 0)], 0.0)
        f1 = jnp.where(m1, f_col[jnp.where(m1, idx1, 0)], 0.0)
        f2 = jnp.where(m2, f_col[jnp.where(m2, idx2, 0)], 0.0)
        f3 = jnp.where(m3, f_col[jnp.where(m3, idx3, 0)], 0.0)
        
        df_ds = (dw0 * f0 + dw1 * f1 + dw2 * f2 + dw3 * f3)
        
        return jnp.sum(l_col * df_ds) * (dt / dv)

    return jax.vmap(process_col, in_axes=1)(lambda_in, f_star, s)


def advect_adj(lambda_f: jnp.ndarray, f_star: jnp.ndarray, efield: jnp.ndarray, grid: Grid, dt: float, order: int) -> jnp.ndarray:
    """Adjoint of Strang split: x(dt/2) v(dt) x(dt/2). Only cubic (order 3) is implemented."""
    if order != 3:
        raise NotImplementedError(f"Only interp.order=3 is implemented (got {order}).")
    
    lambda_f = _adv_x_adj(lambda_f, grid, dt / 2.0)
    
    mu = compute_mu(lambda_f.copy(), f_star, grid, efield, dt)

    lambda_f = _adv_v_adj(lambda_f, grid, efield, dt)

    lambda_f = _adv_x_adj(lambda_f, grid, dt / 2.0)
    
    return lambda_f, mu
