"""Vlasov–Poisson field solve (MATLAB ``vPoisson``)."""

from __future__ import annotations

import jax.numpy as jnp

from .config import Grid

import jax
jax.config.update("jax_enable_x64", True)


def compute_density(f: jnp.ndarray, dv: float) -> jnp.ndarray:
    """∫ f dv along velocity axis (first axis is v, second is x)."""
    return 1 - jnp.sum(f, axis=0) * dv


def compute_density_adj(lambda_rho: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    lambda_f = jnp.broadcast_to(lambda_rho, (grid.nv, grid.nx))
    return - lambda_f * grid.dv


def vpoisson(rho: jnp.ndarray, grid: Grid, charge: float) -> jnp.ndarray:
    """Return E(x) on the spatial grid (real space, length ``nx``)."""
    rho_hat = jnp.fft.fft(rho)

    phi_hat = rho_hat * grid.kx2_inverse
    phi_hat = phi_hat.at[0].set(0.0)
    dphi_hat = 1j * grid.kx * phi_hat
    return -jnp.real(jnp.fft.ifft(dphi_hat))


def vpoisson_adj(lambda_E: jnp.ndarray, grid: Grid, charge: float) -> jnp.ndarray:
    """
    Adjoint du solveur de Poisson.
    L'opérateur étant anti-symétrique (à cause de la dérivée spatiale), 
    son adjoint est exactement son opposé !
    """
    return - vpoisson(lambda_E, grid, charge)
