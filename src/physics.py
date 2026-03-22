"""Vlasov–Poisson field solve (MATLAB ``vPoisson``)."""

from __future__ import annotations

import jax.numpy as jnp

from .config import Grid


def compute_density(f: jnp.ndarray, dv: float) -> jnp.ndarray:
    """∫ f dv along velocity axis (first axis is v, second is x)."""
    return jnp.sum(f, axis=0) * dv


def vpoisson(rho: jnp.ndarray, grid: Grid, charge: float) -> jnp.ndarray:
    """Return E(x) on the spatial grid (real space, length ``nx``)."""
    rho_hat = jnp.fft.fft(rho)
    
    phi_hat = rho_hat * grid.kx2_inverse
    phi_hat = phi_hat.at[0].set(0.0)
    dphi_hat = 1j * grid.kx * phi_hat
    return -jnp.real(jnp.fft.ifft(dphi_hat))

