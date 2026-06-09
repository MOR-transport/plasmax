"""Strang-split advection."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Grid

jax.config.update("jax_enable_x64", True)


def _cubic_periodic_weights(xi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Uniform cubic Lagrange weights for nodes at -1, 0, 1, 2 in index space; ``xi`` in [0,1)."""
    w0 = -(xi * (xi - 1) * (xi - 2)) / 6.0
    w1 = ((xi + 1) * (xi - 1) * (xi - 2)) / 2.0
    w2 = -((xi + 1) * xi * (xi - 2)) / 2.0
    w3 = ((xi + 1) * xi * (xi - 1)) / 6.0
    return w0, w1, w2, w3


def _cubic_left_weights(xi: jnp.array) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Uniform cubic Lagrange weights for left boundary."""
    w0 = -((xi - 1) * (xi - 2) * (xi - 3)) / 6.0
    w1 = (xi * (xi - 2) * (xi - 3)) / 2.0
    w2 = -(xi * (xi - 1) * (xi - 3)) / 2.0
    w3 = (xi * (xi - 1) * (xi - 2)) / 6.0
    return w0, w1, w2, w3


def _cubic_periodic_sample(f1d: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    """Sample a periodic uniform 1D field at fractional grid indices ``s`` (any real → wrapped)."""
    n = f1d.shape[0]
    s = jnp.mod(s, n)
    k = jnp.floor(s).astype(jnp.int32)
    xi = s - k
    km1 = (k - 1) % n
    kp1 = (k + 1) % n
    kp2 = (k + 2) % n
    y0 = f1d[km1]
    y1 = f1d[k]
    y2 = f1d[kp1]
    y3 = f1d[kp2]
    w0, w1, w2, w3 = _cubic_periodic_weights(xi)
    return w0 * y0 + w1 * y1 + w2 * y2 + w3 * y3


def _cubic_nonperiodic_sample(f1d: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    """Sample a nonperiodic uniform 1D field at fractional grid indices ``s`` (any real → wrapped)."""
    n = f1d.shape[0]
    s = jnp.clip(s, 0.0, n - 1.0)  # Remove modulo since we are nonperiodic

    # Left boundary
    w0, w1, w2, w3 = _cubic_left_weights(s)
    y_left = w0 * f1d[0] + w1 * f1d[1] + w2 * f1d[2] + w3 * f1d[3]

    # Center
    k = jnp.clip(jnp.floor(s).astype(jnp.int32), 1, n - 3)
    xi = s - k
    w0, w1, w2, w3 = _cubic_periodic_weights(xi)
    y_center = w0 * f1d[k - 1] + w1 * f1d[k] + w2 * f1d[k + 1] + w3 * f1d[k + 2]

    # Right boundary
    x_reflect = s - (n - 4)  # Use symmetry to reuse _cubic_left_weights
    w0, w1, w2, w3 = _cubic_left_weights(x_reflect)
    y_right = w0 * f1d[n - 4] + w1 * f1d[n - 3] + w2 * f1d[n - 2] + w3 * f1d[n - 1]

    # Select the correct y
    return jnp.where(s < 1.0,
                     y_left,
                     jnp.where(
                         s > (n - 3),
                         y_right,
                         y_center))


def _adv_x(f: jnp.ndarray, grid: Grid, dt: float) -> jnp.ndarray:
    """Advect along x with speed v (periodic in x). ``f`` shape (Nv, Nx)."""
    x_new = grid.X - grid.V * dt
    s = jnp.mod(x_new, grid.lx) / float(grid.dx)

    def row_advect(row: jnp.ndarray, sv: jnp.ndarray) -> jnp.ndarray:
        # ``in_axes=(None, 0)`` keeps ``row`` fixed so nested vmap does not add a spurious batch axis.
        return jax.vmap(_cubic_periodic_sample, in_axes=(None, 0))(row, sv)

    return jax.vmap(row_advect)(f, s)


def _adv_v(f: jnp.ndarray, grid: Grid, efield: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Advect along v with acceleration qE/m"""
    lv = float(grid.lv)
    dv = float(grid.dv)
    period_v = 2.0 * lv
    v_new = grid.V + efield[jnp.newaxis, :] * dt
    s = jnp.mod(v_new + lv, period_v) / dv

    def interp_col(f_col: jnp.ndarray, s_col: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(_cubic_nonperiodic_sample, in_axes=(None, 0))(f_col, s_col)

    return jax.vmap(interp_col, in_axes=1, out_axes=1)(f, s)


def advect(f: jnp.ndarray, efield: jnp.ndarray, grid: Grid, dt: float, order: int) -> jnp.ndarray:
    """Strang split: x(dt/2) v(dt) x(dt/2). Only cubic (order 3) is implemented."""
    if order != 3:
        raise NotImplementedError(f"Only interp.order=3 is implemented (got {order}).")
    f = _adv_x(f, grid, dt / 2.0)
    f = _adv_v(f, grid, efield, dt)
    f = _adv_x(f, grid, dt / 2.0)
    return f


def advect_with_source(f: jnp.ndarray, efield: jnp.ndarray, grid: Grid, dt: float, order: int, source) -> jnp.ndarray:
    """Strang split: x(dt/2) v(dt) x(dt/2). Only cubic (order 3) is implemented."""
    if order != 3:
        raise NotImplementedError(f"Only interp.order=3 is implemented (got {order}).")
    f = _adv_x(f, grid, dt / 2.0)
    f = _adv_v(f, grid, efield, dt/2.0)

    src_old = source(f)
    f_12 = f + dt*src_old
    f = f + (dt/2) * (src_old + source(f_12))

    f = _adv_v(f, grid, efield, dt/2.0)
    f = _adv_x(f, grid, dt / 2.0)
    return f


def advect_with_source_hist(f: jnp.ndarray, efield: jnp.ndarray, grid: Grid, dt: float, order: int, source, it):
    """Strang split: x(dt/2) v(dt) x(dt/2). Only cubic (order 3) is implemented."""
    if order != 3:
        raise NotImplementedError(f"Only interp.order=3 is implemented (got {order}).")
    f = _adv_x(f, grid, dt / 2.0)
    f = _adv_v(f, grid, efield, dt/2.0)

    f = f + (dt/2) * (source[it] + source[it-1])

    f = _adv_v(f, grid, efield, dt/2.0)
    f = _adv_x(f, grid, dt / 2.0)
    return f
