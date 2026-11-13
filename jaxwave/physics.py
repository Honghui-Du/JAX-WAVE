"""Second-order wave propagation with zero initial velocity and Dirichlet walls."""
import numpy as np


def laplacian(field, xp=np):
    n = field.shape[-1]
    pad = [(0, 0)] * (field.ndim - 2) + [(1, 1), (1, 1)]
    padded = xp.pad(field, pad)
    return (padded[..., 2:, 1:-1] + padded[..., :-2, 1:-1]
            + padded[..., 1:-1, 2:] + padded[..., 1:-1, :-2]
            - 4 * field) * (n + 1)**2


def validate_grid(speed, T=0.2, steps=160):
    speed = np.asarray(speed)
    if steps < 1 or T <= 0 or speed.shape[-1] != speed.shape[-2]:
        raise ValueError('Use a square grid, positive terminal time and time-step count.')
    if not np.isfinite(speed).all() or np.min(speed) <= 0:
        raise ValueError('Wave speeds must be finite and positive.')
    courant = T / steps * np.max(speed) * (speed.shape[-1] + 1)
    if courant > 1 / np.sqrt(2):
        raise ValueError('Time step violates the two-dimensional Courant condition.')


def propagate_numpy(pressure, speed, T=0.2, steps=160):
    dt = T / steps
    previous = pressure
    current = pressure + 0.5 * dt**2 * speed**2 * laplacian(pressure)
    for _ in range(1, steps):
        previous, current = current, (2 * current - previous
                                     + dt**2 * speed**2 * laplacian(current))
    return current


def propagate(pressure, speed, T=0.2, steps=160):
    import jax
    import jax.numpy as jnp
    dt = T / steps
    first = pressure + 0.5 * dt**2 * speed**2 * laplacian(pressure, jnp)

    def step(pair, _):
        previous, current = pair
        following = 2 * current - previous + dt**2 * speed**2 * laplacian(current, jnp)
        return (current, following), None

    return jax.lax.scan(step, (pressure, first), None, length=steps - 1)[0][1]


def adjoint(residual, speed, T=0.2, steps=160):
    return propagate(speed**2 * residual, speed, T, steps) / speed**2
