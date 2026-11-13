"""Projected FISTA for the same bounded regularized inverse problem."""
import numpy as np
import jax
import jax.numpy as jnp
from .physics import propagate, adjoint, propagate_numpy, validate_grid


def solve(target, speed, iterations=256, regularization=2e-4,
          T=0.2, steps=160, restart=False):
    target, speed = np.asarray(target), np.asarray(speed)
    single = target.ndim == 2
    if single:
        target, speed = target[None], speed[None]
    if target.shape != speed.shape or target.ndim != 3:
        raise ValueError('Use matching target and speed arrays of shape (count, n, n).')
    if not np.isfinite(target).all():
        raise ValueError('Target fields must be finite.')
    if iterations < 0 or regularization <= 0:
        raise ValueError('Use nonnegative iteration counts and positive regularization.')
    validate_grid(speed, T, steps)
    y, c = jnp.asarray(target), jnp.asarray(speed)
    batch = len(y)

    def run():
        def update(_, state):
            u, v, momentum, bound, failures = state
            gradient = adjoint(propagate(v, c, T, steps) - y, c, T, steps) + regularization * v

            def trial(ell):
                candidate = jnp.clip(v - gradient / ell[:, None, None], -1, 1)
                displacement = candidate - v
                response = propagate(displacement, c, T, steps)
                left = jnp.sum(response**2 + regularization * displacement**2, axis=(-2, -1))
                right = ell * jnp.sum(displacement**2, axis=(-2, -1))
                accepted = left <= right * (1 + 2e-6) + 1e-12 * jnp.sum(y**2, axis=(-2, -1))
                return candidate, accepted

            candidate, accepted = trial(bound)

            def retry(state):
                ell, _, ok, count = state
                ell = jnp.where(ok, ell, 2 * ell)
                proposed, ok = trial(ell)
                return ell, proposed, ok, count + 1

            bound, candidate, accepted, _ = jax.lax.while_loop(
                lambda s: jnp.any(~s[2]) & (s[3] < 32), retry,
                (bound, candidate, accepted, jnp.array(0)))
            next_momentum = (1 + jnp.sqrt(1 + 4 * momentum**2)) / 2
            extrapolated = candidate + ((momentum - 1) / next_momentum)[:, None, None] * (candidate - u)
            if restart:
                reset = jnp.sum((v - candidate) * (candidate - u), axis=(-2, -1)) > 0
                extrapolated = jnp.where(reset[:, None, None], candidate, extrapolated)
                next_momentum = jnp.where(reset, 1, next_momentum)
            return candidate, extrapolated, next_momentum, bound, failures + jnp.sum(~accepted)

        zero = jnp.zeros_like(y)
        result = jax.lax.fori_loop(0, iterations, update,
                                  (zero, zero, jnp.ones(batch), jnp.ones(batch), jnp.array(0)))
        return result[0], result[4]

    control, failures = jax.jit(run)()
    if int(failures):
        raise RuntimeError('Backtracking did not find an acceptable step.')
    control = np.asarray(control)
    return control[0] if single else control


def evaluate(control, target, speed, regularization=2e-4, T=0.2, steps=160):
    u, y, c = [np.asarray(x, dtype=np.float64) for x in (control, target, speed)]
    residual = propagate_numpy(u, c, T, steps) - y
    gradient = propagate_numpy(c**2 * residual, c, T, steps) / c**2 + regularization * u
    denominator = np.sum(y**2, axis=(-2, -1))
    return {'relative_l2': np.sqrt(np.sum(residual**2, axis=(-2, -1)) / denominator),
            'normalized_objective': np.sum(residual**2 + regularization * u**2, axis=(-2, -1)) / denominator,
            'normalized_half_objective_gap_bound': np.sum(gradient * u + np.abs(gradient), axis=(-2, -1)) / denominator}
