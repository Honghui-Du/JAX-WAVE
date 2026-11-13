"""Predict bounded initial pressures from target fields and speed maps."""
import argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from .data import load_pairs
from .models import Model
from .physics import propagate_numpy, validate_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--forward-steps', type=int, default=160)
    parser.add_argument('--terminal-time', type=float, default=0.2)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive.')
    target, speed = load_pairs(args.input)
    if target.shape[-1] % 8:
        raise ValueError('The grid size must be divisible by eight.')
    validate_grid(speed, args.terminal_time, args.forward_steps)
    jax.config.update('jax_default_matmul_precision', 'highest')
    model = Model(width=args.width)
    template = model.init(jax.random.PRNGKey(0), jnp.asarray(target[:1]),
                          jnp.asarray(speed[:1]))['params']
    parameters = serialization.from_bytes(template, args.checkpoint.read_bytes())
    predict = jax.jit(lambda y, c: model.apply({'params': parameters}, y, c))
    controls, fields, errors = [], [], []
    for start in range(0, len(target), args.batch_size):
        y, c = target[start:start + args.batch_size], speed[start:start + args.batch_size]
        u = np.asarray(predict(jnp.asarray(y), jnp.asarray(c)))
        field = propagate_numpy(u.astype(np.float64), c.astype(np.float64),
                                args.terminal_time, args.forward_steps)
        controls.append(u)
        fields.append(field)
        errors.extend(np.linalg.norm(field - y, axis=(-2, -1))
                      / np.linalg.norm(y, axis=(-2, -1)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, initial_pressure=np.concatenate(controls),
                        terminal_field=np.concatenate(fields), relative_l2=np.asarray(errors))
    print('Mean relative field error:', float(np.mean(errors)))


if __name__ == '__main__':
    main()
