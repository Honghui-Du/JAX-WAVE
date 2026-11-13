"""Train a bounded inverse map through the discrete wave model."""
import argparse
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from flax.training import train_state
import optax
from .data import checksum, load_pairs
from .models import Model
from .physics import propagate, propagate_numpy, validate_grid


def train(args):
    if min(args.updates, args.batch_size, args.eval_every, args.width) < 1 or args.reg <= 0:
        raise ValueError('Training counts, model width and regularization must be positive.')
    target, speed = load_pairs(args.train)
    validation_target, validation_speed = load_pairs(args.validation)
    if target.shape[-1] % 8 or target.shape[-1] != validation_target.shape[-1]:
        raise ValueError('Training and validation grids must match and be divisible by eight.')
    validate_grid(speed, args.terminal_time, args.forward_steps)
    validate_grid(validation_speed, args.terminal_time, args.forward_steps)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    jax.config.update('jax_default_matmul_precision', 'highest')
    model = Model(width=args.width)
    params = model.init(jax.random.PRNGKey(args.seed),
                        jnp.asarray(target[:args.batch_size]),
                        jnp.asarray(speed[:args.batch_size]))['params']
    schedule = optax.cosine_decay_schedule(3e-4, args.updates, alpha=0.1)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params,
                                         tx=optax.adam(schedule))

    def objective(parameters, y, c):
        control = model.apply({'params': parameters}, y, c)
        field = propagate(control, c, args.terminal_time, args.forward_steps)
        denominator = jnp.sum(y * y, axis=(-2, -1))
        return jnp.mean((jnp.sum((field - y)**2, axis=(-2, -1))
                         + args.reg * jnp.sum(control**2, axis=(-2, -1))) / denominator)

    @jax.jit
    def update(current, y, c):
        loss, gradient = jax.value_and_grad(objective)(current.params, y, c)
        return current.apply_gradients(grads=gradient), loss

    predict = jax.jit(lambda p, y, c: model.apply({'params': p}, y, c))

    def evaluate(parameters):
        errors = []
        for start in range(0, len(validation_target), args.batch_size):
            y = validation_target[start:start + args.batch_size]
            c = validation_speed[start:start + args.batch_size]
            control = np.asarray(predict(parameters, jnp.asarray(y), jnp.asarray(c)))
            field = propagate_numpy(control.astype(np.float64), c.astype(np.float64),
                                    args.terminal_time, args.forward_steps)
            errors.extend(np.linalg.norm(field - y, axis=(-2, -1))
                          / np.linalg.norm(y, axis=(-2, -1)))
        errors = np.asarray(errors)
        return {'validation_mean_squared_relative': float(np.mean(errors**2)),
                'validation_mean_relative_l2': float(errors.mean())}

    config = {'width': args.width, 'seed': args.seed, 'updates': args.updates,
              'batch_size': args.batch_size, 'grid_size': target.shape[-1],
              'terminal_time': args.terminal_time, 'forward_steps': args.forward_steps,
              'regularization': args.reg, 'training_pairs': len(target),
              'validation_pairs': len(validation_target),
              'learning_rate_initial': 3e-4, 'learning_rate_final': 3e-5,
              'training_data_sha256': checksum(args.train),
              'validation_data_sha256': checksum(args.validation)}
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    random = np.random.default_rng(args.seed)
    history, best = [], float('inf')
    for step in range(args.updates + 1):
        loss = None
        if step:
            indices = random.integers(len(target), size=args.batch_size)
            state, loss = update(state, jnp.asarray(target[indices]), jnp.asarray(speed[indices]))
            loss = float(loss)
            if not np.isfinite(loss):
                raise RuntimeError('Nonfinite training loss.')
        if step == 0 or step % args.eval_every == 0 or step == args.updates:
            row = {'step': step, **evaluate(state.params)}
            if loss is not None:
                row['training_objective'] = loss
            if row['validation_mean_squared_relative'] < best:
                best = row['validation_mean_squared_relative']
                (output / 'best.msgpack').write_bytes(serialization.to_bytes(state.params))
            history.append(row)
            (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    (output / 'last.msgpack').write_bytes(serialization.to_bytes(state.params))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--updates', type=int, default=60000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--forward-steps', type=int, default=160)
    parser.add_argument('--terminal-time', type=float, default=0.2)
    parser.add_argument('--reg', type=float, default=2e-4)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
