"""U-Net and shared sign projection for initial-pressure design."""
import jax
import jax.numpy as jnp
from flax import linen as nn

class Block(nn.Module):
    width: int
    @nn.compact
    def __call__(self, x):
        for _ in range(2):
            x = nn.gelu(nn.Conv(self.width, (3,3))(x))
        return x

class Net(nn.Module):
    width: int = 16

    @nn.compact
    def __call__(self, y, c):
        x = jnp.stack([y, c - 1], -1)
        skips = []
        for w in [self.width, 2*self.width, 4*self.width]:
            x = Block(w)(x)
            skips.append(x)
            x = nn.avg_pool(x, (2, 2), (2, 2))
        x = Block(4*self.width)(x)
        for w, skip in zip([4*self.width, 2*self.width, self.width], reversed(skips)):
            x = jax.image.resize(x, (len(y), skip.shape[1], skip.shape[2], x.shape[-1]), 'bilinear')
            x = Block(w)(jnp.concatenate([x, skip], -1))
        x = nn.Conv(1, (1, 1))(x)[..., 0]
        return jnp.tanh(x)

class Model(nn.Module):
    width: int = 16

    @nn.compact
    def __call__(self, y, c):
        net = Net(width=self.width, name='Net_0')
        u = .5 * (net(y, c) - net(-y, c))
        return u
