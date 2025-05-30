from typing import Literal, Callable
from time import time

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from flax.nnx.rnglib import Rngs


def benchmark(
    func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    mask: jnp.ndarray,
    expected: jnp.ndarray,
    n: int,
    rtol: float = 1e-4,
    atol: float = 1e-4
):
    times = []
    func = jax.jit(func)
    for _ in range(n):
        start = time()
        result = func(x, mask).block_until_ready()
        end = time()
        times.append(end - start)
        if not jnp.allclose(result, expected, rtol=rtol, atol=atol):
            raise ValueError("Output does not match expected result.")

    jit_time = times[0]
    times = jnp.array(times[1:])
    avg_time = jnp.mean(times)
    std_time = jnp.std(times)
    print(f"Execution time: {avg_time:.6f} ± {std_time:.6f} seconds (JIT: {jit_time:.6f} seconds)")


class TriangleMultiplicativeModuleOriginal(nnx.Module):
    """The original implementation of the triangle multiplicative module."""
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int | None = None,
        mix: Literal["ingoing", "outgoing"] = "ingoing",
    ):
        assert mix in {"ingoing", "outgoing"}, "mix must be either ingoing or outgoing"
        rngs = Rngs(0)  # Use a fixed RNG for reproducibility

        hidden_dim = hidden_dim if hidden_dim is not None else dim
        self.norm = nnx.LayerNorm(dim, rngs=rngs)

        self.left_proj = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.right_proj = nnx.Linear(dim, hidden_dim, rngs=rngs)

        self.left_gate = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.right_gate = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.out_gate = nnx.Linear(dim, hidden_dim, rngs=rngs)

        if mix == "outgoing":
            self.mix_einsum_eq = "... i k d, ... j k d -> ... i j d"
        elif mix == "ingoing":
            self.mix_einsum_eq = "... k i d, ... k j d -> ... i j d"

        self.to_out_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.to_out = nnx.Linear(hidden_dim, dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, src_mask: jnp.ndarray) -> jnp.ndarray:
        src_mask = src_mask.astype(jnp.float32)
        src_mask = jnp.expand_dims(src_mask, axis=-1)
        # (B, L, 1) * (B, 1, L) -> (B, L, L)
        mask = jnp.matmul(src_mask, src_mask.transpose((0, 2, 1)))
        mask = jnp.expand_dims(mask, axis=-1)

        assert x.shape[1] == x.shape[2], "feature map must be symmetrical"

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        left = left * mask
        right = right * mask

        left_gate = jax.nn.sigmoid(self.left_gate(x))
        right_gate = jax.nn.sigmoid(self.right_gate(x))

        out_gate = jax.nn.sigmoid(self.out_gate(x))

        left = left * left_gate
        right = right * right_gate

        out = jnp.einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class TriangleMultiplicativeModuleMatmul(nnx.Module):
    """Use transpose + matmul instead of einsum."""
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int | None = None,
        mix: Literal["ingoing", "outgoing"] = "ingoing",
    ):
        assert mix in {"ingoing", "outgoing"}, "mix must be either ingoing or outgoing"
        rngs = Rngs(0)  # Use a fixed RNG for reproducibility

        hidden_dim = hidden_dim if hidden_dim is not None else dim
        self.norm = nnx.LayerNorm(dim, rngs=rngs)

        self.left_proj = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.right_proj = nnx.Linear(dim, hidden_dim, rngs=rngs)

        self.left_gate = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.right_gate = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.out_gate = nnx.Linear(dim, hidden_dim, rngs=rngs)

        if mix == "outgoing":
            self.perm_left, self.perm_right = (0, 3, 1, 2), (0, 3, 2, 1)
        elif mix == "ingoing":
            self.perm_left, self.perm_right = (0, 3, 2, 1), (0, 3, 1, 2)

        self.to_out_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.to_out = nnx.Linear(hidden_dim, dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, src_mask: jnp.ndarray) -> jnp.ndarray:
        src_mask = src_mask.astype(jnp.float32)
        src_mask = jnp.expand_dims(src_mask, axis=-1)
        # (B, L, 1) * (B, 1, L) -> (B, L, L)
        mask = jnp.matmul(src_mask, src_mask.transpose((0, 2, 1)))
        mask = jnp.expand_dims(mask, axis=-1)

        assert x.shape[1] == x.shape[2], "feature map must be symmetrical"

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        left = left * mask
        right = right * mask

        left_gate = jax.nn.sigmoid(self.left_gate(x))
        right_gate = jax.nn.sigmoid(self.right_gate(x))

        out_gate = jax.nn.sigmoid(self.out_gate(x))

        left = left * left_gate
        right = right * right_gate

        out = jnp.matmul(
            left.transpose(self.perm_left),
            right.transpose(self.perm_right)
        ).transpose((0, 2, 3, 1))

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class TriangleMultiplicativeModuleOptimizedTranspose(nnx.Module):
    """Use transpose + matmul but try do transpositions before expensive matmul."""
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int | None = None,
        mix: Literal["ingoing", "outgoing"] = "ingoing",
    ):
        assert mix in {"ingoing", "outgoing"}, "mix must be either ingoing or outgoing"
        rngs = Rngs(0)  # Use a fixed RNG for reproducibility

        hidden_dim = hidden_dim if hidden_dim is not None else dim
        self.norm = nnx.LayerNorm(dim, rngs=rngs)

        self.left_proj = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.right_proj = nnx.Linear(dim, hidden_dim, rngs=rngs)

        self.left_gate = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.right_gate = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.out_gate = nnx.Linear(dim, hidden_dim, rngs=rngs)

        if mix == "outgoing":
            self.perm_left, self.perm_right = (0, 3, 1, 2), (0, 3, 2, 1)
        elif mix == "ingoing":
            self.perm_left, self.perm_right = (0, 3, 2, 1), (0, 3, 1, 2)

        self.to_out_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.to_out = nnx.Linear(hidden_dim, dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, src_mask: jnp.ndarray) -> jnp.ndarray:
        src_mask = src_mask.astype(jnp.float32)
        src_mask = jnp.expand_dims(src_mask, axis=-1)
        # (B, L, 1) * (B, 1, L) -> (B, L, L)
        mask = jnp.matmul(src_mask, src_mask.transpose((0, 2, 1)))
        mask = jnp.expand_dims(mask, axis=-1)

        assert x.shape[1] == x.shape[2], "feature map must be symmetrical"

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        left = left * mask
        right = right * mask

        left_gate = jax.nn.sigmoid(self.left_gate(x))
        right_gate = jax.nn.sigmoid(self.right_gate(x))

        out_gate = jax.nn.sigmoid(self.out_gate(x))

        left = left.transpose(self.perm_left) * left_gate.transpose(self.perm_left)
        right = right.transpose(self.perm_right) * right_gate.transpose(self.perm_right)

        out = jnp.matmul(left, right).transpose((0, 2, 3, 1))

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)



if __name__ == "__main__":
    SHAPE = (8, 177, 177, 128)  # (Batch size, Sequence length, Feature dimension)
    IN_OR_OUT = "outgoing"
    N = 100
    
    # Random input data (values not important)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, SHAPE, dtype=jnp.float32)
    mask = jax.random.randint(key, (SHAPE[0], SHAPE[1]), 0, 2, dtype=jnp.int32)

    original = TriangleMultiplicativeModuleOriginal(dim=SHAPE[3], mix=IN_OR_OUT)
    expected = original(x, mask)

    print("\nOriginal")
    benchmark(original, x, mask, expected, n=N)

    print("\nMatmul")
    matmul = TriangleMultiplicativeModuleMatmul(dim=SHAPE[3], mix=IN_OR_OUT)
    benchmark(matmul, x, mask, expected, n=N)

    print("\nOptimized Transpose + Matmul")
    optimized = TriangleMultiplicativeModuleOptimizedTranspose(dim=SHAPE[3], mix=IN_OR_OUT)
    benchmark(optimized, x, mask, expected, n=N)
