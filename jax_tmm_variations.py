from typing import Literal, Callable
from time import time
from functools import partial

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from flax.nnx.rnglib import Rngs
from tqdm import trange


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
    for _ in trange(n):
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
    """Use transpose + matmul but try to do transpositions before expensive matmul."""
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


class TriangleMultiplicativeModuleDot(nnx.Module):
    """vmap-based matmul to see how this relates to built-in matmul."""
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
        self.batched_matmul = jax.vmap(
            jax.vmap(
                self._my_matmul,
                in_axes=(0, 0),
                out_axes=0,
            ),
            in_axes=(0, 0),
            out_axes=0,
        )

        self.to_out_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.to_out = nnx.Linear(hidden_dim, dim, rngs=rngs)

    def _my_matmul(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
    ) -> jnp.ndarray:
        mv = jax.vmap(jnp.vdot, (0, None), 0)
        mm = jax.vmap(mv, (None, 1), 1)
        return mm(a, b)

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

        out = self.batched_matmul(
            left.transpose(self.perm_left),
            right.transpose(self.perm_right)
        ).transpose((0, 2, 3, 1))

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class TriangleMultiplicativeModuleDotOptimized(nnx.Module):
    """vmap-based matmul without transpositions."""
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

        # Note: both of these are equivalent to a simple matmul, but will serve as
        # a blueprint for a local update later
        if mix == "outgoing":
            def _my_matmul(
                a: jnp.ndarray,
                b: jnp.ndarray,
            ) -> jnp.ndarray:
                # This is a little bit slower then "ingoing" because of different memory access patterns
                mv = jax.vmap(jnp.vdot, (0, None), 0)
                mm = jax.vmap(mv, (None, 0), 1)
                return mm(a, b)
        elif mix == "ingoing":
            def _my_matmul(
                a: jnp.ndarray,
                b: jnp.ndarray,
            ) -> jnp.ndarray:
                # This is fast because it accesses memory of a and b contiguously
                mv = jax.vmap(jnp.vdot, (1, None), 0)
                mm = jax.vmap(mv, (None, 1), 1)
                return mm(a, b)
        self.batched_matmul = jax.vmap(
            jax.vmap(
                _my_matmul,
                in_axes=(2, 2),
                out_axes=2,
            ),
            in_axes=(0, 0),
            out_axes=0,
        )

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

        out = self.batched_matmul(left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)



class TriangleMultiplicativeModuleLocal(nnx.Module):
    """vmap-based matmul with local dot products."""
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int | None = None,
        window_size: int = 4,
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
            def _my_matmul(
                a: jnp.ndarray,
                b: jnp.ndarray,
            ) -> jnp.ndarray:
                # This is a little bit slower then "ingoing" because of different memory access patterns
                _local_dot = partial(self._local_dot, window_size=window_size)
                mv = jax.vmap(_local_dot, (0, None, 0), 0)
                mm = jax.vmap(mv, (None, 0, None), 1)
                return mm(a, b, jnp.arange(a.shape[0], dtype=jnp.int32))
        elif mix == "ingoing":
            def _my_matmul(
                a: jnp.ndarray,
                b: jnp.ndarray,
            ) -> jnp.ndarray:
                # This is fast because it accesses memory of a and b contiguously
                _local_dot = partial(self._local_dot, window_size=window_size)
                mv = jax.vmap(_local_dot, (1, None, 0), 0)
                mm = jax.vmap(mv, (None, 1, None), 1)
                return mm(a, b, jnp.arange(a.shape[0], dtype=jnp.int32))
        self.batched_matmul = jax.vmap(
            jax.vmap(
                _my_matmul,
                in_axes=(2, 2),
                out_axes=2,
            ),
            in_axes=(0, 0),
            out_axes=0,
        )

        self.to_out_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.to_out = nnx.Linear(hidden_dim, dim, rngs=rngs)

    @staticmethod
    def _local_dot(
        x: jnp.ndarray,
        y: jnp.ndarray,
        i: jnp.ndarray,
        window_size: int
    ) -> jnp.ndarray:
        ws = 2 * window_size + 1
        return jnp.vdot(
            jax.lax.dynamic_slice(jnp.pad(x, window_size, mode="constant", constant_values=0.0), (i,), (ws,)),
            jax.lax.dynamic_slice(jnp.pad(y, window_size, mode="constant", constant_values=0.0), (i,), (ws,))
        )


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

        out = self.batched_matmul(left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class TriangleMultiplicativeModuleTrivial(nnx.Module):
    """element-wise multiplication as lower bound for performance."""
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

        self.to_out_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.to_out = nnx.Linear(hidden_dim, dim, rngs=rngs)

    @staticmethod
    def _local_dot(
        x: jnp.ndarray,
        y: jnp.ndarray,
        i: int,
        window_size: int
    ) -> jnp.ndarray:
        ws = 2 * window_size + 1
        return jnp.vdot(
            jax.lax.dynamic_slice(jnp.pad(x, window_size, mode="constant", constant_values=0.0), (i,), (ws,)),
            jax.lax.dynamic_slice(jnp.pad(y, window_size, mode="constant", constant_values=0.0), (i,), (ws,))
        )


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

        out = left * right

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

    # print("\nOptimized Transpose + Matmul")
    # optimized = TriangleMultiplicativeModuleOptimizedTranspose(dim=SHAPE[3], mix=IN_OR_OUT)
    # benchmark(optimized, x, mask, expected, n=N)

    # print("\nDot product-based matmul")
    # dot = TriangleMultiplicativeModuleDot(dim=SHAPE[3], mix=IN_OR_OUT)
    # benchmark(dot, x, mask, expected, n=N)

    # print("\nDot product-based matmul (optimized)")
    # dot_optimized = TriangleMultiplicativeModuleDotOptimized(dim=SHAPE[3], mix=IN_OR_OUT)
    # benchmark(dot_optimized, x, mask, expected, n=N)

    print("\nLocal dot product-based matmul")
    # local = TriangleMultiplicativeModuleLocal(dim=SHAPE[3], mix=IN_OR_OUT, window_size=SHAPE[1])
    # benchmark(local, x, mask, expected, n=N)  # to verify correctness
    local = TriangleMultiplicativeModuleLocal(dim=SHAPE[3], mix=IN_OR_OUT, window_size=4)
    benchmark(local, x, mask, expected, n=N, rtol=1e8, atol=1e8)

    print("\nTrivial element-wise multiplication")
    trivial = TriangleMultiplicativeModuleTrivial(dim=SHAPE[3], mix=IN_OR_OUT)
    benchmark(trivial, x, mask, expected, n=N, rtol=1e8, atol=1e8)
