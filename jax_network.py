from typing import Literal
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
from flax.nnx.rnglib import Rngs
from jax.nn import initializers
from functools import (
    partial,
    partialmethod,
)  # partialmethod for Dropout convenience classes

from einops import rearrange


# TENSOR_LOG = []
TENSOR_LOG = None


def log_tensor(name, val):
    if TENSOR_LOG is not None:
        TENSOR_LOG.append((name, np.array(val)))


# Helper for PRNG sequences if needed for repeated dropout masks, though usually dropout handles its own key.
# For this Dropout, we need to generate a mask, so a key is needed in __call__.


def linear_with_torch_initialization(
    in_features: int,
    out_features: int,
    *,
    rngs: nnx.Rngs,
    kernel_init=None,
    bias_init=None,
    **kwargs,
):
    if kernel_init is None:
        kernel_init = uniform_scaled_initializer(
            scale_factor=1, in_features=in_features
        )
    if bias_init is None:
        bias_init = uniform_scaled_initializer(scale_factor=1, in_features=in_features)
    return nnx.Linear(
        in_features,
        out_features,
        kernel_init=kernel_init,
        bias_init=bias_init,
        rngs=rngs,
        **kwargs,
    )


class Dropout(nnx.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: float, batch_dim: int | list[int]):
        """
        Args:
            r: Dropout rate
            batch_dim: Dimension(s) along which the dropout mask is shared
        """
        self.r = r
        if isinstance(batch_dim, int):
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout_layer = nnx.Dropout(rate=self.r)

    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = False,
        *,
        rngs: Rngs | None = None,
    ) -> jnp.ndarray:
        """
        Args:
            x: Tensor to which dropout is applied.
            deterministic: If true, dropout is disabled.
            rngs: Rngs for the dropout operation itself (key 'dropout' typically).
        """
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1

        # Create a mask of the reduced shape
        mask_values = jnp.ones(shape, dtype=x.dtype)
        mask = self.dropout_layer(mask_values, deterministic=deterministic, rngs=rngs)

        # Apply the broadcasted mask
        x = x * mask
        return x


class DropoutRowwise(Dropout):
    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)


class DropoutColumnwise(Dropout):
    __init__ = partialmethod(Dropout.__init__, batch_dim=-2)


def uniform_scaled_initializer(scale_factor: float, in_features: int):
    bound = (1.0 / (in_features**0.5)) * scale_factor

    def init(
        key, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32
    ) -> jnp.ndarray:
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init


class ScaledDotProductAttention(nnx.Module):
    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        self.temperature = temperature
        self.dropout = nnx.Dropout(attn_dropout)

    def __call__(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mask: jnp.ndarray | None = None,  # This is the bias term
        attn_mask: jnp.ndarray | None = None,  # This is for masking specific elements
        deterministic: bool = False,
        *,
        rngs: Rngs | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # K shape is (batch, num_heads, seq_len_k, d_k)
        # Q shape is (batch, num_heads, seq_len_q, d_k)
        # Transpose k for matmul: (batch, num_heads, d_k, seq_len_k)
        attn = jnp.matmul(q, jnp.swapaxes(k, 2, 3)) / self.temperature

        if mask is not None:
            attn = attn + mask

        if attn_mask is not None:
            attn = jnp.where(attn_mask == -1, -1e9, attn)

        attn_weights = jax.nn.softmax(attn, axis=-1)

        # Apply dropout to attention weights
        attn_weights_dropped = self.dropout(
            attn_weights, deterministic=deterministic, rngs=rngs
        )

        output = jnp.matmul(attn_weights_dropped, v)
        return (
            output,
            attn_weights,
        )


class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_k: int,
        d_v: int,
        kernel_init,
        dropout: float,
        attn_dropout: float,
        *,
        rngs: Rngs,
    ):
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nnx.Linear(
            d_model,
            n_head * d_k,
            use_bias=False,
            kernel_init=kernel_init(d_model),
            rngs=rngs,
        )
        self.w_ks = nnx.Linear(
            d_model,
            n_head * d_k,
            use_bias=False,
            kernel_init=kernel_init(d_model),
            rngs=rngs,
        )
        self.w_vs = nnx.Linear(
            d_model,
            n_head * d_v,
            use_bias=False,
            kernel_init=kernel_init(d_model),
            rngs=rngs,
        )
        # self.fc = nnx.Linear(n_head * d_v, d_model, use_bias=False, rngs=rngs) # in original, this fc is commented out

        self.attention = ScaledDotProductAttention(
            temperature=d_k**0.5, attn_dropout=attn_dropout
        )
        # self.dropout = nnx.Dropout(dropout) # Not used if fc is commented out
        # self.layer_norm = nnx.LayerNorm(d_model, rngs=rngs) # Not used in original forward

    def __call__(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mask: jnp.ndarray | None = None,  # Pairwise bias for attention
        src_mask: jnp.ndarray | None = None,  # Source padding mask
        deterministic: bool = False,
        *,
        rngs: Rngs | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.shape
        _, len_k, _ = k.shape
        _, len_v, _ = v.shape

        # residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).reshape(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).reshape(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).reshape(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q = jnp.transpose(q, axes=(0, 2, 1, 3))
        k = jnp.transpose(k, axes=(0, 2, 1, 3))
        v = jnp.transpose(v, axes=(0, 2, 1, 3))

        # The `mask` argument in PyTorch MHA is typically the attention mask (additive)
        # Here, `mask` is passed as `pairwise_bias` to ScaledDotProductAttention
        # `src_mask` is used to generate the `attn_mask` for ScaledDotProductAttention

        attn_mask = None
        if src_mask is not None:
            src_mask = src_mask.astype(jnp.int32)
            src_mask = jnp.expand_dims(src_mask, axis=-1).astype(jnp.int32)  # (B, L, 1)
            src_mask = jnp.where(src_mask == 0, -1, src_mask)  # (B, L, 1), pads are -1

            # So if `matmul_result` is -1, we mask.
            # (B, Lq, Lk)
            matmul_result = jnp.matmul(src_mask, jnp.transpose(src_mask, (0, 2, 1)))
            # (B, 1, Lq, Lk) for broadcasting to heads
            attn_mask = jnp.expand_dims(matmul_result, axis=1)

        # The `mask` argument for MHA is the `pairwise_bias`
        # The `attn_mask` for ScaledDotProductAttention is derived from `src_mask`
        q, attn = self.attention(
            q,
            k,
            v,
            mask=mask,
            attn_mask=attn_mask,
            deterministic=deterministic,
            rngs=rngs,
        )

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = jnp.transpose(q, axes=(0, 2, 1, 3))
        q = q.reshape(sz_b, len_q, -1)

        # Original PyTorch code comments out fc, residual, norm
        # q = self.dropout(self.fc(q), deterministic=deterministic, rngs=rngs)
        # q += residual
        # q = self.layer_norm(q)

        return q, attn


class TriangleMultiplicativeModule(nnx.Module):
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int | None = None,
        mix: Literal["ingoing", "outgoing"] = "ingoing",
        rngs: Rngs,
        kernel_init,
    ):
        assert mix in {"ingoing", "outgoing"}, "mix must be either ingoing or outgoing"

        hidden_dim = hidden_dim if hidden_dim is not None else dim
        self.norm = nnx.LayerNorm(dim, rngs=rngs)

        self.left_proj = nnx.Linear(
            dim, hidden_dim, kernel_init=kernel_init(dim), rngs=rngs
        )
        self.right_proj = nnx.Linear(
            dim, hidden_dim, kernel_init=kernel_init(dim), rngs=rngs
        )

        # Initialize all gates to be identity functions

        self.left_gate = nnx.Linear(
            dim,
            hidden_dim,
            kernel_init=initializers.zeros,
            bias_init=initializers.ones,
            rngs=rngs,
        )
        self.right_gate = nnx.Linear(
            dim,
            hidden_dim,
            kernel_init=initializers.zeros,
            bias_init=initializers.ones,
            rngs=rngs,
        )
        self.out_gate = nnx.Linear(
            dim,
            hidden_dim,
            kernel_init=initializers.zeros,
            bias_init=initializers.ones,
            rngs=rngs,
        )

        if mix == "outgoing":
            self.mix_einsum_eq = "... i k d, ... j k d -> ... i j d"
            self.left_t, self.right_t = (0, 3, 1, 2), (0, 3, 2, 1)
        elif mix == "ingoing":
            self.mix_einsum_eq = "... k i d, ... k j d -> ... i j d"
            self.left_t, self.right_t = (0, 3, 2, 1), (0, 3, 1, 2)

        self.to_out_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.to_out = nnx.Linear(
            hidden_dim, dim, rngs=rngs, kernel_init=kernel_init(hidden_dim)
        )

    def __call__(self, x: jnp.ndarray, src_mask: jnp.ndarray) -> jnp.ndarray:
        src_mask = src_mask.astype(jnp.float32)
        src_mask = jnp.expand_dims(src_mask, axis=-1)
        log_tensor("src_mask", src_mask)
        # (B, L, 1) * (B, 1, L) -> (B, L, L)
        mask = jnp.matmul(src_mask, src_mask.transpose((0, 2, 1)))
        mask = jnp.expand_dims(mask, axis=-1)
        log_tensor("mask", mask)

        assert x.shape[1] == x.shape[2], "feature map must be symmetrical"

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        log_tensor("left", left)
        log_tensor("right", right)

        left = left * mask
        right = right * mask

        log_tensor("left", left)
        log_tensor("right", right)

        left_gate = jax.nn.sigmoid(self.left_gate(x))
        right_gate = jax.nn.sigmoid(self.right_gate(x))

        out_gate = jax.nn.sigmoid(self.out_gate(x))

        left = left.transpose(self.left_t) * left_gate.transpose(self.left_t)
        right = right.transpose(self.right_t) * right_gate.transpose(self.right_t)

        out = jnp.matmul(left, right)
        # out = jnp.einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out.transpose((0, 2, 3, 1)))
        out = out * out_gate
        return self.to_out(out)


class TriangleAttention(nnx.Module):
    def __init__(
        self,
        in_dim: int = 128,
        dim: int = 32,
        n_heads: int = 4,
        wise: Literal["row", "col"] = "row",
        *,
        rngs: Rngs,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.wise = wise
        self.norm = nnx.LayerNorm(in_dim, rngs=rngs)
        self.to_qkv = linear_with_torch_initialization(
            in_dim, dim * 3 * n_heads, use_bias=False, rngs=rngs
        )
        self.linear_for_pair = linear_with_torch_initialization(
            in_dim, n_heads, use_bias=False, rngs=rngs
        )
        self.to_gate = nnx.Sequential(
            linear_with_torch_initialization(in_dim, in_dim, rngs=rngs), jax.nn.sigmoid
        )
        self.to_out = linear_with_torch_initialization(n_heads * dim, in_dim, rngs=rngs)

    def __call__(self, z: jnp.ndarray, src_mask: jnp.ndarray) -> jnp.ndarray:
        """
        how to do masking
        for row tri attention:
        attention matrix is brijh, where b is batch, r is row, h is head
        so mask should be b()ijh, i.e. take self attention mask and unsqueeze(1,-1)
        add negative inf to matrix before softmax

        for col tri attention
        attention matrix is bijlh, so take self attention mask and unsqueeze(3,-1)

        take src_mask and spawn pairwise mask, and unsqueeze accordingly
        """
        # src_mask: (B, L), 1 for token, 0 for pad
        # Create pairwise mask (additive, -inf for masked): (B, L, L)
        # Original: src_mask[src_mask == 0] = -1
        #           attn_mask = torch.matmul(src_mask_unsqueezed, src_mask_permuted)
        #           logits = logits.masked_fill(attn_mask == -1, float("-1e-9"))
        # This means if product is -1, mask.
        src_mask = jnp.where(src_mask == 0, -1, 1).astype(jnp.float32)  # (B,L)
        attn_mask = (
            jnp.expand_dims(src_mask, axis=-1)  # (B,L,1)
            * jnp.expand_dims(src_mask, axis=-2)  # (B,1,L)
        )

        z = self.norm(z)
        q, k, v = jnp.split(self.to_qkv(z), 3, axis=-1)
        q, k, v = map(
            lambda x: rearrange(x, "b i j (h d)->b i j h d", h=self.n_heads), (q, k, v)
        )
        b = self.linear_for_pair(z)  # (B, i, j, n_heads)
        gate = self.to_gate(z)  # (B, i, j, in_dim)
        scale = q.shape[-1] ** 0.5

        if self.wise == "row":
            eq_attn = "brihd,brjhd->brijh"
            eq_multi = "brijh,brjhd->brihd"
            b = rearrange(b, "b i j (r h)->b r i j h", r=1)
            softmax_dim = 3
            attn_mask = rearrange(attn_mask, "b i j->b 1 i j 1")
        elif self.wise == "col":
            eq_attn = "bilhd,bjlhd->bijlh"
            eq_multi = "bijlh,bjlhd->bilhd"
            b = rearrange(b, "b i j (l h)->b i j l h", l=1)
            softmax_dim = 2
            attn_mask = rearrange(attn_mask, "b i j->b i j 1 1")
        else:
            raise ValueError("wise should be col or row!")
        logits = jnp.einsum(eq_attn, q, k) / scale + b
        logits = jnp.where(attn_mask == -1, -1e-9, logits)
        attn = jax.nn.softmax(logits, axis=softmax_dim)
        out = jnp.einsum(eq_multi, attn, v)
        out = gate * rearrange(out, "b i j h d-> b i j (h d)")
        z_ = self.to_out(out)
        return z_


class OuterProductMean(nnx.Module):
    def __init__(
        self,
        in_dim: int = 256,
        dim_msa: int = 32,
        pairwise_dim: int = 64,
        kernel_init=None,
        bias_init=None,
        *,
        rngs: Rngs,
    ):
        self.proj_down1 = linear_with_torch_initialization(
            in_dim,
            dim_msa,
            kernel_init=kernel_init(in_dim) if kernel_init is not None else None,
            bias_init=bias_init,
            rngs=rngs,
        )
        self.proj_down2 = linear_with_torch_initialization(
            dim_msa**2,
            pairwise_dim,
            kernel_init=kernel_init(dim_msa**2) if kernel_init is not None else None,
            bias_init=bias_init,
            rngs=rngs,
        )

    def __call__(
        self, seq_rep: jnp.ndarray, pair_rep: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        # seq_rep: (B, L, in_dim)
        seq_rep_proj = self.proj_down1(seq_rep)  # (B, L, dim_msa)
        # Outer product: (B, L, L, dim_msa, dim_msa)
        outer_product = jnp.einsum("bid,bjc->bijcd", seq_rep_proj, seq_rep_proj)
        outer_product = rearrange(
            outer_product, "b i j c d -> b i j (c d)"
        )  # (B, L, L, dim_msa**2)
        outer_product = self.proj_down2(outer_product)  # (B, L, L, pairwise_dim)

        if pair_rep is not None:
            outer_product = outer_product + pair_rep
        return outer_product


class RelPos(nnx.Module):  # Renamed to avoid conflict if relpos is a function
    def __init__(self, dim: int = 64, max_rel_pos: int = 16, *, rngs: Rngs):
        # max_rel_pos defines the range [-max_rel_pos, max_rel_pos], so 2*max_rel_pos + 1 bins
        self.num_bins = 2 * max_rel_pos + 1
        self.linear = linear_with_torch_initialization(self.num_bins, dim, rngs=rngs)
        self.max_rel_pos = max_rel_pos

    def __call__(self, src: jnp.ndarray) -> jnp.ndarray:
        # src: (B, L, D_model) - only L is used.
        L = src.shape[1]
        res_id = jnp.arange(L)

        # Relative positions: (L, L)
        d = jnp.expand_dims(res_id, axis=1) - jnp.expand_dims(res_id, axis=0)

        # Clip to boundaries
        d_clipped = jnp.clip(d, a_min=-self.max_rel_pos, a_max=self.max_rel_pos)

        # Shift to be non-negative for one-hot encoding indices
        d_shifted = d_clipped + self.max_rel_pos  # Range [0, 2*max_rel_pos]

        # One-hot encode: (L, L, num_bins)
        d_onehot = jax.nn.one_hot(
            d_shifted, num_classes=self.num_bins, dtype=jnp.float32
        )

        # Project to embedding dimension: (L, L, dim)
        p = self.linear(d_onehot)  # Linear applies to the last dimension

        # Expand batch dim: (1, L, L, dim) and broadcast if src has batch
        # Or (B, L, L, dim) if we want to make it batch-dependent (not in original)
        # The original returns (L,L,dim) essentially, and PyTorch broadcasting handles batch.
        # For JAX, ensure it's broadcastable or explicitly tile.
        # Here, we return (L,L,dim) and rely on broadcasting when added to (B,L,L,dim) features.
        return p  # Shape (L, L, dim)


class ConvTransformerEncoderLayer(nnx.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        pairwise_dimension: int,
        use_triangular_attention: bool,
        dim_msa: int,
        dropout: float,
        attn_dropout: float,
        k: int,  # k is unused in original logic
        scale_factor: float,  # For custom init
        *,
        rngs: Rngs,
    ):
        dk = d_model // nhead
        dv = d_model // nhead

        # Self Attention: Apply custom init if applicable, or default like Xavier
        # Original doesn't show specific init for MHA's internal linears here, but RibonanzaNet loop implies it
        # Let's assume MHA internal linears get a default init like Xavier for now unless specified
        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dk,
            dv,
            dropout=dropout,
            attn_dropout=attn_dropout,
            kernel_init=lambda in_features: uniform_scaled_initializer(
                scale_factor, in_features
            ),
            rngs=rngs,
        )

        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout)
        self.dropout2 = nnx.Dropout(dropout)

        # Pairwise bias projection
        self.pairwise_norm = nnx.LayerNorm(pairwise_dimension, rngs=rngs)
        self.pairwise2heads = nnx.Linear(
            pairwise_dimension,
            nhead,
            use_bias=False,
            kernel_init=uniform_scaled_initializer(scale_factor, pairwise_dimension),
            rngs=rngs,
        )

        # Outer Product Mean
        self.outer_product_mean = OuterProductMean(
            in_dim=d_model,
            dim_msa=dim_msa,
            pairwise_dim=pairwise_dimension,
            kernel_init=lambda in_features: uniform_scaled_initializer(
                scale_factor, in_features
            ),
            bias_init=initializers.zeros,
            rngs=rngs,
        )
        # (OPM's internal linears would need scale_factor if recursive_init applied to them)

        # Triangle updates
        self.triangle_update_out = TriangleMultiplicativeModule(
            dim=pairwise_dimension,
            mix="outgoing",
            kernel_init=lambda in_features: uniform_scaled_initializer(
                scale_factor, in_features
            ),
            rngs=rngs,
        )
        self.triangle_update_in = TriangleMultiplicativeModule(
            dim=pairwise_dimension,
            mix="ingoing",
            kernel_init=lambda in_features: uniform_scaled_initializer(
                scale_factor, in_features
            ),
            rngs=rngs,
        )
        self.pair_dropout_out = DropoutRowwise(dropout)  # batch_dim=-3
        self.pair_dropout_in = DropoutRowwise(
            dropout
        )  # batch_dim=-3 (rowwise for (B,L,L,C))

        # Triangular Attention
        self.use_triangular_attention = use_triangular_attention
        if self.use_triangular_attention:
            self.triangle_attention_out = TriangleAttention(
                in_dim=pairwise_dimension,
                dim=pairwise_dimension // 4,  # dim_head for tri_attn
                wise="row",
                rngs=rngs,
            )
            self.triangle_attention_in = TriangleAttention(
                in_dim=pairwise_dimension,
                dim=pairwise_dimension // 4,
                wise="col",
                rngs=rngs,
            )
            self.pair_attention_dropout_out = DropoutRowwise(dropout)
            self.pair_attention_dropout_in = DropoutColumnwise(dropout)  # batch_dim=-2

        # Sequence Transition (FFN)
        self.sequence_transititon = nnx.Sequential(
            nnx.Linear(
                d_model,
                dim_feedforward,
                kernel_init=uniform_scaled_initializer(scale_factor, d_model),
                bias_init=initializers.zeros,
                rngs=rngs,
            ),
            nnx.relu,
            nnx.Linear(
                dim_feedforward,
                d_model,
                kernel_init=uniform_scaled_initializer(scale_factor, dim_feedforward),
                bias_init=initializers.zeros,
                rngs=rngs,
            ),
        )

        # Pair Transition (FFN for pairs)
        self.pair_transition = nnx.Sequential(
            nnx.LayerNorm(pairwise_dimension, rngs=rngs),
            nnx.Linear(
                pairwise_dimension,
                pairwise_dimension * 4,
                kernel_init=uniform_scaled_initializer(
                    scale_factor, pairwise_dimension
                ),
                bias_init=initializers.zeros,
                rngs=rngs,
            ),
            jax.nn.relu,
            nnx.Linear(
                pairwise_dimension * 4,
                pairwise_dimension,
                kernel_init=uniform_scaled_initializer(
                    scale_factor, pairwise_dimension * 4
                ),
                bias_init=initializers.zeros,
                rngs=rngs,
            ),
        )

    def __call__(
        self,
        src: jnp.ndarray,
        pairwise_features: jnp.ndarray,
        src_mask: jnp.ndarray | None,  # (B,L) 1 for token, 0 for pad
        deterministic: bool = False,
        return_aw: bool = False,
        *,
        rngs: Rngs,  # Pass down for dropout, MHA etc.
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray]:
        log_tensor("src", src)
        # Pairwise bias from pairwise_features
        pairwise_bias = self.pairwise2heads(
            self.pairwise_norm(pairwise_features)
        )  # (B, L, L, nhead)
        pairwise_bias = jnp.transpose(pairwise_bias, (0, 3, 1, 2))  # (B, nhead, L, L)
        log_tensor("pairwise_bias", pairwise_bias)

        # Self attention
        res = src
        # src_mask for MHA is (B,L)
        # MHA's mask argument is for the additive bias (pairwise_bias)
        # MHA's src_mask argument is for padding
        src, attention_weights = self.self_attn(
            src,
            src,
            src,
            mask=pairwise_bias,
            src_mask=src_mask,
            deterministic=deterministic,
            rngs=rngs,
        )
        log_tensor("src after self_attn", src)
        src = res + self.dropout1(src, deterministic=deterministic, rngs=rngs)
        log_tensor("src after dropout1", src)
        src = self.norm1(src)
        log_tensor("src after norm1", src)

        # Sequence transition
        res = src
        src = self.sequence_transititon(src)
        log_tensor("src after sequence transition", src)
        src = res + self.dropout2(src, deterministic=deterministic, rngs=rngs)
        log_tensor("src after dropout2", src)
        src = self.norm2(src)
        log_tensor("src after norm2", src)

        # Pair track ops
        pairwise_features = pairwise_features + self.outer_product_mean(src)
        log_tensor("pairwise_features after outer product mean", pairwise_features)
        pairwise_features = pairwise_features + self.pair_dropout_out(
            self.triangle_update_out(pairwise_features, src_mask=src_mask),
            deterministic=deterministic,
            rngs=rngs,
        )
        log_tensor("pairwise_features after triangle update out", pairwise_features)
        pairwise_features = pairwise_features + self.pair_dropout_in(
            self.triangle_update_in(pairwise_features, src_mask=src_mask),
            deterministic=deterministic,
            rngs=rngs,
        )
        log_tensor("pairwise_features after triangle update in", pairwise_features)

        if self.use_triangular_attention:
            log_tensor("Before triangle attention", pairwise_features)
            pairwise_features = pairwise_features + self.pair_attention_dropout_out(
                self.triangle_attention_out(pairwise_features, src_mask=src_mask),
                deterministic=deterministic,
                rngs=rngs,
            )
            log_tensor("After triangle attention out", pairwise_features)
            pairwise_features = pairwise_features + self.pair_attention_dropout_in(
                self.triangle_attention_in(pairwise_features, src_mask=src_mask),
                deterministic=deterministic,
                rngs=rngs,
            )
            log_tensor("After triangle attention in", pairwise_features)

        pairwise_features = pairwise_features + self.pair_transition(pairwise_features)
        log_tensor("pairwise features", pairwise_features)

        if return_aw:
            return src, pairwise_features, attention_weights
        else:
            return src, pairwise_features


def embedding_init(padding_idx: int):
    base_initializer = initializers.normal(stddev=1)

    def init(key, shape: tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
        return base_initializer(key, shape, dtype).at[padding_idx].set(0)

    return init


class RibonanzaNet(nnx.Module):
    def __init__(self, config, *, rngs: Rngs):
        self.config = config
        dim_feedforward = config.ninp * 4  # Standard feedforward dim

        # Embedding layer for source tokens
        self.encoder = nnx.Embed(
            num_embeddings=config.ntoken,
            features=config.ninp,
            embedding_init=embedding_init(padding_idx=4),
            rngs=rngs,
        )

        # Positional encoding (relative, not absolute from original Transformer)
        self.pos_encoder = RelPos(dim=config.pairwise_dimension, rngs=rngs)

        # Initial Outer Product Mean to create pairwise features
        self.outer_product_mean = OuterProductMean(
            in_dim=config.ninp,
            dim_msa=config.dim_msa,
            pairwise_dim=config.pairwise_dimension,
            rngs=rngs,
        )

        self.transformer_encoder = []
        for i in range(config.nlayers):
            if i != config.nlayers - 1:
                k = config.k
            else:
                k = 1
            scale_factor = 1 / (i + 1) ** 0.5

            self.transformer_encoder.append(
                ConvTransformerEncoderLayer(
                    d_model=config.ninp,
                    nhead=config.nhead,
                    dim_feedforward=dim_feedforward,
                    pairwise_dimension=config.pairwise_dimension,
                    use_triangular_attention=config.use_triangular_attention,
                    dim_msa=config.dim_msa,
                    dropout=config.dropout,
                    attn_dropout=config.attn_dropout,
                    k=k,
                    scale_factor=scale_factor,
                    rngs=rngs,
                )
            )

        self.decoder = nnx.Linear(
            in_features=config.ninp,
            out_features=config.nclass,
            kernel_init=uniform_scaled_initializer(scale_factor, config.ninp),
            bias_init=initializers.zeros,
            rngs=rngs,
        )

        self.use_gradient_checkpoint = False  # Control this externally if needed

    def __call__(
        self,
        src_tokens: jnp.ndarray,
        src_mask: jnp.ndarray | None = None,  # (B,L) 1 for token, 0 for pad
        deterministic: bool = True,  # True for eval, False for train
        *,
        rngs: Rngs | None = None,
    ):
        B, L = src_tokens.shape

        src = self.encoder(src_tokens)  # (B, L, ninp)

        # Note: If padding_idx was used in PyTorch, ensure padding tokens are zeroed out
        # or handled by masks appropriately. Here, src_mask will handle it.

        # 2. Create initial pairwise features
        # a. Outer product from sequence embeddings
        pairwise_features = self.outer_product_mean(src)  # (B, L, L, pairwise_dim)
        pairwise_features = pairwise_features + self.pos_encoder(src)

        # 3. Pass through Transformer Encoder Layers
        # src_mask (B,L) is used by MHA and TriangleAttention/Multiplicative modules

        for i, layer in enumerate(self.transformer_encoder):
            src, pairwise_features = layer(
                src,
                pairwise_features,
                src_mask=src_mask,
                deterministic=deterministic,
                rngs=rngs,
            )

        output = self.decoder(src)
        if self.decoder.out_features == 1:
            output = jnp.squeeze(output, axis=-1)
        return output

    def get_embeddings(
        self,
        src_tokens: jnp.ndarray,
        src_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
        *,
        rngs: Rngs | None = None,
    ):
        # This method is very similar to __call__ but returns intermediate embeddings
        if not deterministic and rngs is None:
            raise ValueError("Rngs must be provided for training mode.")

        B, L = src_tokens.shape
        src_embeddings = self.encoder(src_tokens)

        pairwise_features = self.outer_product_mean_initial(src_embeddings)
        pairwise_features = pairwise_features + self.pos_encoder(src_embeddings)

        current_src = src_embeddings
        current_pairwise_features = pairwise_features

        # Original get_embeddings always uses checkpointing for layers.
        # And always sets return_aw to False implicitly for layers.
        # This seems to be about getting the *final* src and pairwise_features.

        for i, layer in enumerate(self.transformer_encoder):
            # The PyTorch version had a self.custom wrapper for checkpointing.
            # For JAX, directly use jax.checkpoint.
            # The layer call requires src, pairwise, src_mask, return_aw=False, deterministic, rngs

            # Always use checkpoint for layers in get_embeddings as per original
            partial_layer_call = partial(
                layer.__call__,
                src_mask=src_mask,
                return_aw=False,  # Not returning attention weights from layers
                deterministic=deterministic,
                rngs=rngs,
            )
            # Inputs to checkpointed function are current_src, current_pairwise_features
            current_src, current_pairwise_features = jax.checkpoint(partial_layer_call)(
                current_src, current_pairwise_features
            )

        return current_src, current_pairwise_features
