# Original file: hr/chropgsxbgkdrwqecisjpu6dkbqiifvxevavqmjtqvxjruq4akww.debug/output_code.py
# Type: Backward
# Generated for TriangleMultiplicativeModule
# PyTorch version: 2.7.0+cu126

# AOT ID: ['0_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/triton_kernels_ybb5buya/e2/ce2m7s4b24gddyxrex4dz4y4irf44gypfmyugi3cja4vdn6gz6j5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_20, [0], True), kwargs = {})
triton_red_fused_sum_0 = async_compile.triton('triton_red_fused_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '2B4DCE383ADC6F6AC9E37B7BE08352A0B047033492CBA4C9904681B32954FEA9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 67968
    r0_numel = 236
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 30208*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/triton_kernels_ybb5buya/7h/c7hlcvjwrjx2ovgb53ngzoepf52fxwy3q27zsjuucczxweddi47r.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_20, [0], True), kwargs = {})
triton_red_fused_sum_1 = async_compile.triton('triton_red_fused_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '2B4DCE383ADC6F6AC9E37B7BE08352A0B047033492CBA4C9904681B32954FEA9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 531
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/triton_kernels_ybb5buya/t7/ct7tzllayasu44fn3b5jpbkjxtgirendwxniyewsrksw6hq6wf4s.py
# Topologically Sorted Source Nodes: [out_1, out_gate], Original ATen: [aten.native_layer_norm, aten.mul, aten.sigmoid, aten.native_layer_norm_backward, aten.sigmoid_backward]
# Source node to ATen node mapping:
#   out_1 => add_3, clone_2, mul_6, mul_7, sub_1
#   out_gate => sigmoid_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_17,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_2, %getitem_3), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %primals_15), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %primals_16), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, %add_3), kwargs = {})
#   %sigmoid_2 : [num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_13,), kwargs = {})
#   %mul_10 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, %sigmoid_2), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_17, %getitem_3), kwargs = {})
#   %mul_11 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_12 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %primals_15), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_12, [3], True), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %mul_11), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_14, [3], True), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_2), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_2, %sub_5), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %mul_22), kwargs = {})
triton_red_fused_mul_native_layer_norm_native_layer_norm_backward_sigmoid_sigmoid_backward_2 = async_compile.triton('triton_red_fused_mul_native_layer_norm_native_layer_norm_backward_sigmoid_sigmoid_backward_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_layer_norm_native_layer_norm_backward_sigmoid_sigmoid_backward_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': '2B4DCE383ADC6F6AC9E37B7BE08352A0B047033492CBA4C9904681B32954FEA9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mul_native_layer_norm_native_layer_norm_backward_sigmoid_sigmoid_backward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 125316
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x0 = (xindex % 31329)
    x1 = xindex // 31329
    tmp10 = tl.load(in_ptr4 + (x0 + 31360*x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0 + 31360*x1), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x3), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (x0 + 31329*r0_2 + 4010112*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr6 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp5 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask & xmask, tmp17, _tmp16)
        tmp18 = tmp13 * tmp4
        tmp20 = tmp18 + tmp19
        tmp21 = tmp0 * tmp20
        tmp22 = 1.0
        tmp23 = tmp22 - tmp2
        tmp24 = tmp2 * tmp23
        tmp25 = tmp21 * tmp24
        tl.store(out_ptr2 + (r0_2 + 128*x3), tmp25, xmask & r0_mask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 31360*x1), tmp7, xmask)
    tl.store(out_ptr1 + (x0 + 31360*x1), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/triton_kernels_ybb5buya/2z/c2zeykhfcnuhp4oloiccjfwoa44x6yotik7jvt2pusug3664f655.py
# Topologically Sorted Source Nodes: [out_gate], Original ATen: [aten.sigmoid, aten.mul, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   out_gate => sigmoid_2
# Graph fragment:
#   %sigmoid_2 : [num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_13,), kwargs = {})
#   %mul_10 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, %sigmoid_2), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_17, %getitem_3), kwargs = {})
#   %mul_11 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %mul_11), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_17, [0, 1, 2]), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_10, [0, 1, 2]), kwargs = {})
triton_red_fused_mul_native_layer_norm_backward_sigmoid_3 = async_compile.triton('triton_red_fused_mul_native_layer_norm_backward_sigmoid_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_layer_norm_backward_sigmoid_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '2B4DCE383ADC6F6AC9E37B7BE08352A0B047033492CBA4C9904681B32954FEA9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mul_native_layer_norm_backward_sigmoid_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 67968
    r0_numel = 236
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 30208*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 30208*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (31329*x0 + 4010112*((r0_2 + 236*x1) // 31329) + (((r0_2 + 236*x1) % 31329))), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (31360*((r0_2 + 236*x1) // 31329) + (((r0_2 + 236*x1) % 31329))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr4 + (31360*((r0_2 + 236*x1) // 31329) + (((r0_2 + 236*x1) % 31329))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp6 = tmp4 - tmp5
        tmp8 = tmp6 * tmp7
        tmp9 = tmp3 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tmp13 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/triton_kernels_ybb5buya/ca/ccaoosvtlhm5hxv47fvvrmvcbiwvwzlacqgshbpyoltzabjb2tcv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_16,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 32768}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '2B4DCE383ADC6F6AC9E37B7BE08352A0B047033492CBA4C9904681B32954FEA9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 31329
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = yindex // 128
    y0 = (yindex % 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 31360*y1), ymask & xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + 128*x2 + 4010112*y1), ymask & xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + 128*x2 + 4010112*y1), ymask & xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x2 + 31360*y1), ymask & xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2 + 31329*y3), ymask & xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x2 + 31360*y1), ymask & xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2 + 31360*y1), ymask & xmask, eviction_policy='evict_last')
    tmp1 = 0.0078125
    tmp2 = tmp0 * tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = 128.0
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 - tmp11
    tmp15 = tmp13 - tmp14
    tmp16 = tmp15 * tmp0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp12 - tmp18
    tmp20 = tmp2 * tmp19
    tl.store(out_ptr0 + (x2 + 31360*y3), tmp20, ymask & xmask)
''', device_str='cuda')


# kernel path: /tmp/triton_kernels_ybb5buya/ts/cts574mpfidjbox4z3iagch26omhoxzn6roli2ztbmbmz65knven.py
# Topologically Sorted Source Nodes: [right_1, right_gate], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.clone]
# Source node to ATen node mapping:
#   right_1 => mul_3
#   right_gate => sigmoid_1
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, %view_3), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, %mul_3), kwargs = {})
#   %sigmoid_1 : [num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_11,), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, %sigmoid_1), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_1), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_1, %sub_6), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %mul_24), kwargs = {})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%mul_25,), kwargs = {memory_format: torch.contiguous_format})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %view_3), kwargs = {})
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%mul_28,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_sigmoid_sigmoid_backward_5 = async_compile.triton('triton_poi_fused_clone_mul_sigmoid_sigmoid_backward_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 32768}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_sigmoid_sigmoid_backward_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '2B4DCE383ADC6F6AC9E37B7BE08352A0B047033492CBA4C9904681B32954FEA9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_sigmoid_sigmoid_backward_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 708
    xnumel = 22656
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex % 128)
    x3 = xindex // 128
    y0 = (yindex % 177)
    y1 = yindex // 177
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 177*x3 + 31329*x2 + 4010112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x5 + 22656*y4), ymask & xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x3 + 177*y4), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x5 + 22656*y4), ymask & xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = tmp0 * tmp6
    tmp12 = tmp11 * tmp2
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + 22656*y4), tmp10, ymask & xmask)
    tl.store(out_ptr0 + (x5 + 22656*y4), tmp12, ymask & xmask)
''', device_str='cuda')


# kernel path: /tmp/triton_kernels_ybb5buya/m4/cm4337h5g4mdnuapqr2xmchoh5kgwa35lrlz33g4oa2evxn5bm66.py
# Topologically Sorted Source Nodes: [left_1, left_gate], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.clone]
# Source node to ATen node mapping:
#   left_1 => mul_2
#   left_gate => sigmoid
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %view_3), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, %mul_2), kwargs = {})
#   %sigmoid : [num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_9,), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, %sigmoid), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %sub_7), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %mul_26), kwargs = {})
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%mul_27,), kwargs = {memory_format: torch.contiguous_format})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %view_3), kwargs = {})
#   %clone_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%mul_29,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_sigmoid_sigmoid_backward_6 = async_compile.triton('triton_poi_fused_clone_mul_sigmoid_sigmoid_backward_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_sigmoid_sigmoid_backward_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '2B4DCE383ADC6F6AC9E37B7BE08352A0B047033492CBA4C9904681B32954FEA9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_sigmoid_sigmoid_backward_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 125316
    xnumel = 128
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 31329)
    y1 = yindex // 31329
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 31329*x2 + 4010112*y1), ymask & xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + 128*y3), ymask & xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + 128*y3), ymask & xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = tmp0 * tmp6
    tmp12 = tmp11 * tmp2
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 128*y3), tmp10, ymask & xmask)
    tl.store(out_ptr0 + (x2 + 128*y3), tmp12, ymask & xmask)
''', device_str='cuda')


# kernel path: /tmp/triton_kernels_ybb5buya/73/c73gssaafojnx2un6dlvbuvjdgwqhml3fcdbaf2el6rofk4wglof.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x => mul, sub
# Graph fragment:
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_29, %view_32), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_35), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_38), kwargs = {})
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_41), kwargs = {})
#   %mul_31 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, %primals_3), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, 128), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_31, [3], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %getitem_1), kwargs = {})
#   %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %mul), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_33, [3], True), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %sum_12), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_32, %sum_11), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_9, %mul_34), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 128), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %sub_10), kwargs = {})
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_7 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 2, 'backend_hash': '2B4DCE383ADC6F6AC9E37B7BE08352A0B047033492CBA4C9904681B32954FEA9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 125316
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x2 = (xindex % 31329)
    x3 = xindex // 31329
    tmp15 = tl.load(in_ptr6 + (x2 + 31360*x3), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2 + 31360*x3), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r0_1 + 128*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r0_1 + 128*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr3 + (r0_1 + 128*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr5 + (r0_1 + 128*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp10 = tmp8 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp10 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
        tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp8, xmask & r0_mask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp25 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr5 + (r0_1 + 128*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp23 = 0.0078125
        tmp24 = tmp17 * tmp23
        tmp27 = tmp25 * tmp26
        tmp28 = 128.0
        tmp29 = tmp27 * tmp28
        tmp30 = tmp29 - tmp12
        tmp32 = tmp31 - tmp15
        tmp33 = tmp32 * tmp17
        tmp34 = tmp33 * tmp21
        tmp35 = tmp30 - tmp34
        tmp36 = tmp24 * tmp35
        tl.store(out_ptr2 + (r0_1 + 128*x0), tmp36, xmask & r0_mask)
''', device_str='cuda')


# kernel path: /tmp/triton_kernels_ybb5buya/5i/c5iyefwzqsrrdrsgfw27lftehd2pd2bbasdynrvhoe6rwceerec5.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   x => mul, sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %getitem_1), kwargs = {})
#   %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, %mul), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_36, [0, 1, 2]), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_7, [0, 1, 2]), kwargs = {})
triton_red_fused_native_layer_norm_native_layer_norm_backward_8 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '2B4DCE383ADC6F6AC9E37B7BE08352A0B047033492CBA4C9904681B32954FEA9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 67968
    r0_numel = 236
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 30208*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 30208*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (31360*((r0_2 + 236*x1) // 31329) + (((r0_2 + 236*x1) % 31329))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (31360*((r0_2 + 236*x1) // 31329) + (((r0_2 + 236*x1) % 31329))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = tmp0 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
        tmp10 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''', device_str='cuda')


# Local Attention Backward BMM Kernel for TriangleMultiplicativeModule outgoing einsum
@triton.jit
def triton_local_attention_backward_bmm_kernel(
    grad_output_ptr, left_ptr, right_ptr, 
    grad_left_ptr, grad_right_ptr,
    batch_size, M, N, K, window_size,
    stride_grad_output_batch, stride_grad_output_m, stride_grad_output_n,
    stride_left_batch, stride_left_m, stride_left_k,
    stride_right_batch, stride_right_k, stride_right_n,
    stride_grad_left_batch, stride_grad_left_m, stride_grad_left_k,
    stride_grad_right_batch, stride_grad_right_k, stride_grad_right_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8, grid_m: tl.constexpr = 177, grid_n: tl.constexpr = 177, ACC_TYPE: tl.constexpr = tl.float32
):
    """
    Computes gradients for left and right tensors with k range limited to window_size around i
    
    Forward: 
        out[i,j] = sum_k left[i,k] * right[j,k] where |i-k| <= window_size
    Backward: 
        grad_left[i,k] = sum_j grad_output[i,j] * right[j,k] where |i-k| <= window_size
        grad_right[j,k] = sum_i grad_output[i,j] * left[i,k] where |i-k| <= window_size
    """
    pid = tl.program_id(0)
    pid_batch = tl.program_id(1)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    
    acc_grad_left = tl.zeros((BLOCK_M, BLOCK_K), dtype=ACC_TYPE)
    acc_grad_right = tl.zeros((BLOCK_K, BLOCK_N), dtype=ACC_TYPE)
    
    m_center = pid_m * BLOCK_M + BLOCK_M // 2  # Center of current m block
    k_start = tl.maximum(0, m_center - window_size)
    k_end = tl.minimum(K, m_center + window_size + 1)
    
    k_start_block = (k_start // BLOCK_K) * BLOCK_K
    k_end_block = ((k_end + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
    
    # Main loop
    rk = tl.arange(0, BLOCK_K)
    
    for k_offset in range(k_start_block, k_end_block, BLOCK_K):
        if k_offset < K:
            k_indices = k_offset + rk
            remaining_k = K - k_offset
            
            # Create local mask for this k block
            m_indices = ram[:, None]  # Shape: (BLOCK_M, 1)
            k_indices_2d = k_indices[None, :]  # Shape: (1, BLOCK_K)
            
            # |m - k| <= window_size
            local_mask = tl.abs(m_indices - k_indices_2d) <= window_size
            
            # Boundary masks
            m_mask = ram < M
            k_mask = k_indices < K
            n_mask = rbn < N
            
            # Combined masks
            mk_mask = (m_mask[:, None] & k_mask[None, :]) & local_mask
            kn_mask = (k_mask[:, None] & n_mask[None, :])
            mn_mask = (m_mask[:, None] & n_mask[None, :])
            
            # Memory pointers
            grad_output_ptrs = (pid_batch * stride_grad_output_batch + 
                              ram[:, None] * stride_grad_output_m + 
                              rbn[None, :] * stride_grad_output_n)
            left_ptrs = (pid_batch * stride_left_batch + 
                        ram[:, None] * stride_left_m + 
                        k_indices[None, :] * stride_left_k)
            right_ptrs = (pid_batch * stride_right_batch +
                         rbn[:, None] * stride_right_k +
                         k_indices[None, :] * stride_right_n)
            
            # Load tensors
            if remaining_k >= BLOCK_K:
                grad_output_block = tl.load(grad_output_ptr + grad_output_ptrs, mask=mn_mask, other=0.0)
                left_block = tl.load(left_ptr + left_ptrs, mask=mk_mask, other=0.0)
                right_block = tl.load(right_ptr + right_ptrs, mask=kn_mask, other=0.0)
            else:
                # k boundary check
                k_boundary_mask = k_indices < K
                mk_final_mask = mk_mask & k_boundary_mask[None, :]
                kn_final_mask = kn_mask & k_boundary_mask[:, None]
                
                grad_output_block = tl.load(grad_output_ptr + grad_output_ptrs, mask=mn_mask, other=0.0)
                left_block = tl.load(left_ptr + left_ptrs, mask=mk_final_mask, other=0.0)
                right_block = tl.load(right_ptr + right_ptrs, mask=kn_final_mask, other=0.0)
            
            # grad_left[i,k] = sum_j grad_output[i,j] * right[j,k] where |i-k| <= window_size
            grad_left_update = tl.dot(grad_output_block, right_block, allow_tf32=True)
            acc_grad_left += tl.where(mk_mask, grad_left_update, 0.0)
            
            # grad_right[j,k] = sum_i grad_output[i,j] * left[i,k] where |i-k| <= window_size
            grad_right_update = tl.dot(left_block, grad_output_block, allow_tf32=True)
            acc_grad_right += tl.where(kn_mask, grad_right_update, 0.0)
    
    # Store grad_left
    grad_left_ptrs = (pid_batch * stride_grad_left_batch + 
                     ram[:, None] * stride_grad_left_m + 
                     rk[None, :] * stride_grad_left_k)
    mk_store_mask = (ram[:, None] < M) & (rk[None, :] < K)
    tl.store(grad_left_ptr + grad_left_ptrs, acc_grad_left, mask=mk_store_mask)
    
    # Store grad_right
    grad_right_ptrs = (pid_batch * stride_grad_right_batch + 
                      rbn[:, None] * stride_grad_right_k + 
                      rk[None, :] * stride_grad_right_n)
    kn_store_mask = (rbn[:, None] < N) & (rk[None, :] < K)
    tl.store(grad_right_ptr + grad_right_ptrs, acc_grad_right, mask=kn_store_mask)


def launch_local_attention_backward_bmm_kernel(grad_output, left, right, grad_left, grad_right, window_size=16):
    """
        grad_output: Gradient of output tensor (batch_size, M, N)
        left: Left input tensor (batch_size, M, K)
        right: Right input tensor (batch_size, N, K)  
        grad_left: Gradient of left tensor (batch_size, M, K)
        grad_right: Gradient of right tensor (batch_size, N, K)
        window_size: Local attention window size (default: 16)
    """
    batch_size, M, N = grad_output.shape
    _, M_left, K_left = left.shape
    _, N_right, K_right = right.shape
    K = K_left

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    GROUP_M = 8

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = grid_m
    total_programs = grid_m * grid_n

    stride_grad_output_batch, stride_grad_output_m, stride_grad_output_n = grad_output.stride()
    stride_left_batch, stride_left_m, stride_left_k = left.stride()
    stride_right_batch, stride_right_k, stride_right_n = right.stride()
    stride_grad_left_batch, stride_grad_left_m, stride_grad_left_k = grad_left.stride()
    stride_grad_right_batch, stride_grad_right_k, stride_grad_right_n = grad_right.stride()
    
    # Launch kernel
    triton_local_attention_backward_bmm_kernel[(total_programs, batch_size)](
        grad_output, left, right, grad_left, grad_right,
        batch_size, M, N, K, window_size,
        stride_grad_output_batch, stride_grad_output_m, stride_grad_output_n,
        stride_left_batch, stride_left_m, stride_left_k,
        stride_right_batch, stride_right_k, stride_right_n,
        stride_grad_left_batch, stride_grad_left_m, stride_grad_left_k,
        stride_grad_right_batch, stride_grad_right_k, stride_grad_right_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, grid_m=grid_m, grid_n=grid_n
    )


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, primals_3, primals_15, primals_16, bmm, getitem_1, rsqrt, view_4, addmm, addmm_1, addmm_2, addmm_3, addmm_4, bmm_1, getitem_3, rsqrt_1, view_18, permute_12, permute_17, permute_18, permute_23, permute_27, permute_31, permute_35, permute_39, tangents_1 = args
    args.clear()
    assert_size_stride(primals_2, (4, 177, 177, 128), (4010112, 22656, 128, 1))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(bmm, (4, 177, 177), (31329, 177, 1))
    assert_size_stride(getitem_1, (4, 177, 177, 1), (31360, 177, 1, 1))
    assert_size_stride(rsqrt, (4, 177, 177, 1), (31360, 177, 1, 1))
    assert_size_stride(view_4, (125316, 128), (128, 1))
    assert_size_stride(addmm, (125316, 128), (128, 1))
    assert_size_stride(addmm_1, (125316, 128), (128, 1))
    assert_size_stride(addmm_2, (125316, 128), (128, 1))
    assert_size_stride(addmm_3, (125316, 128), (128, 1))
    assert_size_stride(addmm_4, (125316, 128), (128, 1))
    assert_size_stride(bmm_1, (512, 177, 177), (31329, 177, 1))
    assert_size_stride(getitem_3, (4, 177, 177, 1), (31360, 177, 1, 1))
    assert_size_stride(rsqrt_1, (4, 177, 177, 1), (31360, 177, 1, 1))
    assert_size_stride(view_18, (125316, 128), (128, 1))
    assert_size_stride(permute_12, (128, 128), (128, 1))
    assert_size_stride(permute_17, (512, 177, 177), (31360, 1, 177))
    assert_size_stride(permute_18, (512, 177, 177), (31360, 1, 177))
    assert_size_stride(permute_23, (128, 128), (128, 1))
    assert_size_stride(permute_27, (128, 128), (128, 1))
    assert_size_stride(permute_31, (128, 128), (128, 1))
    assert_size_stride(permute_35, (128, 128), (128, 1))
    assert_size_stride(permute_39, (128, 128), (128, 1))
    assert_size_stride(tangents_1, (4, 177, 177, 128), (4010112, 22656, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((125316, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (125316, 128), (128, 1), 0), permute_12, out=buf0)
        del permute_12
        buf1 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [permute_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (128, 125316), (1, 128), 0), view_18, out=buf1)
        del view_18
        buf2 = empty_strided_cuda((1, 128, 531), (67968, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_0.run(tangents_1, buf2, 67968, 236, stream=stream0)
        del tangents_1
        buf3 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_1.run(buf2, buf3, 128, 531, stream=stream0)
        buf4 = empty_strided_cuda((4, 177, 177, 1), (31360, 177, 1, 125440), torch.float32)
        buf5 = empty_strided_cuda((4, 177, 177, 1), (31360, 177, 1, 125440), torch.float32)
        buf13 = empty_strided_cuda((4, 177, 177, 128), (4010112, 22656, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_1, out_gate], Original ATen: [aten.native_layer_norm, aten.mul, aten.sigmoid, aten.native_layer_norm_backward, aten.sigmoid_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_native_layer_norm_native_layer_norm_backward_sigmoid_sigmoid_backward_2.run(buf0, addmm_4, primals_15, bmm_1, getitem_3, rsqrt_1, primals_16, buf4, buf5, buf13, 125316, 128, stream=stream0)
        del primals_16
        buf6 = reinterpret_tensor(buf2, (128, 531), (1, 128), 0); del buf2  # reuse
        buf8 = empty_strided_cuda((128, 531), (1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_gate], Original ATen: [aten.sigmoid, aten.mul, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_native_layer_norm_backward_sigmoid_3.run(buf0, addmm_4, bmm_1, getitem_3, rsqrt_1, buf6, buf8, 67968, 236, stream=stream0)
        buf7 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out_gate], Original ATen: [aten.sigmoid, aten.mul, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_1.run(buf6, buf7, 128, 531, stream=stream0)
        buf9 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out_gate], Original ATen: [aten.sigmoid, aten.mul, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_1.run(buf8, buf9, 128, 531, stream=stream0)
        buf10 = empty_strided_cuda((4, 128, 177, 1, 177), (4014080, 31360, 177, 177, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(rsqrt_1, buf0, addmm_4, primals_15, buf4, bmm_1, getitem_3, buf5, buf10, 512, 31329, stream=stream0)
        del addmm_4
        del bmm_1
        del buf4
        del buf5
        del getitem_3
        del primals_15
        del rsqrt_1
        buf11 = reinterpret_tensor(buf0, (512, 177, 177), (31329, 177, 1), 0); del buf0  # reuse
        
        BMM_MODE = 1
        
        if BMM_MODE == 0:
            # Use original bmm operations
            extern_kernels.bmm(permute_17, reinterpret_tensor(buf10, (512, 177, 177), (31360, 177, 1), 0), out=buf11)
            del permute_17
            buf12 = empty_strided_cuda((512, 177, 177), (31329, 177, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf10, (512, 177, 177), (31360, 177, 1), 0), permute_18, out=buf12)
            del buf10
            del permute_18
        else:
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            # Using local attention backward BMM kernel for TriangleMultiplicativeModule outgoing einsum
            # Configuration for local attention backward BMM
            LOCAL_WINDOW_SIZE = 8  # Local attention window size

            grad_output_tensor = reinterpret_tensor(buf10, (512, 177, 177), (31360, 177, 1), 0)
            left_tensor = permute_17  # Left tensor from forward pass
            right_tensor = permute_18  # Right tensor from forward pass

            grad_left_tensor = torch.empty_like(left_tensor)
            grad_right_tensor = torch.empty_like(right_tensor)

            launch_local_attention_backward_bmm_kernel(
                grad_output_tensor, left_tensor, right_tensor,
                grad_left_tensor, grad_right_tensor,
                window_size=LOCAL_WINDOW_SIZE
            )

            buf11 = grad_left_tensor
            buf12 = grad_right_tensor
            
            del permute_17
            del buf10
            del permute_18
        buf14 = empty_strided_cuda((125316, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (125316, 128), (128, 1), 0), permute_23, out=buf14)
        del permute_23
        buf15 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [permute_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (128, 125316), (1, 128), 0), view_4, out=buf15)
        buf16 = reinterpret_tensor(buf8, (1, 128, 531), (67968, 1, 128), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_0.run(buf13, buf16, 67968, 236, stream=stream0)
        buf17 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_1.run(buf16, buf17, 128, 531, stream=stream0)
        buf18 = reinterpret_tensor(addmm_1, (4, 177, 177, 128), (4010112, 22656, 128, 1), 0); del addmm_1  # reuse
        buf28 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [right_1, right_gate], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_mul_sigmoid_sigmoid_backward_5.run(buf18, buf11, bmm, addmm_3, buf28, 708, 22656, stream=stream0)
        del addmm_3
        buf19 = reinterpret_tensor(buf11, (125316, 128), (128, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (125316, 128), (128, 1), 0), permute_27, out=buf19)
        del permute_27
        buf20 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [permute_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (128, 125316), (1, 128), 0), view_4, out=buf20)
        buf21 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_0.run(buf18, buf21, 67968, 236, stream=stream0)
        buf22 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_1.run(buf21, buf22, 128, 531, stream=stream0)
        buf23 = reinterpret_tensor(addmm, (4, 177, 177, 128), (4010112, 22656, 128, 1), 0); del addmm  # reuse
        buf33 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [left_1, left_gate], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_mul_sigmoid_sigmoid_backward_6.run(buf23, buf12, bmm, addmm_2, buf33, 125316, 128, stream=stream0)
        del addmm_2
        del bmm
        buf24 = reinterpret_tensor(buf12, (125316, 128), (128, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (125316, 128), (128, 1), 0), permute_31, out=buf24)
        del permute_31
        buf25 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [permute_34], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (128, 125316), (1, 128), 0), view_4, out=buf25)
        buf26 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_0.run(buf23, buf26, 67968, 236, stream=stream0)
        buf27 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_1.run(buf26, buf27, 128, 531, stream=stream0)
        buf29 = reinterpret_tensor(buf23, (125316, 128), (128, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (125316, 128), (128, 1), 0), permute_35, out=buf29)
        del permute_35
        buf30 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [permute_38], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (128, 125316), (1, 128), 0), view_4, out=buf30)
        buf31 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_0.run(buf28, buf31, 67968, 236, stream=stream0)
        buf32 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_1.run(buf31, buf32, 128, 531, stream=stream0)
        buf34 = reinterpret_tensor(buf28, (125316, 128), (128, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (125316, 128), (128, 1), 0), permute_39, out=buf34)
        del permute_39
        buf35 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [permute_42], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (128, 125316), (1, 128), 0), view_4, out=buf35)
        del view_4
        buf36 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_0.run(buf33, buf36, 67968, 236, stream=stream0)
        buf37 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_1.run(buf36, buf37, 128, 531, stream=stream0)
        buf38 = reinterpret_tensor(buf14, (4, 177, 177, 128), (4010112, 22656, 128, 1), 0); del buf14  # reuse
        buf41 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_7.run(buf38, buf19, buf24, buf29, buf34, primals_3, primals_2, getitem_1, rsqrt, buf41, 125316, 128, stream=stream0)
        del buf19
        del buf24
        del buf29
        del buf34
        del primals_3
        buf42 = reinterpret_tensor(buf36, (128, 531), (1, 128), 0); del buf36  # reuse
        buf44 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_8.run(buf38, primals_2, getitem_1, rsqrt, buf42, buf44, 67968, 236, stream=stream0)
        del buf38
        del getitem_1
        del primals_2
        del rsqrt
        buf43 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_1.run(buf42, buf43, 128, 531, stream=stream0)
        del buf42
        buf45 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_1.run(buf44, buf45, 128, 531, stream=stream0)
        del buf44
    return (None, buf41, buf43, buf45, buf35, reinterpret_tensor(buf37, (128, ), (1, ), 0), buf30, reinterpret_tensor(buf32, (128, ), (1, ), 0), buf25, reinterpret_tensor(buf27, (128, ), (1, ), 0), buf20, reinterpret_tensor(buf22, (128, ), (1, ), 0), buf15, reinterpret_tensor(buf17, (128, ), (1, ), 0), buf7, buf9, buf1, reinterpret_tensor(buf3, (128, ), (1, ), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((4, 177, 177, 128), (4010112, 22656, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    bmm = rand_strided((4, 177, 177), (31329, 177, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((4, 177, 177, 1), (31360, 177, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((4, 177, 177, 1), (31360, 177, 1, 1), device='cuda:0', dtype=torch.float32)
    view_4 = rand_strided((125316, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm = rand_strided((125316, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((125316, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((125316, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_3 = rand_strided((125316, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((125316, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    bmm_1 = rand_strided((512, 177, 177), (31329, 177, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((4, 177, 177, 1), (31360, 177, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((4, 177, 177, 1), (31360, 177, 1, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((125316, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_12 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_17 = rand_strided((512, 177, 177), (31360, 1, 177), device='cuda:0', dtype=torch.float32)
    permute_18 = rand_strided((512, 177, 177), (31360, 1, 177), device='cuda:0', dtype=torch.float32)
    permute_23 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_27 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_31 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_35 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_39 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((4, 177, 177, 128), (4010112, 22656, 128, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_2, primals_3, primals_15, primals_16, bmm, getitem_1, rsqrt, view_4, addmm, addmm_1, addmm_2, addmm_3, addmm_4, bmm_1, getitem_3, rsqrt_1, view_18, permute_12, permute_17, permute_18, permute_23, permute_27, permute_31, permute_35, permute_39, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
