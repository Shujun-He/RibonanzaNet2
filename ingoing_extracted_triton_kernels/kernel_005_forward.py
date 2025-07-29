# Original file: hi/chielldyh6kvf7shx6qbt62owvwg3rb3tjnaaynkkmox73xzepxu.debug/output_code.py
# Type: Forward
# Generated for TriangleMultiplicativeModule
# PyTorch version: 2.7.0+cu126

# AOT ID: ['0_forward']
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


# kernel path: /tmp/triton_kernels_diqdfnfo/tz/ctzr24u6cknm7llgnajs63lu5eqllozyiuasidcl2vdpjb62m2t5.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x => add, add_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_2, [3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_3), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_4), kwargs = {})
triton_red_fused_native_layer_norm_0 = async_compile.triton('triton_red_fused_native_layer_norm_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '2B4DCE383ADC6F6AC9E37B7BE08352A0B047033492CBA4C9904681B32954FEA9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x0 = (xindex % 31329)
    x1 = xindex // 31329
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp5[:, None]
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tl.store(out_ptr0 + (x0 + 31360*x1), tmp2, xmask)
    tmp8 = 128.0
    tmp9 = (tmp3 / tmp8)
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0 + 31360*x1), tmp12, xmask)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp13 = tl.load(in_ptr0 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp13 - tmp2
        tmp15 = tmp14 * tmp12
        tmp17 = tmp15 * tmp16
        tmp19 = tmp17 + tmp18
        tl.store(out_ptr1 + (r0_2 + 128*x3), tmp19, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/triton_kernels_diqdfnfo/uw/cuwt4uyaufskejcyhxhgbljdb3qztdxl64ykedfpzjxozetbucnv.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   out => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_1 = async_compile.triton('triton_poi_fused_clone_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 32768}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '2B4DCE383ADC6F6AC9E37B7BE08352A0B047033492CBA4C9904681B32954FEA9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 708
    xnumel = 22656
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    x3 = xindex // 128
    x2 = (xindex % 128)
    y0 = (yindex % 177)
    y1 = yindex // 177
    tmp0 = tl.load(in_ptr0 + (x5 + 22656*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + 177*y4), ymask & xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x5 + 22656*y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (y0 + 177*x3 + 31360*x2 + 4014080*y1), tmp5, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/triton_kernels_diqdfnfo/pm/cpmj2msa3hy3knghvrdr7eh2z2yyyylusv6xisxggknsd4txqexy.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   out => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '2B4DCE383ADC6F6AC9E37B7BE08352A0B047033492CBA4C9904681B32954FEA9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 125316
    xnumel = 128
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 31329)
    y1 = yindex // 31329
    tmp0 = tl.load(in_ptr0 + (x2 + 128*y3), ymask & xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + 128*y3), ymask & xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (y0 + 31360*x2 + 4014080*y1), tmp5, ymask & xmask)
''', device_str='cuda')


# kernel path: /tmp/triton_kernels_diqdfnfo/gr/cgrk2iq6vn3qllykzdc4ovf4eiiafwk7mzoudwj5xmf2xs5cxzhn.py
# Topologically Sorted Source Nodes: [out_gate, out_1, out_2], Original ATen: [aten.sigmoid, aten.native_layer_norm, aten.mul]
# Source node to ATen node mapping:
#   out_1 => add_2, add_3, clone_2, mul_6, mul_7, rsqrt_1, sub_1, var_mean_1
#   out_2 => mul_8
#   out_gate => sigmoid_2
# Graph fragment:
#   %sigmoid_2 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_13,), kwargs = {})
#   %clone_2 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%view_17,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_2, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_2, %getitem_3), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %primals_15), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %primals_16), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_2), kwargs = {})
triton_red_fused_mul_native_layer_norm_sigmoid_3 = async_compile.triton('triton_red_fused_mul_native_layer_norm_sigmoid_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_layer_norm_sigmoid_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '2B4DCE383ADC6F6AC9E37B7BE08352A0B047033492CBA4C9904681B32954FEA9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mul_native_layer_norm_sigmoid_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 125316
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 31329)
    x1 = xindex // 31329
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 31329*r0_2 + 4010112*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp5[:, None]
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tl.store(out_ptr0 + (x0 + 31360*x1), tmp2, xmask)
    tmp8 = 128.0
    tmp9 = (tmp3 / tmp8)
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0 + 31360*x1), tmp12, xmask)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp13 = tl.load(in_ptr0 + (x0 + 31329*r0_2 + 4010112*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr3 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tmp13 - tmp2
        tmp15 = tmp14 * tmp12
        tmp17 = tmp15 * tmp16
        tmp19 = tmp17 + tmp18
        tmp21 = tl.sigmoid(tmp20)
        tmp22 = tmp19 * tmp21
        tl.store(out_ptr1 + (r0_2 + 128*x3), tmp22, r0_mask & xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

# Local attention BMM kernel for ingoing triangle multiplicative module
@triton.jit
def triton_local_attention_bmm_kernel(
    left_ptr, right_ptr, out_ptr,
    batch_size, M, N, K, window_size,
    stride_left_batch, stride_left_k, stride_left_m,
    stride_right_batch, stride_right_k, stride_right_n,
    stride_out_batch, stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8, ACC_TYPE: tl.constexpr = tl.float32
):
    """
    Computes einsum('... k i d, ... k j d -> ... i j d') with k dimension limited to window around i
    
    left:   [batch_size*feature_dim, K, M]
    right:  [batch_size*feature_dim, K, N]
    output: [batch_size*feature_dim, M, N]
    """
    pid = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    i_center = pid_m * BLOCK_M + BLOCK_M // 2  # Center of current i block
    k_start = tl.maximum(0, i_center - window_size)
    k_end = tl.minimum(K, i_center + window_size + 1)
    
    k_start_block = (k_start // BLOCK_K) * BLOCK_K
    k_end_block = ((k_end + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
    
    # Main loop
    rk = tl.arange(0, BLOCK_K)
    for k_offset in range(k_start_block, k_end_block, BLOCK_K):
        if k_offset < K:
            k_indices = k_offset + rk
            remaining_k = K - k_offset
            
            # Create local mask for this k block
            # For each i position, only include k values within window_size
            i_indices = ram[:, None]  # Shape: (BLOCK_M, 1)
            k_indices_2d = k_indices[None, :]  # Shape: (1, BLOCK_K)
            
            # |i - k| <= window_size
            local_mask = tl.abs(i_indices - k_indices_2d) <= window_size
            
            # Boundary masks
            m_mask = ram < M
            n_mask = rbn < N
            k_mask = k_indices < K
            
            # Combined mask for left tensor (k, i) - note the transposed access pattern
            left_mask = (k_mask[:, None] & m_mask[None, :]) & local_mask.T
            
            # Mask for right tensor (k, j)
            right_mask = k_mask[:, None] & n_mask[None, :]
            
            # Memory pointers
            # left: [batch, k, i, d]
            left_ptrs = (pid_batch * stride_left_batch + 
                        k_indices[:, None] * stride_left_k + 
                        ram[None, :] * stride_left_m)
            # right: [batch, k, j, d]
            right_ptrs = (pid_batch * stride_right_batch +
                        k_indices[:, None] * stride_right_k +
                        rbn[None, :] * stride_right_n)
            
            if remaining_k >= BLOCK_K:
                left_block = tl.load(left_ptr + left_ptrs, mask=left_mask, other=0.0)
                right_block = tl.load(right_ptr + right_ptrs, mask=right_mask, other=0.0)
            else:
                # k boundary check
                k_boundary_mask = k_indices < K
                left_final_mask = left_mask & k_boundary_mask[:, None]
                right_final_mask = right_mask & k_boundary_mask[:, None]
                
                left_block = tl.load(left_ptr + left_ptrs, mask=left_final_mask, other=0.0)
                right_block = tl.load(right_ptr + right_ptrs, mask=right_final_mask, other=0.0)

            acc += tl.dot(left_block.T, right_block, allow_tf32=True)
    
    # Store results
    idx_m = ram[:, None]
    idx_n = rbn[None, :]
    output_mask = (idx_m < M) & (idx_n < N)
    
    out_ptrs = (pid_batch * stride_out_batch + 
               idx_m * stride_out_m + 
               idx_n * stride_out_n)
    tl.store(out_ptr + out_ptrs, acc, mask=output_mask)


def launch_local_attention_bmm_kernel_ingoing(left_input, right_input, output, window_size=16):
    """
        left_input: Left input tensor (batch_size*feature_dim, K, M)
        right_input: Right input tensor (batch_size*feature_dim, K, N)  
        output: Output tensor (batch_size*feature_dim, M, N)
        window_size: Local attention window size (default: 16)
    """
    batch_size, K, M = left_input.shape
    _, K_right, N = right_input.shape

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    GROUP_M = 8
    
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    total_programs = grid_m * grid_n
    
    stride_left_batch, stride_left_k, stride_left_m = left_input.stride()
    stride_right_batch, stride_right_k, stride_right_n = right_input.stride()
    stride_out_batch, stride_out_m, stride_out_n = output.stride()
    
    # Launch kernel
    triton_local_attention_bmm_kernel[(total_programs, batch_size)](
        left_input,
        right_input, 
        output,
        batch_size, M, N, K, window_size,
        stride_left_batch, stride_left_k, stride_left_m,
        stride_right_batch, stride_right_k, stride_right_n,
        stride_out_batch, stride_out_m, stride_out_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M
    )

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18 = args
    args.clear()
    assert_size_stride(primals_1, (4, 177), (177, 1))
    assert_size_stride(primals_2, (4, 177, 177, 128), (4010112, 22656, 128, 1))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, 128), (128, 1))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (128, 128), (128, 1))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, 128), (128, 1))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, 128), (128, 1))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, 128), (128, 1))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (128, 128), (128, 1))
    assert_size_stride(primals_18, (128, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 177, 177), (31329, 177, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mask], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(primals_1, (4, 177, 1), (177, 1, 1), 0), reinterpret_tensor(primals_1, (4, 1, 177), (177, 1, 1), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((4, 177, 177, 1), (31360, 177, 1, 1), torch.float32)
        buf2 = empty_strided_cuda((4, 177, 177, 1), (31360, 177, 1, 125440), torch.float32)
        buf4 = reinterpret_tensor(buf2, (4, 177, 177, 1), (31360, 177, 1, 1), 0); del buf2  # reuse
        buf5 = empty_strided_cuda((4, 177, 177, 128), (4010112, 22656, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_0.run(buf4, primals_2, primals_3, primals_4, buf1, buf5, 125316, 128, stream=stream0)
        del primals_4
        buf6 = empty_strided_cuda((125316, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [left], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, reinterpret_tensor(buf5, (125316, 128), (128, 1), 0), reinterpret_tensor(primals_5, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf6)
        del primals_6
        buf7 = empty_strided_cuda((125316, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [right], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, reinterpret_tensor(buf5, (125316, 128), (128, 1), 0), reinterpret_tensor(primals_7, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf7)
        del primals_8
        buf8 = empty_strided_cuda((125316, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, reinterpret_tensor(buf5, (125316, 128), (128, 1), 0), reinterpret_tensor(primals_9, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf8)
        del primals_10
        buf9 = empty_strided_cuda((125316, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_12, reinterpret_tensor(buf5, (125316, 128), (128, 1), 0), reinterpret_tensor(primals_11, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf9)
        del primals_12
        buf10 = empty_strided_cuda((125316, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_14, reinterpret_tensor(buf5, (125316, 128), (128, 1), 0), reinterpret_tensor(primals_13, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf10)
        del primals_14
        buf11 = empty_strided_cuda((4, 128, 177, 177, 1), (4014080, 31360, 177, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf6, buf0, buf8, buf11, 708, 22656, stream=stream0)
        buf12 = empty_strided_cuda((4, 128, 177, 177, 1), (4014080, 31360, 177, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf7, buf0, buf9, buf12, 125316, 128, stream=stream0)
        buf13 = empty_strided_cuda((512, 177, 177), (31329, 177, 1), torch.float32)
        
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.bmm]
        # Triangle multiplicative module ingoing with optional local attention
        USE_LOCAL_ATTENTION = True  # Set to False to use original bmm
        LOCAL_ATTENTION_WINDOW_SIZE = 8  # Window size for local attention
        if USE_LOCAL_ATTENTION:
            # Use local attention BMM kernel to limit k dimension around i position
            left_tensor = reinterpret_tensor(buf11, (512, 177, 177), (31360, 177, 1), 0)
            right_tensor = reinterpret_tensor(buf12, (512, 177, 177), (31360, 177, 1), 0)
            launch_local_attention_bmm_kernel_ingoing(left_tensor, right_tensor, buf13, LOCAL_ATTENTION_WINDOW_SIZE)
        else:
            # Original code
            extern_kernels.bmm(reinterpret_tensor(buf11, (512, 177, 177), (31360, 177, 1), 0), reinterpret_tensor(buf12, (512, 177, 177), (31360, 177, 1), 0), out=buf13)
        
        buf14 = empty_strided_cuda((4, 177, 177, 1), (31360, 177, 1, 1), torch.float32)
        buf15 = empty_strided_cuda((4, 177, 177, 1), (31360, 177, 1, 125440), torch.float32)
        buf17 = reinterpret_tensor(buf15, (4, 177, 177, 1), (31360, 177, 1, 1), 0); del buf15  # reuse
        buf18 = empty_strided_cuda((4, 177, 177, 128), (4010112, 22656, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_gate, out_1, out_2], Original ATen: [aten.sigmoid, aten.native_layer_norm, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_native_layer_norm_sigmoid_3.run(buf17, buf13, primals_15, primals_16, buf10, buf14, buf18, 125316, 128, stream=stream0)
        buf19 = empty_strided_cuda((125316, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_18, reinterpret_tensor(buf18, (125316, 128), (128, 1), 0), reinterpret_tensor(primals_17, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf19)
        del primals_18
    return (reinterpret_tensor(buf19, (4, 177, 177, 128), (4010112, 22656, 128, 1), 0), primals_2, primals_3, primals_15, primals_16, buf0, buf1, buf4, reinterpret_tensor(buf5, (125316, 128), (128, 1), 0), buf6, buf7, buf8, buf9, buf10, buf13, buf14, buf17, reinterpret_tensor(buf18, (125316, 128), (128, 1), 0), primals_17, reinterpret_tensor(buf11, (512, 177, 177), (31360, 1, 177), 0), reinterpret_tensor(buf12, (512, 177, 177), (31360, 1, 177), 0), primals_13, primals_11, primals_9, primals_7, primals_5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 177), (177, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 177, 177, 128), (4010112, 22656, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
