#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

template <typename scalar_t>
__global__ void triangle_mult_outgoing_forward_kernel(
    const scalar_t* __restrict__ left,
    const scalar_t* __restrict__ right,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int dim,
    const int window_size) {
    
    const int i = blockIdx.x;
    const int j = blockIdx.y;
    const int b = blockIdx.z;
    const int d_idx = threadIdx.x;
    
    if (i >= seq_len || j >= seq_len || b >= batch_size || d_idx >= dim)
        return;
    
    scalar_t sum = 0;
    
    const int k_start = max(0, i - window_size);
    const int k_end = min(seq_len, i + window_size + 1);
    
    for (int k = k_start; k < k_end; ++k) {
        const int left_idx = ((b * seq_len + i) * seq_len + k) * dim + d_idx;
        const int right_idx = ((b * seq_len + j) * seq_len + k) * dim + d_idx;
        sum += left[left_idx] * right[right_idx];
    }
    
    const int out_idx = ((b * seq_len + i) * seq_len + j) * dim + d_idx;
    output[out_idx] = sum;
}

template <typename scalar_t>
__global__ void triangle_mult_outgoing_backward_left_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ right,
    scalar_t* __restrict__ grad_left,
    const int batch_size,
    const int seq_len,
    const int dim,
    const int window_size) {
    
    const int i = blockIdx.x;
    const int k = blockIdx.y;
    const int b = blockIdx.z;
    const int d_idx = threadIdx.x;
    
    if (i >= seq_len || k >= seq_len || b >= batch_size || d_idx >= dim)
        return;
    
    scalar_t sum = 0;
    
    const int j_start = 0;
    const int j_end = seq_len;
    
    for (int j = j_start; j < j_end; ++j) {
        if (abs(i - k) <= window_size) {
            const int grad_idx = ((b * seq_len + i) * seq_len + j) * dim + d_idx;
            const int right_idx = ((b * seq_len + j) * seq_len + k) * dim + d_idx;
            sum += grad_output[grad_idx] * right[right_idx];
        }
    }
    
    const int left_idx = ((b * seq_len + i) * seq_len + k) * dim + d_idx;
    atomicAdd(&grad_left[left_idx], sum);
}

template <typename scalar_t>
__global__ void triangle_mult_outgoing_backward_right_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ left,
    scalar_t* __restrict__ grad_right,
    const int batch_size,
    const int seq_len,
    const int dim,
    const int window_size) {
    
    const int j = blockIdx.x;
    const int k = blockIdx.y;
    const int b = blockIdx.z;
    const int d_idx = threadIdx.x;
    
    if (j >= seq_len || k >= seq_len || b >= batch_size || d_idx >= dim)
        return;
    
    scalar_t sum = 0;
    
    const int i_start = max(0, k - window_size);
    const int i_end = min(seq_len, k + window_size + 1);
    
    for (int i = i_start; i < i_end; ++i) {
        const int grad_idx = ((b * seq_len + i) * seq_len + j) * dim + d_idx;
        const int left_idx = ((b * seq_len + i) * seq_len + k) * dim + d_idx;
        sum += grad_output[grad_idx] * left[left_idx];
    }
    
    const int right_idx = ((b * seq_len + j) * seq_len + k) * dim + d_idx;
    atomicAdd(&grad_right[right_idx], sum);
}

__global__ void triangle_mult_outgoing_forward_kernel_bf16(
    const __nv_bfloat16* __restrict__ left,
    const __nv_bfloat16* __restrict__ right,
    __nv_bfloat16* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int dim,
    const int window_size) {
    
    const int i = blockIdx.x;
    const int j = blockIdx.y;
    const int b = blockIdx.z;
    const int d_idx = threadIdx.x;
    
    if (i >= seq_len || j >= seq_len || b >= batch_size || d_idx >= dim)
        return;
    
    float sum = 0.0f;
    
    const int k_start = max(0, i - window_size);
    const int k_end = min(seq_len, i + window_size + 1);
    
    for (int k = k_start; k < k_end; ++k) {
        const int left_idx = ((b * seq_len + i) * seq_len + k) * dim + d_idx;
        const int right_idx = ((b * seq_len + j) * seq_len + k) * dim + d_idx;
        sum += __bfloat162float(left[left_idx]) * __bfloat162float(right[right_idx]);
    }
    
    const int out_idx = ((b * seq_len + i) * seq_len + j) * dim + d_idx;
    output[out_idx] = __float2bfloat16(sum);
}

__global__ void triangle_mult_outgoing_backward_left_kernel_bf16(
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ right,
    __nv_bfloat16* __restrict__ grad_left,
    const int batch_size,
    const int seq_len,
    const int dim,
    const int window_size) {
    
    const int i = blockIdx.x;
    const int k = blockIdx.y;
    const int b = blockIdx.z;
    const int d_idx = threadIdx.x;
    
    if (i >= seq_len || k >= seq_len || b >= batch_size || d_idx >= dim)
        return;
    
    float sum = 0.0f;
    
    const int j_start = 0;
    const int j_end = seq_len;
    
    for (int j = j_start; j < j_end; ++j) {
        if (abs(i - k) <= window_size) {
            const int grad_idx = ((b * seq_len + i) * seq_len + j) * dim + d_idx;
            const int right_idx = ((b * seq_len + j) * seq_len + k) * dim + d_idx;
            sum += __bfloat162float(grad_output[grad_idx]) * __bfloat162float(right[right_idx]);
        }
    }
    
    const int left_idx = ((b * seq_len + i) * seq_len + k) * dim + d_idx;
    atomicAdd(&grad_left[left_idx], __float2bfloat16(sum));
}

__global__ void triangle_mult_outgoing_backward_right_kernel_bf16(
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ left,
    __nv_bfloat16* __restrict__ grad_right,
    const int batch_size,
    const int seq_len,
    const int dim,
    const int window_size) {
    
    const int j = blockIdx.x;
    const int k = blockIdx.y;
    const int b = blockIdx.z;
    const int d_idx = threadIdx.x;
    
    if (j >= seq_len || k >= seq_len || b >= batch_size || d_idx >= dim)
        return;
    
    float sum = 0.0f;
    
    const int i_start = max(0, k - window_size);
    const int i_end = min(seq_len, k + window_size + 1);
    
    for (int i = i_start; i < i_end; ++i) {
        const int grad_idx = ((b * seq_len + i) * seq_len + j) * dim + d_idx;
        const int left_idx = ((b * seq_len + i) * seq_len + k) * dim + d_idx;
        sum += __bfloat162float(grad_output[grad_idx]) * __bfloat162float(left[left_idx]);
    }
    
    const int right_idx = ((b * seq_len + j) * seq_len + k) * dim + d_idx;
    atomicAdd(&grad_right[right_idx], __float2bfloat16(sum));
}

template <typename scalar_t>
__global__ void triangle_mult_ingoing_forward_kernel(
    const scalar_t* __restrict__ left,
    const scalar_t* __restrict__ right,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int dim,
    const int window_size) {
    
    const int i = blockIdx.x;
    const int j = blockIdx.y;
    const int b = blockIdx.z;
    const int d_idx = threadIdx.x;
    
    if (i >= seq_len || j >= seq_len || b >= batch_size || d_idx >= dim)
        return;
    
    scalar_t sum = 0;
    
    const int k_start = max(0, i - window_size);
    const int k_end = min(seq_len, i + window_size + 1);
    
    for (int k = k_start; k < k_end; ++k) {
        const int left_idx = ((b * seq_len + k) * seq_len + i) * dim + d_idx;
        const int right_idx = ((b * seq_len + k) * seq_len + j) * dim + d_idx;
        sum += left[left_idx] * right[right_idx];
    }
    
    const int out_idx = ((b * seq_len + i) * seq_len + j) * dim + d_idx;
    output[out_idx] = sum;
}

template <typename scalar_t>
__global__ void triangle_mult_ingoing_backward_left_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ right,
    scalar_t* __restrict__ grad_left,
    const int batch_size,
    const int seq_len,
    const int dim,
    const int window_size) {
    
    const int k = blockIdx.x;
    const int i = blockIdx.y;
    const int b = blockIdx.z;
    const int d_idx = threadIdx.x;
    
    if (k >= seq_len || i >= seq_len || b >= batch_size || d_idx >= dim)
        return;
    
    scalar_t sum = 0;
    
    const int j_start = 0;
    const int j_end = seq_len;
    
    for (int j = j_start; j < j_end; ++j) {
        if (abs(i - k) <= window_size) {
            const int grad_idx = ((b * seq_len + i) * seq_len + j) * dim + d_idx;
            const int right_idx = ((b * seq_len + k) * seq_len + j) * dim + d_idx;
            sum += grad_output[grad_idx] * right[right_idx];
        }
    }
    
    const int left_idx = ((b * seq_len + k) * seq_len + i) * dim + d_idx;
    atomicAdd(&grad_left[left_idx], sum);
}

template <typename scalar_t>
__global__ void triangle_mult_ingoing_backward_right_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ left,
    scalar_t* __restrict__ grad_right,
    const int batch_size,
    const int seq_len,
    const int dim,
    const int window_size) {
    
    const int k = blockIdx.x;
    const int j = blockIdx.y;
    const int b = blockIdx.z;
    const int d_idx = threadIdx.x;
    
    if (k >= seq_len || j >= seq_len || b >= batch_size || d_idx >= dim)
        return;
    
    scalar_t sum = 0;
    
    const int i_start = max(0, k - window_size);
    const int i_end = min(seq_len, k + window_size + 1);
    
    for (int i = i_start; i < i_end; ++i) {
        const int grad_idx = ((b * seq_len + i) * seq_len + j) * dim + d_idx;
        const int left_idx = ((b * seq_len + k) * seq_len + i) * dim + d_idx;
        sum += grad_output[grad_idx] * left[left_idx];
    }
    
    const int right_idx = ((b * seq_len + k) * seq_len + j) * dim + d_idx;
    atomicAdd(&grad_right[right_idx], sum);
}

__global__ void triangle_mult_ingoing_forward_kernel_bf16(
    const __nv_bfloat16* __restrict__ left,
    const __nv_bfloat16* __restrict__ right,
    __nv_bfloat16* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int dim,
    const int window_size) {
    
    const int i = blockIdx.x;
    const int j = blockIdx.y;
    const int b = blockIdx.z;
    const int d_idx = threadIdx.x;
    
    if (i >= seq_len || j >= seq_len || b >= batch_size || d_idx >= dim)
        return;
    
    float sum = 0.0f;
    
    const int k_start = max(0, i - window_size);
    const int k_end = min(seq_len, i + window_size + 1);
    
    for (int k = k_start; k < k_end; ++k) {
        const int left_idx = ((b * seq_len + k) * seq_len + i) * dim + d_idx;
        const int right_idx = ((b * seq_len + k) * seq_len + j) * dim + d_idx;
        sum += __bfloat162float(left[left_idx]) * __bfloat162float(right[right_idx]);
    }
    
    const int out_idx = ((b * seq_len + i) * seq_len + j) * dim + d_idx;
    output[out_idx] = __float2bfloat16(sum);
}

__global__ void triangle_mult_ingoing_backward_left_kernel_bf16(
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ right,
    __nv_bfloat16* __restrict__ grad_left,
    const int batch_size,
    const int seq_len,
    const int dim,
    const int window_size) {
    
    const int k = blockIdx.x;
    const int i = blockIdx.y;
    const int b = blockIdx.z;
    const int d_idx = threadIdx.x;
    
    if (k >= seq_len || i >= seq_len || b >= batch_size || d_idx >= dim)
        return;
    
    float sum = 0.0f;
    
    const int j_start = 0;
    const int j_end = seq_len;
    
    for (int j = j_start; j < j_end; ++j) {
        if (abs(i - k) <= window_size) {
            const int grad_idx = ((b * seq_len + i) * seq_len + j) * dim + d_idx;
            const int right_idx = ((b * seq_len + k) * seq_len + j) * dim + d_idx;
            sum += __bfloat162float(grad_output[grad_idx]) * __bfloat162float(right[right_idx]);
        }
    }
    
    const int left_idx = ((b * seq_len + k) * seq_len + i) * dim + d_idx;
    atomicAdd(&grad_left[left_idx], __float2bfloat16(sum));
}

__global__ void triangle_mult_ingoing_backward_right_kernel_bf16(
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ left,
    __nv_bfloat16* __restrict__ grad_right,
    const int batch_size,
    const int seq_len,
    const int dim,
    const int window_size) {
    
    const int k = blockIdx.x;
    const int j = blockIdx.y;
    const int b = blockIdx.z;
    const int d_idx = threadIdx.x;
    
    if (k >= seq_len || j >= seq_len || b >= batch_size || d_idx >= dim)
        return;
    
    float sum = 0.0f;
    
    const int i_start = max(0, k - window_size);
    const int i_end = min(seq_len, k + window_size + 1);
    
    for (int i = i_start; i < i_end; ++i) {
        const int grad_idx = ((b * seq_len + i) * seq_len + j) * dim + d_idx;
        const int left_idx = ((b * seq_len + k) * seq_len + i) * dim + d_idx;
        sum += __bfloat162float(grad_output[grad_idx]) * __bfloat162float(left[left_idx]);
    }
    
    const int right_idx = ((b * seq_len + k) * seq_len + j) * dim + d_idx;
    atomicAdd(&grad_right[right_idx], __float2bfloat16(sum));
}

torch::Tensor triangle_mult_outgoing_forward_cuda(
    torch::Tensor left,
    torch::Tensor right,
    int window_size) {
    
    const auto batch_size = left.size(0);
    const auto seq_len = left.size(1);
    const auto dim = left.size(3);
    
    auto output = torch::zeros_like(left);
    
    const dim3 blocks(seq_len, seq_len, batch_size);
    const dim3 threads(dim);
    
    if (left.scalar_type() == torch::ScalarType::BFloat16) {
        triangle_mult_outgoing_forward_kernel_bf16<<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16*>(left.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(right.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            batch_size,
            seq_len,
            dim,
            window_size);
    } else {
        AT_DISPATCH_FLOATING_TYPES(left.scalar_type(), "triangle_mult_outgoing_forward_cuda", ([&] {
            triangle_mult_outgoing_forward_kernel<scalar_t><<<blocks, threads>>>(
                left.data_ptr<scalar_t>(),
                right.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                seq_len,
                dim,
                window_size);
        }));
    }
    
    return output;
}

std::vector<torch::Tensor> triangle_mult_outgoing_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor left,
    torch::Tensor right,
    int window_size) {
    
    const auto batch_size = grad_output.size(0);
    const auto seq_len = grad_output.size(1);
    const auto dim = grad_output.size(3);
    
    auto grad_left = torch::zeros_like(left);
    auto grad_right = torch::zeros_like(right);
    
    const dim3 blocks_left(seq_len, seq_len, batch_size);
    const dim3 blocks_right(seq_len, seq_len, batch_size);
    const dim3 threads(dim);
    
    if (grad_output.scalar_type() == torch::ScalarType::BFloat16) {
        triangle_mult_outgoing_backward_left_kernel_bf16<<<blocks_left, threads>>>(
            reinterpret_cast<__nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(right.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(grad_left.data_ptr()),
            batch_size,
            seq_len,
            dim,
            window_size);
            
        triangle_mult_outgoing_backward_right_kernel_bf16<<<blocks_right, threads>>>(
            reinterpret_cast<__nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(left.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(grad_right.data_ptr()),
            batch_size,
            seq_len,
            dim,
            window_size);
    } else {
        AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "triangle_mult_outgoing_backward_cuda", ([&] {
            triangle_mult_outgoing_backward_left_kernel<scalar_t><<<blocks_left, threads>>>(
                grad_output.data_ptr<scalar_t>(),
                right.data_ptr<scalar_t>(),
                grad_left.data_ptr<scalar_t>(),
                batch_size,
                seq_len,
                dim,
                window_size);
                
            triangle_mult_outgoing_backward_right_kernel<scalar_t><<<blocks_right, threads>>>(
                grad_output.data_ptr<scalar_t>(),
                left.data_ptr<scalar_t>(),
                grad_right.data_ptr<scalar_t>(),
                batch_size,
                seq_len,
                dim,
                window_size);
        }));
    }
    
    return {grad_left, grad_right};
}

torch::Tensor triangle_mult_ingoing_forward_cuda(
    torch::Tensor left,
    torch::Tensor right,
    int window_size) {
    
    const auto batch_size = left.size(0);
    const auto seq_len = left.size(1);
    const auto dim = left.size(3);
    
    auto output = torch::zeros_like(left);
    
    const dim3 blocks(seq_len, seq_len, batch_size);
    const dim3 threads(dim);
    
    if (left.scalar_type() == torch::ScalarType::BFloat16) {
        triangle_mult_ingoing_forward_kernel_bf16<<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16*>(left.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(right.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            batch_size,
            seq_len,
            dim,
            window_size);
    } else {
        AT_DISPATCH_FLOATING_TYPES(left.scalar_type(), "triangle_mult_ingoing_forward_cuda", ([&] {
            triangle_mult_ingoing_forward_kernel<scalar_t><<<blocks, threads>>>(
                left.data_ptr<scalar_t>(),
                right.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                seq_len,
                dim,
                window_size);
        }));
    }
    
    return output;
}

std::vector<torch::Tensor> triangle_mult_ingoing_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor left,
    torch::Tensor right,
    int window_size) {
    
    const auto batch_size = grad_output.size(0);
    const auto seq_len = grad_output.size(1);
    const auto dim = grad_output.size(3);
    
    auto grad_left = torch::zeros_like(left);
    auto grad_right = torch::zeros_like(right);
    
    const dim3 blocks_left(seq_len, seq_len, batch_size);
    const dim3 blocks_right(seq_len, seq_len, batch_size);
    const dim3 threads(dim);
    
    if (grad_output.scalar_type() == torch::ScalarType::BFloat16) {
        triangle_mult_ingoing_backward_left_kernel_bf16<<<blocks_left, threads>>>(
            reinterpret_cast<__nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(right.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(grad_left.data_ptr()),
            batch_size,
            seq_len,
            dim,
            window_size);
            
        triangle_mult_ingoing_backward_right_kernel_bf16<<<blocks_right, threads>>>(
            reinterpret_cast<__nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(left.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(grad_right.data_ptr()),
            batch_size,
            seq_len,
            dim,
            window_size);
    } else {
        AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "triangle_mult_ingoing_backward_cuda", ([&] {
            triangle_mult_ingoing_backward_left_kernel<scalar_t><<<blocks_left, threads>>>(
                grad_output.data_ptr<scalar_t>(),
                right.data_ptr<scalar_t>(),
                grad_left.data_ptr<scalar_t>(),
                batch_size,
                seq_len,
                dim,
                window_size);
                
            triangle_mult_ingoing_backward_right_kernel<scalar_t><<<blocks_right, threads>>>(
                grad_output.data_ptr<scalar_t>(),
                left.data_ptr<scalar_t>(),
                grad_right.data_ptr<scalar_t>(),
                batch_size,
                seq_len,
                dim,
                window_size);
        }));
    }
    
    return {grad_left, grad_right};
}