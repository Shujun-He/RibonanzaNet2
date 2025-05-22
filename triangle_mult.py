import torch
from torch.autograd import Function
from torch.library import Library, impl
import triangle_mult_cuda

class TriangleMultOutgoingFunction(Function):
    @staticmethod
    def forward(ctx, left, right, window_size):
        ctx.save_for_backward(left, right)
        ctx.window_size = window_size
        return triangle_mult_cuda.triangle_mult_outgoing_forward(left, right, window_size)

    @staticmethod
    def backward(ctx, grad_output):
        left, right = ctx.saved_tensors
        window_size = ctx.window_size
        
        grad_left, grad_right = triangle_mult_cuda.triangle_mult_outgoing_backward(
            grad_output, left, right, window_size)

        return grad_left, grad_right, None

class TriangleMultIngoingFunction(Function):
    @staticmethod
    def forward(ctx, left, right, window_size):
        ctx.save_for_backward(left, right)
        ctx.window_size = window_size
        return triangle_mult_cuda.triangle_mult_ingoing_forward(left, right, window_size)

    @staticmethod
    def backward(ctx, grad_output):
        left, right = ctx.saved_tensors
        window_size = ctx.window_size
        
        grad_left, grad_right = triangle_mult_cuda.triangle_mult_ingoing_backward(
            grad_output, left, right, window_size)
        
        return grad_left, grad_right, None

def triangle_mult_outgoing(left, right, window_size):
    if not isinstance(left, torch.Tensor) or not hasattr(left, "data_ptr"):
        return torch.zeros_like(left)
    return TriangleMultOutgoingFunction.apply(left, right, window_size)

def triangle_mult_ingoing(left, right, window_size):
    if not isinstance(left, torch.Tensor) or not hasattr(left, "data_ptr"):
        return torch.zeros_like(left)
    return TriangleMultIngoingFunction.apply(left, right, window_size)