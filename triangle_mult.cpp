#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
torch::Tensor triangle_mult_outgoing_forward_cuda(
    torch::Tensor left,
    torch::Tensor right,
    int window_size);

std::vector<torch::Tensor> triangle_mult_outgoing_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor left,
    torch::Tensor right,
    int window_size);

torch::Tensor triangle_mult_ingoing_forward_cuda(
    torch::Tensor left,
    torch::Tensor right,
    int window_size);

std::vector<torch::Tensor> triangle_mult_ingoing_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor left,
    torch::Tensor right,
    int window_size);

// C++ interface
torch::Tensor triangle_mult_outgoing_forward(
    torch::Tensor left,
    torch::Tensor right,
    int window_size) {
    return triangle_mult_outgoing_forward_cuda(left, right, window_size);
}

std::vector<torch::Tensor> triangle_mult_outgoing_backward(
    torch::Tensor grad_output,
    torch::Tensor left,
    torch::Tensor right,
    int window_size) {
    return triangle_mult_outgoing_backward_cuda(grad_output, left, right, window_size);
}

torch::Tensor triangle_mult_ingoing_forward(
    torch::Tensor left,
    torch::Tensor right,
    int window_size) {
    return triangle_mult_ingoing_forward_cuda(left, right, window_size);
}

std::vector<torch::Tensor> triangle_mult_ingoing_backward(
    torch::Tensor grad_output,
    torch::Tensor left,
    torch::Tensor right,
    int window_size) {
    return triangle_mult_ingoing_backward_cuda(grad_output, left, right, window_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("triangle_mult_outgoing_forward", &triangle_mult_outgoing_forward, "Triangle Multiplicative (outgoing) forward");
    m.def("triangle_mult_outgoing_backward", &triangle_mult_outgoing_backward, "Triangle Multiplicative (outgoing) backward");
    m.def("triangle_mult_ingoing_forward", &triangle_mult_ingoing_forward, "Triangle Multiplicative (ingoing) forward");
    m.def("triangle_mult_ingoing_backward", &triangle_mult_ingoing_backward, "Triangle Multiplicative (ingoing) backward");
}