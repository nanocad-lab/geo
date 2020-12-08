//#include <ATen/ATen.h>
// #include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
//#include "sc_cuda_kernel.h"

namespace F = torch::nn::functional;

torch::Tensor conv2d_add_partial_cuda_acc(torch::Tensor input, torch::Tensor weight_pos, torch::Tensor weight_neg, int bit_length, bool share);
torch::Tensor conv2d_add_partial_direct_variable_cuda(torch::Tensor input, torch::Tensor weight_pos, torch::Tensor weight_neg, int bit_length, int total_width, int load_width, int load_wait, bool im2col);
torch::Tensor conv2d_addyz_variable_cuda(torch::Tensor input, torch::Tensor weight_pos, torch::Tensor weight_neg, int bit_length, int total_width, int load_width, int load_wait_w, int load_wait_a, int z_unit, bool im2col);
torch::Tensor conv2d_addz_variable_cuda(torch::Tensor input, torch::Tensor weight_pos, torch::Tensor weight_neg, int bit_length, int total_width, int load_width, int load_wait, bool im2col);
torch::Tensor conv2d_or_cuda(torch::Tensor input, torch::Tensor weight_pos, torch::Tensor weight_neg, int bit_length, bool im2col);


at::Tensor conv2d_or_acc(torch::Tensor input, torch::Tensor weight, int bit_length, at::IntArrayRef padding, at::IntArrayRef stride, bool im2col) {
    auto input_pad = F::pad(input, F::PadFuncOptions({padding[0], padding[0], padding[1], padding[1]})/*.mode(torch::kReplicate)*/);
    auto compare_type = torch::kInt32;
    auto input_split = (input_pad*bit_length).clamp(0, bit_length-1).ceil().to(compare_type).clone();
    auto w_pos_split = (weight*bit_length).clamp(0, bit_length-1).ceil().to(compare_type);
    auto w_neg_split = (-(weight*bit_length).clamp(1-bit_length, 0)).ceil().to(compare_type);

    return conv2d_or_cuda(input_split, w_pos_split, w_neg_split, bit_length, im2col);
}

at::Tensor conv2d_add_yz_variable_acc(torch::Tensor input, torch::Tensor weight, int bit_length, at::IntArrayRef padding, at::IntArrayRef stride, int total_width, int load_width, int load_wait_w, int load_wait_a, bool im2col, int z_unit) {
    auto input_pad = F::pad(input, F::PadFuncOptions({padding[0], padding[0], padding[1], padding[1]})/*.mode(torch::kReplicate)*/);
    auto compare_type = torch::kInt32;
    auto input_split = (input_pad*bit_length).clamp(0, bit_length-1).ceil().to(compare_type).clone();
    auto w_pos_split = (weight*bit_length).clamp(0, bit_length-1).ceil().to(compare_type);
    auto w_neg_split = (-(weight*bit_length).clamp(1-bit_length, 0)).ceil().to(compare_type);

    return conv2d_addyz_variable_cuda(input_split, w_pos_split, w_neg_split, bit_length, total_width, load_width, load_wait_w, load_wait_a, z_unit, im2col);
}

at::Tensor conv2d_add_partial_variable_acc(torch::Tensor input, torch::Tensor weight, int bit_length, at::IntArrayRef padding, at::IntArrayRef stride, int total_width, int load_width, int load_wait, bool im2col, int dim) {
    auto input_pad = F::pad(input, F::PadFuncOptions({padding[0], padding[0], padding[1], padding[1]})/*.mode(torch::kReplicate)*/);
    auto compare_type = torch::kInt32;
    auto input_split = (input_pad*bit_length).clamp(0, bit_length-1).ceil().to(compare_type).clone();
    auto w_pos_split = (weight*bit_length).clamp(0, bit_length-1).ceil().to(compare_type);
    auto w_neg_split = (-(weight*bit_length).clamp(1-bit_length, 0)).ceil().to(compare_type);
    
    if (dim==0) {
        return conv2d_addz_variable_cuda(input_split, w_pos_split, w_neg_split, bit_length, total_width, load_width, load_wait, im2col);
    }
    else {
        return conv2d_add_partial_direct_variable_cuda(input_split, w_pos_split, w_neg_split, bit_length, total_width, load_width, load_wait, im2col);
    }
}

at::Tensor conv2d_add_partial_acc_acc(torch::Tensor input, torch::Tensor weight, int bit_length, at::IntArrayRef padding, at::IntArrayRef stride, bool share, int dim) {
    auto input_pad = F::pad(input, F::PadFuncOptions({padding[0], padding[0], padding[1], padding[1]})/*.mode(torch::kReplicate)*/);
    auto compare_type = torch::kInt32;
    auto input_split = (input_pad*bit_length).clamp(0, bit_length-1).ceil().to(compare_type).clone();
    auto w_pos_split = (weight*bit_length).clamp(0, bit_length-1).ceil().to(compare_type);
    auto w_neg_split = (-(weight*bit_length).clamp(1-bit_length, 0)).ceil().to(compare_type);

    return conv2d_add_partial_cuda_acc(input_split, w_pos_split, w_neg_split, bit_length, share);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_add_partial_variable_acc", &conv2d_add_partial_variable_acc, "SC forward conv2d partial (y or z dimension) bin add variable load lfsr generator");
    m.def("conv2d_add_yz_variable_acc", &conv2d_add_yz_variable_acc, "SC forward conv2d partial (yz dimension) bin add variable load lfsr generator");
    m.def("conv2d_add_partial_acc_acc", &conv2d_add_partial_acc_acc, "SC forward conv2d partial bin add accurate random generator");
    m.def("conv2d_or_acc", &conv2d_or_acc, "SC forward conv2d or add random generator");
}
