#include <torch/extension.h>
//#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

torch::Tensor conv2d_add_partial_cuda(torch::Tensor input, torch::Tensor weight_pos, torch::Tensor weight_neg, int bit_packs);
