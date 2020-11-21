#include <ATen/ATen.h>
#include <torch/torch.h>
// #include <torch/extension.h>
#include <iostream>
#include <time.h>


at::IntArrayRef bitstream(torch::Tensor input, int bit_length) {
    auto input_size = input.sizes();
    auto input_leng = input_size.size();
    std::cout << input_leng << std::endl;
//     for(int i=0; i<input_leng; i++){std::cout << input_size[i] << std::endl;}
    at::Tensor a = at::Tensor();
    switch(input_leng)
    {
        case 1:
            a = at::rand({input_size[0],bit_length});
            break;
        case 2:
            a = at::rand({input_size[0],input_size[1],bit_length});
            break;
        case 3:
            a = at::rand({input_size[0],input_size[1],input_size[2],bit_length});
            break;
        case 4:
            a = at::rand({input_size[0],input_size[1],input_size[2],input_size[3],bit_length});
            break;
        default:
            a = at::rand({bit_length});
    }
    auto input_bit_expand = (input.unsqueeze(-1) > a).to(torch::kUInt8);
    auto input_back = input_bit_expand.sum({-1}).to(torch::kFloat32)/bit_length;
    std::cout << input_back << std::endl;
    return input_size;
}

at::Tensor linear_or(torch::Tensor &input, torch::Tensor &weight, int &bit_length, int &add_full) {
//     clock_t start;
//     start = clock();
    auto input_size = input.sizes();
    auto weight_size = weight.sizes();
    auto weight_pos = weight.clamp(0,32);
    auto weight_neg = -weight.clamp(-32,0);
    
    auto range = bit_length - 1;
    auto comp_type = torch::kInt16;
    
    auto split_size = input_size[1]/add_full;
    auto input_split = at::split((input*bit_length).to(comp_type), split_size, 1);
    auto weight_pos_split = at::split((weight*bit_length).clamp(0,bit_length).to(comp_type), split_size, 1);
    auto weight_neg_split = at::split(-(weight*bit_length).clamp(-bit_length,0).to(comp_type), split_size, 1);
    
//     at::Tensor rand_input = torch::rand(input_size);
//     at::Tensor rand_weight_pos = torch::rand(weight_size);
//     at::Tensor rand_weight_neg = torch::rand(weight_size);
    
    auto input_split_size = input_split[0].sizes();
    auto weight_split_size = weight_pos_split[0].sizes();
    auto rand_input = torch::randint(range, input_split_size, comp_type);
    auto rand_weight_pos = torch::randint(range, weight_split_size, comp_type);
    auto rand_weight_neg = torch::randint(range, weight_split_size, comp_type);

    auto input_bit = (input_split[0] > rand_input).to(torch::kUInt8);
    auto weight_pos_bit = (weight_pos_split[0] > rand_weight_pos).to(torch::kUInt8);
    auto weight_neg_bit = (weight_neg_split[0] > rand_weight_neg).to(torch::kUInt8);
    auto result_pos = at::linear(input_bit, weight_pos_bit).sign().to(torch::kInt32);
    auto result_neg = at::linear(input_bit, weight_neg_bit).sign().to(torch::kInt32);
    
    for (int i=1; i<add_full; i++) {
        rand_input.random_(range);
        rand_weight_pos.random_(range);
        rand_weight_neg.random_(range);
        input_bit = (input_split[i] > rand_input).to(torch::kUInt8);
        weight_pos_bit = (weight_pos_split[i] > rand_weight_pos).to(torch::kUInt8);
        weight_neg_bit = (weight_neg_split[i] > rand_weight_neg).to(torch::kUInt8);
        result_pos = at::linear(input_bit, weight_pos_bit).sign().to(torch::kInt32);
        result_neg = at::linear(input_bit, weight_neg_bit).sign().to(torch::kInt32);
    }
    
//     std::cout << (double)(clock() - start) << std::endl;
//     start = clock();
    for(int j=1; j<bit_length; j++) {
        for (int i=1; i<add_full; i++) {
            rand_input.random_(range);
            rand_weight_pos.random_(range);
            rand_weight_neg.random_(range);
            input_bit = (input_split[i] > rand_input).to(torch::kUInt8);
            weight_pos_bit = (weight_pos_split[i] > rand_weight_pos).to(torch::kUInt8);
            weight_neg_bit = (weight_neg_split[i] > rand_weight_neg).to(torch::kUInt8);
            result_pos += at::linear(input_bit, weight_pos_bit).sign().to(torch::kInt32);
            result_neg += at::linear(input_bit, weight_neg_bit).sign().to(torch::kInt32);
    //         std::cout << (double)(clock() - start) << std::endl;
    //         start = clock();
        }
    }
    at::Tensor result = result_pos.to(torch::kFloat32)/bit_length - result_neg.to(torch::kFloat32)/bit_length;
    return result;
}

// at::Tensor conv2d_or(torch::Tensor &input, torch::Tensor &weight, int &bit_length, at::IntArrayRef stride=1, at::IntArrayRef padding=0, at::IntArrayRef dilation=1, int64_t groups=1) {
//     clock_t start;
//     start = clock();
//     std::cout << (double)(clock() - start) << std::endl;
//     auto input_size = input.sizes();
//     auto weight_size = weight.sizes();
    
//     auto device = at::kCUDA;
//     std::cout << (double)(clock() - start) << std::endl;
//     auto input_device = input.to(device);
//     auto weight_pos = weight.clamp(0,32).to(device);
//     auto weight_neg = -weight.clamp(-32,0).to(device);
    
//     std::cout << (double)(clock() - start) << std::endl;
//     at::Tensor rand_input = at::rand(input_size, device);
//     at::Tensor rand_weight_pos = at::rand(weight_size, device);
//     at::Tensor rand_weight_neg = at::rand(weight_size, device);
    
//     std::cout << (double)(clock() - start) << std::endl;
    
//     auto input_type = torch::kFloat16;
//     auto output_type = torch::kFloat16;
    
//     auto input_bit = (input_device > rand_input).to(input_type).to(device);
//     auto weight_pos_bit = (weight_pos > rand_weight_pos).to(input_type).to(device);
//     auto weight_neg_bit = (weight_neg > rand_weight_neg).to(input_type).to(device);
    
    
//     auto result_pos = at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
//     auto result_neg = at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
    
//     std::cout << (double)(clock() - start) << std::endl;
    
//     for(int i=1; i<bit_length; i++) {
//         rand_input.uniform_();
//         rand_weight_pos.uniform_();
//         rand_weight_neg.uniform_();
//         input_bit = (input_device > rand_input).to(input_type);
//         weight_pos_bit = (weight_pos > rand_weight_pos).to(input_type);
//         weight_neg_bit = (weight_neg > rand_weight_neg).to(input_type);
//         result_pos += at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
//         result_neg += at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
//     }
//     std::cout << (double)(clock() - start) << std::endl;
//     at::Tensor result = result_pos.to(torch::kFloat32)/bit_length - result_neg.to(torch::kFloat32)/bit_length;
//     std::cout << (double)(clock() - start) << std::endl;
//     return result;
// }

at::Tensor conv2d_or(torch::Tensor &input, torch::Tensor &weight, int &bit_length, at::IntArrayRef stride=1, at::IntArrayRef padding=0, at::IntArrayRef dilation=1, int64_t groups=1) {
    clock_t start;
    start = clock();
    std::cout << (double)(clock() - start) << std::endl;
    auto input_size = input.sizes();
    auto weight_size = weight.sizes();
 
    auto input_type = torch::kFloat16;
    auto output_type = torch::kFloat16;
    
    auto input_test = input.to(at::kCUDA);
    auto gpu = input_test.device();
    
    auto device = at::device(gpu).dtype(torch::kUInt8);

    std::cout << (double)(clock() - start) << std::endl;
    auto input_device = (input*256).to(device);
    auto weight_pos = (weight*256).clamp(0,1).to(device);
    auto weight_neg = -(weight*256).clamp(-1,0).to(device);
    
    std::cout << (double)(clock() - start) << std::endl;
    
    int64_t upper = 255;
    
    at::Tensor rand_input = torch::randint(255, input_size, at::device(gpu).dtype(torch::kUInt8));
    at::Tensor rand_weight_pos = torch::randint(255, weight_size, at::device(gpu).dtype(torch::kUInt8));
    at::Tensor rand_weight_neg = torch::randint(255, weight_size, at::device(gpu).dtype(torch::kUInt8));
    
    std::cout << (double)(clock() - start) << std::endl;
    
    auto input_bit = (input_device > rand_input).to(input_type);
    auto weight_pos_bit = (weight_pos > rand_weight_pos).to(input_type);
    auto weight_neg_bit = (weight_neg > rand_weight_neg).to(input_type);
    
    auto result_pos = at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
    auto result_neg = at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
    
    std::cout << (double)(clock() - start) << std::endl;
    
    for(int i=1; i<bit_length; i++) {
        rand_input.random_(255);
        rand_weight_pos.random_(255);
        rand_weight_neg.random_(255);
        input_bit = (input_device > rand_input).to(input_type);
        weight_pos_bit = (weight_pos > rand_weight_pos).to(input_type);
        weight_neg_bit = (weight_neg > rand_weight_neg).to(input_type);
        result_pos += at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
        result_neg += at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
    }
    std::cout << (double)(clock() - start) << std::endl;
    at::Tensor result = result_pos.to(torch::kFloat32)/bit_length - result_neg.to(torch::kFloat32)/bit_length;
    std::cout << (double)(clock() - start) << std::endl;
    return result;
}

int main() {
    torch::Tensor input = torch::rand({128,64,28,28});
    torch::Tensor weight = torch::randn({64,64,3,3})/4;
    int bit_length=128;
    int stride=1;
    int padding=2;
//     auto result = conv2d_or(input, weight, bit_length, stride, padding);

//     auto result_cor = at::conv2d(input, weight, {}, stride, padding);
//     std::cout << "Finished convolution" << std::endl;
    
    torch::Tensor input_f = torch::rand({8,8});
    torch::Tensor weight_f = torch::randn({8,8})/4;
    int split_size = 32;
    auto input_split = at::split(input_f, split_size, -1);
    int add_full = 4;
    auto result_or = linear_or(input_f, weight_f, bit_length, add_full);
    std::cout << "Finished linear" << std::endl;
    return 0;
}
