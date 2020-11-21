#include <ATen/ATen.h>
// #include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

namespace F = torch::nn::functional;

// at::IntArrayRef bitstream(torch::Tensor input, int bit_length) {
//     auto input_size = input.sizes();
//     auto input_leng = input_size.size();
//     std::cout << input_leng << std::endl;
// //     for(int i=0; i<input_leng; i++){std::cout << input_size[i] << std::endl;}
//     at::Tensor a = at::Tensor();
//     switch(input_leng)
//     {
//         case 1:
//             a = at::rand({input_size[0],bit_length});
//             break;
//         case 2:
//             a = at::rand({input_size[0],input_size[1],bit_length});
//             break;
//         case 3:
//             a = at::rand({input_size[0],input_size[1],input_size[2],bit_length});
//             break;
//         case 4:
//             a = at::rand({input_size[0],input_size[1],input_size[2],input_size[3],bit_length});
//             break;
//         default:
//             a = at::rand({bit_length});
//     }
//     auto input_bit_expand = (input.unsqueeze(-1) > a).to(torch::kUInt8);
//     auto input_back = input_bit_expand.sum({-1}).to(torch::kFloat32)/bit_length;
//     std::cout << input_back << std::endl;
//     return input_size;
// }

at::Tensor bitstream(torch::Tensor input, int bit_length) {
    int orig_bit_length = bit_length;
    auto rand_input = torch::rand_like(input);
    auto result = (input>rand_input).to(torch::kFloat16);
    for(int i=1; i<bit_length; i++) {
        rand_input.uniform_();
        result += (input>rand_input).to(torch::kFloat16);
    }
    at::Tensor output = result/bit_length;
    return output;
}

at::Tensor linear_or_shared(torch::Tensor input, torch::Tensor weight, int bit_length, int add_full) {
    auto input_size = input.sizes();
    auto weight_size = weight.sizes();
    
    auto device = input.device();
    auto input_type = torch::kFloat16;
    auto output_type = torch::kFloat16;
    auto comp_type = torch::kInt16;
    auto comp_device = at::device(device).dtype(comp_type);
    auto range = bit_length-1;
    auto split_size = input_size[1]/add_full;
    
    auto input_split = at::split((input*bit_length).to(comp_type), split_size, 1);
    auto weight_pos_split = at::split((weight*bit_length).clamp(0,range).to(comp_type), split_size, 1);
    auto weight_neg_split = at::split(-(weight*bit_length).clamp(-range,0).to(comp_type), split_size, 1);
    
    at::IntArrayRef input_split_size = {input_split[0].size(-1)};
    at::IntArrayRef weight_split_size = {weight_pos_split[0].size(-1)};
    auto rand_input = torch::randint(range, input_split_size, comp_device);
    auto rand_weight_pos = torch::randint(range, weight_split_size, comp_device)[0];
    auto rand_weight_neg = torch::randint(range, weight_split_size, comp_device)[0];
    
    auto input_bit = (input_split[0] > rand_input).to(input_type);
    auto weight_pos_bit = (weight_pos_split[0] > rand_weight_pos).to(input_type);
    auto weight_neg_bit = (weight_neg_split[0] > rand_weight_neg).to(input_type);
    auto result_pos = at::linear(input_bit, weight_pos_bit).sign().to(output_type);
    auto result_neg = at::linear(input_bit, weight_neg_bit).sign().to(output_type);
    
    for(int i=1; i<add_full; i++) {
        rand_input.random_(range);
        rand_weight_pos.random_(range);
        rand_weight_neg.random_(range);
        input_bit = (input_split[i] > rand_input).to(input_type);
        weight_pos_bit = (weight_pos_split[i] > rand_weight_pos).to(input_type);
        weight_neg_bit = (weight_neg_split[i] > rand_weight_neg).to(input_type);
        result_pos += at::linear(input_bit, weight_pos_bit).sign().to(output_type);
        result_neg += at::linear(input_bit, weight_neg_bit).sign().to(output_type);
    }
    
    for(int j=1; j<bit_length; j++) {
        for(int i=0;i<add_full;i++) {
            rand_input.random_(range);
            rand_weight_pos.random_(range);
            rand_weight_neg.random_(range);
            input_bit = (input_split[i] > rand_input).to(input_type);
            weight_pos_bit = (weight_pos_split[i] > rand_weight_pos).to(input_type);
            weight_neg_bit = (weight_neg_split[i] > rand_weight_neg).to(input_type);
            result_pos += at::linear(input_bit, weight_pos_bit).sign().to(output_type);
            result_neg += at::linear(input_bit, weight_neg_bit).sign().to(output_type);
        }
    }
//     at::Tensor result = result_pos.to(torch::kFloat32)/bit_length - result_neg.to(torch::kFloat32)/bit_length;
    at::Tensor result_pos_scale = result_pos.to(torch::kFloat32)/bit_length;
    at::Tensor result_neg_scale = result_neg.to(torch::kFloat32)/bit_length;
    at::Tensor result = at::stack({result_pos_scale, result_neg_scale}, 0);
    return result;
}

at::Tensor linear_or(torch::Tensor input, torch::Tensor weight, int bit_length, int add_full) {
    auto input_size = input.sizes();
    auto weight_size = weight.sizes();
    
    auto device = input.device();
    auto input_type = torch::kFloat16;
    auto output_type = torch::kFloat16;
    auto comp_type = torch::kInt16;
    auto comp_device = at::device(device).dtype(comp_type);
    auto range = bit_length-1;
    auto split_size = input_size[1]/add_full;
    
    auto input_split = at::split((input*bit_length).to(comp_type), split_size, 1);
    auto weight_pos_split = at::split((weight*bit_length).clamp(0,range).to(comp_type), split_size, 1);
    auto weight_neg_split = at::split(-(weight*bit_length).clamp(-range,0).to(comp_type), split_size, 1);
    
    auto input_split_size = input_split[0].sizes();
    auto weight_split_size = weight_pos_split[0].sizes();
    auto rand_input = torch::randint(range, input_split_size, comp_device);
    auto rand_weight_pos = torch::randint(range, weight_split_size, comp_device)[0];
    auto rand_weight_neg = torch::randint(range, weight_split_size, comp_device)[0];
    
    auto input_bit = (input_split[0] > rand_input).to(input_type);
    auto weight_pos_bit = (weight_pos_split[0] > rand_weight_pos).to(input_type);
    auto weight_neg_bit = (weight_neg_split[0] > rand_weight_neg).to(input_type);
    auto result_pos = at::linear(input_bit, weight_pos_bit).sign().to(output_type);
    auto result_neg = at::linear(input_bit, weight_neg_bit).sign().to(output_type);
    
    for(int i=1; i<add_full; i++) {
        rand_input.random_(range);
        rand_weight_pos.random_(range);
        rand_weight_neg.random_(range);
        input_bit = (input_split[i] > rand_input).to(input_type);
        weight_pos_bit = (weight_pos_split[i] > rand_weight_pos).to(input_type);
        weight_neg_bit = (weight_neg_split[i] > rand_weight_neg).to(input_type);
        result_pos += at::linear(input_bit, weight_pos_bit).sign().to(output_type);
        result_neg += at::linear(input_bit, weight_neg_bit).sign().to(output_type);
    }
    
    for(int j=1; j<bit_length; j++) {
        for(int i=0;i<add_full;i++) {
            rand_input.random_(range);
            rand_weight_pos.random_(range);
            rand_weight_neg.random_(range);
            input_bit = (input_split[i] > rand_input).to(input_type);
            weight_pos_bit = (weight_pos_split[i] > rand_weight_pos).to(input_type);
            weight_neg_bit = (weight_neg_split[i] > rand_weight_neg).to(input_type);
            result_pos += at::linear(input_bit, weight_pos_bit).sign().to(output_type);
            result_neg += at::linear(input_bit, weight_neg_bit).sign().to(output_type);
        }
    }
//     at::Tensor result = result_pos.to(torch::kFloat32)/bit_length - result_neg.to(torch::kFloat32)/bit_length;
    at::Tensor result_pos_scale = result_pos.to(torch::kFloat32)/bit_length;
    at::Tensor result_neg_scale = result_neg.to(torch::kFloat32)/bit_length;
    at::Tensor result = at::stack({result_pos_scale, result_neg_scale}, 0);
    return result;
}

at::Tensor linear_and(torch::Tensor input, torch::Tensor weight, int bit_length) {
    auto input_size = input.sizes();
    auto weight_size = weight.sizes();
    
    auto device = input.device();
    auto input_type = torch::kFloat16;
    auto output_type = torch::kFloat32;
    auto comp_type = torch::kInt16;
    auto comp_device = at::device(device).dtype(comp_type);
    auto range = bit_length-1;
    
    auto input_split = (input*bit_length).to(comp_type);
    auto weight_pos_split = (weight*bit_length).clamp(0,range).to(comp_type);
    auto weight_neg_split = -(weight*bit_length).clamp(-range,0).to(comp_type);

    auto rand_input = torch::randint(range, input_size, comp_device);
    auto rand_weight_pos = torch::randint(range, weight_size, comp_device);
    auto rand_weight_neg = torch::randint(range, weight_size, comp_device);
    
    auto input_bit = (input_split > rand_input).to(input_type);
    auto weight_pos_bit = (weight_pos_split > rand_weight_pos).to(input_type);
    auto weight_neg_bit = (weight_neg_split > rand_weight_neg).to(input_type);
    auto result_pos = at::linear(input_bit, weight_pos_bit).to(output_type);
    auto result_neg = at::linear(input_bit, weight_neg_bit).to(output_type);
    
    for(int j=1; j<bit_length; j++) {
        rand_input.random_(range);
        rand_weight_pos.random_(range);
        rand_weight_neg.random_(range);
        input_bit = (input_split > rand_input).to(input_type);
        weight_pos_bit = (weight_pos_split > rand_weight_pos).to(input_type);
        weight_neg_bit = (weight_neg_split > rand_weight_neg).to(input_type);
        result_pos += at::linear(input_bit, weight_pos_bit).to(output_type);
        result_neg += at::linear(input_bit, weight_neg_bit).to(output_type);
    }
//     at::Tensor result = result_pos.to(torch::kFloat32)/bit_length - result_neg.to(torch::kFloat32)/bit_length;
    at::Tensor result_pos_scale = result_pos.to(torch::kFloat32)/bit_length;
    at::Tensor result_neg_scale = result_neg.to(torch::kFloat32)/bit_length;
    at::Tensor result = at::stack({result_pos_scale, result_neg_scale}, 0);
    return result;
}

at::Tensor linear_xnor(torch::Tensor input, torch::Tensor weight, int bit_length) {
    auto input_size = input.sizes();
    auto weight_size = weight.sizes();
    
    float orig_bit_length = bit_length;
    
    auto device = input.device();
    auto input_type = torch::kFloat32;
    auto output_type = torch::kFloat32;
    auto comp_type = torch::kInt16;
    auto comp_device = at::device(device).dtype(comp_type);
    auto range = bit_length-1;
    
    auto input_scale = (input*bit_length).to(comp_type).to(input_type);
    auto weight_scale = (weight*bit_length).clamp(-range,range).to(comp_type).to(input_type);
    
    auto rand_input = torch::rand_like(input_scale)*2-1;
    auto rand_weight = torch::rand_like(weight_scale)*2-1;
    
    auto input_actual = input_scale / bit_length;
    auto weight_actual = weight_scale / bit_length;
    
    auto input_bit = (input_actual > rand_input).to(input_type)*2-1;
    auto weight_bit = (weight_actual > rand_weight).to(input_type)*2-1;
    auto result = at::linear(input_bit, weight_bit).to(output_type);
    
    input_scale -= input_bit;
    weight_scale -= weight_bit;
    bit_length--;
    
    while(bit_length>0) {
        rand_input.uniform_(-1,1);
        rand_weight.uniform_(-1,1);
        
        input_actual = input_scale / bit_length;
        weight_actual = weight_scale / bit_length;
        
        input_bit = (input_actual > rand_input).to(input_type)*2-1;
        weight_bit = (weight_actual > rand_weight).to(input_type)*2-1;
        result += at::linear(input_bit, weight_bit).to(output_type);
        
        input_scale -= input_bit;
        weight_scale -= weight_bit;
        bit_length--;
    }
    at::Tensor result_scale = result.to(torch::kFloat32) / orig_bit_length;
    return result_scale;
}

at::Tensor linear_or_acc(torch::Tensor input, torch::Tensor weight, int bit_length, int add_full) {
// std::vector<at::Tensor> linear_or_acc(torch::Tensor input, torch::Tensor weight, int bit_length, int add_full) {
// int linear_or_acc(torch::Tensor input, torch::Tensor weight, int bit_length, int add_full) {
    auto input_size = input.sizes();
    auto weight_size = weight.sizes();
    
    float orig_bit_length = bit_length;
    
    auto device = input.device();
    auto input_type = torch::kFloat16;
    auto output_type = torch::kFloat16;
    auto comp_type = torch::kInt16;
    auto comp_device = at::device(device).dtype(input_type);
    auto range = bit_length-1;
    auto split_size = input_size[1]/add_full;
    
    auto input_split = at::split((input*bit_length).to(comp_type).to(input_type), split_size, 1);
    auto weight_pos_split = at::split((weight*bit_length).clamp(0,range).to(comp_type).to(input_type), split_size, 1);
    auto weight_neg_split = at::split(-(weight*bit_length).clamp(-range,0).to(comp_type).to(input_type), split_size, 1);
    
    auto input_split_size = input_split[0].sizes();
    auto weight_split_size = weight_pos_split[0].sizes();
    
    auto rand_input = torch::rand_like(input_split[0]);
    auto rand_weight_pos = torch::rand_like(weight_pos_split[0])[0];
    auto rand_weight_neg = torch::rand_like(weight_neg_split[0])[0];
    
    auto input_actual = input_split[0] / bit_length;
    auto weight_pos_actual = weight_pos_split[0] / bit_length;
    auto weight_neg_actual = weight_neg_split[0] / bit_length;
    
    auto input_copy = input_split;
    
    auto input_bit = (input_actual > rand_input).to(input_type);
    auto weight_pos_bit = (weight_pos_actual > rand_weight_pos).to(input_type);
    auto weight_neg_bit = (weight_neg_actual > rand_weight_neg).to(input_type);
    auto result_pos = at::linear(input_bit, weight_pos_bit).sign().to(output_type);
    auto result_neg = at::linear(input_bit, weight_neg_bit).sign().to(output_type);
    input_split[0] -= input_bit;
    weight_pos_split[0] -= weight_pos_bit;
    weight_neg_split[0] -= weight_neg_bit;
    
    for(int i=1; i<add_full; i++) {
        rand_input.uniform_();
        rand_weight_pos.uniform_();
        rand_weight_neg.uniform_();
        
        input_actual = input_split[i] / bit_length;
        weight_pos_actual = weight_pos_split[i] / bit_length;
        weight_neg_actual = weight_neg_split[i] / bit_length;
        
        input_bit = (input_actual > rand_input).to(input_type);
        weight_pos_bit = (weight_pos_actual > rand_weight_pos).to(input_type);
        weight_neg_bit = (weight_neg_actual > rand_weight_neg).to(input_type);
        result_pos += at::linear(input_bit, weight_pos_bit).sign().to(output_type);
        result_neg += at::linear(input_bit, weight_neg_bit).sign().to(output_type);
        
        input_split[i] -= input_bit;
//         input_split[i] = input_split[i] - input_bit;
        weight_pos_split[i] -= weight_pos_bit;
        weight_neg_split[i] -= weight_neg_bit;
    }
    bit_length--;
    
//     for(int j=1; j<orig_bit_length; j++) {
    while(bit_length>0){
        for(int i=0;i<add_full;i++) {
            rand_input.uniform_();
            rand_weight_pos.uniform_();
            rand_weight_neg.uniform_();
            
            input_actual = input_split[i] / bit_length;
            weight_pos_actual = weight_pos_split[i] / bit_length;
            weight_neg_actual = weight_neg_split[i] / bit_length;
            
            input_bit = (input_actual > rand_input).to(input_type);
            weight_pos_bit = (weight_pos_actual > rand_weight_pos).to(input_type);
            weight_neg_bit = (weight_neg_actual > rand_weight_neg).to(input_type);
            result_pos += at::linear(input_bit, weight_pos_bit).sign().to(output_type);
            result_neg += at::linear(input_bit, weight_neg_bit).sign().to(output_type);
            
            input_split[i] -= input_bit;
//             input_split[i] = input_split[i] - input_bit;
            weight_pos_split[i] -= weight_pos_bit;
            weight_neg_split[i] -= weight_neg_bit;
        }
        bit_length--;
    }
    at::Tensor result_pos_scale = result_pos.to(torch::kFloat32)/orig_bit_length;
    at::Tensor result_neg_scale = result_neg.to(torch::kFloat32)/orig_bit_length;
    at::Tensor result = at::stack({result_pos_scale, result_neg_scale}, 0);
    return result;
}

at::Tensor linear_count(torch::Tensor input, torch::Tensor weight, int bit_length) {
    auto input_size = input.sizes();
    auto weight_size = weight.sizes();
    
    auto device = input.device();
    auto input_type = torch::kFloat16;
    auto output_type = torch::kFloat16;
    auto comp_type = torch::kInt16;
    auto comp_device = at::device(device).dtype(comp_type);
    auto range = bit_length-1;
    
    auto input_device = (input*bit_length).to(comp_type);
    auto weight_pos = (weight*bit_length).clamp(0,range).to(comp_type);
    auto weight_neg = -(weight*bit_length).clamp(-range,0).to(comp_type);
    
    auto rand_input = torch::randint(range, input_size, comp_device);
    auto rand_weight_pos = torch::randint(range, weight_size, comp_device);
    auto rand_weight_neg = torch::randint(range, weight_size, comp_device);
    
    auto input_bit = (input_device > rand_input).to(input_type);
    auto weight_pos_bit = (weight_pos > rand_weight_pos).to(input_type);
    auto weight_neg_bit = (weight_neg > rand_weight_neg).to(input_type);
    auto result_pos = at::linear(input_bit, weight_pos_bit).to(output_type);
    auto result_neg = at::linear(input_bit, weight_neg_bit).to(output_type);
    
    for(int j=1; j<bit_length; j++) {
        rand_input.random_(range);
        rand_weight_pos.random_(range);
        rand_weight_neg.random_(range);
        input_bit = (input_device > rand_input).to(input_type);
        weight_pos_bit = (weight_pos > rand_weight_pos).to(input_type);
        weight_neg_bit = (weight_neg > rand_weight_neg).to(input_type);
        result_pos += at::linear(input_bit, weight_pos_bit).to(output_type);
        result_neg += at::linear(input_bit, weight_neg_bit).to(output_type);
    }
    at::Tensor result = result_pos.to(torch::kFloat32)/bit_length - result_neg.to(torch::kFloat32)/bit_length;
    return result;
}

at::Tensor conv2d_or_shared(torch::Tensor input, torch::Tensor weight, int bit_length, at::IntArrayRef padding, at::IntArrayRef stride, int add_full) {
    
    auto input_size = input.sizes();
    auto weight_size = weight.sizes();
    
    at::IntArrayRef dilation = {1};
    int64_t groups=1;

    auto device = input.device();
    auto input_type = torch::kFloat16;
    auto output_type = torch::kFloat16;
    auto comp_type = torch::kInt16;
    auto comp_device = at::device(device).dtype(comp_type);    
    auto range = bit_length-1;
    auto split_size = input_size[1]/add_full;
    
    auto input_split = at::split((input*bit_length).to(comp_type), split_size, 1);
    auto weight_pos_split = at::split((weight*bit_length).clamp(0,range).to(comp_type), split_size, 1);
    auto weight_neg_split = at::split(-(weight*bit_length).clamp(-range,0).to(comp_type), split_size, 1);
    
    at::IntArrayRef input_split_size = input_split[0].sizes();
    at::IntArrayRef weight_split_size = weight_pos_split[0].sizes();
    torch::Tensor rand_input = torch::randint(range, input_split_size, comp_device);
    torch::Tensor rand_weight_pos = torch::randint(range, weight_split_size, comp_device)[0];
    torch::Tensor rand_weight_neg = torch::randint(range, weight_split_size, comp_device)[0];   
    
    auto input_bit = (input_split[0] > rand_input).to(input_type);
    auto weight_pos_bit = (weight_pos_split[0] > rand_weight_pos).to(input_type);
    auto weight_neg_bit = (weight_neg_split[0] > rand_weight_neg).to(input_type);
    auto result_pos = at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
    auto result_neg = at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).sign().to(output_type);

    for(int i=1; i<add_full; i++) {
        rand_input.random_(range);
        rand_weight_pos.random_(range);
        rand_weight_neg.random_(range);
        input_bit = (input_split[i] > rand_input).to(input_type);
        weight_pos_bit = (weight_pos_split[i] > rand_weight_pos).to(input_type);
        weight_neg_bit = (weight_neg_split[i] > rand_weight_neg).to(input_type);
        result_pos += at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
        result_neg += at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
    }
    
    for(int j=1; j<bit_length; j++) {
        for(int i=0;i<add_full;i++) {
            rand_input.random_(range);
            rand_weight_pos.random_(range);
            rand_weight_neg.random_(range);
            input_bit = (input_split[i] > rand_input).to(input_type);
            weight_pos_bit = (weight_pos_split[i] > rand_weight_pos).to(input_type);
            weight_neg_bit = (weight_neg_split[i] > rand_weight_neg).to(input_type);
            result_pos += at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
            result_neg += at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
        }
    }
//     at::Tensor result = result_pos.to(torch::kFloat32)/bit_length - result_neg.to(torch::kFloat32)/bit_length;
    at::Tensor result_pos_scale = result_pos.to(torch::kFloat32)/bit_length;
    at::Tensor result_neg_scale = result_neg.to(torch::kFloat32)/bit_length;
    at::Tensor result = at::stack({result_pos_scale, result_neg_scale}, 0);
    return result;
}
   
at::Tensor lfsr_5(torch::Tensor rand_in) {
    auto rand_out = ((rand_in/16)+(rand_in/4)%2)%2+2*(rand_in%16);
    return rand_out;
}

at::Tensor lfsr_7(torch::Tensor rand_in) {
    auto rand_out = ((rand_in/32)%2+rand_in/64)%2+2*(rand_in%64);
    return rand_out;
}

void matmul_outer(int* input, int* weight_pos, int* weight_neg, int* output, int add_size, int inner_size, int cin_size, int cout_size) {
    int o_cout_step = cin_size;

    int i_pw_step = inner_size * cin_size;
    int i_flatten_step = cin_size;
    
    int w_pw_step = inner_size * cout_size;
    int w_flatten_step = cout_size;
    
    for(int pw=0; pw<add_size; pw++) {
        int32_t* input_pw = input + pw * i_pw_step;
        int32_t* weight_pos_pw = weight_pos + pw * w_pw_step;
        int32_t* weight_neg_pw = weight_neg + pw * w_pw_step;
        int i_out_size = 0;

#ifdef __AVX512F__
        int32_t output_pos_16_0[16];
        int32_t output_pos_16_1[16];
        int32_t output_pos_16_2[16];
        int32_t output_pos_16_3[16];
        int32_t output_neg_16_0[16];
        int32_t output_neg_16_1[16];
        int32_t output_neg_16_2[16];
        int32_t output_neg_16_3[16];
        for (; i_out_size+15<cin_size; i_out_size+=16) {
            __m512i input_16_v;
            __m512i weight_pos_16_v;
            __m512i weight_neg_16_v;
            __m512i output_v_pos_16_0;
            __m512i output_v_pos_16_1;
            __m512i output_v_pos_16_2;
            __m512i output_v_pos_16_3;
            __m512i output_v_neg_16_0;
            __m512i output_v_neg_16_1;
            __m512i output_v_neg_16_2;
            __m512i output_v_neg_16_3;
            int w_cout=0;
            // Group of 4 weight channels
            for(; w_cout+3<w_flatten_step; w_cout+=4) {
                output_v_pos_16_0 = _mm512_set1_epi32(0);
                output_v_pos_16_1 = _mm512_set1_epi32(0);
                output_v_pos_16_2 = _mm512_set1_epi32(0);
                output_v_pos_16_3 = _mm512_set1_epi32(0);
                output_v_neg_16_0 = _mm512_set1_epi32(0);
                output_v_neg_16_1 = _mm512_set1_epi32(0);
                output_v_neg_16_2 = _mm512_set1_epi32(0);
                output_v_neg_16_3 = _mm512_set1_epi32(0);
                for(int flatten_in=0; flatten_in<inner_size; flatten_in++) {
                    input_16_v = _mm512_loadu_si512(input_pw + flatten_in*i_flatten_step + i_out_size);
                    weight_pos_16_v = _mm512_set1_epi32(weight_pos_pw[flatten_in*w_flatten_step + w_cout]);
                    output_v_pos_16_0 = _mm512_or_epi32(output_v_pos_16_0, _mm512_and_epi32(input_16_v, weight_pos_16_v));
                    weight_pos_16_v = _mm512_set1_epi32(weight_pos_pw[flatten_in*w_flatten_step + w_cout+1]);
                    output_v_pos_16_1 = _mm512_or_epi32(output_v_pos_16_1, _mm512_and_epi32(input_16_v, weight_pos_16_v));
                    weight_pos_16_v = _mm512_set1_epi32(weight_pos_pw[flatten_in*w_flatten_step + w_cout+2]);
                    output_v_pos_16_2 = _mm512_or_epi32(output_v_pos_16_2, _mm512_and_epi32(input_16_v, weight_pos_16_v));
                    weight_pos_16_v = _mm512_set1_epi32(weight_pos_pw[flatten_in*w_flatten_step + w_cout+3]);
                    output_v_pos_16_3 = _mm512_or_epi32(output_v_pos_16_3, _mm512_and_epi32(input_16_v, weight_pos_16_v));

                    weight_neg_v = _mm512_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout]);
                    output_v_neg_16_0 = _mm512_or_epi32(output_v_neg_16_0, _mm512_and_epi32(input_16_v, weight_neg_16_v));
                    weight_neg_v = _mm512_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout+1]);
                    output_v_neg_16_1 = _mm512_or_epi32(output_v_neg_16_1, _mm512_and_epi32(input_16_v, weight_neg_16_v));
                    weight_neg_v = _mm512_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout+2]);
                    output_v_neg_16_2 = _mm512_or_epi32(output_v_neg_16_2, _mm512_and_epi32(input_16_v, weight_neg_16_v));
                    weight_neg_v = _mm512_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout+3]);
                    output_v_neg_16_3 = _mm512_or_epi32(output_v_neg_16_3, _mm512_and_epi32(input_16_v, weight_neg_16_v));
                }
#ifdef __AVX512VPOPCNTDQ__
                __m512i output_store_0 = _mm512_load_epi32(output_batch + i_out_size + o_cout_step*w_cout);
                __m512i output_store_1 = _mm512_load_epi32(output_batch + i_out_size + o_cout_step*(w_cout+1));
                __m512i output_store_2 = _mm512_load_epi32(output_batch + i_out_size + o_cout_step*(w_cout+2));
                __m512i output_store_3 = _mm512_load_epi32(output_batch + i_out_size + o_cout_step*(w_cout+3));
                if (pw==0) {
                    output_store_0 = _mm512_set1_epi32(0);
                    output_store_1 = _mm512_set1_epi32(0);
                    output_store_2 = _mm512_set1_epi32(0);
                    output_store_3 = _mm512_set1_epi32(0);
                }

                output_store_0 = _mm512_add_epi32(output_store_0, _mm512_sub_epi32(_mm512_popcnt_epi32(output_v_pos_16_0), _mm512_popcnt_epi32(output_v_neg_16_0)));
                output_store_1 = _mm512_add_epi32(output_store_1, _mm512_sub_epi32(_mm512_popcnt_epi32(output_v_pos_16_1), _mm512_popcnt_epi32(output_v_neg_16_1)));
                output_store_2 = _mm512_add_epi32(output_store_2, _mm512_sub_epi32(_mm512_popcnt_epi32(output_v_pos_16_2), _mm512_popcnt_epi32(output_v_neg_16_2)));
                output_store_3 = _mm512_add_epi32(output_store_3, _mm512_sub_epi32(_mm512_popcnt_epi32(output_v_pos_16_3), _mm512_popcnt_epi32(output_v_neg_16_3)));

                _mm512_storeu_si512(output_batch + i_out_size, output_store_0);
                _mm512_storeu_si512(output_batch + i_out_size, output_store_1);
                _mm512_storeu_si512(output_batch + i_out_size, output_store_2);
                _mm512_storeu_si512(output_batch + i_out_size, output_store_3);
#else
                _mm512_storeu_si512(output_pos_16_0, output_v_pos_16_0);
                _mm512_storeu_si512(output_pos_16_1, output_v_pos_16_1);
                _mm512_storeu_si512(output_pos_16_2, output_v_pos_16_2);
                _mm512_storeu_si512(output_pos_16_3, output_v_pos_16_3);
                _mm512_storeu_si512(output_neg_16_0, output_v_neg_16_0);
                _mm512_storeu_si512(output_neg_16_1, output_v_neg_16_1);
                _mm512_storeu_si512(output_neg_16_2, output_v_neg_16_2);
                _mm512_storeu_si512(output_neg_16_3, output_v_neg_16_3);
                int o_store_0_offset = i_out_size + o_cout_step*w_cout;
                int o_store_1_offset = i_out_size + o_cout_step*(w_cout+1);
                int o_store_2_offset = i_out_size + o_cout_step*(w_cout+2);
                int o_store_3_offset = i_out_size + o_cout_step*(w_cout+3);
                for (int v_in=0; v_in<16; v_in++) {
                    if (pw==0) {
                        output[o_store_0_offset+v_in]=0;
                        output[o_store_1_offset+v_in]=0;
                        output[o_store_2_offset+v_in]=0;
                        output[o_store_3_offset+v_in]=0;
                    }
                    output[o_store_0_offset+v_in] += __builtin_popcount(output_pos_16_0[v_in]) - __builtin_popcount(output_neg_16_0[v_in]);
                    output[o_store_1_offset+v_in] += __builtin_popcount(output_pos_16_1[v_in]) - __builtin_popcount(output_neg_16_1[v_in]);
                    output[o_store_2_offset+v_in] += __builtin_popcount(output_pos_16_2[v_in]) - __builtin_popcount(output_neg_16_2[v_in]);
                    output[o_store_3_offset+v_in] += __builtin_popcount(output_pos_16_3[v_in]) - __builtin_popcount(output_neg_16_3[v_in]);
                }
#endif
            }
            //Leftover channels
            for(; w_cout<w_flatten_step; w_cout++) {
                output_v_pos_16_0 = _mm512_set1_epi32(0);
                output_v_neg_16_0 = _mm512_set1_epi32(0);
                for(int flatten_in=0; flatten_in<inner_size; flatten_in++) {
//                     std::cout << "Input address " << pw * i_pw_step + flatten_in*i_flatten_step + i_out_size << std::endl;
//                     std::cout << "Weight address " << pw * w_pw_step + flatten_in*w_flatten_step + w_cout << std::endl;
                    input_16_v = _mm512_loadu_si512(input_pw + flatten_in*i_flatten_step + i_out_size);
                    weight_pos_16_v = _mm512_set1_epi32(weight_pos_pw[flatten_in*w_flatten_step + w_cout]);
                    output_v_pos_16_0 = _mm512_or_epi32(output_v_pos_16_0, _mm512_and_epi32(input_16_v, weight_pos_16_v));
                    weight_neg_16_v = _mm512_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout]);
                    output_v_neg_16_0 = _mm512_or_epi32(output_v_neg_16_0, _mm512_and_epi32(input_16_v, weight_neg_16_v));
                    
                    //Testing purpose
//                     _mm512_storeu_si512(output_pos_0, output_v_pos_0);
//                     std::cout << flatten_in << "output_pos_0 ";
//                     for(int i=0; i<16; i++) {
//                         std::cout << __builtin_popcount(output_pos_0[i]) << " ";
//                     }
//                     std::cout << std::endl;
                }
#ifdef __AVX512VPOPCNTDQ__
                __m512i output_store_0 = _mm512_load_epi32(output_batch + i_out_size + o_cout_step*w_cout);
                if (pw==0) {
                    output_store_0 = _mm512_set1_epi32(0);
                }

                output_store_0 = _mm512_add_epi32(output_store_0, _mm512_sub_epi32(_mm512_popcnt_epi32(output_v_pos_16_0), _mm512_popcnt_epi32(output_v_neg_16_0)));

                _mm512_storeu_si512(output_batch + i_out_size, output_store_0);
#else
                _mm512_storeu_si512(output_pos_16_0, output_v_pos_16_0);
                _mm512_storeu_si512(output_neg_16_0, output_v_neg_16_0);
                int o_store_0_offset = i_out_size + o_cout_step*w_cout;
                for (int v_in=0; v_in<16; v_in++) {
                    if (pw==0) {
                        output[o_store_0_offset+v_in]=0;
                    }
                    output[o_store_0_offset+v_in] += __builtin_popcount(output_pos_16_0[v_in]) - __builtin_popcount(output_neg_16_0[v_in]);
                }
#endif
            }
        }
#endif
        
#ifdef __AVX2__
        int32_t output_pos_8_0[8];
        int32_t output_pos_8_1[8];
        int32_t output_pos_8_2[8];
        int32_t output_pos_8_3[8];
        int32_t output_neg_8_0[8];
        int32_t output_neg_8_1[8];
        int32_t output_neg_8_2[8];
        int32_t output_neg_8_3[8];
        for (; i_out_size+7<cin_size; i_out_size+=8) {
            __m256i input_8_v;
            __m256i weight_pos_8_v;
            __m256i weight_neg_8_v;
            __m256i output_v_pos_8_0;
            __m256i output_v_pos_8_1;
            __m256i output_v_pos_8_2;
            __m256i output_v_pos_8_3;
            __m256i output_v_neg_8_0;
            __m256i output_v_neg_8_1;
            __m256i output_v_neg_8_2;
            __m256i output_v_neg_8_3;
            int w_cout=0;
            // Group of 4 weight channels
            for(; w_cout+3<w_flatten_step; w_cout+=4) {
                output_v_pos_8_0 = _mm256_set1_epi32(0);
                output_v_pos_8_1 = _mm256_set1_epi32(0);
                output_v_pos_8_2 = _mm256_set1_epi32(0);
                output_v_pos_8_3 = _mm256_set1_epi32(0);
                output_v_neg_8_0 = _mm256_set1_epi32(0);
                output_v_neg_8_1 = _mm256_set1_epi32(0);
                output_v_neg_8_2 = _mm256_set1_epi32(0);
                output_v_neg_8_3 = _mm256_set1_epi32(0);
                for(int flatten_in=0; flatten_in<inner_size; flatten_in++) {
                    input_8_v = _mm256_lddqu_si256((__m256i *)(input_pw + flatten_in*i_flatten_step + i_out_size));
                    weight_pos_8_v = _mm256_set1_epi32(weight_pos_pw[flatten_in*w_flatten_step + w_cout]);
                    output_v_pos_8_0 = _mm256_or_si256(output_v_pos_8_0, _mm256_and_si256(input_8_v, weight_pos_8_v));
                    weight_pos_8_v = _mm256_set1_epi32(weight_pos_pw[flatten_in*w_flatten_step + w_cout+1]);
                    output_v_pos_8_1 = _mm256_or_si256(output_v_pos_8_1, _mm256_and_si256(input_8_v, weight_pos_8_v));
                    weight_pos_8_v = _mm256_set1_epi32(weight_pos_pw[flatten_in*w_flatten_step + w_cout+2]);
                    output_v_pos_8_2 = _mm256_or_si256(output_v_pos_8_2, _mm256_and_si256(input_8_v, weight_pos_8_v));
                    weight_pos_8_v = _mm256_set1_epi32(weight_pos_pw[flatten_in*w_flatten_step + w_cout+3]);
                    output_v_pos_8_3 = _mm256_or_si256(output_v_pos_8_3, _mm256_and_si256(input_8_v, weight_pos_8_v));

                    weight_neg_8_v = _mm256_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout]);
                    output_v_neg_8_0 = _mm256_or_si256(output_v_neg_8_0, _mm256_and_si256(input_8_v, weight_neg_8_v));
                    weight_neg_8_v = _mm256_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout+1]);
                    output_v_neg_8_1 = _mm256_or_si256(output_v_neg_8_1, _mm256_and_si256(input_8_v, weight_neg_8_v));
                    weight_neg_8_v = _mm256_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout+2]);
                    output_v_neg_8_2 = _mm256_or_si256(output_v_neg_8_2, _mm256_and_si256(input_8_v, weight_neg_8_v));
                    weight_neg_8_v = _mm256_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout+3]);
                    output_v_neg_8_3 = _mm256_or_si256(output_v_neg_8_3, _mm256_and_si256(input_8_v, weight_neg_8_v));
                }
                _mm256_storeu_si256((__m256i *)output_pos_8_0, output_v_pos_8_0);
                _mm256_storeu_si256((__m256i *)output_pos_8_1, output_v_pos_8_1);
                _mm256_storeu_si256((__m256i *)output_pos_8_2, output_v_pos_8_2);
                _mm256_storeu_si256((__m256i *)output_pos_8_3, output_v_pos_8_3);
                _mm256_storeu_si256((__m256i *)output_neg_8_0, output_v_neg_8_0);
                _mm256_storeu_si256((__m256i *)output_neg_8_1, output_v_neg_8_1);
                _mm256_storeu_si256((__m256i *)output_neg_8_2, output_v_neg_8_2);
                _mm256_storeu_si256((__m256i *)output_neg_8_3, output_v_neg_8_3);
                int o_store_0_offset = i_out_size + o_cout_step*w_cout;
                int o_store_1_offset = i_out_size + o_cout_step*(w_cout+1);
                int o_store_2_offset = i_out_size + o_cout_step*(w_cout+2);
                int o_store_3_offset = i_out_size + o_cout_step*(w_cout+3);
                for (int v_in=0; v_in<8; v_in++) {
                    if (pw==0) {
                        output[o_store_0_offset+v_in]=0;
                        output[o_store_1_offset+v_in]=0;
                        output[o_store_2_offset+v_in]=0;
                        output[o_store_3_offset+v_in]=0;
                    }
                    output[o_store_0_offset+v_in] += __builtin_popcount(output_pos_8_0[v_in]) - __builtin_popcount(output_neg_8_0[v_in]);
                    output[o_store_1_offset+v_in] += __builtin_popcount(output_pos_8_1[v_in]) - __builtin_popcount(output_neg_8_1[v_in]);
                    output[o_store_2_offset+v_in] += __builtin_popcount(output_pos_8_2[v_in]) - __builtin_popcount(output_neg_8_2[v_in]);
                    output[o_store_3_offset+v_in] += __builtin_popcount(output_pos_8_3[v_in]) - __builtin_popcount(output_neg_8_3[v_in]);
                }
            }
            //Leftover channels
            for(; w_cout<w_flatten_step; w_cout++) {
                output_v_pos_8_0 = _mm256_set1_epi32(0);
                output_v_neg_8_0 = _mm256_set1_epi32(0);
                for(int flatten_in=0; flatten_in<inner_size; flatten_in++) {
//                     std::cout << "Input address " << pw * i_pw_step + flatten_in*i_flatten_step + i_out_size << std::endl;
//                     std::cout << "Weight address " << pw * w_pw_step + flatten_in*w_flatten_step + w_cout << std::endl;
                    input_8_v = _mm256_lddqu_si256((__m256i *) (input_pw + flatten_in*i_flatten_step + i_out_size));
                    weight_pos_8_v = _mm256_set1_epi32(weight_pos_pw[flatten_in*w_flatten_step + w_cout]);
                    output_v_pos_8_0 = _mm256_or_si256(output_v_pos_8_0, _mm256_and_si256(input_8_v, weight_pos_8_v));
                    weight_neg_8_v = _mm256_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout]);
                    output_v_neg_8_0 = _mm256_or_si256(output_v_neg_8_0, _mm256_and_si256(input_8_v, weight_neg_8_v));
                    
                    //Testing purpose
//                     _mm512_storeu_si512(output_pos_0, output_v_pos_0);
//                     std::cout << flatten_in << "output_pos_0 ";
//                     for(int i=0; i<16; i++) {
//                         std::cout << __builtin_popcount(output_pos_0[i]) << " ";
//                     }
//                     std::cout << std::endl;
                }
                _mm256_storeu_si256((__m256i *)output_pos_8_0, output_v_pos_8_0);
                _mm256_storeu_si256((__m256i *)output_neg_8_0, output_v_neg_8_0);
                int o_store_0_offset = i_out_size + o_cout_step*w_cout;
                for (int v_in=0; v_in<8; v_in++) {
                    if (pw==0) {
                        output[o_store_0_offset+v_in]=0;
                    }
                    output[o_store_0_offset+v_in] += __builtin_popcount(output_pos_8_0[v_in]) - __builtin_popcount(output_neg_8_0[v_in]);
                }
            }
        }
#endif
        
        for (; i_out_size<i_flatten_step; i_out_size++) {
            int w_cout=0;
            int input_v;
            int weight_pos_v;
            int weight_neg_v;

            for(; w_cout<w_flatten_step; w_cout++) {
                int output_pos_c = 0;
                int output_neg_c = 0;
//                 std::cout << "Read weights" << std::endl;
                for(int flatten_in=0; flatten_in<inner_size; flatten_in++) {
//                     std::cout << "Input address " << pw * i_pw_step + flatten_in*i_flatten_step + i_out_size << std::endl;
                    input_v = input_pw[flatten_in*i_flatten_step + i_out_size];
                    weight_pos_v = weight_pos_pw[flatten_in*w_flatten_step + w_cout];
                    weight_neg_v = weight_neg_pw[flatten_in*w_flatten_step + w_cout];
                    
                    output_pos_c = output_pos_c | (input_v & weight_pos_v);
                    output_neg_c = output_neg_c | (input_v & weight_neg_v);
//                     std::cout<< input_v << " " << weight_pos_v << " " << weight_neg_v << " " << __builtin_popcount(input_v & weight_pos_v) << " " << __builtin_popcount(input_v & weight_neg_v) << std::endl;
                }
                if (pw==0) {
                    output[i_out_size+o_cout_step*w_cout] = 0;
                }
                output[i_out_size+o_cout_step*w_cout] += __builtin_popcount(output_pos_c) - __builtin_popcount(output_neg_c);
//                 std::cout << i_out_size+o_cout_step*w_cout << " " << output[i_out_size+o_cout_step*w_cout] << std::endl;
//                 std::cout << "At the outputs" << std::endl;
//                 std::cout << pw << " " << i_out_size+o_cout_step*w_cout << " " << output[i_out_size+o_cout_step*w_cout] << std::endl;
            }
        }

    }
}

int lfsr_7_s(int value) {
	return ((value/32)%2+value/64)%2+2*(value%64);
}

at::Tensor conv2d_add_partial_new(torch::Tensor input, torch::Tensor weight, int bit_length, at::IntArrayRef padding, at::IntArrayRef stride) {
    auto weight_size = weight.sizes();
    auto input_pad = F::pad(input, F::PadFuncOptions({padding[0], padding[0], padding[1], padding[1]})/*.mode(torch::kReplicate)*/);
    auto compare_type = torch::kInt32;
    auto store_type = torch::kInt32;
    auto input_split = (input_pad*bit_length).clamp(0, bit_length-1).to(compare_type);
    auto input_size = input_split.sizes();
    auto w_pos_split = (weight*bit_length).clamp(0, bit_length-1).to(compare_type);
    auto w_neg_split = -(weight*bit_length).clamp(1-bit_length, 0).to(compare_type);
    int channel_in = weight_size[1];
    int w_weight = weight_size[2];
    int h_weight = weight_size[3];
    int bit_packs = bit_length/32;
    // Weight stream generation
    int32_t* weight_pos_bin_point = w_pos_split.data_ptr<int32_t>();
    int32_t* weight_neg_bin_point = w_neg_split.data_ptr<int32_t>();
    int32_t* weight_pos_stream_new = new int32_t [bit_packs * weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3]];
    int32_t* weight_neg_stream_new = new int32_t [bit_packs * weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3]];
    int pos_seed = 67;
    int neg_seed = 37;
    int* pos_seed_arr = new int [weight_size[1] * weight_size[2]];
    int* neg_seed_arr = new int [weight_size[1] * weight_size[2]];
    for(int i=0; i<weight_size[1] * weight_size[2]; i++) {
        pos_seed_arr[i] = (pos_seed + i + 1)%(bit_length-1);
        neg_seed_arr[i] = (neg_seed + i + 1)%(bit_length-1);
    }
    for(int pack=0; pack<bit_packs; pack++) {
        int32_t* weight_pos_pack = weight_pos_stream_new + pack * weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3];
        int32_t* weight_neg_pack = weight_neg_stream_new + pack * weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3];
        for(int h_in=0; h_in<weight_size[3]; h_in++) {
            int32_t* weight_pos_hin = weight_pos_pack + h_in * weight_size[0] * weight_size[1] * weight_size[2];
            int32_t* weight_neg_hin = weight_neg_pack + h_in * weight_size[0] * weight_size[1] * weight_size[2];
            int flatten_in = 0;
            for(int c_in=0; c_in<weight_size[1]; c_in++) {
                int32_t* weight_pos_cin = weight_pos_hin + c_in * weight_size[0] * weight_size[2];
                int32_t* weight_neg_cin = weight_neg_hin + c_in * weight_size[0] * weight_size[2];
                for(int w_in=0; w_in<weight_size[2]; w_in++) {
                    int32_t* weight_pos_win = weight_pos_cin + w_in * weight_size[0];
                    int32_t* weight_neg_win = weight_neg_cin + w_in * weight_size[0];
                    int pos_seed_cur = pos_seed_arr[flatten_in];
                    int neg_seed_cur = neg_seed_arr[flatten_in];
                    for(int bit=0; bit<32; bit++) {
                        pos_seed_cur = lfsr_7_s(pos_seed_cur);
                        neg_seed_cur = lfsr_7_s(neg_seed_cur);
                        for(int c_out=0; c_out<weight_size[0]; c_out++) {
//                        	std::cout << "c_out" << pack * weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3] + h_in * weight_size[0] * weight_size[1] * weight_size[2] + c_in * weight_size[0] * weight_size[2] + w_in * weight_size[0] + c_out << std::endl;
                            int weight_pos_bin = weight_pos_bin_point[c_out*weight_size[1]*weight_size[2]*weight_size[3] + c_in*weight_size[2]*weight_size[3] + w_in*weight_size[3] + h_in];
                            int weight_neg_bin = weight_neg_bin_point[c_out*weight_size[1]*weight_size[2]*weight_size[3] + c_in*weight_size[2]*weight_size[3] + w_in*weight_size[3] + h_in];
                            if(bit==0) {
                                weight_pos_win[c_out] = 0;
                                weight_neg_win[c_out] = 0;
                            }
                            
                            weight_pos_win[c_out] = (weight_pos_win[c_out]*2) + (weight_pos_bin>pos_seed_cur);
                            weight_neg_win[c_out] = (weight_neg_win[c_out]*2) + (weight_neg_bin>neg_seed_cur);
                        }
                    }
                    if (h_in==weight_size[3]-1) {
                        pos_seed_arr[flatten_in] = pos_seed_cur;
                        neg_seed_arr[flatten_in] = neg_seed_cur;
                    }
                }
                flatten_in += 1;
            }
        }
    }

//	std::cout << "New weight pos new" << std::endl;
//	for(int i=0; i<bit_packs * weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3]; i++) {
//		std::cout << weight_pos_stream_new[i] << " ";
//	}
//	std::cout << std::endl;
//	std::cout << "New weight neg new" << std::endl;
//	for(int i=0; i<bit_packs * weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3]; i++) {
//		std::cout << weight_neg_stream_new[i] << " ";
//	}
//	std::cout << std::endl;

    int32_t* output_point_flat = new int32_t [input_size[0] * weight_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1)];
    int32_t* input_bin_point = input_split.data_ptr<int32_t>();
    int o_batch_step = weight_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int i_bin_batch_step = input_size[1] * input_size[2] * input_size[3];
    int add_size = bit_packs * weight_size[3];
    int inner_size = weight_size[1] * weight_size[2];
    int cin_size = (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int cout_size = weight_size[0];
    #pragma omp parallel for //num_threads(28)
    for(int batch=0; batch<input_size[0]; batch++) {
//    	std::cout << "Batch " << batch << std::endl;
        int32_t* output_batch = output_point_flat + batch*o_batch_step;
        int32_t* input_bin_batch = input_bin_point + batch*i_bin_batch_step;
	 // Input stream generation
        // Due to the current limitations, generation and im2col will be separate
        int32_t* input_stream = new int32_t [bit_packs * input_size[1] * input_size[2] * input_size[3]];
        int32_t* input_point = new int32_t [bit_packs * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) * weight_size[1] * weight_size[2] * weight_size[3]];
        int input_seed = 1;
        int* input_seed_arr = new int [input_size[1] * input_size[2] * input_size[3]];
        for(int i=0; i<input_size[1] * input_size[2] * input_size[3]; i++) {
            input_seed_arr[i] = (input_seed + i)%(bit_length-1);
        }
        for(int pack=0; pack<bit_packs; pack++) {
//        	std::cout << "Pack " << pack << std::endl;
        	int32_t* input_stream_pack = input_stream + pack * input_size[1] * input_size[2] * input_size[3];
        	int32_t* input_point_pack = input_point + pack * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) * weight_size[1] * weight_size[2] * weight_size[3];
        	// Generation
        	int input_seed_ind = 0;
        	for(int c_in=0; c_in<input_size[1]; c_in++) {
        		int32_t* input_bin_cin = input_bin_batch + c_in * input_size[2] * input_size[3];
        		int32_t* input_stream_cin = input_stream_pack + c_in * input_size[2] * input_size[3];
        		for(int w_in=0; w_in<input_size[2]; w_in++) {
        			int32_t* input_bin_win = input_bin_cin + w_in * input_size[3];
        			int32_t* input_stream_win = input_stream_cin + w_in * input_size[3];
        			for(int h_in=0; h_in<input_size[3]; h_in++) {
        				int input_seed_cur = input_seed_arr[input_seed_ind];
        				int input_stream_cur = 0;
        				int input_bin_cur = input_bin_win[h_in];
//        				std::cout << h_in << std::endl;
        				for(int bit=0; bit<32; bit++) {
        					input_seed_cur = lfsr_7_s(input_seed_cur);
        					input_stream_cur = (input_stream_cur*2) + (input_bin_cur > input_seed_cur);
        				}
        				input_seed_arr[input_seed_ind] = input_seed_cur;
        				input_stream_win[h_in] = input_stream_cur;
        				input_seed_ind++;
        			}
        		}
        	}
        	// im2col
        	for(int h_in=0; h_in<weight_size[3]; h_in++) {
        		int32_t* input_point_hin = input_point_pack + h_in * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) * weight_size[1] * weight_size[2];
        		for(int flatten_in=0; flatten_in<weight_size[1] * weight_size[2]; flatten_in++) {
        			int32_t* input_point_or = input_point_hin + flatten_in * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
        			int input_cin = flatten_in / weight_size[2];
        			int input_w_win = flatten_in % weight_size[2];
        			for(int flatten=0; flatten<(input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1); flatten++) {
//        				std::cout << h_in * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) * weight_size[1] * weight_size[2] +
//        						flatten_in * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) + flatten << std::endl;
        				int input_w = flatten / (input_size[3]-weight_size[3]+1);
        				int input_h = flatten % (input_size[3]-weight_size[3]+1);
        				input_point_or[flatten] = input_stream_pack[input_cin*input_size[2]*input_size[3] + (input_w+input_w_win)*input_size[3] + (input_h+h_in)];
        			}
        		}
        	}
		}
//        std::cout << "New input " << batch << std::endl;
//        for(int i=0; i<bit_packs * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) * weight_size[1] * weight_size[2] * weight_size[3]; i++) {
//        	std::cout << input_point[i] << " ";
//        }
//        std::cout << std::endl;
        // Computation
        matmul_outer(input_point, weight_pos_stream_new, weight_neg_stream_new, output_batch, add_size, inner_size, cin_size, cout_size);
        delete [] input_stream;
		delete [] input_point;
		delete [] input_seed_arr;
    }


    auto options = torch::TensorOptions().dtype(torch::kInt32);
    at::Tensor output_tensor = torch::from_blob(output_point_flat, {input_size[0], weight_size[0], (input_size[2]-weight_size[2]+1) , (input_size[3]-weight_size[3]+1)}, options).clone();
    delete [] weight_pos_stream_new;
    delete [] weight_neg_stream_new;
    delete [] pos_seed_arr;
    delete [] neg_seed_arr;
    delete [] output_point_flat;
    return output_tensor;
}

at::Tensor conv2d_add_partial(torch::Tensor input, torch::Tensor weight, int bit_length, at::IntArrayRef padding, at::IntArrayRef stride) {
    
    auto weight_size = weight.sizes();
    auto input_pad = F::pad(input, F::PadFuncOptions({padding[0], padding[0], padding[1], padding[1]})/*.mode(torch::kReplicate)*/);
    auto compare_type = torch::kInt16;
    auto store_type = torch::kInt32;
    auto input_split = (input_pad*bit_length).clamp(0, bit_length-1).to(compare_type);
    auto input_size = input_split.sizes();
    auto w_pos_split = (weight*bit_length).clamp(0, bit_length-1).to(compare_type);
    auto w_neg_split = -(weight*bit_length).clamp(1-bit_length, 0).to(compare_type);
    int channel_in = weight_size[1];
    int w_weight = weight_size[2];
    int h_weight = weight_size[3];
    int bit_packs = bit_length/32;
    
    // LFSR initialization
    int weight_split_size_flat = weight_size[1]*weight_size[2];
    int input_split_size_flat = input_size[1]*input_size[2]*input_size[3];
    auto weight_seed_pos = torch::arange(67, weight_split_size_flat+67)%(bit_length-1) + 1;
    auto weight_seed_neg = torch::arange(37, weight_split_size_flat+37)%(bit_length-1) + 1;
    auto input_seed = torch::arange(0, input_split_size_flat)%(bit_length-1) + 1;
    weight_seed_pos = weight_seed_pos.reshape({weight_size[1], weight_size[2], 1});
    weight_seed_neg = weight_seed_neg.reshape({weight_size[1], weight_size[2], 1});
    input_seed = input_seed.reshape({input_size[1], input_size[2], input_size[3]});
    
    // Generation
    // Currently it isn't packed. Do it later.
    auto input_stream = torch::zeros({bit_packs, input_size[0], input_size[1], input_size[2], input_size[3]}).to(store_type);
    auto weight_pos_stream = torch::zeros({bit_packs, weight_size[0], weight_size[1], weight_size[2], weight_size[3]}).to(store_type);
    auto weight_neg_stream = torch::zeros({bit_packs, weight_size[0], weight_size[1], weight_size[2], weight_size[3]}).to(store_type);
    
    for(int i=0; i<bit_packs; i++) {
        for (int j=0; j<32; j++) {
            input_seed = lfsr_7(input_seed);
            weight_seed_pos = lfsr_7(weight_seed_pos);
            weight_seed_neg = lfsr_7(weight_seed_neg);
            input_stream[i] = ((input_split > input_seed).to(store_type)) + (input_stream[i] * 2);
            weight_pos_stream[i] = ((w_pos_split > weight_seed_pos).to(store_type)) + (weight_pos_stream[i] * 2);
            weight_neg_stream[i] = ((w_neg_split > weight_seed_neg).to(store_type)) + (weight_neg_stream[i] * 2);
        }
    }
    
//     std::cout << input_stream << std::endl;
//     std::cout << weight_pos_stream << std::endl;
//     std::cout << weight_neg_stream << std::endl;
    // Im2Col Computation
    int32_t* input_stream_point = input_stream.data_ptr<int32_t>();
    // Input ordering: batch_size, pack*weight_size[3], weight_size[1]*weight_size[2], out_size
    // Weight ordering: pack*weight_size[3], weight_size[1]*weight_size[2], weight_size[0]
    int32_t* input_point_flat = new int32_t [bit_packs * input_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) * weight_size[1] * weight_size[2] * weight_size[3]];
    int32_t* input_point[input_size[0]];
    int32_t* weight_pos_stream_point = weight_pos_stream./*permute({0,4,2,3,1}).contiguous().*/data_ptr<int32_t>();
    int32_t* weight_neg_stream_point = weight_neg_stream./*permute({0,4,2,3,1}).contiguous().*/data_ptr<int32_t>();
    int32_t* weight_pos_point = new int32_t [bit_packs * weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3]];
    int32_t* weight_neg_point = new int32_t [bit_packs * weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3]];
    #pragma omp parallel for //num_threads(28)
    for(int bit_packs_c=0; bit_packs_c<bit_packs; bit_packs_c++) {
        int32_t* weight_pos_stream_pack = weight_pos_stream_point + bit_packs_c * weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3];
        int32_t* weight_neg_stream_pack = weight_neg_stream_point + bit_packs_c * weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3];
        int32_t* weight_pos_pack = weight_pos_point + bit_packs_c * weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3];
        int32_t* weight_neg_pack = weight_neg_point + bit_packs_c * weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3];
        for(int c_out=0; c_out<weight_size[0]; c_out++) {
            int32_t* weight_pos_stream_cout = weight_pos_stream_pack + c_out * weight_size[1] * weight_size[2] * weight_size[3];
            int32_t* weight_neg_stream_cout = weight_neg_stream_pack + c_out * weight_size[1] * weight_size[2] * weight_size[3];
            for(int c_in=0; c_in<weight_size[1]; c_in++) {
                int32_t* weight_pos_stream_cin = weight_pos_stream_cout + c_in * weight_size[2] * weight_size[3];
                int32_t* weight_neg_stream_cin = weight_neg_stream_cout + c_in * weight_size[2] * weight_size[3];
                for(int w_in=0; w_in<weight_size[2]; w_in++) {
                    int32_t* weight_pos_stream_win = weight_pos_stream_cin + w_in * weight_size[3];
                    int32_t* weight_neg_stream_win = weight_neg_stream_cin + w_in * weight_size[3];
                    for(int h_in=0; h_in<weight_size[3]; h_in++) {
                        weight_pos_pack[h_in * weight_size[0] * weight_size[1] * weight_size[2] + c_in * weight_size[0] * weight_size[2] + w_in * weight_size[0] + c_out] = weight_pos_stream_win[h_in];
                        weight_neg_pack[h_in * weight_size[0] * weight_size[1] * weight_size[2] + c_in * weight_size[0] * weight_size[2] + w_in * weight_size[0] + c_out] = weight_neg_stream_win[h_in];
                    }
                }
            }
        }
    }
//	std::cout << "Weight pos point" << std::endl;
//	for(int i=0; i<bit_packs*weight_size[0]*weight_size[1]*weight_size[2]*weight_size[3]; i++) {
//	 std::cout << weight_pos_point[i] << " " ;
//	}
//	std::cout << std::endl;
//	std::cout << "Weight neg point" << std::endl;
//	for(int i=0; i<bit_packs*weight_size[0]*weight_size[1]*weight_size[2]*weight_size[3]; i++) {
//	 std::cout << weight_neg_point[i] << " " ;
//	}
//	std::cout << std::endl;
    
    #pragma omp parallel for //num_threads(28)
    for(int batch=0; batch<input_size[0]; batch++) {
        int32_t* input_batch = input_point_flat + batch * bit_packs * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) * weight_size[1] * weight_size[2] * weight_size[3];
        input_point[batch] = input_batch;
        for(int pack=0; pack<bit_packs;pack++) {
            int32_t* input_pack = input_batch + pack * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) * weight_size[1] * weight_size[2] * weight_size[3];
            int32_t* input_stream_pack = input_stream_point + pack * input_size[0] * input_size[1] * input_size[2] * input_size[3] + batch * input_size[1] * input_size[2] * input_size[3];
            for(int h_in=0; h_in<weight_size[3]; h_in++) {
                int32_t* input_hin = input_pack + h_in * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) * weight_size[1] * weight_size[2];
                for(int flatten_in=0; flatten_in<weight_size[1] * weight_size[2]; flatten_in++) {
                    int32_t* input_or = input_hin + flatten_in * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
                    int input_cin = flatten_in / weight_size[2];
                    int input_w_win = flatten_in % weight_size[2];
                    for(int flatten=0; flatten<(input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1); flatten++) {
                        int input_w = flatten / (input_size[3]-weight_size[3]+1);
                        int input_h = flatten % (input_size[3]-weight_size[3]+1);
                        input_or[flatten] = input_stream_pack[input_cin*input_size[2]*input_size[3] + (input_w+input_w_win)*input_size[3] + (input_h+h_in)];
                    }
                }
            }
        }
    }
//     for(int i=0; i<bit_packs * input_size[0] * input_size[1] * input_size[2] * input_size[3]; i++) {
//         std::cout << input_stream_point[i] << " ";
//     }
//     std::cout << std::endl;
//     std::cout << std::endl;
//    std::cout << "Old input" << std::endl;
//	for(int i=0; i<bit_packs * input_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) * weight_size[1] * weight_size[2] * weight_size[3]; i++) {
//		std::cout << input_point_flat[i] << " ";
//	}
//	std::cout << std::endl;
    int32_t* output_point_flat = new int32_t [input_size[0] * weight_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1)];
    int32_t* output_point[input_size[0]];
    int o_batch_step = weight_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int i_batch_step = bit_packs * weight_size[3] * weight_size[1] * weight_size[2] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    for(int batch=0; batch<input_size[0]; batch++) output_point[batch] = output_point_flat + batch*o_batch_step;
    int add_size = bit_packs * weight_size[3];
    int inner_size = weight_size[1] * weight_size[2];
    int cin_size = (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int cout_size = weight_size[0];

    #pragma omp parallel for //num_threads(28)
    for(int batch=0; batch<input_size[0]; batch++) {
//         int32_t* output_batch = output_point_flat + batch*o_batch_step;
//         int32_t* input_batch = input_point_flat + batch*i_batch_step;
        int32_t* output_batch = output_point[batch];
        int32_t* input_batch = input_point[batch];
        matmul_outer(input_batch, weight_pos_point, weight_neg_point, output_batch, add_size, inner_size, cin_size, cout_size);
    }

    auto options = torch::TensorOptions().dtype(torch::kInt32);
    at::Tensor output_tensor = torch::from_blob(output_point_flat, {input_size[0], weight_size[0], (input_size[2]-weight_size[2]+1) , (input_size[3]-weight_size[3]+1)}, options).clone();
    delete [] input_point_flat;
    delete [] weight_pos_point;
    delete [] weight_neg_point;
    delete [] output_point_flat;
    return output_tensor;
}

at::Tensor conv2d_or(torch::Tensor input, torch::Tensor weight, int bit_length, at::IntArrayRef padding, at::IntArrayRef stride, int add_full) {
    auto input_size = input.sizes();
    auto weight_size = weight.sizes();
    
    at::IntArrayRef dilation = {1};
    int64_t groups=1;

    auto device = input.device();
    auto input_type = torch::kFloat16;
    auto output_type = torch::kFloat16;
    auto comp_type = torch::kInt16;
    auto comp_device = at::device(device).dtype(comp_type);    
    auto range = bit_length-1;
    auto split_size = input_size[1]/add_full;
    
    auto input_split = at::split((input*bit_length).to(comp_type), split_size, 1);
    auto weight_pos_split = at::split((weight*bit_length).clamp(0,range).to(comp_type), split_size, 1);
    auto weight_neg_split = at::split(-(weight*bit_length).clamp(-range,0).to(comp_type), split_size, 1);
    
    auto input_split_size = input_split[0].sizes();
    auto weight_split_size = weight_pos_split[0].sizes();
    torch::Tensor rand_input = torch::randint(range, input_split_size, comp_device);
    torch::Tensor rand_weight_pos = torch::randint(range, weight_split_size, comp_device)[0];
    torch::Tensor rand_weight_neg = torch::randint(range, weight_split_size, comp_device)[0];   
    
    auto input_bit = (input_split[0] > rand_input).to(input_type);
    auto weight_pos_bit = (weight_pos_split[0] > rand_weight_pos).to(input_type);
    auto weight_neg_bit = (weight_neg_split[0] > rand_weight_neg).to(input_type);
    auto result_pos = at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
    auto result_neg = at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).sign().to(output_type);

    for(int i=1; i<add_full; i++) {
        rand_input.random_(range);
        rand_weight_pos.random_(range);
        rand_weight_neg.random_(range);
        input_bit = (input_split[i] > rand_input).to(input_type);
        weight_pos_bit = (weight_pos_split[i] > rand_weight_pos).to(input_type);
        weight_neg_bit = (weight_neg_split[i] > rand_weight_neg).to(input_type);
        result_pos += at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
        result_neg += at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
    }
    
    for(int j=1; j<bit_length; j++) {
        for(int i=0;i<add_full;i++) {
            rand_input.random_(range);
            rand_weight_pos.random_(range);
            rand_weight_neg.random_(range);
            input_bit = (input_split[i] > rand_input).to(input_type);
            weight_pos_bit = (weight_pos_split[i] > rand_weight_pos).to(input_type);
            weight_neg_bit = (weight_neg_split[i] > rand_weight_neg).to(input_type);
            result_pos += at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
            result_neg += at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
        }
    }
//     at::Tensor result = result_pos.to(torch::kFloat32)/bit_length - result_neg.to(torch::kFloat32)/bit_length;
    at::Tensor result_pos_scale = result_pos.to(torch::kFloat32)/bit_length;
    at::Tensor result_neg_scale = result_neg.to(torch::kFloat32)/bit_length;
    at::Tensor result = at::stack({result_pos_scale, result_neg_scale}, 0);
    return result;
}

at::Tensor conv2d_and(torch::Tensor input, torch::Tensor weight, int bit_length, at::IntArrayRef padding, at::IntArrayRef stride) {
    auto input_size = input.sizes();
    auto weight_size = weight.sizes();
    
    at::IntArrayRef dilation = {1};
    int64_t groups=1;

    auto device = input.device();
    auto input_type = torch::kFloat16;
    auto output_type = torch::kFloat32;
    auto comp_type = torch::kInt16;
    auto comp_device = at::device(device).dtype(comp_type);    
    auto range = bit_length-1;
    
    auto input_split = (input*bit_length).to(comp_type);
    auto weight_pos_split = (weight*bit_length).clamp(0,range).to(comp_type);
    auto weight_neg_split = -(weight*bit_length).clamp(-range,0).to(comp_type);
    
    torch::Tensor rand_input = torch::randint(range, input_size, comp_device);
    torch::Tensor rand_weight_pos = torch::randint(range, weight_size, comp_device);
    torch::Tensor rand_weight_neg = torch::randint(range, weight_size, comp_device);   
    
    auto input_bit = (input_split > rand_input).to(input_type);
    auto weight_pos_bit = (weight_pos_split > rand_weight_pos).to(input_type);
    auto weight_neg_bit = (weight_neg_split > rand_weight_neg).to(input_type);
    auto result_pos = at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).to(output_type);
    auto result_neg = at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).to(output_type);
    
    for(int j=1; j<bit_length; j++) {
        rand_input.random_(range);
        rand_weight_pos.random_(range);
        rand_weight_neg.random_(range);
        input_bit = (input_split > rand_input).to(input_type);
        weight_pos_bit = (weight_pos_split > rand_weight_pos).to(input_type);
        weight_neg_bit = (weight_neg_split > rand_weight_neg).to(input_type);
        result_pos += at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).to(output_type);
        result_neg += at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).to(output_type);
    }
//     at::Tensor result = result_pos.to(torch::kFloat32)/bit_length - result_neg.to(torch::kFloat32)/bit_length;
    at::Tensor result_pos_scale = result_pos.to(torch::kFloat32)/bit_length;
    at::Tensor result_neg_scale = result_neg.to(torch::kFloat32)/bit_length;
    at::Tensor result = at::stack({result_pos_scale, result_neg_scale}, 0);
    return result;
}

at::Tensor conv2d_xnor(torch::Tensor input, torch::Tensor weight, int bit_length, at::IntArrayRef padding, at::IntArrayRef stride) {
    auto input_size = input.sizes();
    auto weight_size = weight.sizes();
    
    at::IntArrayRef dilation = {1};
    int64_t groups=1;
    
    float orig_bit_length = bit_length;
    
    auto device = input.device();
    auto input_type = torch::kFloat32;
    auto output_type = torch::kFloat32;
    auto comp_type = torch::kInt16;
    auto comp_device = at::device(device).dtype(comp_type);
    auto range = bit_length-1;
    
    auto input_scale = (input*bit_length).to(comp_type).to(input_type);
    auto weight_scale = (weight*bit_length).clamp(-range,range).to(comp_type).to(input_type);
    
    auto rand_input = torch::rand_like(input_scale)*2-1;
    auto rand_weight = torch::rand_like(weight_scale)*2-1;
    
    auto input_actual = input_scale / bit_length;
    auto weight_actual = weight_scale / bit_length;
    
    auto input_bit = (input_actual > rand_input).to(input_type)*2-1;
    auto weight_bit = (weight_actual > rand_weight).to(input_type)*2-1;
    auto result = at::conv2d(input_bit, weight_bit, {}, stride, padding, dilation, groups).to(output_type);
    
    input_scale -= input_bit;
    weight_scale -= weight_bit;
    bit_length--;
    
    while(bit_length>0) {
        rand_input.uniform_(-1,1);
        rand_weight.uniform_(-1,1);
        
//         rand_input = rand_input*2-1;
//         rand_weight = rand_weight*2-1;
        
        input_actual = input_scale / bit_length;
        weight_actual = weight_scale / bit_length;
        
        input_bit = (input_actual > rand_input).to(input_type)*2-1;
        weight_bit = (weight_actual > rand_weight).to(input_type)*2-1;
        result += at::conv2d(input_bit, weight_bit, {}, stride, padding, dilation, groups).to(output_type);
        
        input_scale -= input_bit;
        weight_scale -= weight_bit;
        bit_length--;
    }
    at::Tensor result_scale = result.to(torch::kFloat32) / orig_bit_length;
    return result_scale;
}

at::Tensor conv2d_or_acc(torch::Tensor input, torch::Tensor weight, int bit_length, at::IntArrayRef padding, at::IntArrayRef stride, int add_full) {
    auto input_size = input.sizes();
    auto weight_size = weight.sizes();
    
    at::IntArrayRef dilation = {1};
    int64_t groups=1;
    float orig_bit_length = bit_length;
    
    auto device = input.device();
    auto input_type = torch::kFloat16;
    auto output_type = torch::kFloat16;
    auto comp_type = torch::kInt16;
    auto comp_device = at::device(device).dtype(input_type);
    auto range = bit_length-1;
    
    auto input_split = (input*bit_length).to(comp_type).to(input_type);
    auto weight_pos_split = (weight*bit_length).clamp(0,range).to(comp_type).to(input_type);
    auto weight_neg_split = -(weight*bit_length).clamp(-range,0).to(comp_type).to(input_type);
    
    auto rand_input = torch::rand_like(input_split[0]);
    auto rand_weight_pos = torch::rand_like(weight_pos_split[0])[0];
    auto rand_weight_neg = torch::rand_like(weight_neg_split[0])[0];
    
    auto input_actual = input_split / bit_length;
    auto weight_pos_actual = weight_pos_split / bit_length;
    auto weight_neg_actual = weight_neg_split / bit_length;
    
    auto input_bit = (input_actual > rand_input).to(input_type);
    auto weight_pos_bit = (weight_pos_actual > rand_weight_pos).to(input_type);
    auto weight_neg_bit = (weight_neg_actual > rand_weight_neg).to(input_type);
    auto result_pos = at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
    auto result_neg = at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
    input_split -= input_bit;
    weight_pos_split -= weight_pos_bit;
    weight_neg_split -= weight_neg_bit;
    bit_length--;
    
    while (bit_length>0) {
        rand_input.uniform_(0,1);
        rand_weight_pos.uniform_(0,1);
        rand_weight_neg.uniform_(0,1);

        input_actual = input_split / bit_length;
        weight_pos_actual = weight_pos_split/ bit_length;
        weight_neg_actual = weight_neg_split/ bit_length;

        input_bit = (input_actual > rand_input).to(input_type);
        weight_pos_bit = (weight_pos_actual > rand_weight_pos).to(input_type);
        weight_neg_bit = (weight_neg_actual > rand_weight_neg).to(input_type);
        result_pos += at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
        result_neg += at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).sign().to(output_type);
//         result_pos += 1;
//         result_neg += 1;

        input_split -= input_bit;
        weight_pos_split -= weight_pos_bit;
        weight_neg_split -= weight_neg_bit;
        bit_length--;
    }
    at::Tensor result_pos_scale = result_pos.to(torch::kFloat32)/orig_bit_length;
    at::Tensor result_neg_scale = result_neg.to(torch::kFloat32)/orig_bit_length;
    at::Tensor result = at::stack({result_pos_scale, result_neg_scale}, 0);
    return result;
}

at::Tensor conv2d_count(torch::Tensor input, torch::Tensor weight, int bit_length, at::IntArrayRef padding, at::IntArrayRef stride) {
    auto input_size = input.sizes();
    auto weight_size = weight.sizes();
    
    at::IntArrayRef dilation = {1};
    int64_t groups=1;

    auto device = input.device();
    auto input_type = torch::kFloat16;
    auto output_type = torch::kFloat16;
    auto comp_type = torch::kInt16;
    auto comp_device = at::device(device).dtype(comp_type);    
    auto range = bit_length-1;
    
    auto input_device = (input*bit_length).to(comp_type);
    auto weight_pos = (weight*bit_length).clamp(0,range).to(comp_type);
    auto weight_neg = -(weight*bit_length).clamp(-range,0).to(comp_type);

    torch::Tensor rand_input = torch::randint(range, input_size, comp_device);
    torch::Tensor rand_weight_pos = torch::randint(range, weight_size, comp_device);
    torch::Tensor rand_weight_neg = torch::randint(range, weight_size, comp_device);   
    
    auto input_bit = (input_device > rand_input).to(input_type);
    auto weight_pos_bit = (weight_pos > rand_weight_pos).to(input_type);
    auto weight_neg_bit = (weight_neg > rand_weight_neg).to(input_type);
    auto result_pos = at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).to(output_type);
    auto result_neg = at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).to(output_type);
    
    for(int j=1; j<bit_length; j++) {
        rand_input.random_(range);
        rand_weight_pos.random_(range);
        rand_weight_neg.random_(range);
        input_bit = (input_device > rand_input).to(input_type);
        weight_pos_bit = (weight_pos > rand_weight_pos).to(input_type);
        weight_neg_bit = (weight_neg > rand_weight_neg).to(input_type);
        result_pos += at::conv2d(input_bit, weight_pos_bit, {}, stride, padding, dilation, groups).to(output_type);
        result_neg += at::conv2d(input_bit, weight_neg_bit, {}, stride, padding, dilation, groups).to(output_type);
    }
    at::Tensor result = result_pos.to(torch::kFloat32)/bit_length - result_neg.to(torch::kFloat32)/bit_length;
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bitstream", &bitstream, "bit stream generator (and back)");
    m.def("linear_or", &linear_or, "SC forward linear or");
    m.def("conv2d_or", &conv2d_or, "SC forward conv2d or");
    m.def("linear_count", &linear_count, "SC forward linear count");
    m.def("conv2d_count", &conv2d_count, "SC forward conv2d count");
    m.def("linear_acc", &linear_or_acc, "SC forward linear or (accurate)");
    m.def("conv2d_acc", &conv2d_or_acc, "SC forward conv2d or (accurate)");

    m.def("linear_xnor", &linear_xnor, "SC forward linear xnor");
    m.def("conv2d_xnor", &conv2d_xnor, "SC forward conv2d xnor");
    m.def("linear_and", &linear_and, "SC forward linear and");
    m.def("conv2d_and", &conv2d_and, "SC forward conv2d and");
    
    m.def("conv2d_add_partial", &conv2d_add_partial, "SC forward conv2d partial bin add");
    m.def("conv2d_add_partial_new", &conv2d_add_partial_new, "SC forward conv2d partial bin add (new version)");
    
    m.def("lfsr_5", &lfsr_5, "lfsr_5");
    m.def("lfsr_7", &lfsr_7, "lfsr_7");
}
