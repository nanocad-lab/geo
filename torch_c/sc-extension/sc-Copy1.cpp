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
    #pragma omp parallel for num_threads(28)
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
//     std::cout << "Weight point" << std::endl;
//     for(int i=0; i<bit_packs*weight_size[0]*weight_size[1]*weight_size[2]*weight_size[3]; i++) {
//         std::cout << (int) weight_pos_point[i] << " " ;
//     }
//     std::cout << std::endl;
    
    #pragma omp parallel for num_threads(28)
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
//     for(int i=0; i<bit_packs * input_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) * weight_size[1] * weight_size[2] * weight_size[3]; i++) {
//         std::cout << input_point_flat[i] << " ";
//     }
//     std::cout << std::endl;
    int32_t* output_point_flat = new int32_t [input_size[0] * weight_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1)];
    int32_t* output_point[input_size[0]];
    int o_batch_step = weight_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int i_batch_step = bit_packs * weight_size[3] * weight_size[1] * weight_size[2] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    for(int batch=0; batch<input_size[0]; batch++) output_point[batch] = output_point_flat + batch*o_batch_step;
    int add_size = bit_packs * weight_size[3];
    int inner_size = weight_size[1] * weight_size[2];
    int cin_size = (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int cout_size = weight_size[0];

    #pragma omp parallel for num_threads(28)
    for(int batch=0; batch<input_size[0]; batch++) {
//         int32_t* output_batch = output_point_flat + batch*o_batch_step;
//         int32_t* input_batch = input_point_flat + batch*i_batch_step;
        int32_t* output_batch = output_point[batch];
        int32_t* input_batch = input_point[batch];
        matmul_outer(input_batch, weight_pos_point, weight_neg_point, output_batch, add_size, inner_size, cin_size, cout_size);
    }

    auto options = torch::TensorOptions().dtype(torch::kInt32);
    at::Tensor output_tensor = torch::from_blob(output_point_flat, {input_size[0], weight_size[0], (input_size[2]-weight_size[2]+1) , (input_size[3]-weight_size[3]+1)}, options);
    delete [] input_point_flat;
    delete [] weight_pos_point;
    delete [] weight_neg_point;
    delete [] output_point_flat;
    return output_tensor;
}
