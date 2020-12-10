#include <ATen/ATen.h>
#include <torch/extension.h>
#include <immintrin.h>
#include <omp.h>

namespace F = torch::nn::functional;

/*
 * Accelerated CPU implementation. Only y-dimension fixed-point accumulation with LFSR generator is supported right now
 */

void matmul_outer(int* input, int* weight_pos, int* weight_neg, int* output, int add_size, int inner_size, int cin_size, int cout_size) {
    int o_cout_step = cin_size;

    int i_pw_step = inner_size * cin_size;
    int i_flatten_step = cin_size;
    
    int w_pw_step = inner_size * cout_size;
    int w_flatten_step = cout_size;
    
    for(int pw=0; pw<add_size; pw++) {
    	// Different pw is added together. This includes different 32-bit sections in the same stream and parts of accumulation done using fixed point
        int32_t* input_pw = input + pw * i_pw_step;
        int32_t* weight_pos_pw = weight_pos + pw * w_pw_step;
        int32_t* weight_neg_pw = weight_neg + pw * w_pw_step;
        int i_out_size = 0;

        // AVX512 implementation
#ifdef __AVX512F__
        // 4-way unroll, 16x4 outputs
        int32_t output_pos_16_0[16];
        int32_t output_pos_16_1[16];
        int32_t output_pos_16_2[16];
        int32_t output_pos_16_3[16];
        int32_t output_neg_16_0[16];
        int32_t output_neg_16_1[16];
        int32_t output_neg_16_2[16];
        int32_t output_neg_16_3[16];
        for (; i_out_size+15<cin_size; i_out_size+=16) {
        	// 512 = 16x32
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

                    weight_neg_16_v = _mm512_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout]);
                    output_v_neg_16_0 = _mm512_or_epi32(output_v_neg_16_0, _mm512_and_epi32(input_16_v, weight_neg_16_v));
                    weight_neg_16_v = _mm512_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout+1]);
                    output_v_neg_16_1 = _mm512_or_epi32(output_v_neg_16_1, _mm512_and_epi32(input_16_v, weight_neg_16_v));
                    weight_neg_16_v = _mm512_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout+2]);
                    output_v_neg_16_2 = _mm512_or_epi32(output_v_neg_16_2, _mm512_and_epi32(input_16_v, weight_neg_16_v));
                    weight_neg_16_v = _mm512_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout+3]);
                    output_v_neg_16_3 = _mm512_or_epi32(output_v_neg_16_3, _mm512_and_epi32(input_16_v, weight_neg_16_v));
                }
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
            }
            //Leftover channels
            for(; w_cout<w_flatten_step; w_cout++) {
                output_v_pos_16_0 = _mm512_set1_epi32(0);
                output_v_neg_16_0 = _mm512_set1_epi32(0);
                for(int flatten_in=0; flatten_in<inner_size; flatten_in++) {
                    input_16_v = _mm512_loadu_si512(input_pw + flatten_in*i_flatten_step + i_out_size);
                    weight_pos_16_v = _mm512_set1_epi32(weight_pos_pw[flatten_in*w_flatten_step + w_cout]);
                    output_v_pos_16_0 = _mm512_or_epi32(output_v_pos_16_0, _mm512_and_epi32(input_16_v, weight_pos_16_v));
                    weight_neg_16_v = _mm512_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout]);
                    output_v_neg_16_0 = _mm512_or_epi32(output_v_neg_16_0, _mm512_and_epi32(input_16_v, weight_neg_16_v));
                }
                _mm512_storeu_si512(output_pos_16_0, output_v_pos_16_0);
                _mm512_storeu_si512(output_neg_16_0, output_v_neg_16_0);
                int o_store_0_offset = i_out_size + o_cout_step*w_cout;
                for (int v_in=0; v_in<16; v_in++) {
                    if (pw==0) {
                        output[o_store_0_offset+v_in]=0;
                    }
                    output[o_store_0_offset+v_in] += __builtin_popcount(output_pos_16_0[v_in]) - __builtin_popcount(output_neg_16_0[v_in]);
                }
            }
        }
#endif
        // AVX2 implementation
#ifdef __AVX2__
        // 4-way unroll, 8x4 outputs
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
                    input_8_v = _mm256_lddqu_si256((__m256i *) (input_pw + flatten_in*i_flatten_step + i_out_size));
                    weight_pos_8_v = _mm256_set1_epi32(weight_pos_pw[flatten_in*w_flatten_step + w_cout]);
                    output_v_pos_8_0 = _mm256_or_si256(output_v_pos_8_0, _mm256_and_si256(input_8_v, weight_pos_8_v));
                    weight_neg_8_v = _mm256_set1_epi32(weight_neg_pw[flatten_in*w_flatten_step + w_cout]);
                    output_v_neg_8_0 = _mm256_or_si256(output_v_neg_8_0, _mm256_and_si256(input_8_v, weight_neg_8_v));
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
        // Leftover outputs + compatibility code
        for (; i_out_size<i_flatten_step; i_out_size++) {
            int w_cout=0;
            int input_v;
            int weight_pos_v;
            int weight_neg_v;

            for(; w_cout<w_flatten_step; w_cout++) {
                int output_pos_c = 0;
                int output_neg_c = 0;
                for(int flatten_in=0; flatten_in<inner_size; flatten_in++) {
                    input_v = input_pw[flatten_in*i_flatten_step + i_out_size];
                    weight_pos_v = weight_pos_pw[flatten_in*w_flatten_step + w_cout];
                    weight_neg_v = weight_neg_pw[flatten_in*w_flatten_step + w_cout];
                    

                    output_pos_c = output_pos_c | (input_v & weight_pos_v);
                    output_neg_c = output_neg_c | (input_v & weight_neg_v);
                }
                if (pw==0) {
                    output[i_out_size+o_cout_step*w_cout] = 0;
                }
                output[i_out_size+o_cout_step*w_cout] += __builtin_popcount(output_pos_c) - __builtin_popcount(output_neg_c);
            }
        }
    }
}

// LFSR generator functions
template <class T>
inline T lfsr_8_s(T value) {
    return ((value/128)+(value/32)%2+(value/16)%2+(value/8)%2)%2+2*(value%128);
}

template <class T>
inline T lfsr_7_s(T value) {
    return ((value/32)%2+value/64)%2+2*(value%64);
}

template <class T>
inline T lfsr_6_s(T value) {
    return ((value/32)+(value/16)%2)%2+2*(value%32);
}

template <class T>
inline T lfsr_5_s(T value) {
    return ((value/16)+(value/4)%2)%2+2*(value%16);
}

template <class T>
inline T lfsr_4_s(T value) {
    return ((value/8)+(value/4)%2)%2+2*(value%8);
}

template <class T>
inline T lfsr_3_s(T value) {
    return ((value/4)+(value/2)%2)%2+2*(value%4);
}

at::Tensor conv2d_add_partial_new(torch::Tensor input, torch::Tensor weight, int bit_length, at::IntArrayRef padding, at::IntArrayRef stride) {
	/*
	 * Conv2d with fixed-point accumulation in y dimension
	 */
	// Input padding and scaling
    auto weight_size = weight.sizes();
    auto input_pad = F::pad(input, F::PadFuncOptions({padding[0], padding[0], padding[1], padding[1]}));
    auto compare_type = torch::kInt32;
    auto store_type = torch::kInt32;
    auto input_split = (input_pad*bit_length).clamp(0, bit_length-1).ceil().to(compare_type);
    auto input_size = input_split.sizes();
    auto w_pos_split = (weight*bit_length).clamp(0, bit_length-1).ceil().to(compare_type);
    auto w_neg_split = (-(weight*bit_length).clamp(1-bit_length, 0)).ceil().to(compare_type);
    int channel_in = weight_size[1];
    int w_weight = weight_size[2];
    int h_weight = weight_size[3];
    int bit_packs = (bit_length+31)/32;
    int (*lfsr)(int);
    // Streams are processed in groups of 32-bits. If stream length is smaller than 32, it will still be processed using 32-bit ints.
    // LFSR length is tied to stream length. Stream length = 2**LFSR length
    int bit_unit = 32;
    switch(bit_length) {
    case 8:
        lfsr=&lfsr_3_s;
        bit_unit=8;
        break;
    case 16:
        lfsr=&lfsr_4_s;
        bit_unit=16;
        break;
    case 32:
        lfsr=&lfsr_5_s;
        break;
    case 64:
        lfsr=&lfsr_6_s;
        break;
    case 128:
        lfsr=&lfsr_7_s;
        break;
    case 256:
        lfsr=&lfsr_8_s;
        break;
    }
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
        pos_seed_arr[i] = (pos_seed + i)%(bit_length-1) + 1;
        neg_seed_arr[i] = (neg_seed + i)%(bit_length-1) + 1;
    }

    // Weight generation + im2col generation on the weight side. omp helps very little here, and overall this part takes little time
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
                    for(int bit=0; bit<bit_unit; bit++) {
                        pos_seed_cur = (*lfsr)(pos_seed_cur);
                        neg_seed_cur = (*lfsr)(neg_seed_cur);
                        for(int c_out=0; c_out<weight_size[0]; c_out++) {
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
                    flatten_in += 1;
                }
            }
        }
    }

    int32_t* output_point_flat = new int32_t [input_size[0] * weight_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1)];
    int32_t* input_bin_point = input_split.data_ptr<int32_t>();
    int o_batch_step = weight_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int i_bin_batch_step = input_size[1] * input_size[2] * input_size[3];
    int add_size = bit_packs * weight_size[3];
    int inner_size = weight_size[1] * weight_size[2];
    int cin_size = (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int cout_size = weight_size[0];

    #pragma omp parallel for
    // By default this only uses half the number of threads in a CPU with hyperthreading/simultaneous multithreading.
    // Specify the max number of threads if you want to take advantage of HT/SMT
    for(int batch=0; batch<input_size[0]; batch++) {
        int32_t* output_batch = output_point_flat + batch*o_batch_step;
        int32_t* input_bin_batch = input_bin_point + batch*i_bin_batch_step;
        // Input stream generation
        // Due to the current limitations, generation and im2col will be separate
        int32_t* input_stream = new int32_t [bit_packs * input_size[1] * input_size[2] * input_size[3]];
        int32_t* input_point = new int32_t [bit_packs * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) * weight_size[1] * weight_size[2] * weight_size[3]];
        int input_seed = 0;
        int* input_seed_arr = new int [input_size[1] * input_size[2] * input_size[3]];
        for(int i=0; i<input_size[1] * input_size[2] * input_size[3]; i++) {
            input_seed_arr[i] = (input_seed + i)%(bit_length-1) + 1;
        }
        for(int pack=0; pack<bit_packs; pack++) {
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
                        for(int bit=0; bit<bit_unit; bit++) {
                            input_seed_cur = (*lfsr)(input_seed_cur);
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
                        int input_w = flatten / (input_size[3]-weight_size[3]+1);
                        int input_h = flatten % (input_size[3]-weight_size[3]+1);
                        input_point_or[flatten] = input_stream_pack[input_cin*input_size[2]*input_size[3] + (input_w+input_w_win)*input_size[3] + (input_h+h_in)];
                    }
                }
            }
            for(int i=0; i<bit_packs * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) * weight_size[1] * weight_size[2] * weight_size[3]; i++) {
            }
        }
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_add_partial_new", &conv2d_add_partial_new, "SC forward conv2d partial bin add (new version)");
}
