#include <torch/extension.h>
//
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

/*
 * Accelerated GPU implementation. Kernel functions
 */

#define BLOCK_INPUT 16
#define BLOCK_WEIGHT 64
#define BATCH_SIZE 128
#define SHARE_SIZE 96

// LFSR functions
__device__ __forceinline__ int d_lfsr_8(int value) {
    return ((value/128)+(value/32)%2+(value/16)%2+(value/8)%2)%2+2*(value%128);
}

__device__ __forceinline__ int d_lfsr_7(int value) {
    return ((value/32)%2+value/64)%2+2*(value%64);
}

__device__ __forceinline__ int d_lfsr_6(int value) {
    return ((value/32)+(value/16)%2)%2+2*(value%32);
}

__device__ __forceinline__ int d_lfsr_5(int value) {
    return ((value/16)+(value/4)%2)%2+2*(value%16);
}

__device__ __forceinline__ int d_lfsr_4(int value) {
    return ((value/8)+(value/4)%2)%2+2*(value%8);
}

__device__ __forceinline__ int d_lfsr_3(int value) {
    return ((value/4)+(value/2)%2)%2+2*(value%4);
}

// Matrix multiplicatiohn with partial binary accumulation
__device__ __forceinline__ void matmul_outer(
        const int32_t* __restrict__ input_point,
        const int32_t* __restrict__ weight_pos_stream,
        const int32_t* __restrict__ weight_neg_stream,
        int32_t* __restrict__ output_stream,
        int index_gen,
        int pws,
        int i_pw_step,
        int w_pw_step,
        int i_flatten_step,
        int w_flatten_step,
        int i_kernel_size,
        int w_kernel_size,
        int inner_size,
        int i_point_batch_offset,
        int o_batch_offset,
        int o_cout_step) {
	/*
	 * Accumulation in pw is performed using fixed-point adders
	 * Computation performed on a grid of i_kernel_size x w_kernel_size
	 */
    for(int pw=0; pw<pws; pw++) {
        int i_point_pw_offset = pw * i_pw_step;
        int w_pw_offset = pw * w_pw_step;
        const int* input_point_pw = input_point + i_point_batch_offset + i_point_pw_offset;
        const int* weight_pos_pw = weight_pos_stream + w_pw_offset;
        const int* weight_neg_pw = weight_neg_stream + w_pw_offset;
        for(int i_out_size=index_gen%i_kernel_size; i_out_size<i_flatten_step; i_out_size+=i_kernel_size){
            for(int w_cout=index_gen/i_kernel_size; w_cout<w_flatten_step; w_cout+=w_kernel_size) {
                int output_pos_c = 0;
                int output_neg_c = 0;
                for(int flatten_in=0; flatten_in<inner_size; flatten_in++) {
                    int input_v = input_point_pw[flatten_in*i_flatten_step + i_out_size];
                    int weight_pos_v = weight_pos_pw[flatten_in*w_flatten_step + w_cout];
                    int weight_neg_v = weight_neg_pw[flatten_in*w_flatten_step + w_cout];

                    output_pos_c |= input_v & weight_pos_v;
                    output_neg_c |= input_v & weight_neg_v;
                }
                output_stream[o_batch_offset+i_out_size+o_cout_step*w_cout] += __popc(output_pos_c) - __popc(output_neg_c);
            }
        }
    }
}

// Weight generation and transpose for y-dimension fixed-point accumulation using LFSR
// Transpose from (c_out, c_in, w, h) to (pack*h, c_in*w, c_out)
// pack = (stream length + 31) / 32
__global__
void stream_generation_transpose_add_variable(
        const int32_t* __restrict__ weight_pos,
        const int32_t* __restrict__ weight_neg,
        int32_t* __restrict__ weight_pos_stream,
        int32_t* __restrict__ weight_neg_stream,
        int* __restrict__ pos_seed_arr,
        int* __restrict__ neg_seed_arr,
        int bit_length,
        int c_outs,
        int c_ins,
        int w_ins,
        int h_ins,
        int total_width,
        int load_width,
        int load_wait) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    int bit_packs = (bit_length+31)/32;
    int (*lfsr)(int);
    int bit_unit = 32;
    switch(bit_length) {
    case 8:
        lfsr=&d_lfsr_3;
        bit_unit=8;
        break;
    case 16:
        lfsr=&d_lfsr_4;
        bit_unit=16;
        break;
    case 32:
        lfsr=&d_lfsr_5;
        break;
    case 64:
        lfsr=&d_lfsr_6;
        break;
    case 128:
        lfsr=&d_lfsr_7;
        break;
    case 256:
        lfsr=&d_lfsr_8;
        break;
    }
    int cur_width;
    int cur_bit;

    for(int i=index_gen; i<c_outs * c_ins * w_ins * h_ins; i+=stride_gen) {
        int h_in=i / (c_outs*c_ins*w_ins);
        int c_in=(i % (c_outs*c_ins*w_ins)) / (c_outs*w_ins);
        int w_in=((i % (c_outs*c_ins*w_ins)) % (c_outs*w_ins)) / c_outs;
        int c_out=i % c_outs;
        int flatten_in = (i % (c_outs*c_ins*w_ins)) / c_outs;
        int pos_seed_cur = pos_seed_arr[flatten_in];
        int neg_seed_cur = neg_seed_arr[flatten_in];
        int weight_pos_bin = weight_pos[c_out*c_ins*w_ins*h_ins + c_in*w_ins*h_ins + w_in*h_ins + h_in];
        int weight_neg_bin = weight_neg[c_out*c_ins*w_ins*h_ins + c_in*w_ins*h_ins + w_in*h_ins + h_in];
        int weight_pos_actual;
        int weight_neg_actual;
        cur_bit=0;
        for(int pack=0; pack<bit_packs; pack++) {
            int weight_pos_stream_c = 0;
            int weight_neg_stream_c = 0;
            int pack_offset = pack * c_outs * c_ins * w_ins * h_ins;
            for(int bit=0; bit<bit_unit; bit++) {
                cur_bit+=1;
                cur_width = (cur_bit/load_wait + 1)*load_width - 1;
                if (cur_width > total_width) cur_width=total_width;
                weight_pos_actual = (weight_pos_bin >> (total_width-cur_width)) << (total_width-cur_width);
                weight_neg_actual = (weight_neg_bin >> (total_width-cur_width)) << (total_width-cur_width);
                pos_seed_cur = (*lfsr)(pos_seed_cur);
                neg_seed_cur = (*lfsr)(neg_seed_cur);
                weight_pos_stream_c = weight_pos_stream_c*2 + (weight_pos_actual>pos_seed_cur);
                weight_neg_stream_c = weight_neg_stream_c*2 + (weight_neg_actual>neg_seed_cur);
            }
            weight_pos_stream[pack_offset+i] = weight_pos_stream_c;
            weight_neg_stream[pack_offset+i] = weight_neg_stream_c;
        }
    }
}

// Weight generation and transpose for y & partial-z-dimension fixed-point accumulation using LFSR
// Transpose from (c_out, c_in, w, h) to (pack*h*(c_in/z_unit), z_unit*w, c_out)
// pack = (stream length + 31) / 32
// z_unit is the number of input channels to accumulate in OR
__global__
void stream_generation_transpose_addyz_variable(
        const int32_t* __restrict__ weight_pos,
        const int32_t* __restrict__ weight_neg,
        int32_t* __restrict__ weight_pos_stream,
        int32_t* __restrict__ weight_neg_stream,
        int* __restrict__ pos_seed_arr,
        int* __restrict__ neg_seed_arr,
        int bit_length,
        int z_units,
        int c_outs,
        int c_ins,
        int w_ins,
        int h_ins,
        int total_width,
        int load_width,
        int load_wait) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    int bit_packs = (bit_length+31)/32;
    int (*lfsr)(int);
    int bit_unit = 32;
    switch(bit_length) {
    case 8:
        lfsr=&d_lfsr_3;
        bit_unit=8;
        break;
    case 16:
        lfsr=&d_lfsr_4;
        bit_unit=16;
        break;
    case 32:
        lfsr=&d_lfsr_5;
        break;
    case 64:
        lfsr=&d_lfsr_6;
        break;
    case 128:
        lfsr=&d_lfsr_7;
        break;
    case 256:
        lfsr=&d_lfsr_8;
        break;
    }
    int cur_width;
    int cur_bit;
    int z_packs = (c_ins+z_units-1)/z_units;

    for(int i=index_gen; i<z_packs*h_ins*z_units*w_ins*c_outs; i+=stride_gen) {
        int z_packs=i / (h_ins*z_units*w_ins*c_outs);
        int h_in=(i % (h_ins*z_units*w_ins*c_outs)) / (z_units*w_ins*c_outs);
        int z_unit=(i % (z_units*w_ins*c_outs)) / (w_ins*c_outs);
        int c_in=z_packs*z_units + z_unit;
        int w_in=(i % (w_ins*c_outs)) / c_outs;
        int c_out=i % c_outs;
        int flatten_in = z_unit*w_ins + w_in;
        int pos_seed_cur = pos_seed_arr[flatten_in];
        int neg_seed_cur = neg_seed_arr[flatten_in];
        int weight_pos_bin;
        int weight_neg_bin;
        if (c_in<c_ins) {
            weight_pos_bin = weight_pos[c_out*c_ins*w_ins*h_ins + c_in*w_ins*h_ins + w_in*h_ins + h_in];
            weight_neg_bin = weight_neg[c_out*c_ins*w_ins*h_ins + c_in*w_ins*h_ins + w_in*h_ins + h_in];
        }
        else{
            weight_pos_bin = 0;
            weight_neg_bin = 0;
        }
        int weight_pos_actual;
        int weight_neg_actual;
        cur_bit=0;
        for(int pack=0; pack<bit_packs; pack++) {
            int weight_pos_stream_c = 0;
            int weight_neg_stream_c = 0;
            int pack_offset = pack * c_outs * c_ins * w_ins * h_ins;
            for(int bit=0; bit<bit_unit; bit++) {
                cur_bit+=1;
                cur_width = (cur_bit/load_wait + 1)*load_width - 1;
                if (cur_width > total_width) cur_width=total_width;
                weight_pos_actual = (weight_pos_bin >> (total_width-cur_width)) << (total_width-cur_width);
                weight_neg_actual = (weight_neg_bin >> (total_width-cur_width)) << (total_width-cur_width);
                pos_seed_cur = (*lfsr)(pos_seed_cur);
                neg_seed_cur = (*lfsr)(neg_seed_cur);
                weight_pos_stream_c = weight_pos_stream_c*2 + (weight_pos_actual>pos_seed_cur);
                weight_neg_stream_c = weight_neg_stream_c*2 + (weight_neg_actual>neg_seed_cur);
            }
            weight_pos_stream[pack_offset+i] = weight_pos_stream_c;
            weight_neg_stream[pack_offset+i] = weight_neg_stream_c;
        }
    }
}

// Weight generation and transpose for y & partial-z-dimension fixed-point accumulation using LFSR
// Transpose from (c_out, c_in, w, h) to (pack*c_in, w*h, c_out)
// pack = (stream length + 31) / 32
__global__
void stream_generation_transpose_addz_variable(
        const int32_t* __restrict__ weight_pos,
        const int32_t* __restrict__ weight_neg,
        int32_t* __restrict__ weight_pos_stream,
        int32_t* __restrict__ weight_neg_stream,
        int* __restrict__ pos_seed_arr,
        int* __restrict__ neg_seed_arr,
        int bit_length,
        int c_outs,
        int c_ins,
        int w_ins,
        int h_ins,
        int total_width,
        int load_width,
        int load_wait) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    int bit_packs = (bit_length+31)/32;
    int (*lfsr)(int);
    int bit_unit = 32;
    switch(bit_length) {
    case 8:
        lfsr=&d_lfsr_3;
        bit_unit=8;
        break;
    case 16:
        lfsr=&d_lfsr_4;
        bit_unit=16;
        break;
    case 32:
        lfsr=&d_lfsr_5;
        break;
    case 64:
        lfsr=&d_lfsr_6;
        break;
    case 128:
        lfsr=&d_lfsr_7;
        break;
    case 256:
        lfsr=&d_lfsr_8;
        break;
    }
    int cur_width;
    int cur_bit;

    for(int i=index_gen; i<c_ins * w_ins * h_ins * c_outs; i+=stride_gen) {
        int c_in=i / (w_ins*h_ins*c_outs);
        int c_out=i % c_outs;
        int flatten_in=(i % (w_ins*h_ins*c_outs)) / c_outs;
        int pos_seed_cur = pos_seed_arr[flatten_in];
        int neg_seed_cur = neg_seed_arr[flatten_in];
        int weight_pos_bin = weight_pos[c_out*c_ins*w_ins*h_ins + c_in*w_ins*h_ins + flatten_in];
        int weight_neg_bin = weight_neg[c_out*c_ins*w_ins*h_ins + c_in*w_ins*h_ins + flatten_in];
        int weight_pos_actual;
        int weight_neg_actual;
        cur_bit=0;
        for(int pack=0; pack<bit_packs; pack++) {
            int weight_pos_stream_c = 0;
            int weight_neg_stream_c = 0;
            int pack_offset = pack * c_outs * c_ins * w_ins * h_ins;
            for(int bit=0; bit<bit_unit; bit++) {
                cur_bit+=1;
                cur_width = (cur_bit/load_wait + 1)*load_width - 1;
                if (cur_width > total_width) cur_width=total_width;
                weight_pos_actual = (weight_pos_bin >> (total_width-cur_width)) << (total_width-cur_width);
                weight_neg_actual = (weight_neg_bin >> (total_width-cur_width)) << (total_width-cur_width);
                pos_seed_cur = (*lfsr)(pos_seed_cur);
                neg_seed_cur = (*lfsr)(neg_seed_cur);
                weight_pos_stream_c = weight_pos_stream_c*2 + (weight_pos_actual>pos_seed_cur);
                weight_neg_stream_c = weight_neg_stream_c*2 + (weight_neg_actual>neg_seed_cur);
            }
            weight_pos_stream[pack_offset+i] = weight_pos_stream_c;
            weight_neg_stream[pack_offset+i] = weight_neg_stream_c;
        }
    }
}

// Weight generation and transpose for y-dimension fixed-point accumulation using accurate random generator
// Transpose from (c_out, c_in, w, h) to (pack*h, c_in*w, c_out)
// pack = (stream length + 31) / 32
__global__
void stream_generation_transpose_add_acc(
        const int32_t* __restrict__ weight_pos,
        const int32_t* __restrict__ weight_neg,
        int32_t* __restrict__ weight_pos_stream,
        int32_t* __restrict__ weight_neg_stream,
        int bit_length,
        int bit_unit,
        int c_outs,
        int c_ins,
        int w_ins,
        int h_ins,
        curandState_t* __restrict__ states_pos,
        curandState_t* __restrict__ states_neg,
        int seed,
        bool share) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;
    share = 0;

    if(share){
        int index_seed = (index_gen % (c_ins*w_ins*c_outs)) / c_outs;
        curand_init(seed, index_seed, 0, states_pos+index_gen);
        curand_init(seed+1, index_seed, 0, states_neg+index_gen);
    }
    else{
        curand_init(seed, index_gen, 0, states_pos+index_gen);
        curand_init(seed+1, index_gen, 0, states_neg+index_gen);
    }


    int bit_packs = (bit_length+31)/32;

    for(int i=index_gen; i<h_ins * c_ins * w_ins * c_outs; i+=stride_gen) {
        int h_in=i / (c_outs*c_ins*w_ins);
        int c_in=(i % (c_outs*c_ins*w_ins)) / (c_outs*w_ins);
        int w_in=((i % (c_outs*c_ins*w_ins)) % (c_outs*w_ins)) / c_outs;
        int c_out=i % c_outs;
        int weight_pos_bin = weight_pos[c_out*c_ins*w_ins*h_ins + c_in*w_ins*h_ins + w_in*h_ins + h_in];
        int weight_neg_bin = weight_neg[c_out*c_ins*w_ins*h_ins + c_in*w_ins*h_ins + w_in*h_ins + h_in];
        int total_bit = bit_length;
        for(int pack=0; pack<bit_packs; pack++) {
            int weight_pos_stream_c = 0;
            int weight_neg_stream_c = 0;
            int pack_offset = pack * c_outs * c_ins * w_ins * h_ins;
            for(int bit=0; bit<bit_unit; bit++) {
                float rand_num_pos = curand_uniform(states_pos+index_gen);
                float rand_num_neg = curand_uniform(states_neg+index_gen);
                float weight_pos_adj = (float)weight_pos_bin / (float)total_bit;
                float weight_neg_adj = (float)weight_neg_bin / (float)total_bit;
                int weight_pos_bit = weight_pos_adj > rand_num_pos;
                int weight_neg_bit = weight_neg_adj > rand_num_neg;

                weight_pos_stream_c = weight_pos_stream_c*2 + weight_pos_bit;
                weight_neg_stream_c = weight_neg_stream_c*2 + weight_neg_bit;
                weight_pos_bin -= weight_pos_bit;
                weight_neg_bin -= weight_neg_bit;
                total_bit -= 1;
            }
            weight_pos_stream[pack_offset+i] = weight_pos_stream_c;
            weight_neg_stream[pack_offset+i] = weight_neg_stream_c;
        }
    }
}

// Weight generation and transpose for full-OR accumuilation using random generator
// Transpose from (c_out, c_in, w, h) to (pack, c_in*w*h, c_out)
// pack = (stream length + 31) / 32
__global__
void stream_generation_transpose(
        const int32_t* __restrict__ weight_pos,
        const int32_t* __restrict__ weight_neg,
        int32_t* __restrict__ weight_pos_stream,
        int32_t* __restrict__ weight_neg_stream,
        int bit_length,
        int bit_unit,
        int c_outs,
        int c_ins,
        int w_ins,
        int h_ins,
        curandState_t* __restrict__ states_pos,
        curandState_t* __restrict__ states_neg,
        int seed) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    curand_init(seed, index_gen, 0, states_pos+index_gen);
    curand_init(seed+1, index_gen, 0, states_neg+index_gen);
    int rand_num_pos;
    int rand_num_neg;

    int bit_packs = (bit_length+31)/32;

    for(int i=index_gen; i<c_outs * c_ins * w_ins * h_ins; i+=stride_gen) {
        int c_in=i / (w_ins*h_ins*c_outs);
        int w_in=(i % (w_ins*h_ins*c_outs)) / (h_ins*c_outs);
        int h_in=((i % (w_ins*h_ins*c_outs)) % (h_ins*c_outs)) / c_outs;
        int c_out=i % c_outs;
        int weight_pos_bin = weight_pos[c_out*c_ins*w_ins*h_ins + c_in*w_ins*h_ins + w_in*h_ins + h_in];
        int weight_neg_bin = weight_neg[c_out*c_ins*w_ins*h_ins + c_in*w_ins*h_ins + w_in*h_ins + h_in];
        for(int pack=0; pack<bit_packs; pack++) {
            int weight_pos_stream_c = 0;
            int weight_neg_stream_c = 0;
            int pack_offset = pack * c_outs * c_ins * w_ins * h_ins;
            for(int bit=0; bit<bit_unit; bit++) {
                rand_num_pos = curand(states_pos+index_gen) % bit_length;
                rand_num_neg = curand(states_neg+index_gen) % bit_length;
                weight_pos_stream_c = weight_pos_stream_c*2 + (weight_pos_bin>rand_num_pos);
                weight_neg_stream_c = weight_neg_stream_c*2 + (weight_neg_bin>rand_num_neg);
            }
            weight_pos_stream[pack_offset+i] = weight_pos_stream_c;
            weight_neg_stream[pack_offset+i] = weight_neg_stream_c;
        }
    }
}

// Activation generation + im2col + computation for accurate random generator + y-dimension fixed-point accumulation
__global__
void stream_compute_add_acc(
        const int32_t* __restrict__ input_bin,
        int32_t* __restrict__ input_stream,
        int32_t* __restrict__ input_point,
        const int32_t* __restrict__ weight_pos_stream,
        const int32_t* __restrict__ weight_neg_stream,
        int32_t* __restrict__ output_stream,
        int bit_length,
        int bit_unit,
        int batches,
        int c_ins,
        int i_w_ins,
        int i_h_ins,
        int c_outs,
        int w_w_ins,
        int w_h_ins,
        int i_kernel_size,
        int w_kernel_size,
        curandState_t* __restrict__ states,
        int seed,
        bool share
        ) {
    int bit_packs = (bit_length+31)/32;
    share = 0;

    const int inner_size = c_ins*w_w_ins;
    const int cin_size = (i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1);
    const int cout_size = c_outs;

    const int o_cout_step = cin_size;

    const int i_pw_step = inner_size * cin_size;
    const int i_flatten_step = cin_size;

    const int w_pw_step = inner_size * cout_size;
    const int w_flatten_step = cout_size;
    const int o_batch_step = c_outs * cin_size;
    const int i_bin_batch_step = c_ins * i_w_ins * i_h_ins;
    const int i_stream_batch_step = bit_packs * c_ins * i_w_ins * i_h_ins;
    const int i_point_batch_step = bit_packs * cin_size * c_ins * w_w_ins * w_h_ins;

    int index_gen = threadIdx.x;
    int stride_gen = blockDim.x;
    int index_batch = blockIdx.x;
    int stride_batch = gridDim.x;
    int index_overall = blockIdx.x*gridDim.x + threadIdx.x;

    if(!share) curand_init(seed, index_gen, 0, states+index_overall);
    for(int batch=index_batch; batch<batches; batch+=stride_batch) {
        int o_batch_offset = batch*o_batch_step;
        int i_bin_batch_offset = batch*i_bin_batch_step;
        int i_stream_batch_offset = index_batch*i_stream_batch_step;
        int i_point_batch_offset = index_batch*i_point_batch_step;
        // Generation
        for(int i=index_gen; i<c_ins * i_w_ins * i_h_ins; i += stride_gen) {
            int input_stream_cur = 0;
            int input_bin_cur = input_bin[i_bin_batch_offset+i];
            int total_bit = bit_length;
            if(share) {
                int index_seed = i/i_h_ins;
                curand_init(seed, index_seed, 0, states+i_bin_batch_offset+i);
            }
            for(int pack=0; pack<bit_packs; pack++) {
                int i_stream_pack_offset = pack*c_ins*i_w_ins*i_h_ins;
                for(int bit=0; bit<bit_unit; bit++) {
                    float input_rand;
                    if(share) input_rand = curand_uniform(states+i_bin_batch_offset+i);
                    else input_rand = curand_uniform(states+index_overall);
                    float input_adj = (float)input_bin_cur / (float)total_bit;
                    int input_bit = input_adj > input_rand;
                    input_stream_cur = (input_stream_cur*2) + input_bit;
                    input_bin_cur -= input_bit;
                    total_bit -= 1;
                }
                input_stream[i_stream_batch_offset+i_stream_pack_offset+i] = input_stream_cur;
            }
        }
        __syncthreads();
        // Im2col
        for(int pack=0; pack<bit_packs; pack++) {
            int i_stream_pack_offset = pack*c_ins*i_w_ins*i_h_ins;
            int i_point_pack_offset = pack * (i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1) * c_ins * w_w_ins * w_h_ins;
            for(int i=index_gen; i<w_h_ins*(i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1)*c_ins*w_w_ins; i+= stride_gen) {
                int h_in = i / ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1) * c_ins * w_w_ins);
                int flatten_in = (i % ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1) * c_ins * w_w_ins)) / ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1));
                int input_cin = flatten_in / w_w_ins;
                int input_w_win = flatten_in % w_w_ins;
                int flatten = i % ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1));
                int input_w = flatten / (i_h_ins-w_h_ins+1);
                int input_h = flatten % (i_h_ins-w_h_ins+1);

                input_point[i_point_batch_offset+i_point_pack_offset+i] = input_stream[i_stream_batch_offset+i_stream_pack_offset + input_cin*i_w_ins*i_h_ins + (input_w+input_w_win)*i_h_ins + (input_h+h_in)];
            }
        }
        __syncthreads();
        // Computation
        matmul_outer(input_point,
                weight_pos_stream,
                weight_neg_stream,
                output_stream,
                index_gen,
                bit_packs*w_h_ins,
                i_pw_step,
                w_pw_step,
                i_flatten_step,
                w_flatten_step,
                i_kernel_size,
                w_kernel_size,
                inner_size,
                i_point_batch_offset,
                o_batch_offset,
                o_cout_step);
    }
}

// Activation generation + im2col + computation for LFSR generator + y-dimension fixed-point accumulation
// This is the only function doing direct convolution instead of im2col method, and achieves ~2X throughput.
// This will be expanded later
__global__
void stream_compute_add_direct_variable(
        const int32_t* __restrict__ input_bin,
        int32_t* __restrict__ input_seed_arr,
        int32_t* __restrict__ input_stream,
        const int32_t* __restrict__ weight_pos_stream,
        const int32_t* __restrict__ weight_neg_stream,
        int32_t* __restrict__ output_stream,
        int32_t* __restrict__ output_pos,
        int32_t* __restrict__ output_neg,
        int bit_length,
        int batches,
        int c_ins,
        int i_w_ins,
        int i_h_ins,
        int c_outs,
        int w_w_ins,
        int w_h_ins,
        int i_kernel_size,
        int w_kernel_size,
        int total_width,
        int load_width,
        int load_wait
        ) {
    int bit_packs = (bit_length+31)/32;
    int (*lfsr)(int);
    int bit_unit = 32;
    switch(bit_length) {
    case 8:
        lfsr=&d_lfsr_3;
        bit_unit = 8;
        break;
    case 16:
        lfsr=&d_lfsr_4;
        bit_unit = 16;
        break;
    case 32:
        lfsr=&d_lfsr_5;
        break;
    case 64:
        lfsr=&d_lfsr_6;
        break;
    case 128:
        lfsr=&d_lfsr_7;
        break;
    case 256:
        lfsr=&d_lfsr_8;
        break;
    }

    const int inner_size = c_ins*w_w_ins;
    const int cin_size = (i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1);
    const int cout_size = c_outs;

    const int o_cout_step = cin_size;

    const int i_flatten_step = cin_size;

    const int w_pw_step = inner_size * cout_size;
    const int w_flatten_step = cout_size;
    const int o_batch_step = c_outs * cin_size;
    const int i_bin_batch_step = c_ins * i_w_ins * i_h_ins;
    const int i_stream_batch_step = bit_packs * c_ins * i_w_ins * i_h_ins;
    const int i_stream_pack_step = c_ins * i_w_ins * i_h_ins;

    int cur_width;
    int cur_bit;

    int index_gen = threadIdx.x;
    int stride_gen = blockDim.x;
    int index_batch = blockIdx.x;
    int stride_batch = gridDim.x;
    for(int batch=index_batch; batch<batches; batch+=stride_batch) {
        int o_batch_offset = batch*o_batch_step;
        int i_bin_batch_offset = batch*i_bin_batch_step;
        int i_stream_batch_offset = index_batch*i_stream_batch_step;
        int* input_stream_batch = input_stream + i_stream_batch_offset;
        int input_seed=0;
        for(int i=index_gen; i<c_ins * i_w_ins * i_h_ins; i += stride_gen) {
            input_seed_arr[i] = (input_seed + i)%(bit_length-1) + 1;
        }
        __syncthreads();
        for(int i=index_gen; i<c_ins * i_w_ins * i_h_ins; i += stride_gen) {
            int input_seed_cur = input_seed_arr[i];
            int input_stream_cur = 0;
            int input_bin_cur = input_bin[i_bin_batch_offset+i];
            int input_bin_actual;
            cur_bit=0;
            for(int pack=0; pack<bit_packs; pack++) {
                int i_stream_pack_offset = pack*c_ins*i_w_ins*i_h_ins;
                int* input_stream_pack = input_stream_batch + i_stream_pack_offset;
                for(int bit=0; bit<bit_unit; bit++) {
                    cur_bit+=1;
                    cur_width = (cur_bit/load_wait + 1)*load_width - 1;
                    if (cur_width > total_width) cur_width=total_width;
                    input_bin_actual = (input_bin_cur >> (total_width-cur_width)) << (total_width-cur_width);
                    input_seed_cur = (*lfsr)(input_seed_cur);
                    input_stream_cur = (input_stream_cur*2) + (input_bin_actual > input_seed_cur);
                }
                input_stream_pack[i] = input_stream_cur;
            }
        }
        __syncthreads();
        // Input stationary
        for(int pw=0; pw<bit_packs*w_h_ins; pw++) {
            int bit_pack = pw / w_h_ins;
            int h_in = pw % w_h_ins;
            int i_stream_pack_offset = bit_pack * i_stream_pack_step;
            int w_pw_offset = pw * w_pw_step;
            int input_dim = index_gen%i_kernel_size;
            int weight_dim = index_gen/i_kernel_size;
            const int* weight_pos_pw = weight_pos_stream + w_pw_offset;
            const int* weight_neg_pw = weight_neg_stream + w_pw_offset;
            int* input_stream_pack = input_stream_batch + i_stream_pack_offset;

            __shared__ int input_s [SHARE_SIZE][BLOCK_INPUT];
            __shared__ int input_s_1 [SHARE_SIZE][BLOCK_INPUT];
            __shared__ int input_s_2 [SHARE_SIZE][BLOCK_INPUT];
            __shared__ int input_s_3 [SHARE_SIZE][BLOCK_INPUT];
            __shared__ int input_s_4 [SHARE_SIZE][BLOCK_INPUT];
            __shared__ int input_s_5 [SHARE_SIZE][BLOCK_INPUT];
            __shared__ int input_s_6 [SHARE_SIZE][BLOCK_INPUT];
            __shared__ int input_s_7 [SHARE_SIZE][BLOCK_INPUT];

            for(int flatten_in_cur=0; flatten_in_cur<inner_size; flatten_in_cur+=SHARE_SIZE) {
                int i_out_size=input_dim;

                // Process eight inputs at the same time
                for(; i_out_size-input_dim+8*i_kernel_size-1<i_flatten_step; i_out_size+=8*i_kernel_size) {
                    int input_w = i_out_size / (i_h_ins-w_h_ins+1);
                    int input_h = i_out_size % (i_h_ins-w_h_ins+1);
                    int input_w_1 = (i_out_size+i_kernel_size) / (i_h_ins-w_h_ins+1);
                    int input_h_1 = (i_out_size+i_kernel_size) % (i_h_ins-w_h_ins+1);
                    int input_w_2 = (i_out_size+i_kernel_size*2) / (i_h_ins-w_h_ins+1);
                    int input_h_2 = (i_out_size+i_kernel_size*2) % (i_h_ins-w_h_ins+1);
                    int input_w_3 = (i_out_size+i_kernel_size*3) / (i_h_ins-w_h_ins+1);
                    int input_h_3 = (i_out_size+i_kernel_size*3) % (i_h_ins-w_h_ins+1);
                    int input_w_4 = (i_out_size+i_kernel_size*4) / (i_h_ins-w_h_ins+1);
                    int input_h_4 = (i_out_size+i_kernel_size*4) % (i_h_ins-w_h_ins+1);
                    int input_w_5 = (i_out_size+i_kernel_size*5) / (i_h_ins-w_h_ins+1);
                    int input_h_5 = (i_out_size+i_kernel_size*5) % (i_h_ins-w_h_ins+1);
                    int input_w_6 = (i_out_size+i_kernel_size*6) / (i_h_ins-w_h_ins+1);
                    int input_h_6 = (i_out_size+i_kernel_size*6) % (i_h_ins-w_h_ins+1);
                    int input_w_7 = (i_out_size+i_kernel_size*7) / (i_h_ins-w_h_ins+1);
                    int input_h_7 = (i_out_size+i_kernel_size*7) % (i_h_ins-w_h_ins+1);
                    for(int i=index_gen; i<SHARE_SIZE*i_kernel_size; i+=stride_gen) {
                        int flatten_in = i / i_kernel_size;
                        int c_in = (flatten_in + flatten_in_cur) / w_w_ins;
                        int w_in = (flatten_in + flatten_in_cur) % w_w_ins;
                        if (flatten_in + flatten_in_cur<inner_size) {
                            input_s[flatten_in][input_dim] = input_stream_pack[c_in*i_w_ins*i_h_ins + (input_w+w_in)*i_h_ins + (input_h+h_in)];
                            input_s_1[flatten_in][input_dim] = input_stream_pack[c_in*i_w_ins*i_h_ins + (input_w_1+w_in)*i_h_ins + (input_h_1+h_in)];
                            input_s_2[flatten_in][input_dim] = input_stream_pack[c_in*i_w_ins*i_h_ins + (input_w_2+w_in)*i_h_ins + (input_h_2+h_in)];
                            input_s_3[flatten_in][input_dim] = input_stream_pack[c_in*i_w_ins*i_h_ins + (input_w_3+w_in)*i_h_ins + (input_h_3+h_in)];
                            input_s_4[flatten_in][input_dim] = input_stream_pack[c_in*i_w_ins*i_h_ins + (input_w_4+w_in)*i_h_ins + (input_h_4+h_in)];
                            input_s_5[flatten_in][input_dim] = input_stream_pack[c_in*i_w_ins*i_h_ins + (input_w_5+w_in)*i_h_ins + (input_h_5+h_in)];
                            input_s_6[flatten_in][input_dim] = input_stream_pack[c_in*i_w_ins*i_h_ins + (input_w_6+w_in)*i_h_ins + (input_h_6+h_in)];
                            input_s_7[flatten_in][input_dim] = input_stream_pack[c_in*i_w_ins*i_h_ins + (input_w_7+w_in)*i_h_ins + (input_h_7+h_in)];
                        }
                    }
                    __syncthreads();
                    for(int w_cout=weight_dim; w_cout<w_flatten_step; w_cout+=w_kernel_size) {
                        int output_pos_c = 0;
                        int output_neg_c = 0;
                        int output_pos_c_1 = 0;
                        int output_neg_c_1 = 0;
                        int output_pos_c_2 = 0;
                        int output_neg_c_2 = 0;
                        int output_pos_c_3 = 0;
                        int output_neg_c_3 = 0;
                        int output_pos_c_4 = 0;
                        int output_neg_c_4 = 0;
                        int output_pos_c_5 = 0;
                        int output_neg_c_5 = 0;
                        int output_pos_c_6 = 0;
                        int output_neg_c_6 = 0;
                        int output_pos_c_7 = 0;
                        int output_neg_c_7 = 0;
                        if(flatten_in_cur==0) {
                            output_pos[o_batch_offset+i_out_size+o_cout_step*w_cout] = 0;
                            output_neg[o_batch_offset+i_out_size+o_cout_step*w_cout] = 0;
                            output_pos[o_batch_offset+i_out_size+i_kernel_size+o_cout_step*w_cout] = 0;
                            output_neg[o_batch_offset+i_out_size+i_kernel_size+o_cout_step*w_cout] = 0;
                            output_pos[o_batch_offset+i_out_size+i_kernel_size*2+o_cout_step*w_cout] = 0;
                            output_neg[o_batch_offset+i_out_size+i_kernel_size*2+o_cout_step*w_cout] = 0;
                            output_pos[o_batch_offset+i_out_size+i_kernel_size*3+o_cout_step*w_cout] = 0;
                            output_neg[o_batch_offset+i_out_size+i_kernel_size*3+o_cout_step*w_cout] = 0;
                            output_pos[o_batch_offset+i_out_size+i_kernel_size*4+o_cout_step*w_cout] = 0;
                            output_neg[o_batch_offset+i_out_size+i_kernel_size*4+o_cout_step*w_cout] = 0;
                            output_pos[o_batch_offset+i_out_size+i_kernel_size*5+o_cout_step*w_cout] = 0;
                            output_neg[o_batch_offset+i_out_size+i_kernel_size*5+o_cout_step*w_cout] = 0;
                            output_pos[o_batch_offset+i_out_size+i_kernel_size*6+o_cout_step*w_cout] = 0;
                            output_neg[o_batch_offset+i_out_size+i_kernel_size*6+o_cout_step*w_cout] = 0;
                            output_pos[o_batch_offset+i_out_size+i_kernel_size*7+o_cout_step*w_cout] = 0;
                            output_neg[o_batch_offset+i_out_size+i_kernel_size*7+o_cout_step*w_cout] = 0;
                        }
                        for(int flatten_in=0; flatten_in<SHARE_SIZE; flatten_in++) {
                            if (flatten_in + flatten_in_cur<inner_size) {
                                int input_v = input_s[flatten_in][input_dim];
                                int weight_pos_v = weight_pos_pw[(flatten_in+flatten_in_cur)*w_flatten_step + w_cout];
                                int weight_neg_v = weight_neg_pw[(flatten_in+flatten_in_cur)*w_flatten_step + w_cout];
                                output_pos_c |= (input_v & weight_pos_v);
                                output_neg_c |= (input_v & weight_neg_v);

                                int input_v_1 = input_s_1[flatten_in][input_dim];
                                output_pos_c_1 |= (input_v_1 & weight_pos_v);
                                output_neg_c_1 |= (input_v_1 & weight_neg_v);

                                int input_v_2 = input_s_2[flatten_in][input_dim];
                                output_pos_c_2 |= (input_v_2 & weight_pos_v);
                                output_neg_c_2 |= (input_v_2 & weight_neg_v);

                                int input_v_3 = input_s_3[flatten_in][input_dim];
                                output_pos_c_3 |= (input_v_3 & weight_pos_v);
                                output_neg_c_3 |= (input_v_3 & weight_neg_v);

                                int input_v_4 = input_s_4[flatten_in][input_dim];
                                output_pos_c_4 |= (input_v_4 & weight_pos_v);
                                output_neg_c_4 |= (input_v_4 & weight_neg_v);

                                int input_v_5 = input_s_5[flatten_in][input_dim];
                                output_pos_c_5 |= (input_v_5 & weight_pos_v);
                                output_neg_c_5 |= (input_v_5 & weight_neg_v);

                                int input_v_6 = input_s_6[flatten_in][input_dim];
                                output_pos_c_6 |= (input_v_6 & weight_pos_v);
                                output_neg_c_6 |= (input_v_6 & weight_neg_v);

                                int input_v_7 = input_s_7[flatten_in][input_dim];
                                output_pos_c_7 |= (input_v_7 & weight_pos_v);
                                output_neg_c_7 |= (input_v_7 & weight_neg_v);
                            }
                        }
                        output_pos[o_batch_offset+i_out_size+o_cout_step*w_cout] |= output_pos_c;
                        output_neg[o_batch_offset+i_out_size+o_cout_step*w_cout] |= output_neg_c;
                        output_pos[o_batch_offset+i_out_size+i_kernel_size+o_cout_step*w_cout] |= output_pos_c_1;
                        output_neg[o_batch_offset+i_out_size+i_kernel_size+o_cout_step*w_cout] |= output_neg_c_1;
                        output_pos[o_batch_offset+i_out_size+i_kernel_size*2+o_cout_step*w_cout] |= output_pos_c_2;
                        output_neg[o_batch_offset+i_out_size+i_kernel_size*2+o_cout_step*w_cout] |= output_neg_c_2;
                        output_pos[o_batch_offset+i_out_size+i_kernel_size*3+o_cout_step*w_cout] |= output_pos_c_3;
                        output_neg[o_batch_offset+i_out_size+i_kernel_size*3+o_cout_step*w_cout] |= output_neg_c_3;
                        output_pos[o_batch_offset+i_out_size+i_kernel_size*4+o_cout_step*w_cout] |= output_pos_c_4;
                        output_neg[o_batch_offset+i_out_size+i_kernel_size*4+o_cout_step*w_cout] |= output_neg_c_4;
                        output_pos[o_batch_offset+i_out_size+i_kernel_size*5+o_cout_step*w_cout] |= output_pos_c_5;
                        output_neg[o_batch_offset+i_out_size+i_kernel_size*5+o_cout_step*w_cout] |= output_neg_c_5;
                        output_pos[o_batch_offset+i_out_size+i_kernel_size*6+o_cout_step*w_cout] |= output_pos_c_6;
                        output_neg[o_batch_offset+i_out_size+i_kernel_size*6+o_cout_step*w_cout] |= output_neg_c_6;
                        output_pos[o_batch_offset+i_out_size+i_kernel_size*7+o_cout_step*w_cout] |= output_pos_c_7;
                        output_neg[o_batch_offset+i_out_size+i_kernel_size*7+o_cout_step*w_cout] |= output_neg_c_7;
                    }
                    __syncthreads();
                }
                // Process four inputs at the same time
                for(; i_out_size-input_dim+4*i_kernel_size-1<i_flatten_step; i_out_size+=4*i_kernel_size) {
                    int input_w = i_out_size / (i_h_ins-w_h_ins+1);
                    int input_h = i_out_size % (i_h_ins-w_h_ins+1);
                    int input_w_1 = (i_out_size+i_kernel_size) / (i_h_ins-w_h_ins+1);
                    int input_h_1 = (i_out_size+i_kernel_size) % (i_h_ins-w_h_ins+1);
                    int input_w_2 = (i_out_size+i_kernel_size*2) / (i_h_ins-w_h_ins+1);
                    int input_h_2 = (i_out_size+i_kernel_size*2) % (i_h_ins-w_h_ins+1);
                    int input_w_3 = (i_out_size+i_kernel_size*3) / (i_h_ins-w_h_ins+1);
                    int input_h_3 = (i_out_size+i_kernel_size*3) % (i_h_ins-w_h_ins+1);
                    for(int i=index_gen; i<SHARE_SIZE*i_kernel_size; i+=stride_gen) {
                        int flatten_in = i / i_kernel_size;
                        int c_in = (flatten_in + flatten_in_cur) / w_w_ins;
                        int w_in = (flatten_in + flatten_in_cur) % w_w_ins;
                        if (flatten_in + flatten_in_cur<inner_size) {
                            input_s[flatten_in][input_dim] = input_stream_pack[c_in*i_w_ins*i_h_ins + (input_w+w_in)*i_h_ins + (input_h+h_in)];
                            input_s_1[flatten_in][input_dim] = input_stream_pack[c_in*i_w_ins*i_h_ins + (input_w_1+w_in)*i_h_ins + (input_h_1+h_in)];
                            input_s_2[flatten_in][input_dim] = input_stream_pack[c_in*i_w_ins*i_h_ins + (input_w_2+w_in)*i_h_ins + (input_h_2+h_in)];
                            input_s_3[flatten_in][input_dim] = input_stream_pack[c_in*i_w_ins*i_h_ins + (input_w_3+w_in)*i_h_ins + (input_h_3+h_in)];
                        }
                    }
                    __syncthreads();
                    for(int w_cout=weight_dim; w_cout<w_flatten_step; w_cout+=w_kernel_size) {
                        int output_pos_c = 0;
                        int output_neg_c = 0;
                        int output_pos_c_1 = 0;
                        int output_neg_c_1 = 0;
                        int output_pos_c_2 = 0;
                        int output_neg_c_2 = 0;
                        int output_pos_c_3 = 0;
                        int output_neg_c_3 = 0;
                        if(flatten_in_cur==0) {
                            output_pos[o_batch_offset+i_out_size+o_cout_step*w_cout] = 0;
                            output_neg[o_batch_offset+i_out_size+o_cout_step*w_cout] = 0;
                            output_pos[o_batch_offset+i_out_size+i_kernel_size+o_cout_step*w_cout] = 0;
                            output_neg[o_batch_offset+i_out_size+i_kernel_size+o_cout_step*w_cout] = 0;
                            output_pos[o_batch_offset+i_out_size+i_kernel_size*2+o_cout_step*w_cout] = 0;
                            output_neg[o_batch_offset+i_out_size+i_kernel_size*2+o_cout_step*w_cout] = 0;
                            output_pos[o_batch_offset+i_out_size+i_kernel_size*3+o_cout_step*w_cout] = 0;
                            output_neg[o_batch_offset+i_out_size+i_kernel_size*3+o_cout_step*w_cout] = 0;
                        }
                        for(int flatten_in=0; flatten_in<SHARE_SIZE; flatten_in++) {
                            if (flatten_in + flatten_in_cur<inner_size) {
                                int input_v = input_s[flatten_in][input_dim];
                                int weight_pos_v = weight_pos_pw[(flatten_in+flatten_in_cur)*w_flatten_step + w_cout];
                                int weight_neg_v = weight_neg_pw[(flatten_in+flatten_in_cur)*w_flatten_step + w_cout];
                                output_pos_c |= (input_v & weight_pos_v);
                                output_neg_c |= (input_v & weight_neg_v);

                                int input_v_1 = input_s_1[flatten_in][input_dim];
                                output_pos_c_1 |= (input_v_1 & weight_pos_v);
                                output_neg_c_1 |= (input_v_1 & weight_neg_v);

                                int input_v_2 = input_s_2[flatten_in][input_dim];
                                output_pos_c_2 |= (input_v_2 & weight_pos_v);
                                output_neg_c_2 |= (input_v_2 & weight_neg_v);

                                int input_v_3 = input_s_3[flatten_in][input_dim];
                                output_pos_c_3 |= (input_v_3 & weight_pos_v);
                                output_neg_c_3 |= (input_v_3 & weight_neg_v);

                            }
                        }
                        output_pos[o_batch_offset+i_out_size+o_cout_step*w_cout] |= output_pos_c;
                        output_neg[o_batch_offset+i_out_size+o_cout_step*w_cout] |= output_neg_c;
                        output_pos[o_batch_offset+i_out_size+i_kernel_size+o_cout_step*w_cout] |= output_pos_c_1;
                        output_neg[o_batch_offset+i_out_size+i_kernel_size+o_cout_step*w_cout] |= output_neg_c_1;
                        output_pos[o_batch_offset+i_out_size+i_kernel_size*2+o_cout_step*w_cout] |= output_pos_c_2;
                        output_neg[o_batch_offset+i_out_size+i_kernel_size*2+o_cout_step*w_cout] |= output_neg_c_2;
                        output_pos[o_batch_offset+i_out_size+i_kernel_size*3+o_cout_step*w_cout] |= output_pos_c_3;
                        output_neg[o_batch_offset+i_out_size+i_kernel_size*3+o_cout_step*w_cout] |= output_neg_c_3;
                    }
                    __syncthreads();
                }
                // Process two inputs at the same time
                for(; i_out_size-input_dim+2*i_kernel_size-1<i_flatten_step; i_out_size+=2*i_kernel_size) {
                    int input_w = i_out_size / (i_h_ins-w_h_ins+1);
                    int input_h = i_out_size % (i_h_ins-w_h_ins+1);
                    int input_w_1 = (i_out_size+i_kernel_size) / (i_h_ins-w_h_ins+1);
                    int input_h_1 = (i_out_size+i_kernel_size) % (i_h_ins-w_h_ins+1);
                    for(int i=index_gen; i<SHARE_SIZE*i_kernel_size; i+=stride_gen) {
                        int flatten_in = i / i_kernel_size;
                        int c_in = (flatten_in + flatten_in_cur) / w_w_ins;
                        int w_in = (flatten_in + flatten_in_cur) % w_w_ins;
                        if (flatten_in + flatten_in_cur<inner_size) {
                            input_s[flatten_in][input_dim] = input_stream_pack[c_in*i_w_ins*i_h_ins + (input_w+w_in)*i_h_ins + (input_h+h_in)];
                            input_s_1[flatten_in][input_dim] = input_stream_pack[c_in*i_w_ins*i_h_ins + (input_w_1+w_in)*i_h_ins + (input_h_1+h_in)];
                        }
                    }
                    __syncthreads();
                    for(int w_cout=weight_dim; w_cout<w_flatten_step; w_cout+=w_kernel_size) {
                        int output_pos_c = 0;
                        int output_neg_c = 0;
                        int output_pos_c_1 = 0;
                        int output_neg_c_1 = 0;
                        if(flatten_in_cur==0) {
                            output_pos[o_batch_offset+i_out_size+o_cout_step*w_cout] = 0;
                            output_neg[o_batch_offset+i_out_size+o_cout_step*w_cout] = 0;
                            output_pos[o_batch_offset+i_out_size+i_kernel_size+o_cout_step*w_cout] = 0;
                            output_neg[o_batch_offset+i_out_size+i_kernel_size+o_cout_step*w_cout] = 0;
                        }
                        for(int flatten_in=0; flatten_in<SHARE_SIZE; flatten_in++) {
                            if (flatten_in + flatten_in_cur<inner_size) {
                                int input_v = input_s[flatten_in][input_dim];
                                int weight_pos_v = weight_pos_pw[(flatten_in+flatten_in_cur)*w_flatten_step + w_cout];
                                int weight_neg_v = weight_neg_pw[(flatten_in+flatten_in_cur)*w_flatten_step + w_cout];
                                output_pos_c |= (input_v & weight_pos_v);
                                output_neg_c |= (input_v & weight_neg_v);

                                int input_v_1 = input_s_1[flatten_in][input_dim];
                                output_pos_c_1 |= (input_v_1 & weight_pos_v);
                                output_neg_c_1 |= (input_v_1 & weight_neg_v);
                            }
                        }
                        output_pos[o_batch_offset+i_out_size+o_cout_step*w_cout] |= output_pos_c;
                        output_neg[o_batch_offset+i_out_size+o_cout_step*w_cout] |= output_neg_c;
                        output_pos[o_batch_offset+i_out_size+i_kernel_size+o_cout_step*w_cout] |= output_pos_c_1;
                        output_neg[o_batch_offset+i_out_size+i_kernel_size+o_cout_step*w_cout] |= output_neg_c_1;
                    }
                    __syncthreads();
                }
                // Process the remaining input grid
                for(; i_out_size-input_dim<i_flatten_step; i_out_size+=i_kernel_size){
                    int input_w = i_out_size / (i_h_ins-w_h_ins+1);
                    int input_h = i_out_size % (i_h_ins-w_h_ins+1);
                    for(int i=index_gen; i<SHARE_SIZE*i_kernel_size; i+=stride_gen) {
                        int flatten_in = i / i_kernel_size;
                        int c_in = (flatten_in + flatten_in_cur) / w_w_ins;
                        int w_in = (flatten_in + flatten_in_cur) % w_w_ins;
                        if ((i_out_size<i_flatten_step) && (flatten_in+flatten_in_cur<inner_size)) input_s[flatten_in][input_dim] = input_stream_pack[c_in*i_w_ins*i_h_ins + (input_w+w_in)*i_h_ins + (input_h+h_in)];
                    }
                    __syncthreads();
                    if (i_out_size<i_flatten_step) {
                        for(int w_cout=weight_dim; w_cout<w_flatten_step; w_cout+=w_kernel_size) {
                            int output_pos_c = 0;
                            int output_neg_c = 0;
                            if(flatten_in_cur==0) {
                                output_pos[o_batch_offset+i_out_size+o_cout_step*w_cout] = 0;
                                output_neg[o_batch_offset+i_out_size+o_cout_step*w_cout] = 0;
                            }
                            for(int flatten_in=0; flatten_in<SHARE_SIZE; flatten_in++) {
                                if (flatten_in + flatten_in_cur < inner_size) {
                                    int input_v = input_s[flatten_in][input_dim];
                                    int weight_pos_v = weight_pos_pw[(flatten_in+flatten_in_cur)*w_flatten_step + w_cout];
                                    int weight_neg_v = weight_neg_pw[(flatten_in+flatten_in_cur)*w_flatten_step + w_cout];

                                    output_pos_c |= (input_v & weight_pos_v);
                                    output_neg_c |= (input_v & weight_neg_v);
                                }
                            }
                            output_pos[o_batch_offset+i_out_size+o_cout_step*w_cout] |= output_pos_c;
                            output_neg[o_batch_offset+i_out_size+o_cout_step*w_cout] |= output_neg_c;
                        }
                    }
                    __syncthreads();
                }
            }
            for(int i=index_gen; i<o_batch_step; i+=stride_gen) {
                output_stream[o_batch_offset+i] += __popc(output_pos[o_batch_offset+i]) - __popc(output_neg[o_batch_offset+i]);
            }
        }
    }
}

// Activation generation + im2col + computation for LFSR generator + y + partial-z-dimension fixed-point accumulation
__global__
void stream_compute_addyz_variable(
        const int32_t* __restrict__ input_bin,
        int32_t* __restrict__ input_seed_arr,
        int32_t* __restrict__ input_stream,
        int32_t* __restrict__ input_point,
        const int32_t* __restrict__ weight_pos_stream,
        const int32_t* __restrict__ weight_neg_stream,
        int32_t* __restrict__ output_stream,
        int bit_length,
        int z_units,
        int batches,
        int c_ins,
        int i_w_ins,
        int i_h_ins,
        int c_outs,
        int w_w_ins,
        int w_h_ins,
        int i_kernel_size,
        int w_kernel_size,
        int total_width,
        int load_width,
        int load_wait
        ) {
    int bit_packs = (bit_length+31)/32;
    int (*lfsr)(int);
    int bit_unit = 32;
    switch(bit_length) {
    case 8:
        lfsr=&d_lfsr_3;
        bit_unit = 8;
        break;
    case 16:
        lfsr=&d_lfsr_4;
        bit_unit = 16;
        break;
    case 32:
        lfsr=&d_lfsr_5;
        break;
    case 64:
        lfsr=&d_lfsr_6;
        break;
    case 128:
        lfsr=&d_lfsr_7;
        break;
    case 256:
        lfsr=&d_lfsr_8;
        break;
    }

    const int z_packs = (c_ins+z_units-1) / z_units;
    const int inner_size = z_units*w_w_ins;
    const int cin_size = (i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1);

    const int i_pw_step = inner_size * cin_size;
    const int i_flatten_step = cin_size;

    const int w_pw_step = inner_size * c_outs;
    const int o_batch_step = c_outs * cin_size;
    const int i_bin_batch_step = c_ins * i_w_ins * i_h_ins;
    const int i_stream_batch_step = bit_packs * z_packs * z_units * i_w_ins * i_h_ins;
    const int i_point_batch_step = bit_packs * z_packs * w_h_ins * cin_size * z_units * w_w_ins;

    volatile int index_gen = threadIdx.x;
    volatile int stride_gen = blockDim.x;
    volatile int index_batch = blockIdx.x;
    volatile int stride_batch = gridDim.x;
    for(int batch=index_batch; batch<batches; batch+=stride_batch) {
        int o_batch_offset = batch*o_batch_step;
        int i_bin_batch_offset = batch*i_bin_batch_step;
        int i_stream_batch_offset = index_batch*i_stream_batch_step;
        int i_point_batch_offset = index_batch*i_point_batch_step;
        for(int i=index_gen; i<z_packs * z_units * i_w_ins * i_h_ins; i += stride_gen) {
            int input_seed_cur = input_seed_arr[i % (z_units*i_w_ins*i_h_ins)];
            int z_pack = i / (z_units*i_w_ins*i_h_ins);
            int z_unit = (i % (z_units*i_w_ins*i_h_ins)) / (i_w_ins*i_h_ins);
            int c_in = z_pack*z_units + z_unit;
            int flatten = i%(i_w_ins * i_h_ins);
            int input_stream_cur = 0;
            int input_bin_cur = 0;
            if (c_in < c_ins) input_bin_cur = input_bin[i_bin_batch_offset+c_in*i_w_ins*i_h_ins+flatten];
            int input_bin_actual;
            int cur_bit=0;
            for(int pack=0; pack<bit_packs; pack++) {
                int i_stream_pack_offset = pack*z_packs*z_units*i_w_ins*i_h_ins;
                for(int bit=0; bit<bit_unit; bit++) {
                    cur_bit+=1;
                    int cur_width = (cur_bit/load_wait + 1)*load_width - 1;
                    if (cur_width > total_width) cur_width=total_width;
                    input_bin_actual = (input_bin_cur >> (total_width-cur_width)) << (total_width-cur_width);
                    input_seed_cur = (*lfsr)(input_seed_cur);
                    input_stream_cur = (input_stream_cur*2) + (input_bin_actual > input_seed_cur);
                }
                input_stream[i_stream_batch_offset+i_stream_pack_offset+i] = input_stream_cur;
            }
        }
        __syncthreads();
        for(int pack=0; pack<bit_packs; pack++) {
            int i_stream_pack_offset = pack*z_packs*z_units*i_w_ins*i_h_ins;
            int i_point_pack_offset = pack * z_packs * w_h_ins * z_units * w_w_ins * (i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1);
            for(int i=index_gen; i<z_packs * w_h_ins * z_units * w_w_ins * (i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1); i+= stride_gen) {
                int z_pack = i / (w_h_ins*z_units*w_w_ins*cin_size);
                int h_in = (i%(w_h_ins*z_units*w_w_ins*cin_size)) / (z_units * w_w_ins * cin_size);
                int z_unit = (i%(z_units*w_w_ins*cin_size)) / (w_w_ins*cin_size);
                int c_in = z_pack*z_units + z_unit;
                int w_in = (i%(w_w_ins * cin_size)) / cin_size;
                int flatten = i % cin_size;
                int input_w = flatten / (i_h_ins-w_h_ins+1);
                int input_h = flatten % (i_h_ins-w_h_ins+1);

                input_point[i_point_batch_offset+i_point_pack_offset+i] = input_stream[i_stream_batch_offset+i_stream_pack_offset + c_in*i_w_ins*i_h_ins + (input_w+w_in)*i_h_ins + (input_h+h_in)];
            }
        }
        __syncthreads();
        matmul_outer(input_point,
                weight_pos_stream,
                weight_neg_stream,
                output_stream,
                index_gen,
                bit_packs*z_packs*w_h_ins,
                i_pw_step,
                w_pw_step,
                i_flatten_step,
                c_outs,
                i_kernel_size,
                w_kernel_size,
                inner_size,
                i_point_batch_offset,
                o_batch_offset,
                cin_size);
    }
}

// Activation generation + im2col + computation for LFSR generator + z-dimension fixed-point accumulation
__global__
void stream_compute_addz_variable(
        const int32_t* __restrict__ input_bin,
        int32_t* __restrict__ input_seed_arr,
        int32_t* __restrict__ input_stream,
        int32_t* __restrict__ input_point,
        const int32_t* __restrict__ weight_pos_stream,
        const int32_t* __restrict__ weight_neg_stream,
        int32_t* __restrict__ output_stream,
        int bit_length,
        int batches,
        int c_ins,
        int i_w_ins,
        int i_h_ins,
        int c_outs,
        int w_w_ins,
        int w_h_ins,
        int i_kernel_size,
        int w_kernel_size,
        int total_width,
        int load_width,
        int load_wait
        ) {
    int bit_packs = (bit_length+31)/32;
    int (*lfsr)(int);
    int bit_unit = 32;
    switch(bit_length) {
    case 8:
        lfsr=&d_lfsr_3;
        bit_unit = 8;
        break;
    case 16:
        lfsr=&d_lfsr_4;
        bit_unit = 16;
        break;
    case 32:
        lfsr=&d_lfsr_5;
        break;
    case 64:
        lfsr=&d_lfsr_6;
        break;
    case 128:
        lfsr=&d_lfsr_7;
        break;
    case 256:
        lfsr=&d_lfsr_8;
        break;
    }

    const int inner_size = w_w_ins*w_h_ins;
    const int cin_size = (i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1);
    const int cout_size = c_outs;

    const int o_cout_step = cin_size;

    const int i_pw_step = inner_size * cin_size;
    const int i_flatten_step = cin_size;

    const int w_pw_step = inner_size * cout_size;
    const int w_flatten_step = cout_size;
    const int o_batch_step = c_outs * cin_size;
    const int i_bin_batch_step = c_ins * i_w_ins * i_h_ins;
    const int i_stream_batch_step = bit_packs * c_ins * i_w_ins * i_h_ins;
    const int i_point_batch_step = bit_packs * cin_size * c_ins * w_w_ins * w_h_ins;

    int cur_width;
    int cur_bit;

    int index_gen = threadIdx.x;
    int stride_gen = blockDim.x;
    int index_batch = blockIdx.x;
    int stride_batch = gridDim.x;
    for(int batch=index_batch; batch<batches; batch+=stride_batch) {
        int o_batch_offset = batch*o_batch_step;
        int i_bin_batch_offset = batch*i_bin_batch_step;
        int i_stream_batch_offset = index_batch*i_stream_batch_step;
        int i_point_batch_offset = index_batch*i_point_batch_step;
        for(int i=index_gen; i<c_ins * i_w_ins * i_h_ins; i += stride_gen) {
            int input_seed_cur = input_seed_arr[i];
            int input_stream_cur = 0;
            int input_bin_cur = input_bin[i_bin_batch_offset+i];
            int input_bin_actual;
            cur_bit=0;
            for(int pack=0; pack<bit_packs; pack++) {
                int i_stream_pack_offset = pack*c_ins*i_w_ins*i_h_ins;
                for(int bit=0; bit<bit_unit; bit++) {
                    cur_bit+=1;
                    cur_width = (cur_bit/load_wait + 1)*load_width - 1;
                    if (cur_width > total_width) cur_width=total_width;
                    input_bin_actual = (input_bin_cur >> (total_width-cur_width)) << (total_width-cur_width);
                    input_seed_cur = (*lfsr)(input_seed_cur);
                    input_stream_cur = (input_stream_cur*2) + (input_bin_actual > input_seed_cur);
                }
                input_stream[i_stream_batch_offset+i_stream_pack_offset+i] = input_stream_cur;
            }
        }
        __syncthreads();
        for(int pack=0; pack<bit_packs; pack++) {
            int i_stream_pack_offset = pack*c_ins*i_w_ins*i_h_ins;
            int i_point_pack_offset = pack * (i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1) * c_ins * w_w_ins * w_h_ins;
            for(int i=index_gen; i<c_ins*w_w_ins*w_h_ins*(i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1); i+= stride_gen) {
                int c_in = i / ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1) * w_w_ins * w_h_ins);
                int flatten_in = (i % ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1) * w_w_ins * w_h_ins)) / ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1));
                int w_in = flatten_in / w_h_ins;
                int h_in = flatten_in % w_h_ins;
                int flatten = i % ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1));
                int input_w = flatten / (i_h_ins-w_h_ins+1);
                int input_h = flatten % (i_h_ins-w_h_ins+1);
                input_point[i_point_batch_offset+i_point_pack_offset+i] = input_stream[i_stream_batch_offset+i_stream_pack_offset + c_in*i_w_ins*i_h_ins + (input_w+w_in)*i_h_ins + (input_h+h_in)];
            }
        }
        __syncthreads();

        matmul_outer(input_point,
                weight_pos_stream,
                weight_neg_stream,
                output_stream,
                index_gen,
                bit_packs*c_ins,
                i_pw_step,
                w_pw_step,
                i_flatten_step,
                w_flatten_step,
                i_kernel_size,
                w_kernel_size,
                inner_size,
                i_point_batch_offset,
                o_batch_offset,
                o_cout_step);
    }
}

// Activation generation + im2col + computation for random generator + full or accumulation
__global__
void stream_compute_or(
        const int32_t* __restrict__ input_bin,
        int32_t* __restrict__ input_stream,
        int32_t* __restrict__ input_point,
        const int32_t* __restrict__ weight_pos_stream,
        const int32_t* __restrict__ weight_neg_stream,
        int32_t* __restrict__ output_stream,
        int bit_length,
        int bit_unit,
        int batches,
        int c_ins,
        int i_w_ins,
        int i_h_ins,
        int c_outs,
        int w_w_ins,
        int w_h_ins,
        int i_kernel_size,
        int w_kernel_size,
        curandState_t* __restrict__ states,
        int seed
        ) {
    int bit_packs = (bit_length+31)/32;

    const int inner_size = c_ins*w_w_ins*w_h_ins;
    const int cin_size = (i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1);

    const int i_pw_step = inner_size * cin_size;

    const int w_pw_step = inner_size * c_outs;
    const int o_batch_step = c_outs * cin_size;
    const int i_bin_batch_step = c_ins * i_w_ins * i_h_ins;
    const int i_stream_batch_step = bit_packs * c_ins * i_w_ins * i_h_ins;
    const int i_point_batch_step = bit_packs * cin_size * c_ins * w_w_ins * w_h_ins;

    int input_rand;

    int index_gen = threadIdx.x;
    int stride_gen = blockDim.x;
    int index_batch = blockIdx.x;
    int stride_batch = gridDim.x;
    int index_overall = blockIdx.x*gridDim.x + threadIdx.x;

    curand_init(seed, index_gen, 0, states+index_overall);
    for(int batch=index_batch; batch<batches; batch+=stride_batch) {
        int o_batch_offset = batch*o_batch_step;
        int i_bin_batch_offset = batch*i_bin_batch_step;
        int i_stream_batch_offset = index_batch*i_stream_batch_step;
        int i_point_batch_offset = index_batch*i_point_batch_step;
        for(int i=index_gen; i<c_ins * i_w_ins * i_h_ins; i += stride_gen) {
            int input_stream_cur = 0;
            int input_bin_cur = input_bin[i_bin_batch_offset+i];
            for(int pack=0; pack<bit_packs; pack++) {
                int i_stream_pack_offset = pack*c_ins*i_w_ins*i_h_ins;
                for(int bit=0; bit<bit_unit; bit++) {
                    input_rand = curand(states+index_overall) % bit_length;
                    input_stream_cur = (input_stream_cur*2) + (input_bin_cur > input_rand);
                }
                input_stream[i_stream_batch_offset+i_stream_pack_offset+i] = input_stream_cur;
            }
        }
        __syncthreads();
        for(int pack=0; pack<bit_packs; pack++) {
            int i_stream_pack_offset = pack*c_ins*i_w_ins*i_h_ins;
            int i_point_pack_offset = pack * (i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1) * c_ins * w_w_ins * w_h_ins;
            for(int i=index_gen; i<(i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1)*c_ins*w_w_ins*w_h_ins; i+= stride_gen) {
                int flatten_in = i/((i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1));
                int c_in = flatten_in / (w_w_ins*w_h_ins);
                int w_in = (flatten_in % (w_w_ins*w_h_ins)) / w_h_ins;
                int h_in = flatten_in % w_h_ins;
                int flatten = i % ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1));
                int input_w = flatten / (i_h_ins-w_h_ins+1);
                int input_h = flatten % (i_h_ins-w_h_ins+1);

                input_point[i_point_batch_offset+i_point_pack_offset+i] = input_stream[i_stream_batch_offset+i_stream_pack_offset + c_in*i_w_ins*i_h_ins + (input_w+w_in)*i_h_ins + (input_h+h_in)];
            }
        }
        __syncthreads();

        matmul_outer(input_point,
                weight_pos_stream,
                weight_neg_stream,
                output_stream,
                index_gen,
                bit_packs,
                i_pw_step,
                w_pw_step,
                cin_size,
                c_outs,
                i_kernel_size,
                w_kernel_size,
                inner_size,
                i_point_batch_offset,
                o_batch_offset,
                cin_size);
    }
}

// Activation generation + im2col + computation for LFSR generator + y-dimesnion fixed-point accumulation. Generation is done after im2col
__global__
void stream_compute_add_im2col_variable(
        const int32_t* __restrict__ input_bin,
        int32_t* __restrict__ input_seed_arr,
        int32_t* __restrict__ input_point,
        const int32_t* __restrict__ weight_pos_stream,
        const int32_t* __restrict__ weight_neg_stream,
        int32_t* __restrict__ output_stream,
        int bit_length,
        int batches,
        int c_ins,
        int i_w_ins,
        int i_h_ins,
        int c_outs,
        int w_w_ins,
        int w_h_ins,
        int i_kernel_size,
        int w_kernel_size,
        int total_width,
        int load_width,
        int load_wait
        ) {
    int bit_packs = (bit_length+31)/32;
    int (*lfsr)(int);
    int bit_unit = 32;
    switch(bit_length) {
    case 8:
        lfsr=&d_lfsr_3;
        bit_unit = 8;
        break;
    case 16:
        lfsr=&d_lfsr_4;
        bit_unit = 16;
        break;
    case 32:
        lfsr=&d_lfsr_5;
        break;
    case 64:
        lfsr=&d_lfsr_6;
        break;
    case 128:
        lfsr=&d_lfsr_7;
        break;
    case 256:
        lfsr=&d_lfsr_8;
        break;
    }

    const int inner_size = c_ins*w_w_ins;
    const int cin_size = (i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1);
    const int cout_size = c_outs;

    const int o_cout_step = cin_size;

    const int i_pw_step = inner_size * cin_size;
    const int i_flatten_step = cin_size;

    const int w_pw_step = inner_size * cout_size;
    const int w_flatten_step = cout_size;
    const int o_batch_step = c_outs * cin_size;
    const int i_bin_batch_step = c_ins * i_w_ins * i_h_ins;
    const int i_point_batch_step = bit_packs * cin_size * c_ins * w_w_ins * w_h_ins;

    int cur_width;
    int cur_bit;

    int index_gen = threadIdx.x;
    int stride_gen = blockDim.x;
    int index_batch = blockIdx.x;
    int stride_batch = gridDim.x;
    for(int batch=index_batch; batch<batches; batch+=stride_batch) {
        int o_batch_offset = batch*o_batch_step;
        int i_bin_batch_offset = batch*i_bin_batch_step;
        int i_point_batch_offset = index_batch*i_point_batch_step;
        int input_seed=0;
        for(int i=index_gen; i<c_ins * w_w_ins; i += stride_gen) {
            input_seed_arr[i] = (input_seed + i)%(bit_length-1) + 1;
        }
        __syncthreads();
        for(int i=index_gen; i<w_h_ins*(i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1)*c_ins*w_w_ins; i+= stride_gen) {
            int h_in = i / ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1) * c_ins * w_w_ins);
            int flatten_in = (i % ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1) * c_ins * w_w_ins)) / ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1));
            int input_cin = flatten_in / w_w_ins;
            int input_w_win = flatten_in % w_w_ins;
            int flatten = i % ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1));
            int input_w = flatten / (i_h_ins-w_h_ins+1);
            int input_h = flatten % (i_h_ins-w_h_ins+1);

            int input_bin_cur = input_bin[i_bin_batch_offset + input_cin*i_w_ins*i_h_ins + (input_w+input_w_win)*i_h_ins + (input_h+h_in)];
            int input_bin_actual;
            cur_bit=0;
            int input_stream_cur = 0;
            int input_seed_cur = input_seed_arr[input_cin*w_w_ins + input_w_win];
            for(int pack=0; pack<bit_packs; pack++) {
                int i_point_pack_offset = pack * (i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1) * c_ins * w_w_ins* w_h_ins;
                for(int bit=0; bit<bit_unit; bit++) {
                    cur_bit+=1;
                    cur_width = (cur_bit/load_wait + 1)*load_width - 1;
                    if (cur_width > total_width) cur_width=total_width;
                    input_bin_actual = (input_bin_cur >> (total_width-cur_width)) << (total_width-cur_width);
                    input_seed_cur = (*lfsr)(input_seed_cur);
                    input_stream_cur = (input_stream_cur*2) + (input_bin_actual > input_seed_cur);
                }
                input_point[i_point_batch_offset+i_point_pack_offset+i] = input_stream_cur;
            }
        }
        __syncthreads();
        matmul_outer(input_point,
                weight_pos_stream,
                weight_neg_stream,
                output_stream,
                index_gen,
                bit_packs*w_h_ins,
                i_pw_step,
                w_pw_step,
                i_flatten_step,
                w_flatten_step,
                i_kernel_size,
                w_kernel_size,
                inner_size,
                i_point_batch_offset,
                o_batch_offset,
                o_cout_step);
    }
}

// Activation generation + im2col + computation for LFSR generator + z-dimesnion fixed-point accumulation. Generation is done after im2col
__global__
void stream_compute_addz_im2col_variable(
        const int32_t* __restrict__ input_bin,
        int32_t* __restrict__ input_seed_arr,
        int32_t* __restrict__ input_point,
        const int32_t* __restrict__ weight_pos_stream,
        const int32_t* __restrict__ weight_neg_stream,
        int32_t* __restrict__ output_stream,
        int bit_length,
        int batches,
        int c_ins,
        int i_w_ins,
        int i_h_ins,
        int c_outs,
        int w_w_ins,
        int w_h_ins,
        int i_kernel_size,
        int w_kernel_size,
        int total_width,
        int load_width,
        int load_wait
        ) {
    int bit_packs = (bit_length+31)/32;
    int (*lfsr)(int);
    int bit_unit = 32;
    switch(bit_length) {
    case 8:
        lfsr=&d_lfsr_3;
        bit_unit = 8;
        break;
    case 16:
        lfsr=&d_lfsr_4;
        bit_unit = 16;
        break;
    case 32:
        lfsr=&d_lfsr_5;
        break;
    case 64:
        lfsr=&d_lfsr_6;
        break;
    case 128:
        lfsr=&d_lfsr_7;
        break;
    case 256:
        lfsr=&d_lfsr_8;
        break;
    }

    const int inner_size = c_ins*w_w_ins;
    const int cin_size = (i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1);
    const int cout_size = c_outs;

    const int o_cout_step = cin_size;

    const int i_pw_step = inner_size * cin_size;
    const int i_flatten_step = cin_size;

    const int w_pw_step = inner_size * cout_size;
    const int w_flatten_step = cout_size;
    const int o_batch_step = c_outs * cin_size;
    const int i_bin_batch_step = c_ins * i_w_ins * i_h_ins;
    const int i_point_batch_step = bit_packs * cin_size * c_ins * w_w_ins * w_h_ins;

    int cur_width;
    int cur_bit;

    int index_gen = threadIdx.x;
    int stride_gen = blockDim.x;
    int index_batch = blockIdx.x;
    int stride_batch = gridDim.x;
    for(int batch=index_batch; batch<batches; batch+=stride_batch) {
        int o_batch_offset = batch*o_batch_step;
        int i_bin_batch_offset = batch*i_bin_batch_step;
        int i_point_batch_offset = index_batch*i_point_batch_step;
        int input_seed=0;
        for(int i=index_gen; i<c_ins * w_w_ins; i += stride_gen) {
            input_seed_arr[i] = (input_seed + i)%(bit_length-1) + 1;
        }
        __syncthreads();
        for(int i=index_gen; i<w_h_ins*(i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1)*c_ins*w_w_ins; i+= stride_gen) {
            int h_in = i / ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1) * c_ins * w_w_ins);
            int flatten_in = (i % ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1) * c_ins * w_w_ins)) / ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1));
            int input_cin = flatten_in / w_w_ins;
            int input_w_win = flatten_in % w_w_ins;
            int flatten = i % ((i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1));
            int input_w = flatten / (i_h_ins-w_h_ins+1);
            int input_h = flatten % (i_h_ins-w_h_ins+1);

            int input_bin_cur = input_bin[i_bin_batch_offset + input_cin*i_w_ins*i_h_ins + (input_w+input_w_win)*i_h_ins + (input_h+h_in)];
            int input_bin_actual;
            cur_bit=0;
            int input_stream_cur = 0;
            int input_seed_cur = input_seed_arr[input_cin*w_w_ins + input_w_win];
            for(int pack=0; pack<bit_packs; pack++) {
                int i_point_pack_offset = pack * (i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1) * c_ins * w_w_ins* w_h_ins;
                for(int bit=0; bit<bit_unit; bit++) {
                    cur_bit+=1;
                    cur_width = (cur_bit/load_wait + 1)*load_width - 1;
                    if (cur_width > total_width) cur_width=total_width;
                    input_bin_actual = (input_bin_cur >> (total_width-cur_width)) << (total_width-cur_width);
                    input_seed_cur = (*lfsr)(input_seed_cur);
                    input_stream_cur = (input_stream_cur*2) + (input_bin_actual > input_seed_cur);
                }
                input_point[i_point_batch_offset+i_point_pack_offset+i] = input_stream_cur;
            }
        }
        __syncthreads();
        matmul_outer(input_point,
                weight_pos_stream,
                weight_neg_stream,
                output_stream,
                index_gen,
                bit_packs*w_h_ins,
                i_pw_step,
                w_pw_step,
                i_flatten_step,
                w_flatten_step,
                i_kernel_size,
                w_kernel_size,
                inner_size,
                i_point_batch_offset,
                o_batch_offset,
                o_cout_step);
    }
}

// Activation generation + im2col + computation for random generator + full-or accumulation. Generation is done after im2col
__global__
void stream_compute_or_im2col(
        const int32_t* __restrict__ input_bin,
        int32_t* __restrict__ input_stream,
        int32_t* __restrict__ input_point_1,
        int32_t* __restrict__ input_point,
        const int32_t* __restrict__ weight_pos_stream,
        const int32_t* __restrict__ weight_neg_stream,
        int32_t* __restrict__ output_stream,
        int bit_length,
        int bit_unit,
        int batches,
        int c_ins,
        int i_w_ins,
        int i_h_ins,
        int c_outs,
        int w_w_ins,
        int w_h_ins,
        int i_kernel_size,
        int w_kernel_size,
        curandState_t* __restrict__ states,
        int seed
        ) {
    int bit_packs = (bit_length+31)/32;

    const int inner_size = c_ins*w_w_ins*w_h_ins;
    const int cin_size = (i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1);

    const int i_pw_step = inner_size * cin_size;

    const int w_pw_step = inner_size * c_outs;
    const int o_batch_step = c_outs * cin_size;
    const int i_bin_batch_step = c_ins * i_w_ins * i_h_ins;
    const int i_point_batch_step = bit_packs * cin_size * inner_size;

    int input_rand=0;

    int index_gen = threadIdx.x;
    int stride_gen = blockDim.x;
    int index_batch = blockIdx.x;
    int stride_batch = gridDim.x;

    int index_overall = blockIdx.x*gridDim.x + threadIdx.x;
    curand_init(seed, index_overall, 0, states+index_overall);
    for(int batch=index_batch; batch<batches; batch+=stride_batch) {
        int o_batch_offset = batch*o_batch_step;
        int i_bin_batch_offset = batch*i_bin_batch_step;
        int i_stream_batch_offset = index_batch*i_pw_step;
        int i_point_batch_offset = index_batch*i_point_batch_step;
        for(int i=index_gen; i<(i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1)*c_ins*w_w_ins*w_h_ins; i+= stride_gen) {
            int flatten_in = i%(c_ins*w_w_ins*w_h_ins);
            int c_in = flatten_in / (w_w_ins*w_h_ins);
            int w_in = (flatten_in % (w_w_ins*w_h_ins)) / w_h_ins;
            int h_in = flatten_in % w_h_ins;
            int flatten = i/(c_ins*w_w_ins*w_h_ins);
            int input_w = flatten / (i_h_ins-w_h_ins+1);
            int input_h = flatten % (i_h_ins-w_h_ins+1);

            int input_bin_cur = input_bin[i_bin_batch_offset + c_in*i_w_ins*i_h_ins + (input_w+w_in)*i_h_ins + (input_h+h_in)];
            input_stream[i_stream_batch_offset+i] = input_bin_cur;
        }
        __syncthreads();
        for(int i=index_gen; i<(i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1)*c_ins*w_w_ins*w_h_ins; i+= stride_gen) {
            int input_bin_cur = input_stream[i_stream_batch_offset+i];
            int input_stream_cur = 0;
            for(int pack=0; pack<bit_packs; pack++) {
                int i_point_pack_offset = pack * (i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1) * c_ins * w_w_ins* w_h_ins;
                for(int bit=0; bit<bit_unit; bit++) {
                    input_rand = curand(states + index_overall) % bit_length;
                    input_stream_cur = (input_stream_cur*2) + (input_bin_cur > input_rand);
                }
                input_point_1[i_point_batch_offset+i_point_pack_offset+i] = input_stream_cur;
            }
        }
        __syncthreads();
        for(int pack=0; pack<bit_packs; pack++) {
            int i_point_pack_offset = pack * (i_w_ins-w_w_ins+1) * (i_h_ins-w_h_ins+1) * c_ins * w_w_ins* w_h_ins;
            for(int i=index_gen; i<(i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1)*c_ins*w_w_ins*w_h_ins; i+= stride_gen){
                int flatten_in = i%(c_ins*w_w_ins*w_h_ins);
                int flatten = i/(c_ins*w_w_ins*w_h_ins);
                input_point[i_point_batch_offset+i_point_pack_offset + flatten_in*(i_w_ins-w_w_ins+1)*(i_h_ins-w_h_ins+1) + flatten] =
                        input_point_1[i_point_batch_offset+i_point_pack_offset + i];
            }
        }
        __syncthreads();
        matmul_outer(input_point,
                weight_pos_stream,
                weight_neg_stream,
                output_stream,
                index_gen,
                bit_packs,
                i_pw_step,
                w_pw_step,
                cin_size,
                c_outs,
                i_kernel_size,
                w_kernel_size,
                inner_size,
                i_point_batch_offset,
                o_batch_offset,
                cin_size);
    }
}

// Conv2d with LFSR generation + y-dimension fixed-point accumulation
torch::Tensor conv2d_add_partial_direct_variable_cuda(torch::Tensor input,
        torch::Tensor weight_pos,
        torch::Tensor weight_neg,
        int bit_length,
        int total_width,
        int load_width,
        int load_wait,
        bool im2col) {
    auto weight_size = weight_pos.sizes();
    auto input_size = input.sizes();
    auto device = weight_pos.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));
    int bit_packs = (bit_length+31)/32;

    int32_t *weight_pos_stream, *weight_neg_stream;
    cudaMallocManaged(&weight_pos_stream, bit_packs*weight_size[3]*weight_size[1]*weight_size[2]*weight_size[0]*sizeof(int32_t));
    cudaMallocManaged(&weight_neg_stream, bit_packs*weight_size[3]*weight_size[1]*weight_size[2]*weight_size[0]*sizeof(int32_t));
    int weight_stream_size = bit_packs*weight_size[0]*weight_size[1]*weight_size[2]*weight_size[3];

    const int threads = 1024;
    const int numBlocks_gen = (weight_size[0]*weight_size[1]*weight_size[2]*weight_size[3] + threads -1)/threads;
    int *pos_seed_arr, *neg_seed_arr;
    cudaMallocManaged(&pos_seed_arr, weight_size[1]*weight_size[2]*sizeof(int32_t));
    cudaMallocManaged(&neg_seed_arr, weight_size[1]*weight_size[2]*sizeof(int32_t));
    int pos_seed = 67;
    int neg_seed = 37;

    for(int i=0; i<weight_size[1] * weight_size[2]; i++) {
        pos_seed_arr[i] = (pos_seed + i)%(bit_length-1) + 1;
        neg_seed_arr[i] = (neg_seed + i)%(bit_length-1) + 1;
    }
    stream_generation_transpose_add_variable<<<numBlocks_gen, threads>>>(
            weight_pos.data_ptr<int32_t>(),
            weight_neg.data_ptr<int32_t>(),
            weight_pos_stream,
            weight_neg_stream,
            pos_seed_arr,
            neg_seed_arr,
            bit_length,
            weight_size[0],
            weight_size[1],
            weight_size[2],
            weight_size[3],
            total_width,
            load_width,
            load_wait);
    auto output_tensor = torch::zeros({input_size[0], weight_size[0], (input_size[2]-weight_size[2]+1), (input_size[3]-weight_size[3]+1)}, at::TensorOptions().dtype(torch::kInt32).device(device));
    int o_batch_step = weight_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int i_bin_batch_step = input_size[1] * input_size[2] * input_size[3];
    int add_size = bit_packs * weight_size[3];
    int inner_size = weight_size[1] * weight_size[2];
    int cin_size = (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int cout_size = weight_size[0];

    const int numBlocks_comp = BATCH_SIZE;
    int32_t *input_stream;
    cudaMallocManaged(&input_stream, bit_packs*numBlocks_comp*input_size[1]*input_size[2]*input_size[3]*sizeof(int32_t));
    int* input_seed_arr;
    cudaMallocManaged(&input_seed_arr, input_size[1]*input_size[2]*input_size[3]*sizeof(int32_t));

    int32_t* output_point = output_tensor.data_ptr<int32_t>();
    int32_t *output_pos, *output_neg;
    cudaMallocManaged(&output_pos, input_size[0]*o_batch_step*sizeof(int32_t));
    cudaMallocManaged(&output_neg, input_size[0]*o_batch_step*sizeof(int32_t));

    int i_kernel = BLOCK_INPUT;
    int w_kernel = BLOCK_WEIGHT;
    const int threads_c = i_kernel * w_kernel;

    stream_compute_add_direct_variable<<<numBlocks_comp, threads_c>>>(
            input.data_ptr<int32_t>(),
            input_seed_arr,
            input_stream,
            weight_pos_stream,
            weight_neg_stream,
            output_point,
            output_pos,
            output_neg,
            bit_length,
            input_size[0],
            input_size[1],
            input_size[2],
            input_size[3],
            weight_size[0],
            weight_size[2],
            weight_size[3],
            i_kernel,
            w_kernel,
            total_width,
            load_width,
            load_wait
            );
    cudaFree(input_stream);
    cudaFree(input_seed_arr);
    cudaFree(weight_pos_stream);
    cudaFree(weight_neg_stream);
    cudaFree(pos_seed_arr);
    cudaFree(neg_seed_arr);
    cudaFree(output_pos);
    cudaFree(output_neg);
    return output_tensor;
}

// Conv2d with accurate random generation + y-dimension fixed-point accumulation
torch::Tensor conv2d_add_partial_cuda_acc(torch::Tensor input,
        torch::Tensor weight_pos,
        torch::Tensor weight_neg,
        int bit_length,
        bool share) {
    auto weight_size = weight_pos.sizes();
    int weight_size_total = weight_size[0]*weight_size[1]*weight_size[2]*weight_size[3];
    auto input_size = input.sizes();
    int input_size_total = input_size[0]*input_size[1]*input_size[2]*input_size[3];
    auto device = weight_pos.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));
    int bit_packs = (bit_length+31)/32;
    int bit_unit = 32;
    switch(bit_length) {
    case 8:
        bit_unit=8;
        break;
    case 16:
        bit_unit=16;
        break;
    }

    int32_t *weight_pos_stream, *weight_neg_stream;
    cudaMallocManaged(&weight_pos_stream, bit_packs*weight_size[3]*weight_size[1]*weight_size[2]*weight_size[0]*sizeof(int32_t));
    cudaMallocManaged(&weight_neg_stream, bit_packs*weight_size[3]*weight_size[1]*weight_size[2]*weight_size[0]*sizeof(int32_t));
    int weight_stream_size = bit_packs*weight_size[0]*weight_size[1]*weight_size[2]*weight_size[3];

    const int threads = 256;
    const int numBlocks_gen = (weight_size[0]*weight_size[1]*weight_size[2]*weight_size[3] + threads -1)/threads;
    curandState_t* states_pos;
    curandState_t* states_neg;
    cudaMallocManaged(&states_pos, weight_size_total * sizeof(curandState_t));
    cudaMallocManaged(&states_neg, weight_size_total * sizeof(curandState_t));

    stream_generation_transpose_add_acc<<<numBlocks_gen, threads>>>(
            weight_pos.data_ptr<int32_t>(),
            weight_neg.data_ptr<int32_t>(),
            weight_pos_stream,
            weight_neg_stream,
            bit_length,
            bit_unit,
            weight_size[0],
            weight_size[1],
            weight_size[2],
            weight_size[3],
            states_pos,
            states_neg,
            37,
            share);
    auto output_tensor = torch::zeros({input_size[0], weight_size[0], (input_size[2]-weight_size[2]+1), (input_size[3]-weight_size[3]+1)}, at::TensorOptions().dtype(torch::kInt32).device(device));
    int o_batch_step = weight_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int i_bin_batch_step = input_size[1] * input_size[2] * input_size[3];
    int add_size = bit_packs * weight_size[3];
    int inner_size = weight_size[1] * weight_size[2];
    int cin_size = (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int cout_size = weight_size[0];

    const int numBlocks_comp = BATCH_SIZE;
    int32_t *input_stream, *input_point;
    cudaMallocManaged(&input_stream, bit_packs*numBlocks_comp*input_size[1]*input_size[2]*input_size[3]*sizeof(int32_t));
    cudaMallocManaged(&input_point, bit_packs*numBlocks_comp*(input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) *
            weight_size[1] * weight_size[2] * weight_size[3] * sizeof(int32_t));

    int32_t* output_point = output_tensor.data_ptr<int32_t>();


    const int threads_c = 256;
    curandState_t* states_input;
    if(!share) cudaMallocManaged(&states_input, threads_c*numBlocks_comp * sizeof(curandState_t));
    else cudaMallocManaged(&states_input, input_size_total * sizeof(curandState_t));
    int i_kernel = 16;
    int w_kernel = 16;
    if (weight_size[0]<16) {
        i_kernel = 32;
        w_kernel = 8;
    }
    if (weight_size[0]<8) {
        i_kernel = 64;
        w_kernel = 4;
    }
    if (weight_size[0]<4) {
        i_kernel = 128;
        w_kernel = 2;
    }
    if (weight_size[0]<2) {
        i_kernel = 256;
        w_kernel = 1;
    }
    if (cin_size<16) {
        i_kernel = 8;
        w_kernel = 32;
    }
    if (cin_size<8) {
        i_kernel = 4;
        w_kernel = 64;
    }
    if (cin_size<4) {
        i_kernel = 2;
        w_kernel = 128;
    }
    stream_compute_add_acc<<<numBlocks_comp, threads_c>>>(
            input.data_ptr<int32_t>(),
            input_stream,
            input_point,
            weight_pos_stream,
            weight_neg_stream,
            output_point,
            bit_length,
            bit_unit,
            input_size[0],
            input_size[1],
            input_size[2],
            input_size[3],
            weight_size[0],
            weight_size[2],
            weight_size[3],
            i_kernel,
            w_kernel,
            states_input,
            0,
            0
            );
    cudaFree(input_stream);
    cudaFree(input_point);
    cudaFree(weight_pos_stream);
    cudaFree(weight_neg_stream);
    cudaFree(states_pos);
    cudaFree(states_neg);
    cudaFree(states_input);
    return output_tensor;
}

// Conv2d with LFSR generation + y & partial-z-dimension fixed-point accumulation
torch::Tensor conv2d_addyz_variable_cuda(torch::Tensor input,
        torch::Tensor weight_pos,
        torch::Tensor weight_neg,
        int bit_length,
        int total_width,
        int load_width,
        int load_wait_w,
        int load_wait_a,
        int z_units,
        bool im2col) {
    auto weight_size = weight_pos.sizes();
    auto input_size = input.sizes();
    auto device = weight_pos.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));
    int bit_packs = (bit_length+31)/32;
    if(weight_size[1]<z_units) z_units=weight_size[1];
    int z_packs = (weight_size[1]+z_units-1)/z_units;

    int32_t *weight_pos_stream, *weight_neg_stream;
    cudaMallocManaged(&weight_pos_stream, bit_packs*weight_size[3]*z_packs*z_units*weight_size[2]*weight_size[0]*sizeof(int32_t));
    cudaMallocManaged(&weight_neg_stream, bit_packs*weight_size[3]*z_packs*z_units*weight_size[2]*weight_size[0]*sizeof(int32_t));

    const int threads = 1024;
    const int numBlocks_gen = (weight_size[0]*z_packs*z_units*weight_size[2]*weight_size[3] + threads -1)/threads;
    int *pos_seed_arr, *neg_seed_arr;
    cudaMallocManaged(&pos_seed_arr, z_units*weight_size[2]*sizeof(int32_t));
    cudaMallocManaged(&neg_seed_arr, z_units*weight_size[2]*sizeof(int32_t));
    int pos_seed = 67;
    int neg_seed = 37;

    for(int i=0; i<z_units * weight_size[2]; i++) {
        pos_seed_arr[i] = (pos_seed + i)%(bit_length-1) + 1;
        neg_seed_arr[i] = (neg_seed + i)%(bit_length-1) + 1;
    }
    stream_generation_transpose_addyz_variable<<<numBlocks_gen, threads>>>(
            weight_pos.data_ptr<int32_t>(),
            weight_neg.data_ptr<int32_t>(),
            weight_pos_stream,
            weight_neg_stream,
            pos_seed_arr,
            neg_seed_arr,
            bit_length,
            z_units,
            weight_size[0],
            weight_size[1],
            weight_size[2],
            weight_size[3],
            total_width,
            load_width,
            load_wait_w);
    auto output_tensor = torch::zeros({input_size[0], weight_size[0], (input_size[2]-weight_size[2]+1), (input_size[3]-weight_size[3]+1)}, at::TensorOptions().dtype(torch::kInt32).device(device));
    int o_batch_step = weight_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int i_bin_batch_step = input_size[1] * input_size[2] * input_size[3];
    int add_size = bit_packs * weight_size[3];
    int inner_size = weight_size[1] * weight_size[2];
    int cin_size = (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int cout_size = weight_size[0];

    const int numBlocks_comp = 64;
    int32_t *input_stream, *input_point;
    if (!im2col) {
        cudaMallocManaged(&input_stream, bit_packs*numBlocks_comp*z_packs*z_units*input_size[2]*input_size[3]*sizeof(int32_t));
    }
    cudaMallocManaged(&input_point, bit_packs*numBlocks_comp*(input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) *
            z_packs*z_units * weight_size[2] * weight_size[3] * sizeof(int32_t));
    int* input_seed_arr;
    cudaMallocManaged(&input_seed_arr, z_units*input_size[2]*input_size[3]*sizeof(int32_t));
    for(int i=0; i<z_units*input_size[2]*input_size[3]; i++) {
        input_seed_arr[i] = i%(bit_length-1) + 1;
    }

    int32_t* output_point = output_tensor.data_ptr<int32_t>();

    const int threads_c = 512;

    int i_kernel = 32;
    int w_kernel = 16;

    if (weight_size[0]<16) {
        i_kernel = 64;
        w_kernel = 8;
    }
    if (weight_size[0]<8) {
        i_kernel = 128;
        w_kernel = 4;
    }
    if (weight_size[0]<4) {
        i_kernel = 256;
        w_kernel = 2;
    }
    if (weight_size[0]<2) {
        i_kernel = 512;
        w_kernel = 1;
    }
    if (cin_size<32) {
        i_kernel = 16;
        w_kernel = 32;
    }
    if (cin_size<16) {
        i_kernel = 8;
        w_kernel = 64;
    }
    if (cin_size<8) {
        i_kernel = 4;
        w_kernel = 128;
    }
    if (cin_size<4) {
        i_kernel = 2;
        w_kernel = 256;
    }

    stream_compute_addyz_variable<<<numBlocks_comp, threads_c>>>(
            input.data_ptr<int32_t>(),
            input_seed_arr,
            input_stream,
            input_point,
            weight_pos_stream,
            weight_neg_stream,
            output_point,
            bit_length,
            z_units,
            input_size[0],
            input_size[1],
            input_size[2],
            input_size[3],
            weight_size[0],
            weight_size[2],
            weight_size[3],
            i_kernel,
            w_kernel,
            total_width,
            load_width,
            load_wait_a
            );
    if (!im2col) {
        cudaFree(input_stream);
    }
    cudaFree(input_point);
    cudaFree(input_seed_arr);
    cudaFree(weight_pos_stream);
    cudaFree(weight_neg_stream);
    cudaFree(pos_seed_arr);
    cudaFree(neg_seed_arr);
    return output_tensor;
}

// Conv2d with LFSR generation + z-dimension fixed-point accumulation
torch::Tensor conv2d_addz_variable_cuda(torch::Tensor input,
        torch::Tensor weight_pos,
        torch::Tensor weight_neg,
        int bit_length,
        int total_width,
        int load_width,
        int load_wait,
        bool im2col) {
    auto weight_size = weight_pos.sizes();
    auto input_size = input.sizes();
    auto device = weight_pos.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));
    int bit_packs = (bit_length+31)/32;

    int32_t *weight_pos_stream, *weight_neg_stream;
    cudaMallocManaged(&weight_pos_stream, bit_packs*weight_size[3]*weight_size[1]*weight_size[2]*weight_size[0]*sizeof(int32_t));
    cudaMallocManaged(&weight_neg_stream, bit_packs*weight_size[3]*weight_size[1]*weight_size[2]*weight_size[0]*sizeof(int32_t));
    int weight_stream_size = bit_packs*weight_size[0]*weight_size[1]*weight_size[2]*weight_size[3];

    const int threads = 1024;
    const int numBlocks_gen = (weight_size[0]*weight_size[1]*weight_size[2]*weight_size[3] + threads -1)/threads;
    int *pos_seed_arr, *neg_seed_arr;
    cudaMallocManaged(&pos_seed_arr, weight_size[1]*weight_size[2]*sizeof(int32_t));
    cudaMallocManaged(&neg_seed_arr, weight_size[1]*weight_size[2]*sizeof(int32_t));
    int pos_seed = 67;
    int neg_seed = 37;

    for(int i=0; i<weight_size[1] * weight_size[2]; i++) {
        pos_seed_arr[i] = (pos_seed + i)%(bit_length-1) + 1;
        neg_seed_arr[i] = (neg_seed + i)%(bit_length-1) + 1;
    }
    stream_generation_transpose_addz_variable<<<numBlocks_gen, threads>>>(
            weight_pos.data_ptr<int32_t>(),
            weight_neg.data_ptr<int32_t>(),
            weight_pos_stream,
            weight_neg_stream,
            pos_seed_arr,
            neg_seed_arr,
            bit_length,
            weight_size[0],
            weight_size[1],
            weight_size[2],
            weight_size[3],
            total_width,
            load_width,
            load_wait);
    auto output_tensor = torch::zeros({input_size[0], weight_size[0], (input_size[2]-weight_size[2]+1), (input_size[3]-weight_size[3]+1)}, at::TensorOptions().dtype(torch::kInt32).device(device));
    int o_batch_step = weight_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int i_bin_batch_step = input_size[1] * input_size[2] * input_size[3];
    int add_size = bit_packs * weight_size[3];
    int inner_size = weight_size[1] * weight_size[2];
    int cin_size = (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int cout_size = weight_size[0];

    const int numBlocks_comp = 64;
    int32_t *input_stream, *input_point;
    if (!im2col) {
        cudaMallocManaged(&input_stream, bit_packs*numBlocks_comp*input_size[1]*input_size[2]*input_size[3]*sizeof(int32_t));
    }
    cudaMallocManaged(&input_point, bit_packs*numBlocks_comp*(input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) *
            weight_size[1] * weight_size[2] * weight_size[3] * sizeof(int32_t));
    int* input_seed_arr;
    cudaMallocManaged(&input_seed_arr, input_size[1]*input_size[2]*input_size[3]*sizeof(int32_t));
    for(int i=0; i<input_size[1]*input_size[2]*input_size[3]; i++) {
        input_seed_arr[i] = i%(bit_length-1) + 1;
    }

    int32_t* output_point = output_tensor.data_ptr<int32_t>();

    const int threads_c = 1024;
    int i_kernel = 32;
    int w_kernel = 32;
    if (weight_size[0]<32) {
        i_kernel = 64;
        w_kernel = 16;
    }
    if (weight_size[0]<16) {
        i_kernel = 128;
        w_kernel = 8;
    }
    if (weight_size[0]<8) {
        i_kernel = 256;
        w_kernel = 4;
    }
    if (weight_size[0]<4) {
        i_kernel = 512;
        w_kernel = 2;
    }
    if (weight_size[0]<2) {
        i_kernel = 1024;
        w_kernel = 1;
    }
    if (cin_size<32) {
        i_kernel = 16;
        w_kernel = 64;
    }
    if (cin_size<16) {
        i_kernel = 8;
        w_kernel = 128;
    }
    if (cin_size<8) {
        i_kernel = 4;
        w_kernel = 256;
    }
    if (cin_size<4) {
        i_kernel = 2;
        w_kernel = 512;
    }
    if (im2col) {
        stream_compute_addz_im2col_variable<<<numBlocks_comp, threads_c>>>(
                input.data_ptr<int32_t>(),
                input_seed_arr,
                input_point,
                weight_pos_stream,
                weight_neg_stream,
                output_point,
                bit_length,
                input_size[0],
                input_size[1],
                input_size[2],
                input_size[3],
                weight_size[0],
                weight_size[2],
                weight_size[3],
                i_kernel,
                w_kernel,
                total_width,
                load_width,
                load_wait
                );
    }
    else {
        stream_compute_addz_variable<<<numBlocks_comp, threads_c>>>(
                input.data_ptr<int32_t>(),
                input_seed_arr,
                input_stream,
                input_point,
                weight_pos_stream,
                weight_neg_stream,
                output_point,
                bit_length,
                input_size[0],
                input_size[1],
                input_size[2],
                input_size[3],
                weight_size[0],
                weight_size[2],
                weight_size[3],
                i_kernel,
                w_kernel,
                total_width,
                load_width,
                load_wait
                );
    }
    if (!im2col) {
        cudaFree(input_stream);
    }
    cudaFree(input_point);
    cudaFree(input_seed_arr);
    cudaFree(weight_pos_stream);
    cudaFree(weight_neg_stream);
    cudaFree(pos_seed_arr);
    cudaFree(neg_seed_arr);
    return output_tensor;
}

// Conv2d with random generation + full OR accumulation
torch::Tensor conv2d_or_cuda(torch::Tensor input,
        torch::Tensor weight_pos,
        torch::Tensor weight_neg,
        int bit_length,
        bool im2col) {
    auto weight_size = weight_pos.sizes();
    int weight_size_total = weight_size[0]*weight_size[1]*weight_size[2]*weight_size[3];
    auto input_size = input.sizes();
    auto device = weight_pos.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));
    int bit_packs = (bit_length+31)/32;

    int32_t *weight_pos_stream, *weight_neg_stream;
    cudaMallocManaged(&weight_pos_stream, bit_packs*weight_size_total*sizeof(int32_t));
    cudaMallocManaged(&weight_neg_stream, bit_packs*weight_size_total*sizeof(int32_t));
    const int threads = 256;
    const int numBlocks_gen = (weight_size_total + threads -1)/threads;

    curandState_t* states_pos;
    curandState_t* states_neg;
    cudaMallocManaged(&states_pos, weight_size_total * sizeof(curandState_t));
    cudaMallocManaged(&states_neg, weight_size_total * sizeof(curandState_t));

    int bit_unit = 32;
    switch(bit_length) {
    case 8:
        bit_unit=8;
        break;
    case 16:
        bit_unit=16;
        break;
    }

    stream_generation_transpose<<<numBlocks_gen, threads>>>(
            weight_pos.data_ptr<int32_t>(),
            weight_neg.data_ptr<int32_t>(),
            weight_pos_stream,
            weight_neg_stream,
            bit_length,
            bit_unit,
            weight_size[0],
            weight_size[1],
            weight_size[2],
            weight_size[3],
            states_pos,
            states_neg,
            37);
    auto output_tensor = torch::zeros({input_size[0], weight_size[0], (input_size[2]-weight_size[2]+1), (input_size[3]-weight_size[3]+1)}, at::TensorOptions().dtype(torch::kInt32).device(device));
    int o_batch_step = weight_size[0] * (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int i_bin_batch_step = input_size[1] * input_size[2] * input_size[3];
    int add_size = bit_packs * weight_size[3];
    int inner_size = weight_size[1] * weight_size[2];
    int cin_size = (input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1);
    int cout_size = weight_size[0];

    const int numBlocks_comp = 64;
    int32_t *input_stream, *input_point, *input_point_1;
    if (!im2col) {
        cudaMallocManaged(&input_stream, bit_packs*numBlocks_comp*input_size[1]*input_size[2]*input_size[3]*sizeof(int32_t));
    }
    else {
        cudaMallocManaged(&input_stream, numBlocks_comp*(input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) *
            weight_size[1] * weight_size[2] * weight_size[3] * sizeof(int32_t));
        cudaMallocManaged(&input_point_1, bit_packs*numBlocks_comp*(input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) *
                weight_size[1] * weight_size[2] * weight_size[3] * sizeof(int32_t));
    }
    cudaMallocManaged(&input_point, bit_packs*numBlocks_comp*(input_size[2]-weight_size[2]+1) * (input_size[3]-weight_size[3]+1) *
            weight_size[1] * weight_size[2] * weight_size[3] * sizeof(int32_t));

    int32_t* output_point = output_tensor.data_ptr<int32_t>();
    const int threads_c = 256;

    int i_kernel = 16;
    int w_kernel = 16;
    if (weight_size[0]<16) {
        i_kernel = 32;
        w_kernel = 8;
    }
    if (weight_size[0]<8) {
        i_kernel = 64;
        w_kernel = 4;
    }
    if (weight_size[0]<4) {
        i_kernel = 128;
        w_kernel = 2;
    }
    if (weight_size[0]<2) {
        i_kernel = 256;
        w_kernel = 1;
    }
    if (cin_size<16) {
        i_kernel = 8;
        w_kernel = 32;
    }
    if (cin_size<8) {
        i_kernel = 4;
        w_kernel = 64;
    }
    if (cin_size<4) {
        i_kernel = 2;
        w_kernel = 128;
    }

    curandState_t* states_input;
    cudaMallocManaged(&states_input, threads_c*numBlocks_comp * sizeof(curandState_t));

    if (im2col) {
        stream_compute_or_im2col<<<numBlocks_comp, threads_c>>>(
                input.data_ptr<int32_t>(),
                input_stream,
                input_point_1,
                input_point,
                weight_pos_stream,
                weight_neg_stream,
                output_point,
                bit_length,
                bit_unit,
                input_size[0],
                input_size[1],
                input_size[2],
                input_size[3],
                weight_size[0],
                weight_size[2],
                weight_size[3],
                i_kernel,
                w_kernel,
                states_input,
                0
                );
    }
    else {
        stream_compute_or<<<numBlocks_comp, threads_c>>>(
                input.data_ptr<int32_t>(),
                input_stream,
                input_point,
                weight_pos_stream,
                weight_neg_stream,
                output_point,
                bit_length,
                bit_unit,
                input_size[0],
                input_size[1],
                input_size[2],
                input_size[3],
                weight_size[0],
                weight_size[2],
                weight_size[3],
                i_kernel,
                w_kernel,
                states_input,
                0
                );
    }
    cudaFree(input_stream);
    if (im2col) cudaFree(input_point_1);
    cudaFree(input_point);
    cudaFree(weight_pos_stream);
    cudaFree(weight_neg_stream);
    cudaFree(states_pos);
    cudaFree(states_neg);
    cudaFree(states_input);
    return output_tensor;
}

