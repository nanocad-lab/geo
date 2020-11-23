import os
import torch
import logging
import torchvision
import torchvision.transforms as transforms

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import time
from multiprocessing import Process

from utils import *

import sc_extension
import sc_extension_cuda

from sys import exit

compute_type = torch.float32
compare_type = torch.float32
global_share = True
global_share_more = False
global_share_max = False
global_share_col = False
global_save = False
global_trueor = False
global_usebn = False
relu_warmup = 8
reludrop_thres = 16
relu_thres = -0.7

'''
Helper functions
'''
def quantize(input, quant=False, prec=8):
    prec_2 = 2**prec
    if quant:
        input = (input * prec_2).round().clamp(-prec_2, prec_2-1)/prec_2
    return input

def quantize_shift(tensor, scale=None, scale_only=False):
    if scale_only:
        scale = torch.mean(tensor)*2
        scale = 2**torch.ceil(torch.log2(scale))
        tensor_quant = torch.ones_like(tensor)*scale
        return tensor_quant, scale
    else:
        if scale is None:
    #         scale = torch.sqrt(torch.mean(tensor**2))*3
            scale = torch.mean(tensor)*3
            scale = 2**torch.ceil(torch.log2(scale))
        tensor_quant = tensor / scale
        tensor_quant = (tensor_quant * 128).round().clamp(-127, 127)/128
        tensor_quant = tensor_quant * scale
        return tensor_quant, scale  
    
def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    if quant_mode=='bin':
        return (tensor>=0).type(type(tensor))*2-1
    else:
#         return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)
        return (tensor>torch.rand(tensor.size(),device=tensor.device)).to(dtype=tensor.dtype)
#         bit_size = 1024
#         size = list(tensor.size())
#         size.insert(0,bit_size)
#         size = tuple(size)
#         output = (tensor>(torch.rand(size, device=tensor.device)*2-1)).to(dtype=tensor.dtype).mean(0)
#         return output

'''
Generator functions
'''

def acc_init(w_pos, w_neg, a=None, device=torch.device('cpu'), prec=128):
    w_pos_cur = w_pos.clone()
    w_neg_cur = w_neg.clone()
    if a is not None:
        a_cur = a.clone()
        return a_cur, w_pos_cur, w_neg_cur
    else:
        return None, w_pos_cur, w_neg_cur

def lfsr_init(w_size, a_size=None, device=torch.device('cpu'), prec=128):
    weight_split_size_flat = np.array(w_size).prod()
    weight_seed_pos = np.arange(67, weight_split_size_flat+67)%(prec-1)+1
    weight_seed_neg = np.arange(37, weight_split_size_flat+37)%(prec-1)+1
    rand_weight_pos = torch.from_numpy(weight_seed_pos).reshape(w_size).to(device)
    rand_weight_neg = torch.from_numpy(weight_seed_neg).reshape(w_size).to(device)
    if a_size is not None:
        input_split_size_flat = np.array(a_size).prod()
        input_seed = np.arange(0, input_split_size_flat)%(prec-1)+1
        rand_input = torch.from_numpy(input_seed).reshape(a_size).to(device)
        return rand_input, rand_weight_pos, rand_weight_neg
    else:
        return None, rand_weight_pos, rand_weight_neg

def sync_init(w_size, a_size=None, device=torch.device('cpu'), prec=128):
    seed = torch.from_numpy(np.random.permutation(prec)).to(device).to(compare_type)
    weight_pos_pos = torch.randint(prec, w_size, dtype=torch.int64, device=device)
    weight_neg_pos = torch.randint(prec, w_size, dtype=torch.int64, device=device)
    if a_size is not None:
        input_pos = torch.randint(prec, a_size, dtype=torch.int64, device=device)
        return seed, input_pos, weight_pos_pos, weight_neg_pos
    else:
        return seed, None, weight_pos_pos, weight_neg_pos

def rand_init(w_size, a_size=None, device=torch.device('cpu'), prec=128):
    rand_weight_pos = torch.randint(prec, w_size, dtype=compare_type, device=device)
    rand_weight_neg = torch.randint(prec, w_size, dtype=compare_type, device=device)
    if a_size is not None:
        rand_input = torch.randint(prec, a_size, dtype=compare_type, device=device)
        return rand_input, rand_weight_pos, rand_weight_neg
    else:
        return None, rand_weight_pos, rand_weight_neg
    
def lfsr_5(rand_in):
    rand_out = ((rand_in//16)+(rand_in//4)%2)%2+2*(rand_in%16)
    return rand_out

def lfsr_6(rand_in):
    rand_out = ((rand_in//32)+(rand_in//16)%2)%2+2*(rand_in%32)
    return rand_out
    
def lfsr_7(rand_in):
    rand_out = ((rand_in//32)%2+rand_in//64)%2+2*(rand_in%64)
    return rand_out

def lfsr_7_xnor(rand_in):
    rand_out = 1-((rand_in//32)%2+rand_in//64)%2+2*(rand_in%64)
    return rand_out

def lfsr_8(rand_in):
    rand_out = ((rand_in//128)+(rand_in//32)%2+(rand_in//16)%2+(rand_in//8)%2)%2+2*(rand_in%128)
    return rand_out
    
def lfsr_cont(rand_input, rand_weight_pos, rand_weight_neg, bit_length=128):
    if bit_length==128:
        lfsr_gen = lfsr_7
    elif bit_length==32:
        lfsr_gen = lfsr_5
    elif bit_length==64:
        lfsr_gen = lfsr_6
    elif bit_length==256:
        lfsr_gen = lfsr_8
    if rand_input is not None:
        rand_input = lfsr_gen(rand_input).to(compare_type)
    rand_weight_pos = lfsr_gen(rand_weight_pos).to(compare_type)
    rand_weight_neg = lfsr_gen(rand_weight_neg).to(compare_type)
    return rand_input, rand_weight_pos, rand_weight_neg

def sync_cont(seed, input_pos, weight_pos_pos, weight_neg_pos, prec=128):
    rand_weight_pos = seed[weight_pos_pos]
    rand_weight_neg = seed[weight_neg_pos]
    weight_pos_pos = (weight_pos_pos+1)%prec
    weight_neg_pos = (weight_neg_pos+1)%prec
    if input_pos is not None:
        rand_input = seed[input_pos]
        input_pos = (input_pos+1)%prec
    else:
        rand_input = None
    return rand_input, rand_weight_pos, rand_weight_neg, input_pos, weight_pos_pos, weight_neg_pos

def acc_cont(input_cur, weight_pos_cur, weight_neg_cur, device, k, prec=128):
    weight_pos_norm = weight_pos_cur * prec / (prec-k)
    weight_neg_norm = weight_neg_cur * prec / (prec-k)
    rand_pos = torch.randint(prec, weight_pos_cur.size(), dtype=compare_type, device=device)
    rand_neg = torch.randint(prec, weight_neg_cur.size(), dtype=compare_type, device=device)
    w_pos_bit = (weight_pos_norm > rand_pos).to(compute_type)
    w_neg_bit = (weight_neg_norm > rand_neg).to(compute_type)
    weight_pos_cur -= w_pos_bit
    weight_neg_cur -= w_neg_bit
    if input_cur is not None:
        input_norm = input_cur * prec / (prec-k)
        rand_input = torch.randint(prec, input_cur.size(), dtype=compare_type, device=device)
        a_bit = (input_norm > rand_input).to(compute_type)
        input_cur -= a_bit
        return a_bit, w_pos_bit, w_neg_bit, input_cur, weight_pos_cur, weight_neg_cur
    else:
        return None, w_pos_bit, w_neg_bit, None, weight_pos_cur, weight_neg_cur
'''
Forward functions for training
'''
def linear_or_approx(activation, weight, true_or=(False or global_trueor)):
    if true_or:
        mult_result = activation.unsqueeze(1)*weight
        return 1-torch.prod(1-mult_result, dim=-1)
    else:
        return 1-torch.exp(-F.linear(activation, weight))
def linear_quant_trun(activation, weight, prec=7, trunc_add=False):
#     max_2 = 2**(-prec)*128
    max_2 = 1
    min_2 = max_2*2**(-prec)
    mult_result = activation.unsqueeze(1)*weight
    mult_result.data = quantize(mult_result.data, quant=True, prec=prec)
#     mult_result = F.hardtanh(mult_result, -max_2, max_2)
#     mult_result = F.hardshrink(mult_result, min_2)
    if trunc_add:
        while mult_result.size(-1)>1:
            half_size = mult_result.size(-1)//2
            mult_result_new = (mult_result[...,:half_size] + mult_result[...,half_size:2*half_size]).clamp(-1,1)
            if mult_result.size(-1)%2==1:
                mult_result = torch.cat((mult_result_new, mult_result[...,-1:]), dim=-1)
            else:
                mult_result = mult_result_new
#     print(mult_result.max())
    return mult_result.sum(-1)    
def conv2d_or_approx(activation, weight, padding, stride, true_or=(False or global_trueor)):
    
    if true_or:
        kernel_size = weight.size()[-2:]
        activation_col = F.unfold(activation, kernel_size, dilation=1, padding=padding, stride=stride)
        
        weight_col = weight.view(weight.data.size(0),-1)
        a_size = list(activation_col.size())
        a_size.insert(1,1)
        w_size = list(weight_col.size())
        w_size.append(1)
        activation_col = activation_col.view(a_size)
        weight_col = weight_col.view(w_size)
    
        mult_result = activation_col*weight_col
        add_res = 1-torch.prod(1-mult_result, dim=2)
        size_out = np.sqrt(add_res.size(-1)).astype(int)
        return F.fold(add_res, (size_out, size_out), (1,1))
    else:
        return 1-torch.exp(-F.conv2d(activation, weight, padding=padding, stride=stride))
def conv2d_quant_trun(activation, weight, padding, stride, prec=7, trunc_add=False):
#     max_2 = 2**(-prec)*128
    max_2 = 1
    min_2 = max_2*2**(-prec)
    kernel_size = weight.size()[-2:]
    activation_col = F.unfold(activation, kernel_size, dilation=1, padding=padding, stride=stride)

    weight_col = weight.view(weight.data.size(0),-1)
    a_size = list(activation_col.size())
    a_size.insert(1,1)
    w_size = list(weight_col.size())
    w_size.append(1)
    activation_col = activation_col.view(a_size)
    weight_col = weight_col.view(w_size)

    mult_result = activation_col*weight_col
    mult_result.data = quantize(mult_result.data, quant=True, prec=prec)
#     mult_result = F.hardtanh(mult_result, -max_2, max_2)
#     mult_result = F.hardshrink(mult_result, min_2)
    if trunc_add:
        while mult_result.size(2)>1:
            half_size = mult_result.size(2)//2
            mult_result_new = (mult_result[:,:,:half_size] + mult_result[:,:,half_size:2*half_size]).clamp(-1,1)
            if mult_result.size(2)%2==1:
                mult_result = torch.cat((mult_result_new, mult_result[:,:,-1:]), dim=2)
            else:
                mult_result = mult_result_new
#     print(mult_result.max(), mult_result.min())
    add_res = mult_result.sum(2)
    size_out = np.sqrt(add_res.size(-1)).astype(int)
    return F.fold(add_res, (size_out, size_out), (1,1))
        
def conv2d_or_bin_2d(activation, w_pos, w_neg, padding, stride):
    i_x = activation.size(2)
    i_y = activation.size(3)
    f_x = w_pos.size(2)
    f_y = w_pos.size(3)
    result_pos_value = []
    result_neg_value = []
    for i in range(f_x):
        for j in range(f_y):
            a_sec = activation[...,i:i_x-f_x+i+1,j:i_y-f_y+j+1].clone()
            w_pos_sec = w_pos[...,i:i+1,j:j+1].clone()
            w_neg_sec = w_neg[...,i:i+1,j:j+1].clone()
            result_pos_value.append(1-torch.exp(-F.conv2d(a_sec, w_pos_sec, stride=stride)))
            result_neg_value.append(1-torch.exp(-F.conv2d(a_sec, w_neg_sec, stride=stride)))
    
    result_pos_value = torch.stack(result_pos_value, 0).sum(0)
    result_neg_value = torch.stack(result_neg_value, 0).sum(0)
#     print(result_pos_value.size(), result_neg_value.size())
    return result_pos_value, result_neg_value                      
def conv2d_or_bin_1d(activation, w_pos, w_neg, padding, stride):
    i_y = activation.size(3)
    f_y = w_pos.size(3)
    result_pos_value = []
    result_neg_value = []
    for j in range(f_y):
        a_sec = activation[...,j:i_y-f_y+j+1].clone()
        w_pos_sec = w_pos[...,j:j+1].clone()
        w_neg_sec = w_neg[...,j:j+1].clone()
        result_pos_value.append(1-torch.exp(-F.conv2d(a_sec, w_pos_sec, stride=stride)))
        result_neg_value.append(1-torch.exp(-F.conv2d(a_sec, w_neg_sec, stride=stride)))
    result_pos_value = torch.stack(result_pos_value, 0).sum(0)
    result_neg_value = torch.stack(result_neg_value, 0).sum(0)
    return result_pos_value, result_neg_value

def conv2d_or_bin_z(activation, w_pos, w_neg, padding, stride):
    c_in = activation.size(1)
    result_pos_value = []
    result_neg_value = []
    for c in range(c_in):
        a_sec = activation[:,c:c+1].clone()
        w_pos_sec = w_pos[:,c:c+1].clone()
        w_neg_sec = w_neg[:,c:c+1].clone()
#         print(a_sec.size())
#         print(w_pos_sec.size())
#         print(w_neg_sec.size())
        result_pos_value.append(1-torch.exp(-F.conv2d(a_sec, w_pos_sec, stride=stride)))
        result_neg_value.append(1-torch.exp(-F.conv2d(a_sec, w_neg_sec, stride=stride)))
    result_pos_value = torch.stack(result_pos_value, 0).sum(0)
    result_neg_value = torch.stack(result_neg_value, 0).sum(0)
    return result_pos_value, result_neg_value

def conv2d_or_bin_yz(activation, w_pos, w_neg, padding, stride, z_unit):
    c_in = activation.size(1)
    i_y = activation.size(3)
    f_y = w_pos.size(3)
    result_pos_value = []
    result_neg_value = []
    for c in range(0,c_in,z_unit):
        if c+z_unit<=c_in:
            c_end=c+z_unit
        else:
            c_end=c_in
        for j in range(f_y):
            a_sec = activation[:,c:c_end,:,j:i_y-f_y+j+1].clone()
            w_pos_sec = w_pos[:,c:c_end,:,j:j+1].clone()
            w_neg_sec = w_neg[:,c:c_end,:,j:j+1].clone()
            result_pos_value.append(1-torch.exp(-F.conv2d(a_sec, w_pos_sec, stride=stride)))
            result_neg_value.append(1-torch.exp(-F.conv2d(a_sec, w_neg_sec, stride=stride)))
    result_pos_value = torch.stack(result_pos_value, 0).sum(0)
    result_neg_value = torch.stack(result_neg_value, 0).sum(0)
    return result_pos_value, result_neg_value
            
'''
Forward functions for streams
'''

def subtract_relu_stream(a, b, warmup, bn):
    '''
    Stream subtraction FSM
    '''
    bit_length = a.size(0)
    counter = torch.zeros_like(a[0])
    output = torch.zeros_like(a)
    
    if isinstance(bn, nn.BatchNorm2d):
        mean = bn.running_mean.unsqueeze(-1).unsqueeze(-1)
        var = bn.running_var.unsqueeze(-1).unsqueeze(-1)
        weight = bn.weight.data.unsqueeze(-1).unsqueeze(-1)
        bias = bn.bias.data.unsqueeze(-1).unsqueeze(-1)
        do_bn = True
    elif isinstance(bn, nn.BatchNorm1d):
        mean = bn.running_mean.unsqueeze(-1)
        var = bn.running_var.unsqueeze(-1)
        weight = bn.weight.data.unsqueeze(-1)
        bias = bn.bias.data.unsqueeze(-1)
        do_bn = True
    else:
        do_bn = False
        
    for i in range(warmup):
        if do_bn:
            counter_cur = a[i]-b[i]
            counter += (counter_cur-mean)*weight/torch.sqrt(var+1e-5) + bias
        else:
            counter += a[i]-b[i]
        counter -= (counter>0).to(counter.dtype)
    for i in range(bit_length):
        pos = (i+warmup)%bit_length
        if do_bn:
            counter_cur = a[pos]-b[pos]
            counter += (counter_cur-mean)*weight/torch.sqrt(var+1e-5) + bias
        else:
            counter += a[pos]-b[pos]
        output[i] = counter>0
        counter -= output[i]
    return output

def subtract_relu_pool(res_pos, res_neg, pool, bn):
    result_pos, result_pos_value = res_pos
    result_neg, result_neg_value = res_neg
     
    # Additional synchronization before relu+pool
    if pool is not None:
        result_pos_unpool = unpool(result_pos)
        result_neg_unpool = unpool(result_neg)
        result_pos_value.data = result_pos_unpool.mean(0)
        result_neg_value.data = result_neg_unpool.mean(0)
    result = subtract_relu_stream(result_pos, result_neg, warmup=relu_warmup, bn=bn)
#     if shuffle:
#         bit_short = result.size(0)
#         pool_size = result.size(1)
#         sel_size  = result[0,0].size()
#         res_n = []
#         device = result.device
#         seed = torch.from_numpy(np.random.permutation(pool_size)).to(device)
#         rand_pos = torch.randint(pool_size, sel_size, device=device)
#         for i in range(bit_short):
#             res_n_n = []
#             for j in range(pool_size):
# #                 rand_sel = torch.randint(pool_size, sel_size, device=device)
#                 rand_sel = seed[rand_pos]
#                 rand_pos = (rand_pos+1)%pool_size
#                 rand_0 = rand_sel==0
#                 rand_1 = rand_sel==1
#                 rand_2 = rand_sel==2
#                 rand_3 = rand_sel==3
#                 res_c = result[i,j].clone()
#                 res_c[rand_0] = result[i,0][rand_0]
#                 res_c[rand_1] = result[i,1][rand_1]
#                 res_c[rand_2] = result[i,2][rand_2]
#                 res_c[rand_3] = result[i,3][rand_3]
#                 res_n_n.append(res_c)
#             res_n_n = torch.stack(res_n_n, 0)
#             res_n.append(res_n_n)
#         result = torch.stack(res_n, 0)
    bit_short = result.size(0)
    device = result.device
    result_c = result[0].clone()
    result_c[:] = result.mean((0,1))
    result = bitstream_acc(result_c, bit_short)
            
    result_value = F.relu(bn(result_pos_value - result_neg_value))
    result_value = F.hardtanh(result_value)
#     result_unpool = subtract_relu_stream(result_pos_unpool, result_neg_unpool, warmup=relu_warmup, bn=bn)
#     result_value.data = result_unpool.mean(0)
    
    result_value = F.avg_pool2d(result_value, pool)
    result_value.data = result.mean((0,1)).float()
    return result, result_value

def subtract_relu(a, b, warmup=relu_warmup):
    '''
    Subtract two streams of equal size. Warmup counter with some cycles
    '''
    a, a_v = a
    b, b_v = b
    bit_length = a.size(0)
    counter = torch.zeros_like(a[0])
    output = subtract_relu_stream(a,b,warmup,bn=nn.Identity())
    output_v = F.relu(a_v - b_v)
    output_v.data = output.mean(0)
    return output, output_v

def pool2d_sift(activation, sift, size=2):
    output_temp = []
    groups = activation.size(1)
    for i in range(size):
        for j in range(size):
            output_temp.append(F.conv2d(activation[...,i:,j:], sift, stride=size, groups=groups))
    return torch.stack(output_temp, 0)

def avgpool2d_stream(activation, size=2, config="Normal", shuffle=False):
    a, a_v = activation
    device = a.device
    dtype = a.dtype
    groups = a_v.size(1)
    bit_length = a.size(0)
    
    sift = torch.ones((groups,1,1,1), dtype=dtype, device=device)
    output = []
    if config=="Normal":
        output_0 = pool2d_sift(a[0], sift, size=size).mean(0)
        output_size = output_0.size()
        rand = torch.rand(output_size, dtype=dtype, device=device)
        output.append((output_0 > rand).to(dtype))
        for j in range(1, bit_length):
            rand = torch.rand(output_size, dtype=dtype, device=device)
            output.append((pool2d_sift(a[j], sift, size=size).mean(0)>rand).to(dtype))
        output = torch.stack(output, 0)
    if config=="Hard_skip":
        skip_length = bit_length // (size**2)
        for i in range(size):
            for j in range(size):
                for k in range(skip_length):
                    output.append(F.conv2d(a[k,...,i:,j:], sift, stride=size, groups=groups))
        output = torch.stack(output, 0)
    if config=="Soft_skip":
        skip_length = bit_length // (size**2)
        for i in range(size):
            for j in range(size):
                offset = skip_length*(i*size+j)
                for k in range(skip_length):
                    output.append(F.conv2d(a[offset+k,...,i:,j:], sift, stride=size, groups=groups))
        output = torch.stack(output, 0)
    if config=="Concat":
        skip_length = bit_length
        bit_length *= size**2
        for i in range(size):
            for j in range(size):
                for k in range(skip_length):
                    output.append(F.conv2d(a[k,...,i:,j:], sift, stride=size, groups=groups))
        output = torch.stack(output, 0)
    
    if shuffle:
        output = output.mean(0)
        output = bitstream_acc(output, bit_length)
    output_v = F.avg_pool2d(a_v, size)
    output_v.data = output.mean(0)
    return output, output_v
    
'''
Generic functional layers. All other configurations should be derived from this
'''

def linear_generic(activation, weight, **kwargs):
    try:
        bit_length = kwargs['bit_length']
    except:
        bit_length = 128
    
    try:
        prec = kwargs['prec']
    except:
        prec = 128
        
    try:
        acc = kwargs['acc']
    except:
        acc = False
        
    try:
        share = kwargs['share']
    except:
        share = global_share
        
    try:
        bypass = kwargs['bypass']
    except:
        bypass = False
        
    try:
        generator = kwargs['generator']
    except:
        generator = None
        
    try:
        sync = kwargs['sync']
    except:
        sync = True
        
    try:
        forward = kwargs['forward']
    except:
        forward = 'full_or'
        
    try:
        relu = kwargs['relu']
    except:
        relu = False
        
    try:
        bn = kwargs['bn']
    except:
        bn = nn.Identity()
    
    device = activation.device
    bit_range = prec-1
    
    if generator=='lfsr':
        prec = bit_length
    if sync:
        input_split = (activation.data*prec).to(compare_type)
        w_pos_split = (weight.data*prec).clamp(0,bit_range).to(compare_type)
        w_neg_split = -(weight.data*prec).clamp(-bit_range,0).to(compare_type)

        if share:
            a_size = [activation.size(-1)]
            w_size = [weight.size(-1)]
        else:
            a_size = activation.size()
            w_size = weight.size()

        if generator=='lfsr':
            rand_input, rand_weight_pos, rand_weight_neg = lfsr_init(w_size, a_size, device, prec)
        elif generator=='sync':
            seed, input_pos, weight_pos_pos, weight_neg_pos = sync_init(w_size, a_size, device, prec)
        else:
            rand_input, rand_weight_pos, rand_weight_neg = rand_init(w_size, a_size, device, prec)

        result_pos = []
        result_neg = []

        for k in range(bit_length):
            if generator=='lfsr':
                rand_input, rand_weight_pos, rand_weight_neg = lfsr_cont(rand_input, rand_weight_pos, rand_weight_neg, bit_length=bit_length)
            elif generator=='sync':
                rand_input, rand_weight_pos, rand_weight_neg, input_pos, weight_pos_pos, weight_neg_pos = sync_cont(seed, input_pos, weight_pos_pos, weight_neg_pos, prec)
            else:
                rand_input, rand_weight_pos, rand_weight_neg = rand_init(w_size, a_size, device, prec)
                
            a_bit = (input_split > rand_input).to(compute_type)
            w_pos_bit = (w_pos_split > rand_weight_pos).to(compute_type)
            w_neg_bit = (w_neg_split > rand_weight_neg).to(compute_type)
            if forward == 'full_or':
                result_pos.append(F.linear(a_bit, w_pos_bit).sign())
                result_neg.append(F.linear(a_bit, w_neg_bit).sign())
            elif forward == 'full_bin':
                result_pos.append(F.linear(a_bit, w_pos_bit))
                result_neg.append(F.linear(a_bit, w_neg_bit))
        result_pos = torch.stack(result_pos, 0)
        result_neg = torch.stack(result_neg, 0)
    
    w_pos = weight.clamp(0,100)
    w_neg = -(weight.clamp(-100,0))
    activation = activation.to(w_pos.dtype)
    
    if forward == 'full_or':
        result_pos_value = linear_or_approx(activation, w_pos)
        result_neg_value = linear_or_approx(activation, w_neg)
    elif forward == 'full_bin':
        result_pos_value = F.linear(activation, w_pos)
        result_neg_value = F.linear(activation, w_neg)
        
    device = str(result_pos_value.device)[-1]
        
    if sync:
        result_pos_value.data = result_pos.mean(0)
        result_neg_value.data = result_neg.mean(0)
        
    if relu:
        return F.relu(bn(result_pos_value - result_neg_value))
    else:
        return bn(result_pos_value - result_neg_value)

def conv2d_generic(activation, weight, padding, stride, **kwargs):
    end = time.time()

    try:
        bit_length = kwargs['bit_length']
    except:
        bit_length = 128
    
    try:
        prec = kwargs['prec']
    except:
        prec = 128
        
    try:
        acc = kwargs['acc']
    except:
        acc = False
        
    try:
        share = kwargs['share']
    except:
        share = global_share
        
    try:
        bypass = kwargs['bypass']
    except:
        bypass = False
        
    try:
        generator = kwargs['generator']
    except:
        generator = None
        
    try:
        sync = kwargs['sync']
    except:
        sync = True
        
    try:
        forward = kwargs['forward']
    except:
        forward = 'full_or'
        
    try:
        relu = kwargs['relu']
    except:
        relu = False
        
    try:
        bn = kwargs['bn']
    except:
        bn = nn.Identity()
        use_bn = False
    else:
        if isinstance(bn, nn.BatchNorm1d) or isinstance(bn, nn.BatchNorm2d):
            use_bn = True
        else:
            use_bn = False
    try:
        legacy = kwargs['legacy']
    except:
        legacy = False
        
    try:
        load_unit = kwargs['load_unit']
    except:
        load_unit = 8
    try:
        load_wait_w = kwargs['load_wait_w']
    except:
        load_wait_w = 1
    try:
        load_wait_a = kwargs['load_wait_a']
    except:
        load_wait_a = 1
    '''
    Conv2d specific
    '''
    try:
        pool = kwargs['pool']
    except:
        pool = None
        
    try:
        im2col = kwargs['im2col']
    except:
        im2col = False
    try:
        z_unit = kwargs['z_unit']
    except:
        z_unit = 1024
    
    '''
    End of Conv2d specific
    '''
    
    device = activation.device
    bit_range = prec-1
    cout = weight.size(0)
    
    if generator=='lfsr':
        prec = bit_length
        
    if use_bn:
        bn_mean = bn.running_mean
        bn_std = torch.sqrt(bn.running_var+bn.eps)
        if bn.affine:
            bn_weight = bn.weight.data
            bn_bias = bn.bias.data
        else:
            bn_weight = torch.ones_like(bn_mean)
            bn_bias = torch.zeros_like(bn_mean)
        bn_sign = torch.sign(bn_weight)
        bn_thres = (((relu_thres - bn_bias)*bn_std/bn_weight + bn_mean)*bn_sign)
        bn_sign_ext = bn_sign.view(bn_sign.size(0),1,1)
        relu_thres_adj = (bn_thres * bit_length).view(bn_sign.size(0),1,1)
    
    if im2col:
        # No-stream version. Maximal share
        kernel_size = weight.size()[-2:]
        # Padding is already taken care
        activation_col = F.unfold(activation, kernel_size, dilation=1, padding=padding, stride=stride)
        weight_col = weight.view(weight.size(0),-1)
        activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))
        
        if sync:
            input_split = (activation_col.data*prec).to(compare_type).transpose(1,2)
            w_pos_split = (weight_col.data*prec).clamp(0,bit_range).to(compare_type)
            w_neg_split = -(weight_col.data*prec).clamp(-bit_range,0).to(compare_type)

            # Maximal sharing
            a_size = input_split.size()
            w_size = weight_col.size()
#             a_size = [input_split.size(-1)]
#             w_size = [weight_col.size(-1)]

            # Maximal-length lfsr
            rand_input, rand_weight_pos, rand_weight_neg = rand_init(w_size, a_size, device, prec)
#             rand_input, rand_weight_pos, rand_weight_neg = lfsr_init(w_size, a_size, device, prec)

            result_pos = 0
            result_neg = 0

            for k in range(bit_length):
#                 rand_input, rand_weight_pos, rand_weight_neg = lfsr_cont(rand_input, rand_weight_pos, rand_weight_neg, bit_length=bit_length)
                rand_input, rand_weight_pos, rand_weight_neg = rand_init(w_size, a_size, device, prec)
                a_bit = (input_split > rand_input).to(compute_type)
                w_pos_bit = (w_pos_split > rand_weight_pos).to(compute_type)
                w_neg_bit = (w_neg_split > rand_weight_neg).to(compute_type)
                if forward == 'full_or':
                    result_pos_temp = a_bit.matmul(w_pos_bit.t()).sign().transpose(1,2)
                    result_neg_temp = a_bit.matmul(w_neg_bit.t()).sign().transpose(1,2)
                    size_out = np.sqrt(result_pos_temp.size(-1)).astype(int)
                    result_pos_temp = F.fold(result_pos_temp, (size_out, size_out), (1,1))
                    result_neg_temp = F.fold(result_neg_temp, (size_out, size_out), (1,1))
                    result_pos = result_pos + result_pos_temp
                    result_neg = result_neg + result_neg_temp
            
#             result_pos = torch.stack(result_pos, 0)
#             result_neg = torch.stack(result_neg, 0)
            
        
    else:
        if sync:
            if (forward=='1d_bin') and (not activation.is_cuda):
                activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))
                result_pos = sc_extension.conv2d_add_partial_new(activation.data, weight.data, bit_length, (0,0), stride).float()
                result_neg = torch.zeros_like(result_pos)
            elif (forward=='1d_bin') and (generator=='lfsr') and (not legacy):
                activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))
                result_pos = sc_extension_cuda.conv2d_add_partial_variable_acc(activation.data, weight.data, bit_length, (0,0), stride, 7, 8, 1, False, 2).float()
                result_neg = torch.zeros_like(result_pos)
            elif (forward=='1d_bin') and (generator=='acc') and (not legacy):
                activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))
                result_pos = sc_extension_cuda.conv2d_add_partial_acc(activation.data, weight.data, bit_length, (0,0), stride, share, 2).float()
                result_neg = torch.zeros_like(result_pos)
            elif (forward=='z_bin') and (activation.is_cuda) and (not legacy):
                activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))
                result_pos = sc_extension_cuda.conv2d_add_partial_variable_acc(activation.data, weight.data, bit_length, (0,0), stride, 7, 8, 1, False, 0).float()
                result_neg = torch.zeros_like(result_pos)
            elif (forward=='yz_bin') and (activation.is_cuda) and (not legacy):
                activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))
                result_pos = sc_extension_cuda.conv2d_add_variable_acc(activation.data, weight.data, bit_length, (0,0), stride, 7, load_unit, load_wait_w, load_wait_a, False, z_unit).float()
                result_neg = torch.zeros_like(result_pos)
            elif (forward=='full_or') and (activation.is_cuda) and (not legacy):
                activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))
                result_pos = sc_extension_cuda.conv2d_or_acc(activation.data, weight.data, bit_length, (0,0), stride, False).float()
                result_neg = torch.zeros_like(result_pos)
            else:
                activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))

                result_pos = 0
                result_neg = 0

                if pool is not None:
                    pool_cnt = pool[0]*pool[1]
                    stride_pool = list(stride[:])
                    stride_pool[0] *= pool[0]
                    stride_pool[1] *= pool[1]
                    
                    bit_length = bit_length // pool_cnt

                input_split = (activation.data*prec).to(compare_type)
                w_pos_split = (weight.data*prec).clamp(0,bit_range).to(compare_type)
                w_neg_split = -(weight.data*prec).clamp(-bit_range,0).to(compare_type)

                if forward=='2d_bin':
                    i_x = activation.size(2)
                    i_y = activation.size(3)
                    f_x = weight.size(2)
                    f_y = weight.size(3)
                    f_size = f_x*f_y
                    w_pos_split_temp = []
                    w_neg_split_temp = []
                    for l in range(f_x):
                        for m in range(f_y):
                            w_pos_split_temp.append(w_pos_split[...,l:l+1,m:m+1])
                            w_neg_split_temp.append(w_neg_split[...,l:l+1,m:m+1])
                    w_pos_split = torch.cat(w_pos_split_temp, 0)
                    w_neg_split = torch.cat(w_neg_split_temp, 0)

                if forward=='1d_bin':
                    i_y = activation.size(3)
                    f_y = weight.size(3)
                    w_pos_split_temp = []
                    w_neg_split_temp = []
                    for m in range(f_y):
                        w_pos_split_temp.append(w_pos_split[...,m:m+1])
                        w_neg_split_temp.append(w_neg_split[...,m:m+1])
                    w_pos_split = torch.cat(w_pos_split_temp, 0)
                    w_neg_split = torch.cat(w_neg_split_temp, 0)
                if forward=='bin_mult':
                    i_y = activation.size(3)
                    f_y = weight.size(3)
                    w_pos_split_temp = []
                    w_neg_split_temp = []
                    for m in range(f_y):
                        w_pos_split_temp.append(w_pos_split[...,m:m+1])
                        w_neg_split_temp.append(w_neg_split[...,m:m+1])
                    w_pos_split = torch.cat(w_pos_split_temp, 0)
                    w_neg_split = torch.cat(w_neg_split_temp, 0)
                    kernel_size = w_pos_split.size()[-2:]
                    w_pos_col = w_pos_split.view(w_pos_split.size(0),-1)
                    w_neg_col = w_neg_split.view(w_neg_split.size(0),-1)
                    activation_col = F.unfold(activation, kernel_size, dilation=1, stride=stride)
                    a_size = list(activation_col.size())
                    a_size.insert(1,1)
                    w_size = list(w_pos_col.size())
                    w_size.append(1)
                    activation_col = activation_col.view(a_size)
                    w_pos_col = w_pos_col.view(w_size)
                    w_neg_col = w_neg_col.view(w_size)
                    
                    mult_result_pos = (activation_col*w_pos_col) / prec
                    mult_result_neg = (activation_col*w_neg_col) / prec
                    
                    _, pos_cur, neg_cur = acc_init(mult_result_pos, mult_result_neg)
                    for k in range(bit_length):
                        _, pos_bit, neg_bit, _, pos_cur, neg_cur = acc_cont(None, pos_cur, neg_cur, device, k, prec)
                        result_pos_temp = pos_bit.sum(-2).sign()
                        result_neg_temp = neg_bit.sum(-2).sign()
                        result_pos_temp = F.fold(result_pos_temp, (i_y-f_y+1, i_y), (1,1))
                        result_neg_temp = F.fold(result_neg_temp, (i_y-f_y+1, i_y), (1,1))
                        for m in range(f_y):
                            result_pos = result_pos + result_pos_temp[:,m*cout:(m+1)*cout,...,m:i_y-f_y+m+1]
                            result_neg = result_neg + result_neg_temp[:,m*cout:(m+1)*cout,...,m:i_y-f_y+m+1]
                else:
                    if share:
                        if global_share_more:
                            a_size = list(activation.size()[-1:])
                            w_size = list(w_pos_split.size()[-1:])
                        elif global_share_max:
                            a_size = [1]
                            w_size = [1]
                        else:
                            a_size = list(activation.size()[-3:])
                            w_size = list(w_pos_split.size()[-3:])
                            if global_share_col:
                                a_size[-1] = 1

                    else:
                        a_size = activation.size()
                        w_size = w_pos_split.size()

                    if generator=='lfsr':
                        rand_input, rand_weight_pos, rand_weight_neg = lfsr_init(w_size, a_size, device, prec)
                    elif generator=='sync':
                        seed, input_pos, weight_pos_pos, weight_neg_pos = sync_init(w_size, a_size, device, prec)
                    elif generator=='acc':
                        input_cur, weight_pos_cur, weight_neg_cur = acc_init(w_pos_split, w_neg_split, input_split)
                    else:
                        rand_input, rand_weight_pos, rand_weight_neg = rand_init(w_size, a_size, device, prec)
                        
                    for k in range(bit_length):
                        if generator=='acc':
                            a_bit, w_pos_bit, w_neg_bit, input_cur, weight_pos_cur, weight_neg_cur = acc_cont(input_cur, weight_pos_cur, weight_neg_cur, device, k, prec)
                        else:
                            if generator=='lfsr':
                                rand_input, rand_weight_pos, rand_weight_neg = lfsr_cont(rand_input, rand_weight_pos, rand_weight_neg, bit_length=bit_length)
                            elif generator=='sync':
                                rand_input, rand_weight_pos, rand_weight_neg, input_pos, weight_pos_pos, weight_neg_pos = sync_cont(seed, input_pos, weight_pos_pos, weight_neg_pos, prec)
                            else:
                                rand_input, rand_weight_pos, rand_weight_neg = rand_init(w_size, a_size, device, prec)
                            w_pos_bit = (w_pos_split > rand_weight_pos).to(compute_type)
                            w_neg_bit = (w_neg_split > rand_weight_neg).to(compute_type)
                            a_bit = (input_split > rand_input).to(compute_type)

                        result_pos_temp = F.conv2d(a_bit, w_pos_bit, stride=stride)
                        result_neg_temp = F.conv2d(a_bit, w_neg_bit, stride=stride)
                        if not forward=='full_bin':
                            result_pos_temp = result_pos_temp.sign()
                            result_neg_temp = result_neg_temp.sign()

                        if forward == '1d_bin':
                            for m in range(f_y):
                                result_pos = result_pos + result_pos_temp[:,m*cout:(m+1)*cout,...,m:i_y-f_y+m+1]
                                result_neg = result_neg + result_neg_temp[:,m*cout:(m+1)*cout,...,m:i_y-f_y+m+1]
                        elif forward == '2d_bin':
                            for l in range(f_x):
                                for m in range(f_y):
                                    index = l*f_y + m
                                    result_pos = result_pos + result_pos_temp[:,index*cout:(index+1)*cout, l:i_x-f_x+l+1, m:i_y-f_y+m+1] 
                                    result_neg = result_neg + result_neg_temp[:,index*cout:(index+1)*cout, l:i_x-f_x+l+1, m:i_y-f_y+m+1]
                        elif (forward=='full_or') or (forward=='full_bin'):
                            result_pos = result_pos + result_pos_temp
                            result_neg = result_neg + result_neg_temp

                        if use_bn and global_usebn:
                            result_adj = (result_pos - result_neg) * bn_sign_ext
                            len_save = bit_length - (k+1)
                            result_cut = result_adj < relu_thres_adj
                            try:
                                result_ind_cur = result_ind[result_cut]
                            except:
                                result_ind = torch.zeros_like(result_pos)
                                result_ind[result_cut] = len_save
                            else:
                                result_ind[result_cut] = torch.max(torch.ones_like(result_ind_cur)*len_save, result_ind_cur)
                                
    w_pos = weight.clamp(0,100)
    w_neg = -(weight.clamp(-100,0))
    activation = activation.to(w_pos.dtype)
    
    if forward == 'full_or':
        result_pos_value = conv2d_or_approx(activation, w_pos, (0,0), stride)
        result_neg_value = conv2d_or_approx(activation, w_neg, (0,0), stride)
        bn = nn.Identity()
    elif forward == 'full_bin':
        result_pos_value = F.conv2d(activation, w_pos, padding=(0,0), stride=stride)
        result_neg_value = F.conv2d(activation, w_neg, padding=(0,0), stride=stride)
    elif forward == '2d_bin':
        result_pos_value, result_neg_value = conv2d_or_bin_2d(activation, w_pos, w_neg, (0,0), stride)
    elif (forward == '1d_bin') or (forward == 'bin_mult'):
        result_pos_value, result_neg_value = conv2d_or_bin_1d(activation, w_pos, w_neg, (0,0), stride)
    elif forward == 'z_bin':
        result_pos_value, result_neg_value = conv2d_or_bin_z(activation, w_pos, w_neg, (0,0), stride)
    elif forward == 'yz_bin':
        result_pos_value, result_neg_value = conv2d_or_bin_yz(activation, w_pos, w_neg, (0,0), stride, z_unit)
        
    if sync and (pool is None):
        result_pos_value.data = result_pos / bit_length
        result_neg_value.data = result_neg / bit_length
    
    if relu:
        bn_res = F.relu(bn(result_pos_value - result_neg_value))
    else:
        bn_res = bn(result_pos_value - result_neg_value)
    if use_bn and global_usebn:
        result_ind_cut = result_ind>0
        bn_res.data[result_ind_cut] = 0
    return bn_res
